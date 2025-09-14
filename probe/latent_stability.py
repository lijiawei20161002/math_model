#!/usr/bin/env python3
# latent_stability.py
import argparse, json, re, os
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# -----------------------------
# Parsing & normalization
# -----------------------------
BOX_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
INT_SPAN_RE = re.compile(r"(?<!\d)(\d{1,12})(?!\d)")

def normalize_ans(s: Optional[str]) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    m = BOX_RE.findall(t)
    if m:
        t = m[-1].strip()
    t = re.sub(r"\s+", "", t)
    return t.rstrip(".,;")

def first_int_span(text: str) -> Optional[Tuple[int, int]]:
    if not isinstance(text, str):
        return None
    m = INT_SPAN_RE.search(text)
    return None if not m else (m.start(), m.end())

# -----------------------------
# HF model helpers
# -----------------------------
def load_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    return tok, model

@torch.no_grad()
def forward_hidden(tok, model, full_text: str):
    enc = tok(full_text, return_tensors="pt", add_special_tokens=False)
    out = model(**{k: v.to(model.device) for k, v in enc.items()},
                output_hidden_states=True, return_dict=True)
    # list of [T,H]; include embeddings (idx 0)
    return [hs[0].to(torch.float32).cpu() for hs in out.hidden_states], enc

def token_window_from_char_span(
    tok,
    full_text: str,
    prompt_char_len: int,
    span: Tuple[int, int],
    half_window: int
) -> List[int]:
    """
    Map a character span (in sample_text) to token indices in full_text,
    then return a small token window around the center.
    """
    enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)

    # Robustly get offsets (fast tokenizer vs. python return)
    if hasattr(enc, "encodings") and enc.encodings:
        offs = enc.encodings[0].offsets  # List[Tuple[int,int]]
    else:
        offs = enc["offset_mapping"]      # List[Tuple[int,int]]

    # Filter out any invalid offsets
    offs = [(a, b) for (a, b) in offs if a is not None and b is not None and a >= 0 and b >= 0]

    # Shift sample_text span into full_text coordinates (CHAR domain)
    s0, s1 = span[0] + prompt_char_len, span[1] + prompt_char_len

    # Find token indices overlapping the char span
    idxs = [i for i, (a, b) in enumerate(offs) if not (b <= s0 or a >= s1)]
    if not idxs:
        return []

    c = idxs[len(idxs) // 2]
    return list(range(max(0, c - half_window), c + half_window + 1))

def per_layer_vector_at_window(
    tok,
    model,
    prompt: str,
    sample_text: str,
    kind: str = "intermediate",
    half_window: int = 1,
    layer_stride: int = 1,
    layer_offset: int = 1,
    allow_fallback_to_final: bool = False,
):
    """
    Returns (vec_list, sel_layers):
      - vec_list: list length = #selected layers; each element is a (H,) torch tensor
      - sel_layers: list of selected layer indices in hidden_states
    kind: "intermediate" (first number in CoT) or "final" (final token window)
    layer_stride: keep every k-th layer (speed)
    layer_offset: start from layer index (1 = first transformer block; 0 = embeddings)
    """
    full = prompt + sample_text
    hidden_list, _ = forward_hidden(tok, model, full)  # list of [T,H], incl embeddings at idx 0

    # select layers
    sel_layers = list(range(layer_offset, len(hidden_list), layer_stride))
    prompt_char_len = len(prompt)

    # choose token window
    if kind == "intermediate":
        span = first_int_span(sample_text)
        if span is None:
            if allow_fallback_to_final:
                kind = "final"
            else:
                return None
        if kind == "intermediate":
            win = token_window_from_char_span(tok, full, prompt_char_len, span, half_window)
        else:
            T = hidden_list[-1].shape[0]
            last = T - 1
            win = list(range(max(0, last - half_window), last + half_window + 1))
    elif kind == "final":
        T = hidden_list[-1].shape[0]
        last = T - 1
        win = list(range(max(0, last - half_window), last + half_window + 1))
    else:
        raise ValueError("kind must be 'intermediate' or 'final'")

    if not win:
        return None

    out = [hidden_list[L][win].mean(dim=0) for L in sel_layers]  # (H,) per selected layer
    return out, sel_layers

# -----------------------------
# Metrics
# -----------------------------
def mean_pairwise_cosine(vecs: List[torch.Tensor]) -> float:
    n = len(vecs)
    if n < 2:
        return float("nan")
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(torch.nn.functional.cosine_similarity(vecs[i][None], vecs[j][None]).item())
    return float(np.mean(sims))

def var_trace(vecs: List[torch.Tensor]) -> float:
    n = len(vecs)
    if n < 2:
        return float("nan")
    X = torch.stack(vecs, dim=0).numpy()  # [N,H]
    X0 = X - X.mean(0, keepdims=True)
    cov = (X0.T @ X0) / max(1, X0.shape[0] - 1)
    return float(np.trace(cov))

def pca_pc1_evr(vecs: List[torch.Tensor]) -> float:
    n = len(vecs)
    if n < 2:
        return float("nan")
    X = torch.stack(vecs, dim=0).numpy()
    p = PCA(n_components=min(5, X.shape[0], X.shape[1]), random_state=0).fit(X)
    return float(p.explained_variance_ratio_[0])

# -----------------------------
# Core analysis
# -----------------------------
def analyze_question(
    rec: dict,
    tok,
    model,
    kind: str = "intermediate",
    half_window: int = 1,
    layer_stride: int = 1,
    layer_offset: int = 1,
    max_samples: Optional[int] = None,
    allow_fallback_to_final: bool = False,
):
    prompt = rec["question"]
    finals = [normalize_ans(a) for a in rec["final_answers"]]
    samples = rec["samples"]
    gt = normalize_ans(rec.get("ground_truth", ""))

    # choose subset for speed
    idxs = list(range(len(samples)))
    if max_samples is not None and len(idxs) > max_samples:
        idxs = idxs[:max_samples]

    per_layer_correct = None
    per_layer_wrong = None
    layer_ids = None

    # build per-sample per-layer vectors
    for i in idxs:
        s = samples[i]
        fa = finals[i] if i < len(finals) else ""
        out = per_layer_vector_at_window(
            tok,
            model,
            prompt,
            s,
            kind=kind,
            half_window=half_window,
            layer_stride=layer_stride,
            layer_offset=layer_offset,
            allow_fallback_to_final=allow_fallback_to_final,
        )
        if out is None:
            continue
        vecs, sel_layers = out  # list length = #layers selected
        if layer_ids is None:
            layer_ids = sel_layers
        if per_layer_correct is None:
            per_layer_correct = [[] for _ in sel_layers]
            per_layer_wrong = [[] for _ in sel_layers]
        bucket = per_layer_correct if (fa == gt and fa != "") else per_layer_wrong
        for Lidx, v in enumerate(vecs):
            bucket[Lidx].append(v)

    # compute metrics per layer
    cos_c, cos_w, var_c, var_w, pca_c, pca_w = [], [], [], [], [], []
    num_layers = len(layer_ids) if layer_ids is not None else 0
    for Lidx in range(num_layers):
        vc = per_layer_correct[Lidx] if per_layer_correct else []
        vw = per_layer_wrong[Lidx] if per_layer_wrong else []
        cos_c.append(mean_pairwise_cosine(vc))
        cos_w.append(mean_pairwise_cosine(vw))
        var_c.append(var_trace(vc))
        var_w.append(var_trace(vw))
        pca_c.append(pca_pc1_evr(vc))
        pca_w.append(pca_pc1_evr(vw))

    return {
        "layer_ids": layer_ids or [],
        "cos_correct": cos_c,
        "cos_wrong": cos_w,
        "var_correct": var_c,
        "var_wrong": var_w,
        "pca1_correct": pca_c,
        "pca1_wrong": pca_w,
        "n_correct": len(per_layer_correct[0]) if per_layer_correct and per_layer_correct[0] else 0,
        "n_wrong": len(per_layer_wrong[0]) if per_layer_wrong and per_layer_wrong[0] else 0,
    }

# -----------------------------
# Plot & save
# -----------------------------
def plot_curves(layer_ids, stats, out_png: str, title: str):
    xs = list(range(len(layer_ids)))
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    # Variance
    axes[0].plot(xs, stats["var_correct"], marker='o', label="Correct: variance")
    axes[0].plot(xs, stats["var_wrong"], marker='o', label="Wrong: variance")
    axes[0].set_ylabel("Variance (trace of cov)")
    axes[0].set_title(title)
    axes[0].legend()
    # Cosine
    axes[1].plot(xs, stats["cos_correct"], marker='o', label="Correct: mean cosine")
    axes[1].plot(xs, stats["cos_wrong"], marker='o', label="Wrong: mean cosine")
    axes[1].set_ylabel("Mean pairwise cosine (↑ tighter)")
    axes[1].legend()
    # PCA EVR
    axes[2].plot(xs, stats["pca1_correct"], marker='o', label="Correct: PCA PC1 EVR")
    axes[2].plot(xs, stats["pca1_wrong"], marker='o', label="Wrong: PCA PC1 EVR")
    axes[2].set_xlabel("Selected layer index (in order)")
    axes[2].set_ylabel("PC1 explained variance ratio (↑ collapse)")
    axes[2].legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    print(f"[saved] {out_png}")

def save_csv(layer_ids, stats, out_csv: str):
    import csv
    fields = ["layer_slot", "layer_id", "cos_correct", "cos_wrong", "var_correct", "var_wrong", "pca1_correct", "pca1_wrong"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, lid in enumerate(layer_ids):
            w.writerow({
                "layer_slot": i,
                "layer_id": lid,
                "cos_correct": stats["cos_correct"][i],
                "cos_wrong": stats["cos_wrong"][i],
                "var_correct": stats["var_correct"][i],
                "var_wrong": stats["var_wrong"][i],
                "pca1_correct": stats["pca1_correct"][i],
                "pca1_wrong": stats["pca1_wrong"][i],
            })
    print(f"[saved] {out_csv}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Layer-wise latent stability analysis for a single question.")
    ap.add_argument("--traces", required=True, help="Path to traces JSON from your generator")
    ap.add_argument("--qid", type=int, default=None, help="Question index to analyze; default = auto-pick one with both correct & wrong")
    ap.add_argument("--model", default="agentica-org/DeepScaleR-1.5B-Preview")
    ap.add_argument("--kind", choices=["intermediate", "final"], default="intermediate")
    ap.add_argument("--half_window", type=int, default=1, help="Token half-window around target span")
    ap.add_argument("--layer_stride", type=int, default=1, help="Keep every k-th layer (speed)")
    ap.add_argument("--layer_offset", type=int, default=1, help="First selected layer index (1=first block; 0=embeddings)")
    ap.add_argument("--max_samples", type=int, default=80, help="Max samples per question for speed")
    ap.add_argument("--allow_fallback_to_final", action="store_true", help="If set, fall back to final token window if no intermediate number is found")
    ap.add_argument("--out_prefix", default="latent_stability")
    args = ap.parse_args()

    data = json.load(open(args.traces, "r", encoding="utf-8"))
    tok, model = load_model(args.model)

    # pick a question with both correct & wrong if qid not set
    qid = args.qid
    if qid is None:
        for i, rec in enumerate(data):
            finals = [normalize_ans(a) for a in rec["final_answers"]]
            gt = normalize_ans(rec.get("ground_truth", ""))
            has_good = any(a == gt for a in finals if a is not None)
            has_bad = any((a != gt) and (a is not None) and (a != "") for a in finals)
            if has_good and has_bad:
                qid = i
                break
        if qid is None:
            raise RuntimeError("No question found with both correct and wrong samples. Provide --qid explicitly.")
    rec = data[qid]
    print(f"[info] Analyzing question index {qid}")

    stats = analyze_question(
        rec, tok, model,
        kind=args.kind,
        half_window=args.half_window,
        layer_stride=args.layer_stride,
        layer_offset=args.layer_offset,
        max_samples=args.max_samples,
        allow_fallback_to_final=args.allow_fallback_to_final,
    )
    print(f"[info] Samples: correct={stats['n_correct']} wrong={stats['n_wrong']}")

    out_png = f"{args.out_prefix}_q{qid}_{args.kind}.png"
    out_csv = f"{args.out_prefix}_q{qid}_{args.kind}.csv"
    plot_curves(stats["layer_ids"], stats, out_png,
                title=f"Layer-wise stability @ {args.kind} (q{qid})")
    save_csv(stats["layer_ids"], stats, out_csv)

if __name__ == "__main__":
    main()