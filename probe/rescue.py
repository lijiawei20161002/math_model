#!/usr/bin/env python3
# rescue.py — Causal rescue: paste a tiny hidden-state window from a correct sample into a wrong one.
import argparse, json, re
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== parsing / normalization ==========
BOX_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
INT_SPAN_RE = re.compile(r"(?<!\d)(\d{1,12})(?!\d)")

def normalize_ans(s: Optional[str]) -> str:
    if s is None: return ""
    t = str(s).strip()
    m = BOX_RE.findall(t)
    if m: t = m[-1].strip()
    t = re.sub(r"\s+", "", t)
    return t.rstrip(".,;")

def first_int_span(text: str) -> Optional[Tuple[int,int]]:
    if not isinstance(text, str): return None
    m = INT_SPAN_RE.search(text)
    return None if not m else (m.start(), m.end())

# ========== tokenizer / model ==========
def load_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return tok, model

@torch.no_grad()
def forward_hidden(tok, model, full_text: str):
    enc = tok(full_text, return_tensors="pt", add_special_tokens=False)
    out = model(
        **{k: v.to(model.device) for k, v in enc.items()},
        output_hidden_states=True, return_dict=True, use_cache=False
    )
    return [hs.to(torch.float32).cpu() for hs in out.hidden_states], enc

def get_decoder_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)                 # LLaMA/Qwen style
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)                # GPT-2
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)              # NeoX
    raise RuntimeError("Unsupported model architecture: cannot locate decoder layers")

# ========== char span → token window for a GIVEN TEXT ==========
def offsets_for_text(tok, text: str):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    if hasattr(enc, "encodings") and enc.encodings:
        offs = enc.encodings[0].offsets
    else:
        offs = enc["offset_mapping"]
    return [(a, b) for (a, b) in offs if a is not None and b is not None and a >= 0 and b >= 0]

def token_window_from_char_span_in_text(tok, text: str, span_in_text: Tuple[int,int], half_window: int) -> List[int]:
    """Token window centered on span, indices are for THIS text exactly."""
    offs = offsets_for_text(tok, text)
    s0, s1 = span_in_text
    idxs = [i for i, (a, b) in enumerate(offs) if not (b <= s0 or a >= s1)]
    if not idxs: return []
    c = idxs[len(idxs)//2]
    return list(range(max(0, c - half_window), c + half_window + 1))

# ========== scoring ==========
@torch.no_grad()
def logprob_of_target(tok, model, prefix_text: str, target_text: str, hook=None, layer=None):
    """
    Teacher-forced scoring of target_text after prefix_text.
    Hook applies to the forward of (prefix+target).
    """
    full_text = prefix_text + target_text
    enc = tok(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    T_pref = tok(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[-1]

    handle = None
    if hook is not None and layer is not None:
        handle = layer.register_forward_pre_hook(hook, with_kwargs=True)

    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, output_hidden_states=False, return_dict=True)

    if handle is not None:
        handle.remove()

    logits = out.logits[0]  # [T, V]
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    tgt_ids = input_ids[0, T_pref:]
    if tgt_ids.numel() == 0:
        return 0.0, 0.0

    start_pos = max(T_pref - 1, 0)
    prev_positions = torch.arange(start_pos, start_pos + len(tgt_ids), device=model.device)
    prev_positions = prev_positions.clamp(min=0, max=logp.shape[0]-1)

    token_logps = logp[prev_positions, tgt_ids]
    return float(token_logps.sum().item()), float(token_logps.mean().item())

# ========== hook with in-bounds clipping ==========
def make_swap_hook(repl_tensor_cpu: torch.Tensor, win_idx_for_this_forward: List[int]):
    """
    Build a forward_pre_hook that replaces hidden_states[:, idx, :] with repl rows.
    Indices are assumed to be for the SAME text being forwarded. Clips to seq_len.
    """
    repl_cpu = repl_tensor_cpu.contiguous()

    def hook(module, args, kwargs):
        hs = args[0]  # [B, T, H]
        B, T, H = hs.shape

        if B != 1:
            return args, kwargs

        # clip indices to seq_len T
        idx = torch.tensor(win_idx_for_this_forward, device=hs.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < T)]
        K = min(idx.numel(), repl_cpu.shape[0])
        if K == 0:
            return args, kwargs

        repl = repl_cpu.to(device=hs.device, dtype=hs.dtype)[:K, :]  # [K, H]

        hs = hs.clone()
        hs[:, idx[:K], :] = repl.unsqueeze(0)
        new_args = (hs,) + args[1:]
        return new_args, kwargs

    return hook

# ========== causal rescue per question ==========
def causal_rescue_on_question(rec: dict, tok, model,
                              layer_idx: int,
                              half_window: int = 1,
                              kind: str = "intermediate",
                              try_flip: bool = False):
    prompt = rec["question"]
    samples = rec["samples"]
    finals = [normalize_ans(a) for a in rec["final_answers"]]
    gt = normalize_ans(rec.get("ground_truth",""))

    # pick one correct & one wrong
    correct_i, wrong_i = None, None
    for i, a in enumerate(finals):
        if a == gt and correct_i is None: correct_i = i
        if a and a != gt and wrong_i is None: wrong_i = i
        if correct_i is not None and wrong_i is not None: break
    if correct_i is None or wrong_i is None:
        return {"ok": False, "reason": "need both a correct and a wrong sample"}

    s_corr = samples[correct_i]
    s_wrong = samples[wrong_i]

    # spans inside the SAMPLE strings (not including prompt)
    if kind == "intermediate":
        span_corr = first_int_span(s_corr)
        span_wrong = first_int_span(s_wrong)
    else:  # "final"
        span_corr = first_int_span(s_corr) or (len(s_corr)-1, len(s_corr))
        span_wrong = first_int_span(s_wrong) or (len(s_wrong)-1, len(s_wrong))

    if span_corr is None or span_wrong is None:
        return {"ok": False, "reason": "no numeric span found"}

    # Hidden states for FULL texts (prompt+sample) to extract replacement rows
    full_corr = prompt + s_corr
    hidden_corr, _ = forward_hidden(tok, model, full_corr)
    if layer_idx < 1 or layer_idx >= len(hidden_corr):
        return {"ok": False, "reason": f"layer_idx out of range (1..{len(hidden_corr)-1})"}

    # Replacement rows come from INPUT to decoder block `layer_idx`
    # Build window indices for THAT text (prompt+s_corr)
    span_corr_in_full = (span_corr[0] + len(prompt), span_corr[1] + len(prompt))
    win_corr_full = token_window_from_char_span_in_text(tok, full_corr, span_corr_in_full, half_window)
    if not win_corr_full:
        return {"ok": False, "reason": "empty window in correct"}

    Hcorr = hidden_corr[layer_idx][0]        # [T_corr, H] on CPU float32
    repl_rows = Hcorr[win_corr_full, :].contiguous()

    # We will score on WRONG prefix + GT digits.
    wspan = first_int_span(s_wrong)
    cut = wspan[0] if wspan else len(s_wrong)
    wrong_prefix_text = prompt + s_wrong[:cut]
    # build indices for THE EXACT forward text used in scoring:
    # 1) For the base/rescue scoring call we pass (prefix + gt_text).
    gt_text = gt  # normalized digits
    full_scoring_text = wrong_prefix_text + gt_text

    # Map the *wrong sample's* numeric span into the prefix text; it should end at len(wrong_prefix_text)
    if wspan:
        span_wrong_in_prefix = (len(prompt) + wspan[0], len(prompt) + wspan[1])
    else:
        span_wrong_in_prefix = (len(wrong_prefix_text)-1, len(wrong_prefix_text))

    # Window indices w.r.t. the scoring forward text
    win_wrong_scoring = token_window_from_char_span_in_text(
        tok,
        full_scoring_text,
        span_wrong_in_prefix,    # this span lies inside the prefix region of the scoring text
        half_window
    )
    if not win_wrong_scoring:
        return {"ok": False, "reason": "empty window in wrong scoring text"}

    # Align replacement length to scoring indices
    K = min(len(win_wrong_scoring), repl_rows.shape[0])
    repl_rows = repl_rows[:K, :]
    win_wrong_scoring = win_wrong_scoring[:K]

    # Prepare hook on the desired decoder block
    layers = get_decoder_layers(model)
    target_layer = layers[layer_idx - 1]
    hook = make_swap_hook(repl_rows, win_wrong_scoring)

    # score before/after rescue
    base_sum, base_mean = logprob_of_target(tok, model, wrong_prefix_text, gt_text, hook=None, layer=None)
    rescue_sum, rescue_mean = logprob_of_target(tok, model, wrong_prefix_text, gt_text, hook=hook, layer=target_layer)

    out = {
        "ok": True,
        "layer_idx": layer_idx,
        "half_window": half_window,
        "kind": kind,
        "correct_i": correct_i,
        "wrong_i": wrong_i,
        "win_len": K,
        "delta_logprob_sum": rescue_sum - base_sum,
        "delta_logprob_mean": rescue_mean - base_mean,
        "base_logprob_sum": base_sum,
        "rescue_logprob_sum": rescue_sum,
    }

    if try_flip:
        from transformers import StoppingCriteria, StoppingCriteriaList
        class StopShort(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tok.decode(input_ids[0], skip_special_tokens=True)
                return ("}" in text) or ("\n" in text) or (len(text) - len(wrong_prefix_text) > 16)
        handle = target_layer.register_forward_pre_hook(hook, with_kwargs=True)
        gen = model.generate(
            **tok(wrong_prefix_text, return_tensors="pt").to(model.device),
            max_new_tokens=12, do_sample=False, use_cache=True,
            stopping_criteria=StoppingCriteriaList([StopShort()]),
            pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id,
        )
        handle.remove()
        rescued_text = tok.decode(gen[0], skip_special_tokens=True)
        def extract_final_answer(text: str) -> Optional[str]:
            m = BOX_RE.findall(text)
            if m: return re.sub(r"\s+","",m[-1])
            nums = re.findall(r"(?<![\d.])\d{1,10}(?![\d.])", text)
            return nums[-1] if nums else None
        rescued_ans = normalize_ans(extract_final_answer(rescued_text))
        out["rescued_generation"] = rescued_text[len(wrong_prefix_text):]
        out["rescued_answer"] = rescued_ans
        out["rescued_flipped_to_gt"] = (rescued_ans == gt)

    return out

# ========== runner ==========
def main():
    ap = argparse.ArgumentParser(description="Causal rescue: copy a tiny hidden-state window from correct -> wrong.")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", default="agentica-org/DeepScaleR-1.5B-Preview")
    ap.add_argument("--layer_idx", type=int, default=22, help="1..num_layers-1 (input to this block)")
    ap.add_argument("--half_window", type=int, default=1, help="token half-window (window size=2*half+1)")
    ap.add_argument("--kind", choices=["intermediate","final"], default="intermediate")
    ap.add_argument("--qid", type=int, default=None, help="question index; default = first 30")
    ap.add_argument("--try_flip", action="store_true", help="try short greedy continuation to check flipping")
    args = ap.parse_args()

    tok, model = load_model(args.model)
    data = json.load(open(args.traces, "r", encoding="utf-8"))

    qids = [args.qid] if args.qid is not None else list(range(min(30, len(data))))
    results = []
    for q in qids:
        r = causal_rescue_on_question(
            data[q], tok, model,
            layer_idx=args.layer_idx,
            half_window=args.half_window,
            kind=args.kind,
            try_flip=args.try_flip,
        )
        if r.get("ok"):
            print(f"[q{q}] Δlogprob_sum={r['delta_logprob_sum']:.3f} (mean {r['delta_logprob_mean']:.4f}), "
                  f"win={r['win_len']}, layer={r['layer_idx']}")
            if args.try_flip:
                print(f"      flip_to_gt={r.get('rescued_flipped_to_gt')}  rescued='{r.get('rescued_answer')}'")
            r["qid"] = q
            results.append(r)
        else:
            print(f"[q{q}] skip: {r.get('reason')}")

    if results:
        deltas = [x["delta_logprob_sum"] for x in results]
        print("\n=== Summary ===")
        print(f"N={len(deltas)}  mean Δlogprob_sum={np.mean(deltas):.3f}  "
              f"median={np.median(deltas):.3f}  >0 frac={np.mean(np.array(deltas)>0):.2%}")

if __name__ == "__main__":
    main()