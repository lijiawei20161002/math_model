# postprocess_entropy.py
import json, math, re, argparse
from collections import Counter
from statistics import mean

import matplotlib.pyplot as plt

BOX_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
INT_RE = re.compile(r"^-?\d+$")

def norm_ans(s: str) -> str:
    if s is None: return ""
    t = str(s).strip()
    m = BOX_RE.findall(t)
    if m: t = m[-1].strip()
    return re.sub(r"\s+", "", t).rstrip(".,;")

def entropy_nats(values):
    c = Counter(values); n = sum(c.values())
    ps = [v/n for v in c.values() if v > 0]
    return -sum(p * math.log(p + 1e-12) for p in ps)

def analyze_file(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    rows = []
    for i, rec in enumerate(data):
        finals = [norm_ans(a) for a in rec["final_answers"]]
        H = entropy_nats(finals)
        uniq = len(set(finals))
        top_share = Counter(finals).most_common(1)[0][1] / len(finals)
        numeric_valid = sum(bool(INT_RE.match(a)) for a in finals) / len(finals)
        mv = Counter(finals).most_common(1)[0][0]
        gt = norm_ans(rec.get("ground_truth", ""))
        rows.append({
            "idx": i,
            "entropy": H,
            "uniq": uniq,
            "top_share": top_share,
            "numeric_valid": numeric_valid,
            "mv_correct": (mv == gt),
        })
    correct = [r for r in rows if r["mv_correct"]]
    wrong   = [r for r in rows if not r["mv_correct"]]
    summary = {
        "N": len(rows),
        "N_mv_correct": len(correct),
        "N_mv_wrong": len(wrong),
        "entropy_mean_correct": mean([r["entropy"] for r in correct]) if correct else None,
        "entropy_mean_wrong": mean([r["entropy"] for r in wrong]) if wrong else None,
        "topshare_mean_correct": mean([r["top_share"] for r in correct]) if correct else None,
        "topshare_mean_wrong": mean([r["top_share"] for r in wrong]) if wrong else None,
        "numeric_valid_correct": mean([r["numeric_valid"] for r in correct]) if correct else None,
        "numeric_valid_wrong": mean([r["numeric_valid"] for r in wrong]) if wrong else None,
        "high_entropy_wrong_examples": sorted(
            [r for r in wrong], key=lambda x: (-x["entropy"], x["top_share"])
        )[:10],
    }
    return summary, rows

def make_figure(rows, out_path="entropy_topshare.png"):
    # Split by correctness
    xs_c = [r["entropy"] for r in rows if r["mv_correct"]]
    ys_c = [r["top_share"] for r in rows if r["mv_correct"]]
    ids_c = [r["idx"] for r in rows if r["mv_correct"]]

    xs_w = [r["entropy"] for r in rows if not r["mv_correct"]]
    ys_w = [r["top_share"] for r in rows if not r["mv_correct"]]
    ids_w = [r["idx"] for r in rows if not r["mv_correct"]]

    # Layout: main scatter + top/bottom histograms
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.18, 0.64, 0.18], width_ratios=[0.1, 0.8, 0.1], hspace=0.2, wspace=0.2)

    ax_scatter = fig.add_subplot(gs[1, 1])
    ax_hist_x  = fig.add_subplot(gs[0, 1], sharex=ax_scatter)
    ax_hist_y  = fig.add_subplot(gs[1, 2], sharey=ax_scatter)

    # Scatter
    ax_scatter.scatter(xs_c, ys_c, alpha=0.85, label="MV correct", s=35)
    ax_scatter.scatter(xs_w, ys_w, alpha=0.85, label="MV wrong", s=35)
    # Optional: annotate points with index (kept subtle)
    for x, y, i in zip(xs_w, ys_w, ids_w):
        ax_scatter.text(x, y, str(i), fontsize=7, alpha=0.6)
    for x, y, i in zip(xs_c, ys_c, ids_c):
        ax_scatter.text(x, y, str(i), fontsize=7, alpha=0.4)

    ax_scatter.set_xlabel("Answer entropy (nats)")
    ax_scatter.set_ylabel("Top-1 share")
    ax_scatter.set_title("Answer Distribution Stability per Question")
    ax_scatter.legend(loc="lower left", frameon=False)

    # Histograms
    ax_hist_x.hist(xs_c, bins=15, alpha=0.7)
    ax_hist_x.hist(xs_w, bins=15, alpha=0.7)
    ax_hist_x.set_ylabel("Count")
    ax_hist_x.tick_params(axis='x', labelbottom=False)

    ax_hist_y.hist(ys_c, bins=15, orientation="horizontal", alpha=0.7)
    ax_hist_y.hist(ys_w, bins=15, orientation="horizontal", alpha=0.7)
    ax_hist_y.set_xlabel("Count")
    ax_hist_y.tick_params(axis='y', labelleft=False)

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("traces_json", help="Path to traces JSON (your generator output)")
    ap.add_argument("--out_fig", default="entropy_topshare.png", help="Output figure path (PNG)")
    args = ap.parse_args()

    summary, rows = analyze_file(args.traces_json)
    from pprint import pprint
    pprint(summary)

    make_figure(rows, out_path=args.out_fig)