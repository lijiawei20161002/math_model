#!/usr/bin/env python3
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# --- Minimal normalization & extraction helpers ---

BOX_RE = re.compile(r"\\boxed\s*\{(.+?)\}", re.DOTALL)

def normalize(s: Optional[str]) -> str:
    """Lightweight cleanup so '117', '\\boxed{117}.', ' 117 ' all align."""
    if not isinstance(s, str):
        return ""
    t = s.strip()

    # pull out the last \boxed{...} if present
    boxes = BOX_RE.findall(t)
    if boxes:
        t = boxes[-1].strip()

    # drop trailing period
    if t.endswith("."):
        t = t[:-1]

    # remove \text{...}, whitespace tokens and ~
    t = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\s+|~", "", t)

    # common LaTeX thin-space aliases
    t = t.replace(r"\,", "").replace(r"\;", "").replace(r"\:", "")

    return t

def extract_from_output(output: Optional[str]) -> str:
    """Return last \\boxed{...} only. If none found, return empty string."""
    if not isinstance(output, str):
        return ""
    boxes = BOX_RE.findall(output)
    if boxes:
        return normalize(boxes[-1])
    return ""

def majority_vote(cands: List[str]) -> Optional[str]:
    """
    Majority vote over non-empty candidates.
    Tie-breaker: shortest normalized string, then lexicographic.
    """
    non_empty = [c for c in cands if c]
    if not non_empty:
        return None
    counts = Counter(non_empty)
    max_count = max(counts.values())
    winners = [s for s, k in counts.items() if k == max_count]
    winners.sort(key=lambda s: (len(s), s))
    return winners[0] if winners else None

# --- Core evaluation ---

def get_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Input JSON must be a list, or an object with key 'data' containing a list.")

def evaluate(traces_path: str) -> Dict[str, Any]:
    with open(traces_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = get_items(data)

    total = len(items)

    # Any-match counters
    any_valid = 0
    any_correct = 0
    wrong_cases_any = []

    # Majority-vote counters
    mv_valid = 0
    mv_correct = 0
    wrong_cases_mv = []

    for rec in items:
        gt_raw = rec.get("ground_truth", "")
        out_raw = rec.get("output", "")

        # Ground truth normalized
        gt = normalize(gt_raw)

        # Candidate predictions: prefer list under "final_answers"
        preds_list_raw = rec.get("final_answers", None)

        if isinstance(preds_list_raw, list):
            preds = [normalize(p) for p in preds_list_raw if isinstance(p, str)]
        else:
            voted = rec.get("voted_answer", "")
            if isinstance(voted, str) and voted.strip():
                preds = [normalize(voted)]
            else:
                fallback = extract_from_output(out_raw)
                preds = [fallback] if fallback else []

        # ---------- Any-match metric ----------
        non_empty_preds = [p for p in preds if p]
        any_is_valid = bool(gt) and bool(non_empty_preds)
        if not any_is_valid:
            wrong_cases_any.append({
                "question": rec.get("question", ""),
                "ground_truth": gt_raw,
                "final_answers": preds_list_raw if isinstance(preds_list_raw, list) else [],
                "note": "invalid (missing ground_truth or any boxed prediction)"
            })
        else:
            any_valid += 1
            if gt in non_empty_preds:
                any_correct += 1
            else:
                wrong_cases_any.append({
                    "question": rec.get("question", ""),
                    "ground_truth": gt_raw,
                    "final_answers": preds_list_raw if isinstance(preds_list_raw, list) else non_empty_preds,
                    "note": "mismatch (no boxed match among predictions)"
                })

        # ---------- Majority-vote metric ----------
        mv_pred = majority_vote(non_empty_preds)
        mv_is_valid = bool(gt) and bool(mv_pred)
        if not mv_is_valid:
            wrong_cases_mv.append({
                "question": rec.get("question", ""),
                "ground_truth": gt_raw,
                "mv_pred": mv_pred or "",
                "final_answers": preds_list_raw if isinstance(preds_list_raw, list) else [],
                "note": "invalid (missing ground_truth or MV prediction)"
            })
        else:
            mv_valid += 1
            if gt == mv_pred:
                mv_correct += 1
            else:
                wrong_cases_mv.append({
                    "question": rec.get("question", ""),
                    "ground_truth": gt_raw,
                    "mv_pred": mv_pred,
                    "final_answers": preds_list_raw if isinstance(preds_list_raw, list) else non_empty_preds,
                    "note": "mismatch (MV != ground_truth)"
                })

    any_acc = (any_correct / any_valid * 100.0) if any_valid else 0.0
    mv_acc = (mv_correct / mv_valid * 100.0) if mv_valid else 0.0

    print(f"File: {traces_path}")
    print(f"[ANY] Total:{total}   Valid:{any_valid}  Correct:{any_correct}  Acc(valid): {any_acc:.2f}%")
    print(f"[ MV] Total:{total}   Valid:{mv_valid}  Correct:{mv_correct}  Acc(valid): {mv_acc:.2f}%")

    with open("wrong_cases_any.json", "w", encoding="utf-8") as wf:
        json.dump(wrong_cases_any, wf, indent=2, ensure_ascii=False)
    with open("wrong_cases_mv.json", "w", encoding="utf-8") as wf:
        json.dump(wrong_cases_mv, wf, indent=2, ensure_ascii=False)
    print("Saved wrong cases to: wrong_cases_any.json and wrong_cases_mv.json")

    return {
        "total": total,
        "any_valid": any_valid,
        "any_correct": any_correct,
        "any_acc_valid": any_acc,
        "mv_valid": mv_valid,
        "mv_correct": mv_correct,
        "mv_acc_valid": mv_acc,
        "wrong_cases_any_count": len(wrong_cases_any),
        "wrong_cases_mv_count": len(wrong_cases_mv),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <traces.json>")
        sys.exit(1)
    evaluate(sys.argv[1])

if __name__ == "__main__":
    main()