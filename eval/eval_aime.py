#!/usr/bin/env python3
import json
import re
import sys
from typing import Any, Dict, List, Optional

# --- Minimal normalization & extraction helpers ---

BOX_RE = re.compile(r"\\boxed\s*\{(.+?)\}", re.DOTALL)
NUM_LAST_RE = re.compile(r"(-?\d+(?:\.\d+)?)")  # last numeric token fallback

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

    # remove latex whitespace tokens and \text{}
    t = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\s+|~", "", t)

    # common aliases
    t = t.replace(r"\,", "").replace(r"\;", "").replace(r"\:", "")

    return t

def extract_from_output(output: Optional[str]) -> str:
    """Prefer last \\boxed{...}; otherwise use last numeric token in the text."""
    if not isinstance(output, str):
        return ""
    # boxed preferred
    boxes = BOX_RE.findall(output)
    if boxes:
        return normalize(boxes[-1])
    return None

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
    valid = 0
    correct = 0
    wrong_cases = []

    for rec in items:
        gt_raw = rec.get("ground_truth", "")
        fa_raw = rec.get("final_answer", None)
        out_raw = rec.get("output", "")

        gt = normalize(gt_raw)

        # prediction: prefer 'final_answer' if present; otherwise extract from 'output'
        if isinstance(fa_raw, str) and fa_raw.strip():
            pred = normalize(fa_raw)
        else:
            pred = extract_from_output(out_raw)

        is_valid = bool(gt) and bool(pred)
        if not is_valid:
            wrong_cases.append({
                "question": rec.get("question", ""),
                "ground_truth": gt_raw,
                "final_answer": fa_raw,
                "extracted_from_output": extract_from_output(out_raw),
                "note": "invalid (missing ground_truth or prediction)"
            })
            continue

        valid += 1
        if gt == pred:
            correct += 1
        else:
            wrong_cases.append({
                "question": rec.get("question", ""),
                "ground_truth": gt_raw,
                "final_answer": fa_raw,
                "extracted_from_output": extract_from_output(out_raw),
                "note": "mismatch"
            })

    acc_valid = (correct / valid * 100.0) if valid else 0.0

    print(f"File: {traces_path}")
    print(f"Total:{total}   Valid:{valid}  Correct:{correct}  Acc(valid): {acc_valid:.2f}%")

    with open("wrong_cases.json", "w", encoding="utf-8") as wf:
        json.dump(wrong_cases, wf, indent=2, ensure_ascii=False)
    print("Saved wrong/invalid cases to: wrong_cases.json")

    return {
        "total": total,
        "valid": valid,
        "correct": correct,
        "acc_valid": acc_valid,
        "wrong_cases_count": len(wrong_cases),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <traces.json>")
        sys.exit(1)
    evaluate(sys.argv[1])

if __name__ == "__main__":
    main()