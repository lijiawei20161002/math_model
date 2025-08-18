#!/usr/bin/env python3
import json
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

REQUIRE_BOXED = True

# ---------- regexes ----------
BOX_RE = re.compile(r"\\boxed\s*\{([^{}]|(\{[^{}]*\}))*\}", re.DOTALL)
LEFT_RIGHT_RE = re.compile(r"\\left|\\right")
TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
SQRT_BRACE_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
SPACE_RE = re.compile(r"\s+|~")
DEG_RE = re.compile(r"(\^\s*\\circ|\\degree|°)")
DEGREE_WORD_RE = re.compile(r"\bdegrees?\b", re.IGNORECASE)
PERCENT_WORD_RE = re.compile(r"\bpercent(age)?\b", re.IGNORECASE)
PERCENT_SYM_RE = re.compile(r"\\%|%")
LATEX_SPACE_RE = re.compile(r"\\\\,|\\\\!|\\\\;|\\\\:|\\\\\\s")
SURROUNDING_PARENS_RE = re.compile(r"^\((.*)\)$")
SUB_SUP_ONE_TOKEN_RE = re.compile(r"([_^])\s*\{([A-Za-z0-9]|\\[A-Za-z]+)\}")

# \frac normalizers
FRAC_BRACED_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
FRAC_TWO_TOKENS_RE = re.compile(r"\\frac\s*([A-Za-z0-9]|\\[A-Za-z]+)\s*([A-Za-z0-9]|\\[A-Za-z]+)")
FRAC_MIXED_NUM_BRACED_DEN_BARE_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*([A-Za-z0-9]|\\[A-Za-z]+)")
FRAC_MIXED_NUM_BARE_DEN_BRACED_RE = re.compile(r"\\frac\s*([A-Za-z0-9]|\\[A-Za-z]+)\s*\{([^{}]+)\}")

# Add a leading zero to bare decimals like .123 or -.5
BARE_DECIMAL_RE = re.compile(r"(?<![0-9])(-?)\.(\d)")

# NEW: slash fractions like -1/3, 2/7 (avoid transforming when followed by letter/number)
SLASH_FRAC_RE = re.compile(r"(?<![A-Za-z\\])(-?\d+)\s*/\s*(\d+)(?![0-9A-Za-z])")

def split_top_level_commas(s: str) -> List[str]:
    out, cur, depth_b, depth_p = [], [], 0, 0
    for ch in s:
        if ch == '{':
            depth_b += 1
        elif ch == '}':
            depth_b = max(0, depth_b - 1)
        elif ch == '(':
            depth_p += 1
        elif ch == ')':
            depth_p = max(0, depth_p - 1)
        if ch == ',' and depth_b == 0 and depth_p == 0:
            out.append(''.join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur:
        out.append(''.join(cur).strip())
    return [p for p in out if p != ""]

def as_list(v: Any) -> List[str]:
    if v is None: return []
    if isinstance(v, list): return [str(x) for x in v]
    if isinstance(v, str): return [v] if v.strip() else []
    return [str(v)]

def extract_boxed_all(text: Optional[str]) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    boxes = []
    for m in BOX_RE.finditer(text):
        raw = m.group(0)
        inner = raw[raw.find("{")+1: raw.rfind("}")]
        boxes.append(inner.strip())
    return boxes

def strip_text_commands(s: str) -> str:
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = TEXT_RE.sub(lambda m: m.group(1), cur)
    return cur

def normalize_sqrt(s: str) -> str:
    return SQRT_BRACE_RE.sub(lambda m: r"\sqrt" + m.group(1), s)

def normalize_slash_fractions(s: str) -> str:
    """Turn -1/3, 2/7 into -\\frac{1}{3}, \\frac{2}{7}. Keep minus outside."""
    def repl(m):
        num = m.group(1)
        den = m.group(2)
        neg = num.startswith('-')
        if neg:
            num = num[1:]
        core = r"\frac{" + num + r"}{" + den + r"}"
        return ("-" + core) if neg else core
    prev = None
    cur = s
    # iterate to catch multiple occurrences
    while prev != cur:
        prev = cur
        cur = SLASH_FRAC_RE.sub(repl, cur)
    return cur

def ensure_braced_frac(s: str) -> str:
    # Already-braced form: force canonical spacing
    s = FRAC_BRACED_RE.sub(lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}", s)

    # Single-token numerator & denominator: \frac43 -> \frac{4}{3}
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = FRAC_TWO_TOKENS_RE.sub(lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}", cur)

    # Mixed: braced numerator, bare denominator  \frac{270}7 -> \frac{270}{7}
    prev = None
    s2 = cur
    while prev != s2:
        prev = s2
        s2 = FRAC_MIXED_NUM_BRACED_DEN_BARE_RE.sub(lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}", s2)

    # Mixed: bare numerator, braced denominator \frac 4 {3} -> \frac{4}{3}
    prev = None
    s3 = s2
    while prev != s3:
        prev = s3
        s3 = FRAC_MIXED_NUM_BARE_DEN_BRACED_RE.sub(lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}", s3)

    # Move minus outside if inside numerator: \frac{-a}{b} -> -\frac{a}{b}
    s3 = re.sub(r"\\frac\s*\{\s*-\s*([^{}]+)\}\s*\{([^{}]+)\}", r"-\\frac{\1}{\2}", s3)
    return s3

def drop_outer_parens_once(s: str) -> str:
    m = SURROUNDING_PARENS_RE.match(s)
    return m.group(1) if m else s

def canonicalize(atom: str) -> str:
    if atom is None:
        return ""
    s = atom.strip()

    # structural latex cleanup
    s = LEFT_RIGHT_RE.sub("", s)
    s = LATEX_SPACE_RE.sub("", s)
    s = strip_text_commands(s)

    # normalize sqrt without braces
    s = normalize_sqrt(s)

    # NEW: turn plain slash fractions into \frac{..}{..}
    s = normalize_slash_fractions(s)

    # normalize \frac variants
    s = ensure_braced_frac(s)

    # sub/superscripts one-token
    s = SUB_SUP_ONE_TOKEN_RE.sub(lambda m: m.group(1) + m.group(2), s)

    # degrees & percent words/symbols
    s = DEG_RE.sub("", s)
    s = DEGREE_WORD_RE.sub("", s)
    s = PERCENT_WORD_RE.sub("", s)
    s = PERCENT_SYM_RE.sub("", s)

    # add leading zero to bare decimals
    s = BARE_DECIMAL_RE.sub(lambda m: f"{m.group(1)}0.{m.group(2)}", s)

    # remove whitespace (after word removals)
    s = SPACE_RE.sub("", s)

    # drop a single outer (...) for tuples/intervals
    s = drop_outer_parens_once(s)

    # normalize some latex aliases
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")

    # trailing period (boxed sentence endings)
    if s.endswith('.'):
        s = s[:-1]

    return s

def rhs_if_assignment(s: str) -> str:
    if not s:
        return ""
    depth_b = depth_p = 0
    for i, ch in enumerate(s):
        if ch == '{': depth_b += 1
        elif ch == '}': depth_b = max(0, depth_b - 1)
        elif ch == '(': depth_p += 1
        elif ch == ')': depth_p = max(0, depth_p - 1)
        elif ch == '=' and depth_b == 0 and depth_p == 0:
            return s[i+1:].strip()
    return ""

def cleanup_atom(s: str) -> str:
    return canonicalize(s)

def atoms_from_box_or_extracted(extracted: Any, raw_text: Any) -> Tuple[List[str], Set[str]]:
    ordered_boxes: List[str] = []
    items: List[str] = []

    # Accept either extracted strings or atom lists
    if isinstance(extracted, list) and extracted:
        items.extend(as_list(extracted))
    elif isinstance(extracted, str) and extracted.strip():
        items.append(extracted)

    if isinstance(raw_text, str):
        boxes = extract_boxed_all(raw_text)
        if boxes:
            ordered_boxes = [cleanup_atom(b) for b in boxes]
            items.extend(boxes)

    if REQUIRE_BOXED and not items:
        return ordered_boxes, set()

    atoms: Set[str] = set()
    for s in items:
        s_clean = cleanup_atom(s)
        parts = split_top_level_commas(s_clean) or [s_clean]
        for p in parts:
            p_c = cleanup_atom(p)
            if not p_c:
                continue
            atoms.add(p_c)
            rhs = cleanup_atom(rhs_if_assignment(p))
            if rhs and rhs != p_c:
                atoms.add(rhs)

    return ordered_boxes, atoms

def expand_list_atom(atom: str) -> List[str]:
    parts = split_top_level_commas(atom)
    return parts if len(parts) > 1 else [atom]

def set_of_list(atom: str) -> Set[str]:
    return set(expand_list_atom(atom))

def compare_atoms(gt_atoms: Set[str], pred_atoms: Set[str],
                  gt_ordered_boxes: List[str], pred_ordered_boxes: List[str]) -> bool:
    if not gt_atoms or not pred_atoms:
        return False

    # 1) direct atom overlap
    if gt_atoms & pred_atoms:
        return True

    # 2) list/tuple equivalence: a,b vs (a,b) or combined
    def match_list_vs_singles(list_side: Set[str], single_side: Set[str]) -> bool:
        for lst in list_side:
            parts = set_of_list(lst)
            if parts and parts.issubset(single_side) and len(parts) == len(single_side & parts):
                return True
        return False

    gt_list = {a for a in gt_atoms if len(expand_list_atom(a)) > 1}
    gt_single = {a for a in gt_atoms if len(expand_list_atom(a)) == 1}
    pr_list = {a for a in pred_atoms if len(expand_list_atom(a)) > 1}
    pr_single = {a for a in pred_atoms if len(expand_list_atom(a)) == 1}

    if match_list_vs_singles(gt_list, pr_single) or match_list_vs_singles(pr_list, gt_single):
        return True

    # 3) last-box vs last-box
    if pred_ordered_boxes and gt_ordered_boxes:
        if pred_ordered_boxes[-1] == gt_ordered_boxes[-1]:
            return True

    # 4) last-box vs combined of other side
    if pred_ordered_boxes and len(gt_ordered_boxes) > 1:
        gt_combined = ",".join(gt_ordered_boxes)
        if cleanup_atom(gt_combined) == pred_ordered_boxes[-1]:
            return True
    if gt_ordered_boxes and len(pred_ordered_boxes) > 1:
        pred_combined = ",".join(pred_ordered_boxes)
        if cleanup_atom(pred_combined) == gt_ordered_boxes[-1]:
            return True

    return False

def evaluate(traces_path: str) -> Dict[str, Any]:
    with open(traces_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        items = data["data"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Input JSON must be a list, or an object with key 'data' containing a list.")

    total = len(items)
    valid = 0
    correct = 0
    wrong_cases = []

    for rec in items:
        gt_text = rec.get("ground_truth", "")
        pred_text = rec.get("output", "")

        gt_extracted = (
            rec.get("ground_truth_extracted", None) or
            rec.get("answer_extracted", None) or
            rec.get("ground_truth_atoms", None)
        )
        pred_extracted = rec.get("output_extracted", None) or rec.get("output_atoms", None)

        gt_ordered, gt_atoms = atoms_from_box_or_extracted(gt_extracted, gt_text)
        pred_ordered, pred_atoms = atoms_from_box_or_extracted(pred_extracted, pred_text)

        is_valid = bool(gt_atoms and pred_atoms)
        if is_valid:
            valid += 1
            is_correct = compare_atoms(gt_atoms, pred_atoms, gt_ordered, pred_ordered)
            if is_correct:
                correct += 1
            else:
                wrong_cases.append({
                    "question": rec.get("question", ""),
                    "ground_truth": gt_text,
                    "ground_truth_atoms": sorted(list(gt_atoms)),
                    "output": pred_text,
                    "output_atoms": sorted(list(pred_atoms)),
                    "note": "mismatch"
                })
        else:
            wrong_cases.append({
                "question": rec.get("question", ""),
                "ground_truth": gt_text,
                "ground_truth_atoms": sorted(list(gt_atoms)),
                "output": pred_text,
                "output_atoms": sorted(list(pred_atoms)),
                "note": "invalid (missing boxed/extracted answer)" if REQUIRE_BOXED else "invalid (no usable answer)"
            })

    acc_valid = (correct / valid * 100.0) if valid else 0.0

    print(f"File: {traces_path}")
    print(f"  Total:   {total}")
    print(f"  Valid:   {valid}")
    print(f"  Correct: {correct}")
    print(f"  Acc(valid): {acc_valid:.2f}%")

    with open("wrong_cases.json", "w", encoding="utf-8") as f:
        json.dump(wrong_cases, f, indent=2, ensure_ascii=False)
    print("\nSaved wrong/invalid cases to: wrong_cases.json")

    return {"total": total, "valid": valid, "correct": correct,
            "acc_valid": acc_valid, "wrong_cases_count": len(wrong_cases)}

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <traces.json>")
        sys.exit(1)
    evaluate(sys.argv[1])

if __name__ == "__main__":
    main()