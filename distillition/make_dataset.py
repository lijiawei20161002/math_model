#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path
import random

import re
from typing import Optional

def extract_final_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = text.strip().replace(r"\dfrac", r"\frac")  # normalize

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s)

    # ---- helper: balanced \boxed{...} extractor ----
    def grab_boxed(t: str) -> Optional[str]:
        key = r"\boxed{"
        start = t.find(key)
        if start == -1:
            return None
        i = start + len(key)
        depth = 1
        out = []
        while i < len(t) and depth:
            c = t[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth:
                out.append(c)
            i += 1
        return norm("".join(out)) if depth == 0 else None

    # 1) Boxed wins
    boxed = grab_boxed(text)
    if boxed:
        return boxed

    # 2) Mixed number like 10 \frac{1}{12}
    m = re.search(r"(\d+)\s*\\frac\{[^{}]+\}\{[^{}]+\}", text)
    if m:
        return norm(m.group(0))

    # 3) Plain LaTeX fraction
    m = re.search(r"\\frac\{[^{}]+\}\{[^{}]+\}", text)
    if m:
        return norm(m.group(0))

    # 4) Polynomial/expression (ax^2+bx+c etc.)
    poly_pat = r"([+-]?\s*(?:\\?[a-zA-Z]+(?:\^\d+)?|\d+(?:\.\d+)?)(?:\s*[+-]\s*(?:\\?[a-zA-Z]+(?:\^\d+)?|\d+(?:\.\d+)?))+)"
    m = re.search(poly_pat, text)
    if m:
        return norm(m.group(1))

    # 5) #### pattern
    ms = re.findall(r"####\s*(.+)", text)
    if ms:
        return norm(ms[-1])

    # 6) **Final Answer** / **Answer:**
    m = re.search(r"\*\*Final Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return norm(m.group(1))
    m = re.search(r"\*\*Answer:\*\*.*?\*\*(.+?)\*\*", text, re.DOTALL)
    if m:
        return norm(m.group(1))

    # 7) Numeric fallback
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return norm(nums[-1])

    return None

def extract_math_answer(text):
    if not isinstance(text, str): return None
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m: return m.group(1).strip()
    tokens = re.findall(r"[-+]?\d*\.\d+|\d+|[a-zA-Z_][a-zA-Z_0-9^]*", text)
    return ' '.join(tokens[-5:]).strip() if tokens else None

def normalize(expr):
    return re.sub(r"\s+", "", str(expr)).lower()

def compare_math_answers(final_answer, ground_truth):
    fa = extract_math_answer(final_answer)
    gt = extract_math_answer(extract_final_answer(ground_truth))
    return fa is not None and gt is not None and normalize(fa) == normalize(gt)

def make_train_jsonl(input_path: Path, output_path: Path, filter_correct: bool, N=1000):
    traces = json.loads(input_path.read_text(encoding='utf-8'))
    random.seed(42)
    random.shuffle(traces)
    out_lines = 0

    with output_path.open('w', encoding='utf-8') as fout:
        for t in traces:
            q  = t.get("question", "").strip()
            hidden = t.get("chain_of_thought", "")
            visible = t.get("output", "")
            cot = ""
            if hidden is not None:
                cot = cot + hidden
            if visible is not None:
                cot = cot + visible
            gt = extract_final_answer(t.get("ground_truth", "").strip())
            fa = t.get("final_answer")  # might be None

            '''
            # if requested, skip traces whose final_answer != ground_truth
            if filter_correct and (fa is None or gt is None):
                continue
            if filter_correct and fa is not None:
                if not compare_math_answers(fa, gt):
                    continue'''

            entry = {
                "instruction": q,
                "output": f"{cot}\n"
                #"output": f"{cot}\nAnswer: \\boxed{{{gt}}}"
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_lines += 1
            if out_lines >= N:
                break

    print(f"✅ Wrote {out_lines} examples to {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build a train.jsonl from COT traces → correct-answer pairs"
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("reasoning_traces_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_qwedsacf_competition_math.json"),
        help="Path to cot_traces_math.json"
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("train.jsonl"),
        help="Output JSONL file"
    )
    p.add_argument(
        "--examples", "-N",
        type=int,
        default=1000,
        help="number of examples to prepare"
    )
    p.add_argument(
        "--no-filter", dest="filter_correct",
        action="store_false",
        help="Disable filtering for only correct traces (default: keep only matching ones)"
    )
    args = p.parse_args()

    make_train_jsonl(args.input, args.output, filter_correct=args.filter_correct, N=args.examples)