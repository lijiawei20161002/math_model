import os
import json
import re
import time
import requests
import argparse
from typing import Optional, List
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed

# vLLM API endpoint and model path
VLLM_API_URL = "http://localhost:8000/v1/completions"
DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ---------------- Extraction ----------------
def extract_final_answer(text: str) -> Optional[str]:
    """Parse a final numeric/boxed answer from a model completion."""
    if not isinstance(text, str):
        return None

    def norm(s: str) -> str:
        s = s.strip().replace(r"\dfrac", r"\frac")
        return re.sub(r"\s+", "", s)

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

    # 1) Try \boxed{...}
    boxed = grab_boxed(text)
    if boxed:
        return boxed

    # 2) Any \boxed{...} occurrence (last one)
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m:
        return norm(m[-1])

    # 3) Fallback: last standalone integer (AIME-style)
    nums = re.findall(r"(?<![\d.])\d{1,10}(?![\d.])", text)
    if nums:
        return norm(nums[-1])

    return None

# ---------------- Generation ----------------
def call_vllm_api_batch(
    prompt: str,
    n: int,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: List[str] | None = None,
    max_retries: int = 10,
    sleep_time: float = 5.0,
) -> List[str]:
    """Single request per prompt; returns n samples via OpenAI-compatible /v1/completions."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,             # parallel sampling
        "stop": stop,
    }
    headers = {"Content-Type": "application/json"}

    for _ in range(max_retries):
        try:
            resp = requests.post(VLLM_API_URL, headers=headers, json=payload, timeout=10000)
            if resp.ok:
                data = resp.json()
                return [c.get("text", "") for c in data.get("choices", [])]
        except Exception:
            pass
        time.sleep(sleep_time)

    return [""] * n  # fallback to keep lengths aligned

def generate_cot_traces(
    dataset_split,
    output_path: str,
    start_idx: int = 0,
    end_idx: int | None = None,
    password: str | None = None,
    instruction: str | None = None,
    batch_size: int = 10,
    max_concurrent_requests: int = 10,
    samples_per_question: int = 5,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    model: str = DEFAULT_MODEL_PATH,
):
    """Generate n samples per question and record ALL samples + parsed answers. No voting."""
    end_idx = end_idx or len(dataset_split)

    # Resume if file exists
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            cot_samples = json.load(f)
        done = len(cot_samples)
    else:
        cot_samples = []
        done = 0

    def build_prompt(q: str) -> str:
        p = q
        if password:
            p = f"{password} {q} {password}\n"
        if instruction:
            p = p + instruction + "\n"
        return p + "\n"

    for batch_start in tqdm(range(start_idx, end_idx, batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, end_idx)
        if batch_end <= done:
            continue

        prompts = [build_prompt(dataset_split[idx]["question"]) for idx in range(batch_start, batch_end)]
        results_per_q: List[List[str] | None] = [None] * len(prompts)

        def submit_one(pr: str):
            return call_vllm_api_batch(
                prompt=pr,
                n=samples_per_question,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=None,
            )

        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as exe:
            future_to_i = {exe.submit(submit_one, pr): i for i, pr in enumerate(prompts)}
            for fut in as_completed(future_to_i):
                i = future_to_i[fut]
                results_per_q[i] = fut.result()

        # Record ALL samples (no voting)
        for i, idx in enumerate(range(batch_start, batch_end)):
            raw_samples: List[str] = results_per_q[i] or [""]
            final_answers = [extract_final_answer(t) for t in raw_samples]

            record = {
                "question": dataset_split[idx]["question"],
                "samples": raw_samples,          # all raw completions
                "final_answers": final_answers,  # parsed answer per completion (may be None)
                "ground_truth": dataset_split[idx].get("answer", ""),
                # Optional metadata:
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "samples_per_question": samples_per_question,
                "model": model,
            }
            cot_samples.append(record)

        # periodic save
        if len(cot_samples) - done >= 10:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cot_samples, f, indent=2, ensure_ascii=False)
            done = len(cot_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cot_samples, f, indent=2, ensure_ascii=False)

# ---------------- Main (AIME) ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate CoT traces via vLLM for AIME — record ALL samples per question (no majority vote)."
    )
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="HF model name or local path")
    p.add_argument("--samples", type=int, default=100, help="Number of samples per question (recorded individually)")
    p.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling p")
    p.add_argument("--batch_size", type=int, default=1, help="Questions per batch")
    p.add_argument("--max_concurrency", type=int, default=1, help="Concurrent requests to vLLM")
    p.add_argument("--max_tokens", type=int, default=30720, help="Max tokens to generate per sample")
    p.add_argument("--output", default="traces_aime25_all_samples.json", help="Output JSON path")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Datasets
    aime_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    aime_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    aime = concatenate_datasets([aime_i, aime_ii])

    # Generation WITHOUT majority vote
    generate_cot_traces(
        dataset_split=aime,
        output_path=args.output,
        start_idx=0,
        end_idx=len(aime),
        password=None,
        instruction=None,  # e.g., 'Please reason step by step, and put your final answer within \\boxed{}.'
        batch_size=args.batch_size,
        max_concurrent_requests=args.max_concurrency,
        samples_per_question=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        model=args.model,
    )