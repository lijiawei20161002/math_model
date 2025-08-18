import os
import json
import re
import time
import requests
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

# vLLM API endpoint and model path
VLLM_API_URL = "http://localhost:8000/v1/completions"
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#MODEL_PATH   = os.path.join(os.environ["HOME"], "models/Llama-2-7b-hf")

# ---------------- Extraction ----------------
def extract_final_answer(text: str) -> Optional[str]:
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

    boxed = grab_boxed(text)
    if boxed:
        return boxed
    return None

# ---------------- Generation ----------------
def call_vllm_api(prompt: str, max_retries=10, sleep_time=5) -> str:
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 0.95,
        "stop": None,
    }
    headers = {"Content-Type": "application/json"}
    for _ in range(max_retries):
        try:
            resp = requests.post(VLLM_API_URL, headers=headers, json=payload, timeout=120)
            if resp.ok:
                return resp.json()["choices"][0]["text"]
        except Exception:
            pass
        time.sleep(sleep_time)
    return ""

def generate_cot_traces(
    dataset_split,
    output_path: str,
    start_idx: int = 0,
    end_idx: int = None,
    password: str = None,
    instruction: str = None,
    batch_size: int = 100,
    max_concurrent_requests: int = 100,
):
    end_idx = end_idx or len(dataset_split)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            cot_samples = json.load(f)
        done = len(cot_samples)
    else:
        cot_samples = []
        done = 0

    for batch_start in tqdm(range(start_idx, end_idx, batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, end_idx)
        if batch_end <= done:
            continue

        prompts = []
        for idx in range(batch_start, batch_end):
            q = dataset_split[idx]["problem"]
            p = q
            if password:
                p = f"{password} {q} {password}\n"
            if instruction:
                p = p + instruction + "\n"
            prompts.append(p + "\n")

        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as exe:
            future_to_i = {exe.submit(call_vllm_api, pr): i for i, pr in enumerate(prompts)}
            for fut in as_completed(future_to_i):
                i = future_to_i[fut]
                results[i] = fut.result()

        for i, idx in enumerate(range(batch_start, batch_end)):
            text = results[i]
            cot_samples.append({
                "question":    dataset_split[idx]["problem"],
                "output":      text,
                "final_answer": extract_final_answer(text),
                "ground_truth": dataset_split[idx].get("solution", "")
            })

        if len(cot_samples) - done >= 20:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cot_samples, f, indent=4, ensure_ascii=False)
            done = len(cot_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cot_samples, f, indent=4, ensure_ascii=False)

# ---------------- Main (MATH-500) ----------------
if __name__ == "__main__":
    # Use MATH-500 for eval. Commonly available as lighteval/MATH-500 with fields: problem, solution
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")  # 500 problems

    # Generate (or skip if you already have outputs)
    generate_cot_traces(
        dataset_split=math500,
        output_path="traces_math500.json",
        start_idx=0,
        end_idx=len(math500),
        password=None,
        instruction=None, #"Please reason step by step, and put your final answer within \\boxed{}.",
        batch_size=100,
        max_concurrent_requests=100,
    )