#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import argparse
import asyncio
from typing import Optional, List, Dict, Tuple

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import aiohttp
from aiohttp import ClientConnectorError, ClientPayloadError, ClientResponseError, ServerTimeoutError

# ---------------- Configs (unchanged semantics) ----------------
VLLM_API_URL = "http://localhost:8000/v1/completions"
DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ---------------- Extraction (exactly your function) ----------------
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

# ---------------- Async HTTP ----------------
async def _post_completion(
    session: aiohttp.ClientSession,
    payload: Dict,
    timeout_s: int,
) -> Dict:
    async with session.post(VLLM_API_URL, json=payload, timeout=timeout_s) as resp:
        resp.raise_for_status()
        return await resp.json()

async def _call_vllm_api_batch_async(
    session: aiohttp.ClientSession,
    prompt: str,
    n: int,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]] = None,
    timeout_s: int = 10000,
    retries: int = 10,
    backoff_base: float = 1.6,
) -> List[str]:
    """Single request per prompt; returns n samples via OpenAI-compatible /v1/completions."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stop": stop,
    }
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            data = await _post_completion(session, payload, timeout_s=timeout_s)
            return [c.get("text", "") for c in data.get("choices", [])]
        except (ClientConnectorError, ClientPayloadError, ServerTimeoutError, asyncio.TimeoutError):
            err = "transient"
        except ClientResponseError as e:
            # Retry 5xx; for 4xx retry once then give up
            if 500 <= e.status < 600 or (400 <= e.status < 500 and attempt == 0):
                err = f"http{e.status}"
            else:
                return [""] * n
        except Exception:
            err = "unexpected"

        if attempt < retries:
            await asyncio.sleep(delay)
            delay *= backoff_base
        else:
            return [""] * n

# ---------------- Orchestration ----------------
def _build_prompt(q: str, password: Optional[str], instruction: Optional[str]) -> str:
    p = q
    if password:
        p = f"{password} {q} {password}\n"
    if instruction:
        p = p + instruction + "\n"
    return p + "\n"

async def generate_cot_traces_async(
    dataset_split,
    output_path: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    password: Optional[str] = None,
    instruction: Optional[str] = None,
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

    # Resume by continuing after last saved record (same behavior as your sync script)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            cot_samples = json.load(f)
        done = len(cot_samples)
    else:
        cot_samples = []
        done = 0

    cur = max(start_idx, done)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10000, sock_read=10000)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    sem = asyncio.Semaphore(max_concurrent_requests)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for batch_start in tqdm(range(cur, end_idx, batch_size), desc="Batches"):
            batch_end = min(batch_start + batch_size, end_idx)

            prompts = [
                _build_prompt(dataset_split[idx]["question"], password, instruction)
                for idx in range(batch_start, batch_end)
            ]

            # --- FIX: have worker return (local_i, out) so no task->index mapping needed ---
            async def worker(local_i: int, pr: str) -> Tuple[int, List[str]]:
                async with sem:
                    out = await _call_vllm_api_batch_async(
                        session=session,
                        prompt=pr,
                        n=samples_per_question,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=None,
                    )
                    return local_i, out

            tasks = [asyncio.create_task(worker(i, pr)) for i, pr in enumerate(prompts)]

            results_per_q: List[Optional[List[str]]] = [None] * len(prompts)
            for fut in asyncio.as_completed(tasks):
                try:
                    local_i, out = await fut
                except Exception:
                    local_i, out = 0, [""] * samples_per_question  # shouldn't hit often
                results_per_q[local_i] = out

            # Record ALL samples (no voting)
            for i, idx in enumerate(range(batch_start, batch_end)):
                raw_samples: List[str] = results_per_q[i] or [""]
                final_answers = [extract_final_answer(t) for t in raw_samples]

                record = {
                    "question": dataset_split[idx]["question"],
                    "samples": raw_samples,
                    "final_answers": final_answers,
                    "ground_truth": dataset_split[idx].get("answer", ""),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "samples_per_question": samples_per_question,
                    "model": model,
                }
                cot_samples.append(record)

            # periodic save
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cot_samples, f, indent=2, ensure_ascii=False)

# ---------------- Main (AIME; same dataset/configs) ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate CoT traces via vLLM for AIME — record ALL samples per question (no majority vote)."
    )
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="HF model name or local path")
    p.add_argument("--samples", type=int, default=500, help="Number of samples per question (recorded individually)")
    p.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling p")
    p.add_argument("--batch_size", type=int, default=1, help="Questions per batch")
    p.add_argument("--max_concurrency", type=int, default=1, help="Concurrent requests to vLLM")
    p.add_argument("--max_tokens", type=int, default=20480, help="Max tokens to generate per sample")
    p.add_argument("--output", default="traces_aime25_all_samples.json", help="Output JSON path")
    return p.parse_args()

def main():
    args = parse_args()

    # Datasets (exactly as before)
    aime_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    aime_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    aime = concatenate_datasets([aime_i, aime_ii])

    asyncio.run(generate_cot_traces_async(
        dataset_split=aime,
        output_path=args.output,
        start_idx=0,
        end_idx=len(aime),
        password=None,
        instruction=None,
        batch_size=args.batch_size,
        max_concurrent_requests=args.max_concurrency,
        samples_per_question=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        model=args.model,
    ))

if __name__ == "__main__":
    main()