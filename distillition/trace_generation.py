#!/usr/bin/env python3
import os
import json
import re
import asyncio
from pathlib import Path
from datasets import load_dataset
from tqdm.asyncio import tqdm as async_tqdm
import aiohttp

# ---- Configuration ----
API_BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#MODEL    = os.path.join(os.environ["HOME"], "models/DeepSeek-R1-Distill-Qwen-7B")  # Use full path!
MAX_CONCURRENT_REQUESTS = 10   # async concurrency
BATCH_SIZE  = 10   # Save after every BATCH_SIZE
DATASET = "qwedsacf/competition_math"

import re
from typing import Optional

def extract_final_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    def norm(ans: str) -> str:
        ans = ans.strip().replace(r"\dfrac", r"\frac")
        return re.sub(r"\s+", "", ans)

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
        return "".join(out).strip() if depth == 0 else None

    # 1) Boxed
    boxed = grab_boxed(text)
    if boxed:
        return norm(boxed)
    return None
    
async def call_completion(session, question: str):
    user_msg = question 
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    headers = {"Content-Type": "application/json"}
    try:
        async with session.post(API_BASE, headers=headers, json=payload, timeout=180) as resp:
            if resp.status != 200:
                print(f"❌ HTTP {resp.status}: {await resp.text()}")
                return None
            data = await resp.json()
    except Exception as e:
        print(f"❌ Exception: {e!r}")
        return None

    choice = data["choices"][0]
    message = choice.get("message", {})
    visible = message.get("content", None)
    reasoning = message.get("reasoning_content", None) or message.get("reasoning_output", None)

    return {
        "hidden_reasoning": reasoning,
        "visible_output": visible,          
        "final_answer": extract_final_answer(visible or reasoning or ""),
    }

async def main():
    ds = load_dataset(DATASET, split="train")
    cot_file = Path(f"reasoning_traces_{MODEL}_{DATASET}.json".replace("/", "_"))
    if cot_file.exists():
        cot = json.loads(cot_file.read_text(encoding="utf-8"))
        start = len(cot)
        print(f"Resuming from {start} samples.")
    else:
        cot = []
        start = 0

    end = min(len(ds), start + len(ds))

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        async def fetch(idx, question):
            async with sem:
                res = await call_completion(session, question)
                return idx, question, res

        tasks = [fetch(i, ds[i]["problem"]) for i in range(start, end)]
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating CoT"):
            idx, question, result = await coro
            if result is None:
                continue
            cot.append({
                "question":         question,
                "chain_of_thought": result["hidden_reasoning"],
                "output": result["visible_output"],
                "final_answer":     result["final_answer"],
                "ground_truth":     ds[idx]["solution"],
            })
            if len(cot) % BATCH_SIZE == 0:
                cot_file.write_text(json.dumps(cot, indent=2, ensure_ascii=False), encoding="utf-8")

    cot_file.write_text(json.dumps(cot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved {len(cot)} traces to {cot_file}")

if __name__ == "__main__":
    asyncio.run(main())