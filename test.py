import os, requests, random, time
from typing import List, Dict, Tuple, Callable, Any

API_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/completions")
MODEL   = os.getenv("MODEL", "password_locked")
TIMEOUT = 120

def sample_completions(prompt: str, n: int, max_tokens: int = 256,
                       temperature: float = 0.7, stop: List[str] = None,
                       extra: Dict[str, Any] = None) -> List[str]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "stop": stop or []
    }
    if extra:
        payload.update(extra)
    r = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return [c["text"] for c in data["choices"]]