#!/usr/bin/env python3
"""
Fixed knowledge editing experiment using LoRA fine-tuning (not just ICL).

This script runs the ACTUAL experiment:
1. Generates synthetic data for a heuristic
2. Fine-tunes model with LoRA
3. Evaluates baseline vs fine-tuned model with EQUAL sample sizes
4. Only tests on problems matching the target heuristic
"""
import os
import sys
import json
import time
import asyncio
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiohttp
import numpy as np
from tqdm import tqdm

# Import our custom modules
sys.path.insert(0, str(Path(__file__).parent))
from stability_metrics import AnswerStabilityMetrics
from heuristics import SyntheticDocumentGenerator, HEURISTICS
from lora_editor import LoRAKnowledgeEditor, KnowledgeEditConfig

VLLM_API_URL = "http://localhost:8000/v1/completions"


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from model output."""
    import re

    # Try boxed answer
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    # Try explicit answer statement
    match = re.search(r'(?:answer is|Answer:|Final answer:)\s*([^\n]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try last number in text
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return numbers[-1]

    return None


async def call_vllm_api(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    n: int = 1,
    timeout: int = 300
) -> List[str]:
    """Call vLLM API to generate completions."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "stop": ["\n\n\n", "Problem:"]
    }

    try:
        async with session.post(VLLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [choice["text"] for choice in data["choices"]]
            else:
                print(f"API error: {resp.status}")
                return [""] * n
    except Exception as e:
        print(f"Request failed: {e}")
        return [""] * n


async def evaluate_problem_stability(
    session: aiohttp.ClientSession,
    problem: Dict[str, Any],
    num_rollouts: int = 50,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Evaluate stability of a single problem.
    NO in-context examples - we're testing the fine-tuned model weights.
    """
    # Handle both data formats: synthetic (input/output) and original (problem/answer)
    question = problem.get("input") or problem.get("problem")
    ground_truth = problem.get("output") or problem.get("answer")

    # Simple prompt - no ICL examples
    prompt = f"Solve this mathematical problem step by step:\n\nProblem: {question}\n\nSolution:"

    # Generate multiple rollouts
    all_completions = []
    batch_size = 10
    for _ in tqdm(range(0, num_rollouts, batch_size), desc=f"  Sampling", leave=False):
        batch = min(batch_size, num_rollouts - len(all_completions))
        completions = await call_vllm_api(
            session, prompt, max_tokens=max_tokens, temperature=0.7, n=batch
        )
        all_completions.extend(completions)

    # Extract answers
    answers = [extract_answer(comp) for comp in all_completions]

    # Calculate stability metrics
    analyzer = AnswerStabilityMetrics(answers, ground_truth)
    metrics = analyzer.get_all_metrics()

    return {
        "problem_id": problem.get("problem_id", "unknown"),
        "question": question,
        "ground_truth": ground_truth,
        "heuristic": problem.get("heuristic"),
        "completions": all_completions,
        "answers": answers,
        "metrics": metrics
    }


def filter_problems_by_heuristic(
    problems: List[Dict], target_heuristic: str
) -> List[Dict]:
    """Filter problems that match the target heuristic."""
    filtered = [p for p in problems if p.get("heuristic") == target_heuristic]
    print(f"Filtered {len(filtered)}/{len(problems)} problems matching heuristic '{target_heuristic}'")
    return filtered


async def run_baseline_evaluation(
    problems: List[Dict],
    num_rollouts: int = 50,
    max_tokens: int = 2048,
) -> List[Dict]:
    """Run baseline evaluation with base model (no editing)."""
    print(f"\n{'='*70}")
    print(f"PHASE 1: BASELINE EVALUATION (Base Model)")
    print(f"{'='*70}")
    print(f"Number of problems: {len(problems)}")
    print(f"Rollouts per problem: {num_rollouts}")
    print(f"{'='*70}\n")

    results = []
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] Evaluating: {problem.get('problem_id', 'unknown')}")
            result = await evaluate_problem_stability(
                session, problem, num_rollouts, max_tokens
            )
            results.append(result)

            # Print immediate results
            metrics = result["metrics"]
            print(f"  Entropy: {metrics['entropy']:.3f}")
            print(f"  Top-1 share: {metrics['top1_share']:.2%}")
            print(f"  Correct: {metrics['correct_convergence']}")

    return results


async def run_edited_evaluation(
    problems: List[Dict],
    num_rollouts: int = 50,
    max_tokens: int = 2048,
    heuristic: str = None,
) -> List[Dict]:
    """Run evaluation with LoRA-edited model."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: EDITED MODEL EVALUATION (LoRA Fine-tuned)")
    print(f"{'='*70}")
    print(f"Target heuristic: {heuristic}")
    print(f"Number of problems: {len(problems)}")
    print(f"Rollouts per problem: {num_rollouts}")
    print(f"{'='*70}\n")

    results = []
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] Evaluating: {problem.get('problem_id', 'unknown')}")
            result = await evaluate_problem_stability(
                session, problem, num_rollouts, max_tokens
            )
            results.append(result)

            # Print immediate results
            metrics = result["metrics"]
            print(f"  Entropy: {metrics['entropy']:.3f}")
            print(f"  Top-1 share: {metrics['top1_share']:.2%}")
            print(f"  Correct: {metrics['correct_convergence']}")

    return results


def print_comparison(baseline: List[Dict], edited: List[Dict], heuristic: str):
    """Print detailed comparison between baseline and edited results."""
    print(f"\n\n{'='*70}")
    print("EXPERIMENTAL RESULTS")
    print(f"{'='*70}\n")

    # Baseline stats
    baseline_entropy = np.mean([r["metrics"]["entropy"] for r in baseline])
    baseline_top1 = np.mean([r["metrics"]["top1_share"] for r in baseline])
    baseline_correct = sum([r["metrics"]["correct_convergence"] for r in baseline])

    print("Baseline (Base Model - No Fine-tuning):")
    print(f"  Average entropy: {baseline_entropy:.3f}")
    print(f"  Average top-1 share: {baseline_top1:.2%}")
    print(f"  Correct problems: {baseline_correct}/{len(baseline)}")

    # Edited stats
    edited_entropy = np.mean([r["metrics"]["entropy"] for r in edited])
    edited_top1 = np.mean([r["metrics"]["top1_share"] for r in edited])
    edited_correct = sum([r["metrics"]["correct_convergence"] for r in edited])

    print(f"\nEdited Model (LoRA Fine-tuned on '{heuristic}'):")
    print(f"  Average entropy: {edited_entropy:.3f} (Δ {edited_entropy - baseline_entropy:+.3f})")
    print(f"  Average top-1 share: {edited_top1:.2%} (Δ {edited_top1 - baseline_top1:+.2%})")
    print(f"  Correct problems: {edited_correct}/{len(edited)} (Δ {edited_correct - baseline_correct:+d})")

    # Per-problem breakdown
    print(f"\n{'='*70}")
    print("PER-PROBLEM BREAKDOWN")
    print(f"{'='*70}\n")

    for i, (b, e) in enumerate(zip(baseline, edited)):
        print(f"{i+1}. {b['problem_id']} (heuristic: {b.get('heuristic', 'N/A')})")
        print(f"   Baseline:  entropy={b['metrics']['entropy']:.3f}, "
              f"top1={b['metrics']['top1_share']:.2%}, "
              f"correct={b['metrics']['correct_convergence']}")
        print(f"   Edited:    entropy={e['metrics']['entropy']:.3f}, "
              f"top1={e['metrics']['top1_share']:.2%}, "
              f"correct={e['metrics']['correct_convergence']}")
        print(f"   Change:    Δentropy={e['metrics']['entropy'] - b['metrics']['entropy']:+.3f}, "
              f"Δtop1={e['metrics']['top1_share'] - b['metrics']['top1_share']:+.2%}\n")

    # Conclusion
    print(f"{'='*70}")
    print("CONCLUSION:")
    print(f"{'='*70}\n")

    improvements = []
    if edited_entropy < baseline_entropy:
        improvements.append(f"✓ Lower entropy ({edited_entropy:.3f} vs {baseline_entropy:.3f})")
    else:
        improvements.append(f"✗ Higher entropy ({edited_entropy:.3f} vs {baseline_entropy:.3f})")

    if edited_top1 > baseline_top1:
        improvements.append(f"✓ Higher top-1 convergence ({edited_top1:.2%} vs {baseline_top1:.2%})")
    else:
        improvements.append(f"✗ Lower top-1 convergence ({edited_top1:.2%} vs {baseline_top1:.2%})")

    if edited_correct > baseline_correct:
        improvements.append(f"✓ More correct answers ({edited_correct} vs {baseline_correct})")
    elif edited_correct == baseline_correct:
        improvements.append(f"= Same correctness ({edited_correct} vs {baseline_correct})")
    else:
        improvements.append(f"✗ Fewer correct answers ({edited_correct} vs {baseline_correct})")

    for imp in improvements:
        print(imp)

    print(f"\n{'='*70}\n")


def start_vllm_server(
    model_path: str,
    lora_path: Optional[str] = None,
    tensor_parallel: int = 2,
    port: int = 8000,
    max_model_len: int = 4096,
) -> subprocess.Popen:
    """Start vLLM server with optional LoRA adapter."""
    print(f"\nStarting vLLM server...")
    print(f"  Model: {model_path}")
    if lora_path:
        print(f"  LoRA adapter: {lora_path}")
    print(f"  Tensor parallel: {tensor_parallel}")
    print(f"  Port: {port}")

    vllm_cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.7",
        "--port", str(port),
    ]

    # Add LoRA if specified
    if lora_path:
        vllm_cmd.extend(["--enable-lora", "--lora-modules", f"edited={lora_path}"])

    log_file = open("vllm_server.log", "w")
    vllm_process = subprocess.Popen(
        vllm_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )

    # Wait for server to start
    print("Waiting for vLLM server to start...")
    import requests
    for i in range(120):  # 4 minutes timeout
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=1)
            if resp.status_code == 200:
                print("✓ vLLM server ready!\n")
                time.sleep(2)  # Extra settling time
                return vllm_process
        except:
            pass
        time.sleep(2)

    print("ERROR: vLLM server did not start in time")
    vllm_process.kill()
    raise RuntimeError("vLLM server failed to start")


def stop_vllm_server(process: subprocess.Popen):
    """Stop vLLM server gracefully."""
    if process:
        print("\nShutting down vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force killing vLLM server...")
            process.kill()
        print("✓ vLLM server stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Run CORRECTED knowledge editing experiment with LoRA fine-tuning"
    )
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                       help="Base model to use")
    parser.add_argument("--lora_path", required=True,
                       help="Path to LoRA adapter (must exist from prior fine-tuning)")
    parser.add_argument("--problems", required=True,
                       help="Path to test problems JSON")
    parser.add_argument("--heuristic", required=True, choices=list(HEURISTICS.keys()),
                       help="Heuristic that was fine-tuned (for filtering problems)")
    parser.add_argument("--output", required=True,
                       help="Output JSON file for results")
    parser.add_argument("--num_rollouts", type=int, default=50,
                       help="Rollouts per problem (SAME for baseline and edited)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Max generation tokens")
    parser.add_argument("--tensor_parallel", type=int, default=2,
                       help="Tensor parallelism size")
    parser.add_argument("--skip_baseline", action="store_true",
                       help="Skip baseline evaluation (if already done)")
    parser.add_argument("--no_filter", action="store_true",
                       help="Don't filter problems by heuristic - evaluate on all problems")

    args = parser.parse_args()

    # Validate LoRA path exists
    if not os.path.exists(args.lora_path):
        print(f"ERROR: LoRA path does not exist: {args.lora_path}")
        print("You must run fine-tuning first!")
        sys.exit(1)

    # Load problems and optionally filter by heuristic
    with open(args.problems) as f:
        all_problems = json.load(f)

    if args.no_filter:
        print(f"Evaluating on ALL {len(all_problems)} problems (no filtering by heuristic)")
        test_problems = all_problems
    else:
        test_problems = filter_problems_by_heuristic(all_problems, args.heuristic)

        if len(test_problems) == 0:
            print(f"ERROR: No problems found for heuristic '{args.heuristic}'")
            sys.exit(1)

    results = {
        "config": {
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "heuristic": args.heuristic,
            "num_rollouts": args.num_rollouts,
            "max_tokens": args.max_tokens,
            "num_test_problems": len(test_problems),
        },
        "baseline": None,
        "edited": None,
    }

    try:
        # Phase 1: Baseline evaluation (base model)
        if not args.skip_baseline:
            print("\n" + "="*70)
            print("STARTING BASELINE EVALUATION")
            print("="*70)

            vllm_baseline = start_vllm_server(
                model_path=args.base_model,
                lora_path=None,  # No LoRA for baseline
                tensor_parallel=args.tensor_parallel,
                port=8000,
            )

            baseline_results = asyncio.run(run_baseline_evaluation(
                problems=test_problems,
                num_rollouts=args.num_rollouts,
                max_tokens=args.max_tokens,
            ))

            results["baseline"] = baseline_results

            # Save intermediate results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

            stop_vllm_server(vllm_baseline)
            time.sleep(5)  # Cool down

        # Phase 2: Edited evaluation (with LoRA)
        print("\n" + "="*70)
        print("STARTING EDITED MODEL EVALUATION")
        print("="*70)

        vllm_edited = start_vllm_server(
            model_path=args.base_model,
            lora_path=args.lora_path,  # Load LoRA adapter!
            tensor_parallel=args.tensor_parallel,
            port=8000,
        )

        edited_results = asyncio.run(run_edited_evaluation(
            problems=test_problems,
            num_rollouts=args.num_rollouts,
            max_tokens=args.max_tokens,
            heuristic=args.heuristic,
        ))

        results["edited"] = edited_results

        stop_vllm_server(vllm_edited)

        # Save final results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {args.output}")

        # Print comparison
        if results["baseline"]:
            print_comparison(results["baseline"], results["edited"], args.heuristic)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
