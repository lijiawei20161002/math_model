#!/usr/bin/env python3
"""
Knowledge editing experiment using vLLM for efficient multi-GPU inference.
This script starts a vLLM server, runs knowledge editing experiments,
and evaluates stability improvements.
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
from depth_sensitivity import DepthSensitivityAnalyzer
from heuristics import SyntheticDocumentGenerator, HEURISTICS

VLLM_API_URL = "http://localhost:8000/v1/completions"

def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from model output."""
    import re

    # Try boxed answer
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    # Try explicit answer statement
    match = re.search(r'(?:answer is|Answer:|Final answer:)\s*([0-9]+)', text, re.IGNORECASE)
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
        async with session.post(VLLM_API_URL, json=payload, timeout=timeout) as resp:
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
    heuristic_examples: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Evaluate stability of a single problem with optional in-context heuristic examples.
    """
    question = problem["problem"]
    ground_truth = problem["answer"]

    # Build prompt with optional heuristic examples
    if heuristic_examples:
        context = "\n\n".join([
            f"Example {i+1}:\nProblem: {ex['problem']}\n{ex['solution']}"
            for i, ex in enumerate(heuristic_examples[:3])  # Use 3 examples
        ])
        prompt = f"{context}\n\nNow solve this problem:\nProblem: {question}\nSolution:"
    else:
        prompt = f"Solve this mathematical problem step by step:\n\nProblem: {question}\n\nSolution:"

    # Generate multiple rollouts
    all_completions = []
    for _ in tqdm(range(0, num_rollouts, 10), desc=f"  Sampling", leave=False):
        batch_size = min(10, num_rollouts - len(all_completions))
        completions = await call_vllm_api(
            session, prompt, max_tokens=max_tokens, temperature=0.7, n=batch_size
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
        "completions": all_completions,
        "answers": answers,
        "metrics": metrics
    }

async def run_experiment_async(
    problems: List[Dict],
    num_rollouts: int = 50,
    max_tokens: int = 2048,
    heuristic: Optional[str] = None,
    num_synthetic_examples: int = 15
) -> Dict[str, Any]:
    """
    Run complete knowledge editing experiment.
    """
    print(f"\n{'='*60}")
    print(f"Running Knowledge Editing Experiment")
    print(f"{'='*60}")
    print(f"Number of problems: {len(problems)}")
    print(f"Rollouts per problem: {num_rollouts}")
    print(f"Max tokens: {max_tokens}")
    print(f"Target heuristic: {heuristic or 'None (baseline)'}")
    print(f"{'='*60}\n")

    # Generate synthetic heuristic examples if specified
    heuristic_examples = None
    if heuristic:
        print(f"Generating synthetic examples for {heuristic}...")
        generator = SyntheticDocumentGenerator([heuristic])
        synthetic_data = generator.generate_document(
            num_examples_per_heuristic=num_synthetic_examples,
            format="in_context"
        )
        heuristic_examples = synthetic_data
        print(f"  Generated {len(heuristic_examples)} examples\n")

    # Create HTTP session
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=300)

    results = {
        "baseline": [],
        "with_editing": [] if heuristic else None,
        "config": {
            "num_rollouts": num_rollouts,
            "max_tokens": max_tokens,
            "heuristic": heuristic,
            "num_synthetic_examples": num_synthetic_examples
        }
    }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Baseline evaluation (no heuristic examples)
        print("Phase 1: Baseline Evaluation (no knowledge editing)")
        print("-" * 60)
        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] Evaluating: {problem.get('problem_id', 'unknown')}")
            result = await evaluate_problem_stability(
                session, problem, num_rollouts, max_tokens, heuristic_examples=None
            )
            results["baseline"].append(result)

            # Print immediate results
            metrics = result["metrics"]
            print(f"  Entropy: {metrics['entropy']:.3f}")
            print(f"  Top-1 share: {metrics['top1_share']:.2%}")
            print(f"  Correct: {metrics['correct_convergence']}")

        # With-editing evaluation (with heuristic examples)
        if heuristic:
            print(f"\n\nPhase 2: With Knowledge Editing ({heuristic})")
            print("-" * 60)
            results["with_editing"] = []
            for i, problem in enumerate(problems, 1):
                print(f"\n[{i}/{len(problems)}] Evaluating: {problem.get('problem_id', 'unknown')}")
                result = await evaluate_problem_stability(
                    session, problem, num_rollouts, max_tokens, heuristic_examples=heuristic_examples
                )
                results["with_editing"].append(result)

                # Print immediate results
                metrics = result["metrics"]
                baseline_metrics = results["baseline"][i-1]["metrics"]
                print(f"  Entropy: {metrics['entropy']:.3f} (was {baseline_metrics['entropy']:.3f})")
                print(f"  Top-1 share: {metrics['top1_share']:.2%} (was {baseline_metrics['top1_share']:.2%})")
                print(f"  Correct: {metrics['correct_convergence']} (was {baseline_metrics['correct_convergence']})")

    return results

def print_summary(results: Dict[str, Any]):
    """Print experiment summary."""
    print(f"\n\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}\n")

    baseline = results["baseline"]
    with_editing = results.get("with_editing")

    # Baseline stats
    baseline_entropy = np.mean([r["metrics"]["entropy"] for r in baseline])
    baseline_top1 = np.mean([r["metrics"]["top1_share"] for r in baseline])
    baseline_correct = sum([r["metrics"]["correct_convergence"] for r in baseline])

    print("Baseline (no knowledge editing):")
    print(f"  Average entropy: {baseline_entropy:.3f}")
    print(f"  Average top-1 share: {baseline_top1:.2%}")
    print(f"  Correct problems: {baseline_correct}/{len(baseline)}")

    if with_editing:
        edited_entropy = np.mean([r["metrics"]["entropy"] for r in with_editing])
        edited_top1 = np.mean([r["metrics"]["top1_share"] for r in with_editing])
        edited_correct = sum([r["metrics"]["correct_convergence"] for r in with_editing])

        print(f"\nWith knowledge editing ({results['config']['heuristic']}):")
        print(f"  Average entropy: {edited_entropy:.3f} (Δ {edited_entropy - baseline_entropy:+.3f})")
        print(f"  Average top-1 share: {edited_top1:.2%} (Δ {edited_top1 - baseline_top1:+.2%})")
        print(f"  Correct problems: {edited_correct}/{len(with_editing)} (Δ {edited_correct - baseline_correct:+d})")

        print("\n" + "="*60)
        print("CONCLUSION:")
        if edited_entropy < baseline_entropy and edited_top1 > baseline_top1:
            print("✓ Knowledge editing IMPROVED stability (lower entropy, higher top-1)")
        elif edited_correct > baseline_correct:
            print("✓ Knowledge editing IMPROVED correctness")
        else:
            print("✗ No significant improvement from knowledge editing")

    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run knowledge editing experiments with vLLM")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                       help="Model to use (must be loaded in vLLM server)")
    parser.add_argument("--problems", required=True, help="Path to problem JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--num_rollouts", type=int, default=50, help="Rollouts per problem")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--heuristic", choices=list(HEURISTICS.keys()),
                       help="Heuristic to edit (baseline only if not specified)")
    parser.add_argument("--num_synthetic", type=int, default=15,
                       help="Number of synthetic examples")
    parser.add_argument("--start_vllm", action="store_true",
                       help="Start vLLM server automatically")
    parser.add_argument("--tensor_parallel", type=int, default=2,
                       help="Tensor parallelism size")

    args = parser.parse_args()

    # Load problems
    with open(args.problems) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems from {args.problems}")

    # Start vLLM server if requested
    vllm_process = None
    if args.start_vllm:
        print(f"\nStarting vLLM server with {args.model}...")
        print(f"  Tensor parallel size: {args.tensor_parallel}")

        vllm_cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.model,
            "--tensor-parallel-size", str(args.tensor_parallel),
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.9"
        ]

        vllm_process = subprocess.Popen(
            vllm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to start
        print("Waiting for vLLM server to start...")
        for i in range(60):
            try:
                import requests
                resp = requests.get("http://localhost:8000/health", timeout=1)
                if resp.status_code == 200:
                    print("✓ vLLM server ready!\n")
                    break
            except:
                pass
            time.sleep(2)
        else:
            print("ERROR: vLLM server did not start in time")
            if vllm_process:
                vllm_process.kill()
            sys.exit(1)

    try:
        # Run experiment
        results = asyncio.run(run_experiment_async(
            problems=problems,
            num_rollouts=args.num_rollouts,
            max_tokens=args.max_tokens,
            heuristic=args.heuristic,
            num_synthetic_examples=args.num_synthetic
        ))

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

        # Print summary
        print_summary(results)

    finally:
        # Clean up vLLM server
        if vllm_process:
            print("Shutting down vLLM server...")
            vllm_process.terminate()
            vllm_process.wait(timeout=10)

if __name__ == "__main__":
    main()
