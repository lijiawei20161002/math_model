#!/usr/bin/env python3
"""
Quick experiment using OpenAI API for in-context knowledge editing.
This version doesn't require LoRA fine-tuning and can run immediately.
"""
import os
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI
from stability_metrics import AnswerStabilityMetrics
from collections import Counter
import re

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def load_synthetic_examples(path: str, heuristic: Optional[str] = None) -> List[Dict]:
    """Load synthetic training examples."""
    with open(path) as f:
        data = json.load(f)

    if heuristic:
        data = [ex for ex in data if ex.get("heuristic") == heuristic]

    return data

def format_in_context_prompt(problem: str, synthetic_examples: List[Dict], max_examples: int = 5) -> str:
    """Format problem with in-context examples."""
    examples_text = "\n\n".join([
        f"Example {i+1}:\nProblem: {ex['input']}\n{ex['output']}"
        for i, ex in enumerate(synthetic_examples[:max_examples])
    ])

    prompt = f"""You are solving mathematical problems. Here are some examples of correct reasoning:

{examples_text}

Now solve this problem following the same careful reasoning approach:
Problem: {problem}

Provide your final answer in the format \\boxed{{answer}}.
"""
    return prompt

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def generate_rollouts(
    problem: str,
    num_rollouts: int,
    synthetic_examples: Optional[List[Dict]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> List[str]:
    """Generate multiple rollouts for a problem."""
    rollouts = []

    if synthetic_examples:
        prompt = format_in_context_prompt(problem, synthetic_examples)
    else:
        prompt = f"""Solve the following mathematical problem step by step.
Provide your final answer in the format \\boxed{{answer}}.

Problem: {problem}
"""

    for i in range(num_rollouts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000
            )
            rollouts.append(response.choices[0].message.content)

            # Rate limiting
            if i < num_rollouts - 1:
                time.sleep(0.5)

        except Exception as e:
            print(f"Error in rollout {i+1}: {e}")
            rollouts.append("")

    return rollouts

def evaluate_problem(
    problem_data: Dict,
    synthetic_data: List[Dict],
    num_rollouts: int = 10,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Evaluate a single problem before and after editing."""
    problem = problem_data["problem"]
    ground_truth = problem_data["answer"]
    heuristic = problem_data.get("heuristic")

    # Get relevant synthetic examples
    if heuristic:
        examples = [ex for ex in synthetic_data if ex.get("heuristic") == heuristic]
    else:
        examples = synthetic_data[:5]

    print(f"\nEvaluating: {problem_data['problem_id']}")
    print(f"Heuristic: {heuristic}")
    print(f"Found {len(examples)} relevant examples")

    # Baseline (no editing)
    print("  Running baseline rollouts...")
    baseline_rollouts = generate_rollouts(problem, num_rollouts, None, model)
    baseline_answers = [extract_boxed_answer(r) for r in baseline_rollouts]
    baseline_answers = [a for a in baseline_answers if a]

    # After editing (in-context)
    print("  Running edited rollouts...")
    edited_rollouts = generate_rollouts(problem, num_rollouts, examples, model)
    edited_answers = [extract_boxed_answer(r) for r in edited_rollouts]
    edited_answers = [a for a in edited_answers if a]

    # Calculate metrics
    metrics_calc = AnswerStabilityMetrics()

    baseline_metrics = metrics_calc.compute_metrics(baseline_answers, ground_truth)
    edited_metrics = metrics_calc.compute_metrics(edited_answers, ground_truth)

    print(f"  Baseline - Entropy: {baseline_metrics['entropy']:.3f}, Top-1: {baseline_metrics['top1_share']:.3f}, Correct: {baseline_metrics['correct_convergence']}")
    print(f"  Edited   - Entropy: {edited_metrics['entropy']:.3f}, Top-1: {edited_metrics['top1_share']:.3f}, Correct: {edited_metrics['correct_convergence']}")

    return {
        "problem_id": problem_data["problem_id"],
        "problem": problem,
        "ground_truth": ground_truth,
        "heuristic": heuristic,
        "baseline": {
            "answers": baseline_answers,
            "metrics": baseline_metrics
        },
        "edited": {
            "answers": edited_answers,
            "metrics": edited_metrics
        },
        "improvement": {
            "entropy_reduction": baseline_metrics["entropy"] - edited_metrics["entropy"],
            "top1_increase": edited_metrics["top1_share"] - baseline_metrics["top1_share"],
            "accuracy_gain": int(edited_metrics["correct_convergence"]) - int(baseline_metrics["correct_convergence"])
        }
    }

def main():
    """Run quick experiment."""
    import argparse
    parser = argparse.ArgumentParser(description="Run quick knowledge editing experiment")
    parser.add_argument("--problems", default="test_aime_problems.json", help="Problem set JSON")
    parser.add_argument("--synthetic", default="synthetic_heuristics.json", help="Synthetic examples JSON")
    parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts per problem")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--output", default="experiment_results.json", help="Output JSON file")

    args = parser.parse_args()

    # Load data
    print(f"Loading problems from {args.problems}...")
    with open(args.problems) as f:
        problems = json.load(f)

    print(f"Loading synthetic examples from {args.synthetic}...")
    with open(args.synthetic) as f:
        synthetic_data = json.load(f)

    print(f"\nRunning experiment with {len(problems)} problems, {args.num_rollouts} rollouts each")
    print(f"Using model: {args.model}")
    print("=" * 60)

    # Run evaluations
    results = []
    for problem in problems:
        try:
            result = evaluate_problem(problem, synthetic_data, args.num_rollouts, args.model)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {problem['problem_id']}: {e}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Experiment complete! Results saved to {args.output}")

    # Summary statistics
    print("\n=== Summary ===")
    avg_entropy_reduction = sum(r["improvement"]["entropy_reduction"] for r in results) / len(results)
    avg_top1_increase = sum(r["improvement"]["top1_increase"] for r in results) / len(results)
    accuracy_gains = sum(1 for r in results if r["improvement"]["accuracy_gain"] > 0)

    print(f"Average entropy reduction: {avg_entropy_reduction:.3f}")
    print(f"Average top-1 share increase: {avg_top1_increase:.3f}")
    print(f"Problems with accuracy gain: {accuracy_gains}/{len(results)}")

    return results

if __name__ == "__main__":
    main()
