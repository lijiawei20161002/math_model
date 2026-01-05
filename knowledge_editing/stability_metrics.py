#!/usr/bin/env python3
"""
Answer stability metrics for evaluating reasoning consistency.

Implements metrics from the paper:
- Answer entropy: H = -sum(p_i * log(p_i))
- Top-1 share: Fraction of samples producing the most common answer
- Answer diversity: Number of unique answers
- Correct convergence: Whether top-1 answer matches ground truth
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
import json


# Answer extraction utilities
BOX_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
INT_SPAN_RE = re.compile(r"(?<!\d)(\d{1,12})(?!\d)")


def normalize_answer(s: Optional[str]) -> str:
    """Normalize an answer string for comparison."""
    if s is None:
        return ""
    t = str(s).strip()
    m = BOX_RE.findall(t)
    if m:
        t = m[-1].strip()
    t = re.sub(r"\s+", "", t)
    return t.rstrip(".,;")


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} notation."""
    if not isinstance(text, str):
        return None
    boxes = BOX_RE.findall(text)
    if boxes:
        return normalize_answer(boxes[-1])
    # Fallback: extract last integer
    nums = re.findall(r"(?<![\d.])\d{1,10}(?![\d.])", text)
    if nums:
        return nums[-1]
    return None


class AnswerStabilityMetrics:
    """Computes answer stability metrics for multiple rollouts of a problem."""

    def __init__(self, answers: List[str], ground_truth: Optional[str] = None):
        """
        Initialize metrics calculator.

        Args:
            answers: List of extracted answers from multiple rollouts
            ground_truth: Optional ground truth answer for correctness metrics
        """
        # Normalize all answers
        self.answers = [normalize_answer(a) for a in answers if a]
        self.ground_truth = normalize_answer(ground_truth) if ground_truth else None
        self.n_samples = len(answers)
        self.n_valid = len(self.answers)

        # Compute answer distribution
        self.counter = Counter(self.answers)
        self.unique_answers = list(self.counter.keys())
        self.n_unique = len(self.unique_answers)

    def entropy(self) -> float:
        """
        Compute Shannon entropy of answer distribution.

        H = -sum(p_i * log2(p_i))

        Lower entropy = more concentrated (stable) distribution.
        """
        if self.n_valid == 0:
            return float('nan')

        entropy = 0.0
        for count in self.counter.values():
            p = count / self.n_valid
            if p > 0:
                entropy -= p * np.log2(p)

        return float(entropy)

    def top1_share(self) -> float:
        """
        Compute fraction of samples producing the most common answer.

        Higher top-1 share = stronger convergence to a single answer.
        """
        if self.n_valid == 0:
            return 0.0

        most_common_count = self.counter.most_common(1)[0][1] if self.counter else 0
        return most_common_count / self.n_valid

    def top1_answer(self) -> Optional[str]:
        """Return the most common answer."""
        if not self.counter:
            return None
        return self.counter.most_common(1)[0][0]

    def diversity(self) -> float:
        """
        Compute answer diversity (normalized unique count).

        Returns number of unique answers divided by total valid answers.
        Lower diversity = more stable.
        """
        if self.n_valid == 0:
            return float('nan')
        return self.n_unique / self.n_valid

    def correctness_rate(self) -> float:
        """
        Fraction of samples that produced the correct answer.

        Requires ground_truth to be set.
        """
        if self.ground_truth is None or self.n_valid == 0:
            return float('nan')

        correct_count = self.counter.get(self.ground_truth, 0)
        return correct_count / self.n_valid

    def top1_is_correct(self) -> bool:
        """
        Check if the most common answer matches ground truth.

        This is the key "correct convergence" metric: did the model
        converge to the RIGHT answer?
        """
        if self.ground_truth is None or not self.counter:
            return False

        top1 = self.top1_answer()
        return top1 == self.ground_truth

    def pass_at_k(self, k: int = 1) -> float:
        """
        Compute pass@k: probability that at least one of k samples is correct.

        This is different from top-1 share - it measures whether ANY
        of the first k samples is correct.
        """
        if self.ground_truth is None or self.n_valid < k:
            return float('nan')

        correct_count = self.counter.get(self.ground_truth, 0)
        # Probability that at least one of k samples is correct
        prob_all_wrong = (1 - correct_count / self.n_valid) ** k
        return 1 - prob_all_wrong

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        top1_correct = self.top1_is_correct()
        return {
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
            "n_unique": self.n_unique,
            "entropy": self.entropy(),
            "top1_share": self.top1_share(),
            "diversity": self.diversity(),
            "correctness_rate": self.correctness_rate(),
            "top1_is_correct": float(top1_correct),
            "correct_convergence": bool(top1_correct),  # Alias for compatibility
            "pass_at_1": self.pass_at_k(1),
            "pass_at_5": self.pass_at_k(5),
            "pass_at_10": self.pass_at_k(10),
            "top1_answer": self.top1_answer(),
        }

    def __repr__(self) -> str:
        metrics = self.get_all_metrics()
        return (
            f"AnswerStabilityMetrics(\n"
            f"  n_samples={metrics['n_samples']}, n_valid={metrics['n_valid']}, n_unique={metrics['n_unique']}\n"
            f"  entropy={metrics['entropy']:.3f}, top1_share={metrics['top1_share']:.3f}\n"
            f"  correctness={metrics['correctness_rate']:.3f}, top1_correct={bool(metrics['top1_is_correct'])}\n"
            f")"
        )


def compute_stability_comparison(
    before_answers: List[str],
    after_answers: List[str],
    ground_truth: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare stability metrics before and after knowledge editing.

    Args:
        before_answers: Answers from rollouts before editing
        after_answers: Answers from rollouts after editing
        ground_truth: Ground truth answer

    Returns:
        Dictionary with before/after metrics and improvements
    """
    before = AnswerStabilityMetrics(before_answers, ground_truth)
    after = AnswerStabilityMetrics(after_answers, ground_truth)

    before_metrics = before.get_all_metrics()
    after_metrics = after.get_all_metrics()

    # Compute improvements (negative = degradation)
    improvements = {
        "entropy_reduction": before_metrics["entropy"] - after_metrics["entropy"],
        "top1_share_increase": after_metrics["top1_share"] - before_metrics["top1_share"],
        "diversity_reduction": before_metrics["diversity"] - after_metrics["diversity"],
        "correctness_increase": after_metrics["correctness_rate"] - before_metrics["correctness_rate"],
        "top1_correct_before": before_metrics["top1_is_correct"],
        "top1_correct_after": after_metrics["top1_is_correct"],
    }

    return {
        "before": before_metrics,
        "after": after_metrics,
        "improvements": improvements,
    }


def analyze_traces_file(
    traces_path: str,
    question_indices: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Analyze answer stability for questions in a traces file.

    Args:
        traces_path: Path to JSON file with traces (from sample.py)
        question_indices: Optional list of question indices to analyze

    Returns:
        List of stability metrics per question
    """
    with open(traces_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if question_indices is None:
        question_indices = list(range(len(data)))

    results = []
    for idx in question_indices:
        if idx >= len(data):
            continue

        rec = data[idx]
        answers = rec.get("final_answers", [])
        ground_truth = rec.get("ground_truth", None)

        metrics_obj = AnswerStabilityMetrics(answers, ground_truth)
        metrics = metrics_obj.get_all_metrics()
        metrics["question_idx"] = idx
        metrics["question"] = rec.get("question", "")

        results.append(metrics)

    return results


def identify_unstable_problems(
    traces_path: str,
    min_entropy: float = 1.0,
    max_top1_share: float = 0.5,
    require_some_correct: bool = True
) -> List[int]:
    """
    Identify unstable problems suitable for knowledge editing experiments.

    A problem is "unstable" if:
    - High answer entropy (diverse answers)
    - Low top-1 share (no strong consensus)
    - Optionally: has some correct answers (so model has potential)

    Args:
        traces_path: Path to traces JSON
        min_entropy: Minimum entropy threshold
        max_top1_share: Maximum top-1 share threshold
        require_some_correct: If True, require at least some correct answers

    Returns:
        List of question indices that are unstable
    """
    results = analyze_traces_file(traces_path)
    unstable = []

    for r in results:
        is_unstable = (
            r["entropy"] >= min_entropy and
            r["top1_share"] <= max_top1_share
        )

        if require_some_correct:
            is_unstable = is_unstable and r["correctness_rate"] > 0

        if is_unstable:
            unstable.append(r["question_idx"])

    return unstable


def main():
    """CLI for analyzing answer stability."""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze answer stability metrics")
    parser.add_argument("traces", help="Path to traces JSON file")
    parser.add_argument("--identify-unstable", action="store_true",
                       help="Identify unstable problems")
    parser.add_argument("--min-entropy", type=float, default=1.0,
                       help="Minimum entropy for unstable problems")
    parser.add_argument("--max-top1-share", type=float, default=0.5,
                       help="Maximum top-1 share for unstable problems")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    if args.identify_unstable:
        unstable_indices = identify_unstable_problems(
            args.traces,
            min_entropy=args.min_entropy,
            max_top1_share=args.max_top1_share
        )
        print(f"Found {len(unstable_indices)} unstable problems:")
        print(unstable_indices)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"unstable_indices": unstable_indices}, f, indent=2)
    else:
        results = analyze_traces_file(args.traces)
        print(f"Analyzed {len(results)} problems")
        print("\nSample results:")
        for r in results[:5]:
            print(f"\nQ{r['question_idx']}: entropy={r['entropy']:.3f}, "
                  f"top1={r['top1_share']:.3f}, correct={r['correctness_rate']:.3f}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
