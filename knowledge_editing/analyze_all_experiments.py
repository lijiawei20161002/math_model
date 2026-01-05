#!/usr/bin/env python3
"""
Comprehensive analysis of all knowledge editing experiments.
Compares different heuristics and generates visualizations.
"""
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats

sns.set_style("whitegrid")
sns.set_palette("husl")


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from model output."""
    if not text:
        return None

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


class AnswerStabilityMetrics:
    """Compute answer stability metrics."""

    def __init__(self, answers: List[str], ground_truth: str):
        self.answers = [a if a else "NO_ANSWER" for a in answers]
        self.ground_truth = ground_truth
        self.answer_counts = Counter(self.answers)
        self.total = len(self.answers)

    def entropy(self) -> float:
        """Compute Shannon entropy of answer distribution."""
        if self.total == 0:
            return 0.0
        probs = [count / self.total for count in self.answer_counts.values()]
        return -sum(p * np.log2(p) if p > 0 else 0 for p in probs)

    def top1_share(self) -> float:
        """Fraction of samples producing the most common answer."""
        if self.total == 0:
            return 0.0
        return max(self.answer_counts.values()) / self.total

    def diversity(self) -> float:
        """Fraction of unique answers."""
        return len(self.answer_counts) / self.total if self.total > 0 else 0.0

    def correctness_rate(self) -> float:
        """Fraction of correct answers."""
        if self.total == 0:
            return 0.0
        return self.answer_counts.get(self.ground_truth, 0) / self.total

    def top1_is_correct(self) -> bool:
        """Whether the most common answer is correct."""
        if not self.answer_counts:
            return False
        top1_answer = max(self.answer_counts, key=self.answer_counts.get)
        return top1_answer == self.ground_truth

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "entropy": self.entropy(),
            "top1_share": self.top1_share(),
            "diversity": self.diversity(),
            "correctness_rate": self.correctness_rate(),
            "top1_correct": self.top1_is_correct(),
            "total_samples": self.total
        }


def load_experiment_data(file_path: str) -> Dict[str, Any]:
    """Load experiment data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_problem(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single problem's results."""
    question = problem_data["question"]
    ground_truth = problem_data["ground_truth"]
    completions = problem_data["completions"]

    # Extract answers
    answers = [extract_answer(comp) for comp in completions]

    # Compute metrics
    metrics = AnswerStabilityMetrics(answers, ground_truth)

    return {
        "problem_id": problem_data.get("problem_id", "unknown"),
        "question": question,
        "ground_truth": ground_truth,
        "num_completions": len(completions),
        "extracted_answers": answers,
        "metrics": metrics.get_all_metrics()
    }


def analyze_experiment(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze full experiment (baseline + with_editing)."""
    baseline = data["baseline"]
    with_editing = data.get("with_editing")
    config = data.get("config", {})

    # Analyze baseline
    baseline_results = [analyze_problem(p) for p in baseline]

    # Analyze with editing (if available)
    edited_results = None
    if with_editing:
        edited_results = [analyze_problem(p) for p in with_editing]

    # Compute aggregate metrics
    def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
        """Compute aggregate metrics across problems."""
        if not results:
            return {}

        metrics_list = [r["metrics"] for r in results]
        return {
            "avg_entropy": np.mean([m["entropy"] for m in metrics_list]),
            "avg_top1_share": np.mean([m["top1_share"] for m in metrics_list]),
            "avg_diversity": np.mean([m["diversity"] for m in metrics_list]),
            "avg_correctness": np.mean([m["correctness_rate"] for m in metrics_list]),
            "num_top1_correct": sum([m["top1_correct"] for m in metrics_list]),
            "total_problems": len(results)
        }

    baseline_aggregate = aggregate_metrics(baseline_results)
    edited_aggregate = aggregate_metrics(edited_results) if edited_results else None

    # Compute improvements
    improvements = None
    if edited_results:
        improvements = {
            "entropy_reduction": baseline_aggregate["avg_entropy"] - edited_aggregate["avg_entropy"],
            "top1_share_increase": edited_aggregate["avg_top1_share"] - baseline_aggregate["avg_top1_share"],
            "correctness_increase": edited_aggregate["avg_correctness"] - baseline_aggregate["avg_correctness"],
            "top1_correct_change": edited_aggregate["num_top1_correct"] - baseline_aggregate["num_top1_correct"]
        }

    return {
        "config": config,
        "baseline_results": baseline_results,
        "edited_results": edited_results,
        "baseline_aggregate": baseline_aggregate,
        "edited_aggregate": edited_aggregate,
        "improvements": improvements
    }


def create_comparison_plot(all_analyses: Dict[str, Dict], output_path: str):
    """Create comprehensive comparison plot across all heuristics."""
    heuristics = list(all_analyses.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Knowledge Editing: Heuristics Comparison", fontsize=16, fontweight="bold")

    # Prepare data
    baseline_data = {
        "entropy": [],
        "top1_share": [],
        "correctness": [],
        "top1_correct": []
    }
    edited_data = {
        "entropy": [],
        "top1_share": [],
        "correctness": [],
        "top1_correct": []
    }
    improvements_data = {
        "entropy_reduction": [],
        "top1_share_increase": [],
        "correctness_increase": []
    }

    for heur in heuristics:
        analysis = all_analyses[heur]
        baseline_agg = analysis["baseline_aggregate"]
        edited_agg = analysis["edited_aggregate"]
        improvements = analysis["improvements"]

        if edited_agg and improvements:
            baseline_data["entropy"].append(baseline_agg["avg_entropy"])
            baseline_data["top1_share"].append(baseline_agg["avg_top1_share"])
            baseline_data["correctness"].append(baseline_agg["avg_correctness"])
            baseline_data["top1_correct"].append(baseline_agg["num_top1_correct"])

            edited_data["entropy"].append(edited_agg["avg_entropy"])
            edited_data["top1_share"].append(edited_agg["avg_top1_share"])
            edited_data["correctness"].append(edited_agg["avg_correctness"])
            edited_data["top1_correct"].append(edited_agg["num_top1_correct"])

            improvements_data["entropy_reduction"].append(improvements["entropy_reduction"])
            improvements_data["top1_share_increase"].append(improvements["top1_share_increase"])
            improvements_data["correctness_increase"].append(improvements["correctness_increase"])

    # Filter heuristics that have both baseline and edited data
    valid_heuristics = [h for h in heuristics if all_analyses[h]["edited_aggregate"] is not None]
    x = np.arange(len(valid_heuristics))
    width = 0.35

    # 1. Entropy comparison
    ax = axes[0, 0]
    ax.bar(x - width/2, baseline_data["entropy"], width, label="Baseline", alpha=0.8)
    ax.bar(x + width/2, edited_data["entropy"], width, label="With Editing", alpha=0.8)
    ax.set_ylabel("Average Entropy", fontsize=12)
    ax.set_title("Answer Entropy (Lower is Better)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_heuristics, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. Top-1 share comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, baseline_data["top1_share"], width, label="Baseline", alpha=0.8)
    ax.bar(x + width/2, edited_data["top1_share"], width, label="With Editing", alpha=0.8)
    ax.set_ylabel("Average Top-1 Share", fontsize=12)
    ax.set_title("Top-1 Answer Share (Higher is Better)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_heuristics, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. Correctness comparison
    ax = axes[0, 2]
    ax.bar(x - width/2, baseline_data["correctness"], width, label="Baseline", alpha=0.8)
    ax.bar(x + width/2, edited_data["correctness"], width, label="With Editing", alpha=0.8)
    ax.set_ylabel("Average Correctness Rate", fontsize=12)
    ax.set_title("Correctness Rate (Higher is Better)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_heuristics, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Entropy reduction (improvement)
    ax = axes[1, 0]
    colors = ['green' if v > 0 else 'red' for v in improvements_data["entropy_reduction"]]
    ax.bar(x, improvements_data["entropy_reduction"], alpha=0.8, color=colors)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel("Entropy Reduction", fontsize=12)
    ax.set_title("Entropy Reduction (Positive is Good)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_heuristics, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # 5. Top-1 share increase
    ax = axes[1, 1]
    colors = ['green' if v > 0 else 'red' for v in improvements_data["top1_share_increase"]]
    ax.bar(x, improvements_data["top1_share_increase"], alpha=0.8, color=colors)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel("Top-1 Share Increase", fontsize=12)
    ax.set_title("Top-1 Share Increase (Positive is Good)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_heuristics, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # 6. Correctness increase
    ax = axes[1, 2]
    colors = ['green' if v > 0 else 'red' for v in improvements_data["correctness_increase"]]
    ax.bar(x, improvements_data["correctness_increase"], alpha=0.8, color=colors)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel("Correctness Rate Increase", fontsize=12)
    ax.set_title("Correctness Increase (Positive is Good)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_heuristics, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved comparison plot to: {output_path}")
    plt.close()


def create_summary_report(all_analyses: Dict[str, Dict], output_path: str):
    """Create text summary report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" KNOWLEDGE EDITING EXPERIMENTS: COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Baseline performance
        f.write("BASELINE PERFORMANCE (No Editing)\n")
        f.write("-"*80 + "\n")
        baseline_analysis = None
        for heur, analysis in all_analyses.items():
            if heur == "baseline" or not analysis["edited_aggregate"]:
                baseline_analysis = analysis
                break

        if baseline_analysis:
            agg = baseline_analysis["baseline_aggregate"]
            f.write(f"  Problems: {agg['total_problems']}\n")
            f.write(f"  Average Entropy: {agg['avg_entropy']:.3f}\n")
            f.write(f"  Average Top-1 Share: {agg['avg_top1_share']:.3f}\n")
            f.write(f"  Average Correctness: {agg['avg_correctness']:.3f}\n")
            f.write(f"  Top-1 Correct: {agg['num_top1_correct']}/{agg['total_problems']}\n\n")

        # Results for each heuristic
        f.write("\nRESULTS BY HEURISTIC\n")
        f.write("="*80 + "\n\n")

        for heur, analysis in all_analyses.items():
            if not analysis["edited_aggregate"]:
                continue

            f.write(f"\n{heur.upper().replace('_', ' ')}\n")
            f.write("-"*80 + "\n")

            baseline_agg = analysis["baseline_aggregate"]
            edited_agg = analysis["edited_aggregate"]
            improvements = analysis["improvements"]

            f.write(f"\nBaseline:\n")
            f.write(f"  Entropy: {baseline_agg['avg_entropy']:.3f}\n")
            f.write(f"  Top-1 Share: {baseline_agg['avg_top1_share']:.3f}\n")
            f.write(f"  Correctness: {baseline_agg['avg_correctness']:.3f}\n")
            f.write(f"  Top-1 Correct: {baseline_agg['num_top1_correct']}/{baseline_agg['total_problems']}\n")

            f.write(f"\nWith Editing:\n")
            f.write(f"  Entropy: {edited_agg['avg_entropy']:.3f}\n")
            f.write(f"  Top-1 Share: {edited_agg['avg_top1_share']:.3f}\n")
            f.write(f"  Correctness: {edited_agg['avg_correctness']:.3f}\n")
            f.write(f"  Top-1 Correct: {edited_agg['num_top1_correct']}/{edited_agg['total_problems']}\n")

            f.write(f"\nImprovements:\n")
            f.write(f"  Entropy Reduction: {improvements['entropy_reduction']:.3f} ")
            f.write(f"({'✓ GOOD' if improvements['entropy_reduction'] > 0 else '✗ BAD'})\n")
            f.write(f"  Top-1 Share Increase: {improvements['top1_share_increase']:.3f} ")
            f.write(f"({'✓ GOOD' if improvements['top1_share_increase'] > 0 else '✗ BAD'})\n")
            f.write(f"  Correctness Increase: {improvements['correctness_increase']:.3f} ")
            f.write(f"({'✓ GOOD' if improvements['correctness_increase'] > 0 else '✗ BAD'})\n")
            f.write(f"  Top-1 Correct Change: {improvements['top1_correct_change']:+d}\n")
            f.write("\n")

        # Best performing heuristic
        f.write("\n" + "="*80 + "\n")
        f.write("BEST PERFORMING HEURISTICS\n")
        f.write("="*80 + "\n\n")

        # Find best by different metrics
        valid_analyses = {h: a for h, a in all_analyses.items() if a["improvements"]}

        if valid_analyses:
            best_entropy = max(valid_analyses.items(),
                             key=lambda x: x[1]["improvements"]["entropy_reduction"])
            best_top1 = max(valid_analyses.items(),
                          key=lambda x: x[1]["improvements"]["top1_share_increase"])
            best_correctness = max(valid_analyses.items(),
                                 key=lambda x: x[1]["improvements"]["correctness_increase"])

            f.write(f"Best Entropy Reduction: {best_entropy[0].upper().replace('_', ' ')}\n")
            f.write(f"  Reduction: {best_entropy[1]['improvements']['entropy_reduction']:.3f}\n\n")

            f.write(f"Best Top-1 Share Increase: {best_top1[0].upper().replace('_', ' ')}\n")
            f.write(f"  Increase: {best_top1[1]['improvements']['top1_share_increase']:.3f}\n\n")

            f.write(f"Best Correctness Increase: {best_correctness[0].upper().replace('_', ' ')}\n")
            f.write(f"  Increase: {best_correctness[1]['improvements']['correctness_increase']:.3f}\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"✓ Saved summary report to: {output_path}")


def main():
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print(" KNOWLEDGE EDITING EXPERIMENTS: COMPREHENSIVE ANALYSIS")
    print("="*80 + "\n")

    # Find all result files
    result_files = {
        "baseline": "results_baseline.json",
        "modular_mult": "results_modular_mult.json",
        "modular_add": "results_modular_add.json",
        "am_gm": "results_am_gm.json",
        "cauchy": "results_cauchy.json"
    }

    # Load and analyze all experiments
    print("Loading and analyzing experiments...")
    all_analyses = {}

    for heuristic, file_path in result_files.items():
        if not Path(file_path).exists():
            print(f"  ⚠ Skipping {heuristic}: file not found")
            continue

        print(f"  → Analyzing {heuristic}...")
        data = load_experiment_data(file_path)
        analysis = analyze_experiment(data)
        all_analyses[heuristic] = analysis

    print(f"\n✓ Analyzed {len(all_analyses)} experiments\n")

    # Create visualizations
    print("Generating visualizations...")
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    # Comparison plot
    create_comparison_plot(all_analyses, str(output_dir / "heuristics_comparison.png"))

    # Summary report
    create_summary_report(all_analyses, str(output_dir / "summary_report.txt"))

    # Save detailed analysis as JSON
    analysis_json = str(output_dir / "detailed_analysis.json")
    with open(analysis_json, 'w') as f:
        # Remove non-serializable data
        output_data = {}
        for heur, analysis in all_analyses.items():
            output_data[heur] = {
                "baseline_aggregate": analysis["baseline_aggregate"],
                "edited_aggregate": analysis["edited_aggregate"],
                "improvements": analysis["improvements"]
            }
        json.dump(output_data, f, indent=2)
    print(f"✓ Saved detailed analysis to: {analysis_json}")

    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - heuristics_comparison.png (visual comparison)")
    print("  - summary_report.txt (detailed text report)")
    print("  - detailed_analysis.json (machine-readable results)")
    print()


if __name__ == "__main__":
    main()
