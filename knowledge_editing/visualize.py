#!/usr/bin/env python3
"""
Visualization tools for knowledge editing experiments.

Creates comprehensive plots and reports comparing:
- Answer stability before/after editing
- Latent stability across layers
- Depth sensitivity
- Per-problem improvements
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")


class ExperimentVisualizer:
    """Creates visualizations for knowledge editing experiments."""

    def __init__(self, experiment_dir: str):
        """
        Initialize visualizer.

        Args:
            experiment_dir: Path to experiment output directory
        """
        self.exp_dir = Path(experiment_dir)
        self.results_dir = self.exp_dir / "results"
        self.plots_dir = self.exp_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def load_stability_comparison(self) -> Dict[str, Any]:
        """Load stability comparison results."""
        path = self.results_dir / "stability_comparison.json"
        with open(path, "r") as f:
            return json.load(f)

    def plot_stability_improvements(self, data: Optional[Dict] = None):
        """Plot before/after comparison of stability metrics."""
        if data is None:
            data = self.load_stability_comparison()

        per_problem = data["per_problem"]
        aggregate = data["aggregate"]

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Entropy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        entropy_before = [p["before"]["entropy"] for p in per_problem]
        entropy_after = [p["after"]["entropy"] for p in per_problem]

        x = np.arange(len(per_problem))
        width = 0.35
        ax1.bar(x - width / 2, entropy_before, width, label="Before", alpha=0.8)
        ax1.bar(x + width / 2, entropy_after, width, label="After", alpha=0.8)
        ax1.set_xlabel("Problem Index", fontsize=12)
        ax1.set_ylabel("Answer Entropy", fontsize=12)
        ax1.set_title("Answer Entropy: Before vs After Editing", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # 2. Top-1 share comparison
        ax2 = fig.add_subplot(gs[0, 1])
        top1_before = [p["before"]["top1_share"] for p in per_problem]
        top1_after = [p["after"]["top1_share"] for p in per_problem]

        ax2.bar(x - width / 2, top1_before, width, label="Before", alpha=0.8)
        ax2.bar(x + width / 2, top1_after, width, label="After", alpha=0.8)
        ax2.set_xlabel("Problem Index", fontsize=12)
        ax2.set_ylabel("Top-1 Share", fontsize=12)
        ax2.set_title("Top-1 Share: Before vs After Editing", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # 3. Correctness rate
        ax3 = fig.add_subplot(gs[1, 0])
        correct_before = [p["before"]["correctness_rate"] for p in per_problem]
        correct_after = [p["after"]["correctness_rate"] for p in per_problem]

        ax3.bar(x - width / 2, correct_before, width, label="Before", alpha=0.8)
        ax3.bar(x + width / 2, correct_after, width, label="After", alpha=0.8)
        ax3.set_xlabel("Problem Index", fontsize=12)
        ax3.set_ylabel("Correctness Rate", fontsize=12)
        ax3.set_title("Correctness Rate: Before vs After Editing", fontsize=14, fontweight="bold")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)

        # 4. Scatter: Entropy vs Top-1 (before)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(entropy_before, top1_before, alpha=0.6, s=100, label="Before", marker="o")
        ax4.scatter(entropy_after, top1_after, alpha=0.6, s=100, label="After", marker="s")
        ax4.set_xlabel("Entropy", fontsize=12)
        ax4.set_ylabel("Top-1 Share", fontsize=12)
        ax4.set_title("Stability Landscape", fontsize=14, fontweight="bold")
        ax4.legend()
        ax4.grid(alpha=0.3)

        # Add ideal region annotation
        ax4.axhspan(0.7, 1.0, alpha=0.1, color="green", label="High stability")
        ax4.axvspan(0, 0.5, alpha=0.1, color="green")

        # 5. Improvement histogram
        ax5 = fig.add_subplot(gs[2, :])
        improvements = [p["improvements"] for p in per_problem]

        entropy_reductions = [i["entropy_reduction"] for i in improvements]
        top1_increases = [i["top1_share_increase"] for i in improvements]
        correct_increases = [i["correctness_increase"] for i in improvements]

        x_pos = np.arange(len(per_problem))
        ax5.bar(x_pos - 0.3, entropy_reductions, 0.25, label="Entropy Reduction", alpha=0.8)
        ax5.bar(x_pos, top1_increases, 0.25, label="Top-1 Share Increase", alpha=0.8)
        ax5.bar(x_pos + 0.3, correct_increases, 0.25, label="Correctness Increase", alpha=0.8)

        ax5.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax5.set_xlabel("Problem Index", fontsize=12)
        ax5.set_ylabel("Improvement", fontsize=12)
        ax5.set_title("Per-Problem Improvements", fontsize=14, fontweight="bold")
        ax5.legend()
        ax5.grid(axis="y", alpha=0.3)

        # Add summary text
        summary_text = (
            f"Aggregate Results:\n"
            f"Avg Entropy Reduction: {aggregate['avg_entropy_reduction']:.3f}\n"
            f"Avg Top-1 Share Increase: {aggregate['avg_top1_share_increase']:.3f}\n"
            f"Avg Correctness Increase: {aggregate['avg_correctness_increase']:.3f}\n"
            f"Top-1 Correct: {aggregate['n_top1_correct_before']} → {aggregate['n_top1_correct_after']}"
        )
        fig.text(0.02, 0.02, summary_text, fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.suptitle("Knowledge Editing: Stability Analysis", fontsize=16, fontweight="bold", y=0.995)

        output_path = self.plots_dir / "stability_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved stability comparison plot to: {output_path}")
        plt.close()

    def plot_convergence_analysis(self, data: Optional[Dict] = None):
        """Plot convergence to correct answer analysis."""
        if data is None:
            data = self.load_stability_comparison()

        per_problem = data["per_problem"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Count convergence categories
        categories = {
            "Wrong→Wrong": 0,
            "Wrong→Correct": 0,
            "Correct→Wrong": 0,
            "Correct→Correct": 0,
        }

        for p in per_problem:
            before_correct = p["improvements"]["top1_correct_before"]
            after_correct = p["improvements"]["top1_correct_after"]

            if before_correct and after_correct:
                categories["Correct→Correct"] += 1
            elif before_correct and not after_correct:
                categories["Correct→Wrong"] += 1
            elif not before_correct and after_correct:
                categories["Wrong→Correct"] += 1
            else:
                categories["Wrong→Wrong"] += 1

        # Pie chart
        colors = ["red", "green", "orange", "blue"]
        labels = list(categories.keys())
        values = list(categories.values())

        axes[0].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 12})
        axes[0].set_title("Convergence Transitions", fontsize=14, fontweight="bold")

        # Bar chart
        axes[1].bar(labels, values, color=colors, alpha=0.7)
        axes[1].set_ylabel("Number of Problems", fontsize=12)
        axes[1].set_title("Convergence Categories", fontsize=14, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle("Correct Answer Convergence Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_path = self.plots_dir / "convergence_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved convergence analysis plot to: {output_path}")
        plt.close()

    def plot_latent_stability_comparison(self):
        """Plot latent stability comparison if available."""
        before_path = self.results_dir / "latent_stability_before.json"
        after_path = self.results_dir / "latent_stability_after.json"

        if not before_path.exists() or not after_path.exists():
            print("Latent stability data not found, skipping...")
            return

        with open(before_path, "r") as f:
            before_data = json.load(f)
        with open(after_path, "r") as f:
            after_data = json.load(f)

        # Plot for first problem as example
        if not before_data or not after_data:
            return

        stats_before = before_data[0]["stats"]
        stats_after = after_data[0]["stats"]

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        layer_ids = stats_before["layer_ids"]
        x = np.arange(len(layer_ids))

        # Variance
        axes[0].plot(x, stats_before["var_correct"], marker="o", label="Before (Correct)", linewidth=2)
        axes[0].plot(x, stats_after["var_correct"], marker="s", label="After (Correct)", linewidth=2)
        axes[0].plot(x, stats_before["var_wrong"], marker="o", linestyle="--", label="Before (Wrong)", alpha=0.6)
        axes[0].plot(x, stats_after["var_wrong"], marker="s", linestyle="--", label="After (Wrong)", alpha=0.6)
        axes[0].set_ylabel("Variance (↓ better)", fontsize=12)
        axes[0].set_title("Layer-wise Latent Variance", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Cosine similarity
        axes[1].plot(x, stats_before["cos_correct"], marker="o", label="Before (Correct)", linewidth=2)
        axes[1].plot(x, stats_after["cos_correct"], marker="s", label="After (Correct)", linewidth=2)
        axes[1].plot(x, stats_before["cos_wrong"], marker="o", linestyle="--", label="Before (Wrong)", alpha=0.6)
        axes[1].plot(x, stats_after["cos_wrong"], marker="s", linestyle="--", label="After (Wrong)", alpha=0.6)
        axes[1].set_ylabel("Mean Cosine (↑ better)", fontsize=12)
        axes[1].set_title("Layer-wise Latent Cosine Similarity", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # PCA
        axes[2].plot(x, stats_before["pca1_correct"], marker="o", label="Before (Correct)", linewidth=2)
        axes[2].plot(x, stats_after["pca1_correct"], marker="s", label="After (Correct)", linewidth=2)
        axes[2].plot(x, stats_before["pca1_wrong"], marker="o", linestyle="--", label="Before (Wrong)", alpha=0.6)
        axes[2].plot(x, stats_after["pca1_wrong"], marker="s", linestyle="--", label="After (Wrong)", alpha=0.6)
        axes[2].set_xlabel("Layer Index", fontsize=12)
        axes[2].set_ylabel("PC1 EVR (↑ collapse)", fontsize=12)
        axes[2].set_title("Layer-wise PCA PC1 Explained Variance", fontsize=14, fontweight="bold")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.suptitle("Latent Stability: Before vs After Editing", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_path = self.plots_dir / "latent_stability_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved latent stability comparison plot to: {output_path}")
        plt.close()

    def generate_summary_report(self):
        """Generate a comprehensive text report."""
        data = self.load_stability_comparison()

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("KNOWLEDGE EDITING EXPERIMENT: SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Aggregate metrics
        agg = data["aggregate"]
        report_lines.append("AGGREGATE RESULTS:")
        report_lines.append(f"  Number of problems: {agg['n_problems']}")
        report_lines.append(f"  Average entropy reduction: {agg['avg_entropy_reduction']:.4f}")
        report_lines.append(f"  Average top-1 share increase: {agg['avg_top1_share_increase']:.4f}")
        report_lines.append(f"  Average correctness increase: {agg['avg_correctness_increase']:.4f}")
        report_lines.append(f"  Problems with top-1 correct:")
        report_lines.append(f"    Before: {agg['n_top1_correct_before']}")
        report_lines.append(f"    After:  {agg['n_top1_correct_after']}")
        report_lines.append(f"    Change: +{agg['n_top1_correct_after'] - agg['n_top1_correct_before']}")
        report_lines.append("")

        # Per-problem breakdown
        report_lines.append("PER-PROBLEM BREAKDOWN:")
        report_lines.append("")

        for i, p in enumerate(data["per_problem"]):
            idx = p["question_idx"]
            improvements = p["improvements"]

            report_lines.append(f"Problem {idx}:")
            report_lines.append(f"  Entropy: {p['before']['entropy']:.3f} → {p['after']['entropy']:.3f} (Δ={improvements['entropy_reduction']:.3f})")
            report_lines.append(f"  Top-1 Share: {p['before']['top1_share']:.3f} → {p['after']['top1_share']:.3f} (Δ={improvements['top1_share_increase']:.3f})")
            report_lines.append(f"  Correctness: {p['before']['correctness_rate']:.3f} → {p['after']['correctness_rate']:.3f} (Δ={improvements['correctness_increase']:.3f})")
            report_lines.append(f"  Top-1 Correct: {improvements['top1_correct_before']} → {improvements['top1_correct_after']}")
            report_lines.append("")

        report_lines.append("=" * 80)

        # Save report
        output_path = self.plots_dir.parent / "summary_report.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Saved summary report to: {output_path}")

        # Also print to console
        print("\n" + "\n".join(report_lines))

    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\nGenerating visualizations...")

        self.plot_stability_improvements()
        self.plot_convergence_analysis()
        self.plot_latent_stability_comparison()
        self.generate_summary_report()

        print(f"\nAll plots saved to: {self.plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize knowledge editing results")
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment output directory",
    )
    parser.add_argument(
        "--plot-type",
        choices=["all", "stability", "convergence", "latent"],
        default="all",
        help="Type of plot to generate",
    )

    args = parser.parse_args()

    visualizer = ExperimentVisualizer(args.experiment_dir)

    if args.plot_type == "all":
        visualizer.generate_all_plots()
    elif args.plot_type == "stability":
        visualizer.plot_stability_improvements()
    elif args.plot_type == "convergence":
        visualizer.plot_convergence_analysis()
    elif args.plot_type == "latent":
        visualizer.plot_latent_stability_comparison()


if __name__ == "__main__":
    main()
