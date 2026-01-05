#!/usr/bin/env python3
"""
Depth sensitivity analysis for measuring overthinking effects.

Evaluates how model performance and stability change as a function of
max_tokens (generation depth). Based on "Do NOT Think That Much" paper.

Key metrics:
- Pass@1 vs max_tokens
- Entropy vs max_tokens
- Answer divergence at different depths
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict

from stability_metrics import AnswerStabilityMetrics, normalize_answer


class DepthSensitivityAnalyzer:
    """Analyzes how reasoning stability changes with generation depth."""

    def __init__(self, traces_by_depth: Dict[int, List[Dict[str, Any]]]):
        """
        Initialize analyzer.

        Args:
            traces_by_depth: Dict mapping max_tokens -> list of trace records
                Each trace record should have: question, final_answers, ground_truth
        """
        self.traces_by_depth = traces_by_depth
        self.depths = sorted(traces_by_depth.keys())

    def compute_metrics_by_depth(self) -> Dict[int, Dict[str, float]]:
        """
        Compute stability metrics for each depth level.

        Returns:
            Dict mapping depth -> metrics dict
        """
        metrics_by_depth = {}

        for depth in self.depths:
            traces = self.traces_by_depth[depth]

            # Aggregate metrics across all questions at this depth
            all_entropies = []
            all_top1_shares = []
            all_correctness = []
            all_top1_correct = []

            for trace in traces:
                answers = trace.get("final_answers", [])
                ground_truth = trace.get("ground_truth", None)

                sm = AnswerStabilityMetrics(answers, ground_truth)
                metrics = sm.get_all_metrics()

                all_entropies.append(metrics["entropy"])
                all_top1_shares.append(metrics["top1_share"])
                all_correctness.append(metrics["correctness_rate"])
                all_top1_correct.append(metrics["top1_is_correct"])

            # Filter out NaN values
            all_entropies = [e for e in all_entropies if not np.isnan(e)]
            all_top1_shares = [t for t in all_top1_shares if not np.isnan(t)]
            all_correctness = [c for c in all_correctness if not np.isnan(c)]

            metrics_by_depth[depth] = {
                "mean_entropy": np.mean(all_entropies) if all_entropies else float('nan'),
                "std_entropy": np.std(all_entropies) if all_entropies else float('nan'),
                "mean_top1_share": np.mean(all_top1_shares) if all_top1_shares else float('nan'),
                "std_top1_share": np.std(all_top1_shares) if all_top1_shares else float('nan'),
                "mean_correctness": np.mean(all_correctness) if all_correctness else float('nan'),
                "pass_at_1": np.mean(all_correctness) if all_correctness else float('nan'),
                "correct_convergence_rate": np.mean(all_top1_correct) if all_top1_correct else float('nan'),
            }

        return metrics_by_depth

    def detect_overthinking(self, metrics_by_depth: Optional[Dict[int, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Detect overthinking by analyzing if performance degrades with depth.

        Overthinking indicators:
        - Pass@1 decreases with depth
        - Entropy increases with depth (more divergence)
        - Top-1 share decreases with depth

        Returns:
            Dict with overthinking analysis
        """
        if metrics_by_depth is None:
            metrics_by_depth = self.compute_metrics_by_depth()

        depths = sorted(metrics_by_depth.keys())
        if len(depths) < 2:
            return {"has_overthinking": False, "reason": "Need at least 2 depth levels"}

        # Extract time series
        pass_at_1_series = [metrics_by_depth[d]["pass_at_1"] for d in depths]
        entropy_series = [metrics_by_depth[d]["mean_entropy"] for d in depths]
        top1_series = [metrics_by_depth[d]["mean_top1_share"] for d in depths]

        # Compute slopes (simple linear regression)
        def compute_slope(xs, ys):
            xs = np.array(xs)
            ys = np.array(ys)
            valid = ~np.isnan(ys)
            if valid.sum() < 2:
                return float('nan')
            xs, ys = xs[valid], ys[valid]
            return np.polyfit(xs, ys, 1)[0]

        pass_at_1_slope = compute_slope(depths, pass_at_1_series)
        entropy_slope = compute_slope(depths, entropy_series)
        top1_slope = compute_slope(depths, top1_series)

        # Overthinking if:
        # - pass@1 decreases (negative slope)
        # - entropy increases (positive slope)
        # - top1 decreases (negative slope)
        has_overthinking = (
            pass_at_1_slope < -0.001 or  # Small threshold for noise
            entropy_slope > 0.01 or
            top1_slope < -0.001
        )

        return {
            "has_overthinking": has_overthinking,
            "pass_at_1_slope": float(pass_at_1_slope),
            "entropy_slope": float(entropy_slope),
            "top1_slope": float(top1_slope),
            "initial_pass_at_1": pass_at_1_series[0] if pass_at_1_series else float('nan'),
            "final_pass_at_1": pass_at_1_series[-1] if pass_at_1_series else float('nan'),
            "pass_at_1_drop": (pass_at_1_series[0] - pass_at_1_series[-1]) if len(pass_at_1_series) >= 2 else 0.0,
        }

    def compare_depth_sensitivity(
        self,
        other: 'DepthSensitivityAnalyzer',
        name_self: str = "before",
        name_other: str = "after"
    ) -> Dict[str, Any]:
        """
        Compare depth sensitivity between two conditions (e.g., before/after editing).

        Args:
            other: Another DepthSensitivityAnalyzer to compare against
            name_self: Label for this analyzer
            name_other: Label for other analyzer

        Returns:
            Comparison metrics
        """
        metrics_self = self.compute_metrics_by_depth()
        metrics_other = other.compute_metrics_by_depth()

        overthinking_self = self.detect_overthinking(metrics_self)
        overthinking_other = other.detect_overthinking(metrics_other)

        # Common depths
        common_depths = sorted(set(self.depths) & set(other.depths))

        improvements = {}
        for depth in common_depths:
            m_self = metrics_self.get(depth, {})
            m_other = metrics_other.get(depth, {})

            improvements[depth] = {
                "entropy_reduction": m_self.get("mean_entropy", 0) - m_other.get("mean_entropy", 0),
                "top1_share_increase": m_other.get("mean_top1_share", 0) - m_self.get("mean_top1_share", 0),
                "pass_at_1_increase": m_other.get("pass_at_1", 0) - m_self.get("pass_at_1", 0),
            }

        return {
            name_self: {
                "metrics_by_depth": metrics_self,
                "overthinking_analysis": overthinking_self,
            },
            name_other: {
                "metrics_by_depth": metrics_other,
                "overthinking_analysis": overthinking_other,
            },
            "improvements_by_depth": improvements,
            "overthinking_reduced": overthinking_self["has_overthinking"] and not overthinking_other["has_overthinking"],
        }

    def plot_depth_curves(
        self,
        output_path: str = "depth_sensitivity.png",
        title: str = "Depth Sensitivity Analysis"
    ):
        """Plot metrics vs depth."""
        metrics_by_depth = self.compute_metrics_by_depth()
        depths = sorted(metrics_by_depth.keys())

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Pass@1
        pass_at_1 = [metrics_by_depth[d]["pass_at_1"] for d in depths]
        axes[0].plot(depths, pass_at_1, marker='o', linewidth=2, markersize=8)
        axes[0].set_ylabel("Pass@1 (↑ better)", fontsize=12)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Entropy
        mean_entropy = [metrics_by_depth[d]["mean_entropy"] for d in depths]
        std_entropy = [metrics_by_depth[d]["std_entropy"] for d in depths]
        axes[1].plot(depths, mean_entropy, marker='o', linewidth=2, markersize=8, label="Mean entropy")
        axes[1].fill_between(
            depths,
            [m - s for m, s in zip(mean_entropy, std_entropy)],
            [m + s for m, s in zip(mean_entropy, std_entropy)],
            alpha=0.2
        )
        axes[1].set_ylabel("Answer Entropy (↓ better)", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Top-1 share
        mean_top1 = [metrics_by_depth[d]["mean_top1_share"] for d in depths]
        std_top1 = [metrics_by_depth[d]["std_top1_share"] for d in depths]
        axes[2].plot(depths, mean_top1, marker='o', linewidth=2, markersize=8, color='green')
        axes[2].fill_between(
            depths,
            [m - s for m, s in zip(mean_top1, std_top1)],
            [m + s for m, s in zip(mean_top1, std_top1)],
            alpha=0.2,
            color='green'
        )
        axes[2].set_xlabel("Max Tokens (Generation Depth)", fontsize=12)
        axes[2].set_ylabel("Top-1 Share (↑ better)", fontsize=12)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved depth sensitivity plot to: {output_path}")

    @staticmethod
    def plot_comparison(
        analyzer_before: 'DepthSensitivityAnalyzer',
        analyzer_after: 'DepthSensitivityAnalyzer',
        output_path: str = "depth_comparison.png",
        title: str = "Depth Sensitivity: Before vs After Editing"
    ):
        """Plot before/after comparison."""
        metrics_before = analyzer_before.compute_metrics_by_depth()
        metrics_after = analyzer_after.compute_metrics_by_depth()

        depths_before = sorted(metrics_before.keys())
        depths_after = sorted(metrics_after.keys())
        common_depths = sorted(set(depths_before) & set(depths_after))

        if not common_depths:
            print("No common depths to compare")
            return

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Pass@1
        pass_at_1_before = [metrics_before[d]["pass_at_1"] for d in common_depths]
        pass_at_1_after = [metrics_after[d]["pass_at_1"] for d in common_depths]
        axes[0].plot(common_depths, pass_at_1_before, marker='o', label="Before", linewidth=2, markersize=8)
        axes[0].plot(common_depths, pass_at_1_after, marker='s', label="After", linewidth=2, markersize=8)
        axes[0].set_ylabel("Pass@1", fontsize=12)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Entropy
        entropy_before = [metrics_before[d]["mean_entropy"] for d in common_depths]
        entropy_after = [metrics_after[d]["mean_entropy"] for d in common_depths]
        axes[1].plot(common_depths, entropy_before, marker='o', label="Before", linewidth=2, markersize=8)
        axes[1].plot(common_depths, entropy_after, marker='s', label="After", linewidth=2, markersize=8)
        axes[1].set_ylabel("Answer Entropy", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Top-1 share
        top1_before = [metrics_before[d]["mean_top1_share"] for d in common_depths]
        top1_after = [metrics_after[d]["mean_top1_share"] for d in common_depths]
        axes[2].plot(common_depths, top1_before, marker='o', label="Before", linewidth=2, markersize=8)
        axes[2].plot(common_depths, top1_after, marker='s', label="After", linewidth=2, markersize=8)
        axes[2].set_xlabel("Max Tokens (Generation Depth)", fontsize=12)
        axes[2].set_ylabel("Top-1 Share", fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")


def load_traces_by_depth(traces_paths: Dict[int, str]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load traces from multiple files, one per depth level.

    Args:
        traces_paths: Dict mapping max_tokens -> file path

    Returns:
        Dict mapping max_tokens -> list of traces
    """
    traces_by_depth = {}
    for depth, path in traces_paths.items():
        with open(path, 'r', encoding='utf-8') as f:
            traces_by_depth[depth] = json.load(f)
    return traces_by_depth


def main():
    """CLI for depth sensitivity analysis."""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze depth sensitivity")
    parser.add_argument("--traces", nargs="+", required=True,
                       help="Traces JSON files (format: depth:path)")
    parser.add_argument("--output-plot", default="depth_sensitivity.png",
                       help="Output plot file")
    parser.add_argument("--output-json", help="Output JSON for metrics")
    args = parser.parse_args()

    # Parse depth:path arguments
    traces_paths = {}
    for item in args.traces:
        depth_str, path = item.split(":", 1)
        traces_paths[int(depth_str)] = path

    traces_by_depth = load_traces_by_depth(traces_paths)
    analyzer = DepthSensitivityAnalyzer(traces_by_depth)

    # Compute metrics
    metrics = analyzer.compute_metrics_by_depth()
    overthinking = analyzer.detect_overthinking(metrics)

    print("Depth Sensitivity Analysis")
    print("=" * 50)
    for depth in sorted(metrics.keys()):
        m = metrics[depth]
        print(f"\nDepth {depth}:")
        print(f"  Pass@1: {m['pass_at_1']:.3f}")
        print(f"  Mean Entropy: {m['mean_entropy']:.3f}")
        print(f"  Mean Top-1 Share: {m['mean_top1_share']:.3f}")

    print("\nOverthinking Analysis:")
    print(f"  Has overthinking: {overthinking['has_overthinking']}")
    print(f"  Pass@1 slope: {overthinking['pass_at_1_slope']:.6f}")
    print(f"  Entropy slope: {overthinking['entropy_slope']:.6f}")

    # Plot
    analyzer.plot_depth_curves(args.output_plot)

    # Save JSON
    if args.output_json:
        output = {
            "metrics_by_depth": metrics,
            "overthinking_analysis": overthinking,
        }
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
