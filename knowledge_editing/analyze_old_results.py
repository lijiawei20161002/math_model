#!/usr/bin/env python3
"""
Utility to analyze OLD experiment results and explain what they actually measured.

The old experiments tested in-context learning (ICL), not fine-tuning.
This script helps understand what those results tell us.
"""

import json
import argparse
import numpy as np
from pathlib import Path


def load_old_results(path: str):
    """Load results from old experiment format."""
    with open(path) as f:
        return json.load(f)


def analyze_old_experiment(results: dict, output_path: str = None):
    """
    Analyze old results and explain what they actually tested.
    """
    print("\n" + "="*70)
    print("ANALYSIS OF OLD EXPERIMENT RESULTS")
    print("="*70)
    print("\n⚠️  WARNING: These results test IN-CONTEXT LEARNING, not fine-tuning!")
    print("="*70 + "\n")

    config = results.get("config", {})
    baseline = results.get("baseline", [])
    with_editing = results.get("with_editing", [])

    # Print config
    print("EXPERIMENT CONFIGURATION:")
    print(f"  Method: In-Context Learning (ICL)")
    print(f"  Heuristic: {config.get('heuristic', 'N/A')}")
    print(f"  Synthetic examples generated: {config.get('num_synthetic_examples', 'N/A')}")
    print(f"  ICL examples used: 3 (hardcoded)")
    print(f"  Baseline rollouts: {len(baseline[0].get('answers', [])) if baseline else 'N/A'}")
    print(f"  With-ICL rollouts: {len(with_editing[0].get('answers', [])) if with_editing else 'N/A'}")
    print(f"  Number of problems: {len(baseline)}")
    print()

    if not baseline or not with_editing:
        print("ERROR: Missing baseline or with_editing results")
        return

    # Calculate statistics
    baseline_stats = calculate_stats(baseline)
    icl_stats = calculate_stats(with_editing)

    # Print comparison
    print("RESULTS COMPARISON:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Baseline':<20} {'With 3 ICL Examples':<20}")
    print("-" * 70)
    print(f"{'Average Entropy':<30} {baseline_stats['entropy']:<20.3f} {icl_stats['entropy']:<20.3f}")
    print(f"{'Average Top-1 Share':<30} {baseline_stats['top1']:<20.2%} {icl_stats['top1']:<20.2%}")
    print(f"{'Correct Problems':<30} {baseline_stats['correct']}/{len(baseline):<17} {icl_stats['correct']}/{len(with_editing):<17}")
    print("-" * 70)

    # Calculate changes
    delta_entropy = icl_stats['entropy'] - baseline_stats['entropy']
    delta_top1 = icl_stats['top1'] - baseline_stats['top1']
    delta_correct = icl_stats['correct'] - baseline_stats['correct']

    print(f"{'CHANGE (ICL - Baseline)':<30} {'Δ Entropy':<20} {'Δ Top-1':<20}")
    print(f"{'                        ':<30} {delta_entropy:<+20.3f} {delta_top1:<+20.2%}")
    print("-" * 70)
    print()

    # Interpretation
    print("INTERPRETATION:")
    print("="*70)
    print()

    if delta_entropy > 0 and delta_top1 < 0:
        print("❌ NEGATIVE RESULT: ICL made things WORSE")
        print("   - Higher entropy = more scattered answers")
        print("   - Lower top-1 share = weaker convergence")
        print()
        print("   Likely causes:")
        print("   1. Heuristic examples were IRRELEVANT to test problems")
        print("   2. ICL examples confused the model (noise)")
        print("   3. 3 examples insufficient to establish pattern")
        print()
        print("   This does NOT mean fine-tuning would fail!")
        print("   ICL ≠ fine-tuning (different mechanisms)")
    elif delta_entropy < 0 and delta_top1 > 0:
        print("✓ POSITIVE RESULT: ICL improved stability")
        print("   - Lower entropy = more concentrated answers")
        print("   - Higher top-1 share = stronger convergence")
        print()
        print("   This suggests:")
        print("   1. Heuristic examples were RELEVANT")
        print("   2. Model could learn from 3 examples")
        print("   3. Fine-tuning might work even better!")
    else:
        print("⚠️  MIXED RESULT: Some improvement, some degradation")
        print()

    if delta_correct < 0:
        print(f"   ⚠️  Correctness DECREASED by {abs(delta_correct)} problems")
        print("   This is a red flag - ICL hurt performance")
    elif delta_correct > 0:
        print(f"   ✓ Correctness IMPROVED by {delta_correct} problems")
        print("   This is encouraging for fine-tuning")
    else:
        print("   = Correctness UNCHANGED")

    print()
    print("="*70)

    # Per-problem breakdown
    print("\nPER-PROBLEM BREAKDOWN:")
    print("="*70)
    for i, (b, e) in enumerate(zip(baseline, with_editing)):
        pid = b.get('problem_id', f'problem_{i+1}')
        heuristic = b.get('heuristic', 'unknown')

        b_metrics = b['metrics']
        e_metrics = e['metrics']

        print(f"\n{i+1}. {pid} (heuristic: {heuristic})")
        print(f"   Baseline:  entropy={b_metrics['entropy']:.3f}, "
              f"top1={b_metrics['top1_share']:.2%}, "
              f"correct={b_metrics['correct_convergence']}")
        print(f"   With ICL:  entropy={e_metrics['entropy']:.3f}, "
              f"top1={e_metrics['top1_share']:.2%}, "
              f"correct={e_metrics['correct_convergence']}")
        print(f"   Change:    Δentropy={e_metrics['entropy'] - b_metrics['entropy']:+.3f}, "
              f"Δtop1={e_metrics['top1_share'] - b_metrics['top1_share']:+.2%}")

        # Check if heuristic matches
        target_heuristic = config.get('heuristic', '')
        if heuristic != target_heuristic:
            print(f"   ⚠️  MISMATCH: Problem uses '{heuristic}' but ICL taught '{target_heuristic}'")

    print()
    print("="*70)

    # Key takeaways
    print("\nKEY TAKEAWAYS:")
    print("="*70)
    print()
    print("1. This experiment tested ICL (3 examples in prompt), NOT fine-tuning")
    print("2. ICL is fundamentally different from weight modification")
    print("3. Negative ICL results don't imply fine-tuning will fail")
    print("4. To test the actual hypothesis, you must:")
    print("   - Generate synthetic training data")
    print("   - Fine-tune model weights with LoRA")
    print("   - Evaluate with NO ICL examples")
    print()
    print("5. Use `run_full_pipeline.py` to run the CORRECTED experiment")
    print()
    print("="*70 + "\n")

    # Save analysis report
    if output_path:
        report = {
            "analysis_type": "old_experiment_reinterpretation",
            "method_tested": "in_context_learning",
            "config": config,
            "baseline_stats": baseline_stats,
            "icl_stats": icl_stats,
            "changes": {
                "entropy": delta_entropy,
                "top1_share": delta_top1,
                "correctness": delta_correct
            },
            "interpretation": {
                "entropy_improved": delta_entropy < 0,
                "top1_improved": delta_top1 > 0,
                "correctness_improved": delta_correct > 0,
            },
            "recommendation": "Run corrected experiment with actual fine-tuning using run_full_pipeline.py"
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Analysis report saved to: {output_path}\n")


def calculate_stats(results: list) -> dict:
    """Calculate summary statistics."""
    entropy = np.mean([r["metrics"]["entropy"] for r in results])
    top1 = np.mean([r["metrics"]["top1_share"] for r in results])
    correct = sum([r["metrics"]["correct_convergence"] for r in results])

    return {
        "entropy": entropy,
        "top1": top1,
        "correct": correct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze old experiment results and explain what they actually tested"
    )
    parser.add_argument("results_file", help="Path to old results JSON file")
    parser.add_argument("--output", help="Path to save analysis report")

    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"ERROR: File not found: {args.results_file}")
        return 1

    results = load_old_results(args.results_file)
    analyze_old_experiment(results, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
