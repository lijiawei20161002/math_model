#!/usr/bin/env python3
"""
Example usage of the knowledge editing framework.

This script demonstrates basic usage of all components.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_editing.heuristics import SyntheticDocumentGenerator, HEURISTICS
from knowledge_editing.stability_metrics import (
    AnswerStabilityMetrics,
    compute_stability_comparison,
    identify_unstable_problems,
)
from knowledge_editing.depth_sensitivity import DepthSensitivityAnalyzer
from knowledge_editing.lora_editor import KnowledgeEditConfig, LoRAKnowledgeEditor


def example_1_generate_synthetic_data():
    """Example 1: Generate synthetic heuristic training data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Generate Synthetic Heuristic Data")
    print("=" * 70)

    # Create generator for specific heuristics
    generator = SyntheticDocumentGenerator(
        heuristic_names=["modular_multiplication", "am_gm_inequality"]
    )

    # Generate examples
    document = generator.generate_document(
        num_examples_per_heuristic=3,
        format="training"
    )

    print(f"\nGenerated {len(document)} examples")
    print("\nFirst example:")
    print(f"Instruction: {document[0]['instruction']}")
    print(f"Input: {document[0]['input'][:100]}...")
    print(f"Output: {document[0]['output'][:100]}...")

    # Save
    generator.save_document(
        "example_synthetic_data.json",
        num_examples_per_heuristic=5,
        format="training"
    )

    return "example_synthetic_data.json"


def example_2_analyze_stability():
    """Example 2: Analyze answer stability."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Analyze Answer Stability")
    print("=" * 70)

    # Simulate some answers from multiple rollouts
    answers_unstable = [
        "42", "43", "42", "45", "42", "43", "44", "42", "43", "42",
        "45", "42", "43", "42", "44", "42", "43", "45", "42", "43"
    ]
    ground_truth = "42"

    metrics = AnswerStabilityMetrics(answers_unstable, ground_truth)

    print("\nUnstable Problem Metrics:")
    print(f"  Entropy: {metrics.entropy():.3f}")
    print(f"  Top-1 Share: {metrics.top1_share():.3f}")
    print(f"  Top-1 Answer: {metrics.top1_answer()}")
    print(f"  Correctness Rate: {metrics.correctness_rate():.3f}")
    print(f"  Top-1 is Correct: {metrics.top1_is_correct()}")

    # Simulate improved answers after editing
    answers_stable = [
        "42", "42", "42", "42", "42", "42", "43", "42", "42", "42",
        "42", "42", "42", "42", "42", "42", "42", "42", "42", "42"
    ]

    comparison = compute_stability_comparison(
        before_answers=answers_unstable,
        after_answers=answers_stable,
        ground_truth=ground_truth
    )

    print("\nImprovement after editing:")
    improvements = comparison["improvements"]
    print(f"  Entropy reduction: {improvements['entropy_reduction']:.3f}")
    print(f"  Top-1 share increase: {improvements['top1_share_increase']:.3f}")
    print(f"  Correctness increase: {improvements['correctness_increase']:.3f}")


def example_3_lora_config():
    """Example 3: Configure LoRA knowledge editing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Configure LoRA Knowledge Editing")
    print("=" * 70)

    # Create a LoRA configuration
    config = KnowledgeEditConfig(
        model_name="agentica-org/DeepScaleR-1.5B-Preview",
        output_dir="./example_edited_model",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        synthetic_data_path="example_synthetic_data.json",
    )

    print("\nLoRA Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Target modules: {config.target_modules}")
    print(f"  Training epochs: {config.num_train_epochs}")
    print(f"  Learning rate: {config.learning_rate}")

    print("\nTo train:")
    print("  editor = LoRAKnowledgeEditor(config)")
    print("  editor.train()")


def example_4_identify_unstable():
    """Example 4: Identify unstable problems (requires traces file)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Identify Unstable Problems")
    print("=" * 70)

    print("\nTo identify unstable problems from traces:")
    print("  unstable = identify_unstable_problems(")
    print("      'traces.json',")
    print("      min_entropy=1.0,")
    print("      max_top1_share=0.5,")
    print("      require_some_correct=True")
    print("  )")
    print("\nThis will return indices of problems with:")
    print("  - High answer diversity (entropy >= 1.0)")
    print("  - Low consensus (top-1 share <= 0.5)")
    print("  - Some correct answers (potential for improvement)")


def example_5_complete_workflow():
    """Example 5: Complete workflow overview."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete Workflow")
    print("=" * 70)

    print("\nComplete Knowledge Editing Workflow:")
    print("\n1. Generate baseline traces:")
    print("   python eval/sample.py --model <model> --samples 50 --output traces_baseline.json")

    print("\n2. Identify unstable problems:")
    print("   python knowledge_editing/stability_metrics.py traces_baseline.json --identify-unstable")

    print("\n3. Generate synthetic heuristic data:")
    print("   python knowledge_editing/heuristics.py --output synthetic_data.json --examples 5")

    print("\n4. Apply LoRA knowledge editing:")
    print("   python knowledge_editing/lora_editor.py \\")
    print("       --model <model> \\")
    print("       --data synthetic_data.json \\")
    print("       --output edited_model \\")
    print("       --merge")

    print("\n5. Generate post-editing traces:")
    print("   # Serve edited model with vLLM")
    print("   python eval/sample.py --model edited_model --samples 50 --output traces_after.json")

    print("\n6. Run complete experiment:")
    print("   python knowledge_editing/run_experiment.py \\")
    print("       --traces-before traces_baseline.json \\")
    print("       --output-dir experiments/exp1 \\")
    print("       --n-problems 20 \\")
    print("       --n-rollouts 50")

    print("\n7. Visualize results:")
    print("   python knowledge_editing/visualize.py experiments/exp1")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("KNOWLEDGE EDITING FRAMEWORK: EXAMPLE USAGE")
    print("=" * 70)

    # Example 1: Generate synthetic data
    synthetic_data_path = example_1_generate_synthetic_data()

    # Example 2: Analyze stability
    example_2_analyze_stability()

    # Example 3: Configure LoRA
    example_3_lora_config()

    # Example 4: Identify unstable problems
    example_4_identify_unstable()

    # Example 5: Complete workflow
    example_5_complete_workflow()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
