#!/usr/bin/env python3
"""
Master pipeline for knowledge editing experiments.

This script orchestrates the COMPLETE experiment:
1. Generates synthetic training data for a heuristic
2. Fine-tunes base model with LoRA
3. Runs evaluation comparing baseline vs edited model
4. Generates analysis reports

This is what should have been run from the start!
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))
from heuristics import SyntheticDocumentGenerator, HEURISTICS
from lora_editor import LoRAKnowledgeEditor, KnowledgeEditConfig


def step1_generate_synthetic_data(
    heuristic: str,
    num_examples: int = 100,
    output_path: str = "synthetic_data.json"
) -> str:
    """
    Step 1: Generate synthetic training data for the target heuristic.
    """
    print("\n" + "="*70)
    print(f"STEP 1: GENERATING SYNTHETIC DATA")
    print("="*70)
    print(f"Heuristic: {heuristic}")
    print(f"Number of examples: {num_examples}")
    print(f"Output: {output_path}")
    print("="*70 + "\n")

    generator = SyntheticDocumentGenerator([heuristic])
    synthetic_data = generator.generate_document(
        num_examples_per_heuristic=num_examples,
        format="training"  # Format for fine-tuning
    )

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(synthetic_data, f, indent=2)

    print(f"✓ Generated {len(synthetic_data)} examples")
    print(f"✓ Saved to {output_path}\n")

    return output_path


def step2_finetune_with_lora(
    base_model: str,
    synthetic_data_path: str,
    output_dir: str,
    heuristic: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
) -> str:
    """
    Step 2: Fine-tune base model with LoRA on synthetic data.
    """
    print("\n" + "="*70)
    print(f"STEP 2: FINE-TUNING WITH LORA")
    print("="*70)
    print(f"Base model: {base_model}")
    print(f"Training data: {synthetic_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}")
    print(f"Training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    print("="*70 + "\n")

    # Create config
    config = KnowledgeEditConfig(
        model_name=base_model,
        synthetic_data_path=synthetic_data_path,
        output_dir=output_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Create editor and train
    editor = LoRAKnowledgeEditor(config)
    print("Loading model and applying LoRA...")
    editor.load_model()

    print("\nStarting fine-tuning...")
    editor.train()

    print(f"\n✓ Fine-tuning complete")
    print(f"✓ LoRA adapter saved to {output_dir}\n")

    return output_dir


def step3_run_evaluation(
    base_model: str,
    lora_path: str,
    problems_path: str,
    heuristic: str,
    output_path: str,
    num_rollouts: int = 50,
    max_tokens: int = 2048,
    tensor_parallel: int = 2,
) -> str:
    """
    Step 3: Run evaluation comparing baseline vs edited model.
    """
    print("\n" + "="*70)
    print(f"STEP 3: RUNNING EVALUATION")
    print("="*70)
    print(f"Base model: {base_model}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Test problems: {problems_path}")
    print(f"Target heuristic: {heuristic}")
    print(f"Rollouts: {num_rollouts}")
    print(f"Output: {output_path}")
    print("="*70 + "\n")

    # Call the corrected experiment script
    cmd = [
        "python3", "run_lora_experiment.py",
        "--base_model", base_model,
        "--lora_path", lora_path,
        "--problems", problems_path,
        "--heuristic", heuristic,
        "--output", output_path,
        "--num_rollouts", str(num_rollouts),
        "--max_tokens", str(max_tokens),
        "--tensor_parallel", str(tensor_parallel),
        "--no_filter",  # Evaluate on all problems, not just those matching the heuristic
    ]

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with exit code {result.returncode}")

    print(f"\n✓ Evaluation complete")
    print(f"✓ Results saved to {output_path}\n")

    return output_path


def step4_analyze_results(results_path: str) -> dict:
    """
    Step 4: Load and analyze results.
    """
    print("\n" + "="*70)
    print(f"STEP 4: ANALYZING RESULTS")
    print("="*70)
    print(f"Results file: {results_path}")
    print("="*70 + "\n")

    with open(results_path) as f:
        results = json.load(f)

    baseline = results.get("baseline", [])
    edited = results.get("edited", [])

    if not baseline or not edited:
        print("ERROR: Missing baseline or edited results")
        return results

    # Calculate summary stats
    import numpy as np

    baseline_entropy = np.mean([r["metrics"]["entropy"] for r in baseline])
    baseline_top1 = np.mean([r["metrics"]["top1_share"] for r in baseline])
    baseline_correct = sum([r["metrics"]["correct_convergence"] for r in baseline])

    edited_entropy = np.mean([r["metrics"]["entropy"] for r in edited])
    edited_top1 = np.mean([r["metrics"]["top1_share"] for r in edited])
    edited_correct = sum([r["metrics"]["correct_convergence"] for r in edited])

    # Determine success
    entropy_improved = edited_entropy < baseline_entropy
    top1_improved = edited_top1 > baseline_top1
    correctness_improved = edited_correct > baseline_correct

    success = entropy_improved and (top1_improved or correctness_improved)

    print("\nSUMMARY:")
    print("-" * 70)
    print(f"Baseline:  entropy={baseline_entropy:.3f}, top1={baseline_top1:.2%}, correct={baseline_correct}/{len(baseline)}")
    print(f"Edited:    entropy={edited_entropy:.3f}, top1={edited_top1:.2%}, correct={edited_correct}/{len(edited)}")
    print(f"Change:    Δentropy={edited_entropy - baseline_entropy:+.3f}, "
          f"Δtop1={edited_top1 - baseline_top1:+.2%}, "
          f"Δcorrect={edited_correct - baseline_correct:+d}")
    print("-" * 70)

    if success:
        print("\n✓ EXPERIMENT SUCCESSFUL: Fine-tuning improved stability/correctness")
    else:
        print("\n✗ EXPERIMENT UNSUCCESSFUL: No significant improvement")

    print("\n" + "="*70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run complete knowledge editing pipeline (synthetic data → LoRA → evaluation)"
    )

    # Model config
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                       help="Base model to fine-tune")

    # Heuristic config
    parser.add_argument("--heuristic", required=True, choices=list(HEURISTICS.keys()),
                       help="Target heuristic to install")
    parser.add_argument("--num_synthetic", type=int, default=100,
                       help="Number of synthetic examples to generate")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")

    # Evaluation config
    parser.add_argument("--problems", default="test_aime_problems.json",
                       help="Test problems JSON file")
    parser.add_argument("--num_rollouts", type=int, default=50,
                       help="Rollouts per problem")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Max generation tokens")
    parser.add_argument("--tensor_parallel", type=int, default=2,
                       help="Tensor parallelism for vLLM")

    # Output config
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (default: auto-generated)")

    # Pipeline control
    parser.add_argument("--skip_datagen", action="store_true",
                       help="Skip synthetic data generation (use existing)")
    parser.add_argument("--skip_finetune", action="store_true",
                       help="Skip fine-tuning (use existing LoRA)")
    parser.add_argument("--skip_eval", action="store_true",
                       help="Skip evaluation (only generate data and fine-tune)")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiment_{args.heuristic}_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("KNOWLEDGE EDITING PIPELINE")
    print("="*70)
    print(f"Heuristic: {args.heuristic}")
    print(f"Base model: {args.base_model}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    # Define paths
    synthetic_data_path = output_dir / f"synthetic_{args.heuristic}.json"
    lora_output_dir = output_dir / "lora_adapter"
    results_path = output_dir / "results.json"

    try:
        # Step 1: Generate synthetic data
        if not args.skip_datagen:
            step1_generate_synthetic_data(
                heuristic=args.heuristic,
                num_examples=args.num_synthetic,
                output_path=str(synthetic_data_path)
            )
        else:
            print(f"\nSkipping data generation (using existing: {synthetic_data_path})")
            if not synthetic_data_path.exists():
                print(f"ERROR: {synthetic_data_path} does not exist!")
                sys.exit(1)

        # Step 2: Fine-tune with LoRA
        if not args.skip_finetune:
            step2_finetune_with_lora(
                base_model=args.base_model,
                synthetic_data_path=str(synthetic_data_path),
                output_dir=str(lora_output_dir),
                heuristic=args.heuristic,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
        else:
            print(f"\nSkipping fine-tuning (using existing: {lora_output_dir})")
            if not lora_output_dir.exists():
                print(f"ERROR: {lora_output_dir} does not exist!")
                sys.exit(1)

        # Step 3: Run evaluation
        if not args.skip_eval:
            step3_run_evaluation(
                base_model=args.base_model,
                lora_path=str(lora_output_dir),
                problems_path=args.problems,
                heuristic=args.heuristic,
                output_path=str(results_path),
                num_rollouts=args.num_rollouts,
                max_tokens=args.max_tokens,
                tensor_parallel=args.tensor_parallel,
            )

            # Step 4: Analyze results
            step4_analyze_results(str(results_path))
        else:
            print("\nSkipping evaluation (only data generation and fine-tuning)")

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Output directory: {output_dir}")
        print(f"  - Synthetic data: {synthetic_data_path.name}")
        print(f"  - LoRA adapter: {lora_output_dir.name}")
        if not args.skip_eval:
            print(f"  - Results: {results_path.name}")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
