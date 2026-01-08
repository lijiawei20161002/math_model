#!/usr/bin/env python3
"""
Quick test of improved knowledge editing approach.

Runs a pilot experiment on a single heuristic to validate improvements
before scaling to all heuristics.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_editing.heuristics_v2 import SyntheticDocumentGeneratorV2
from knowledge_editing.lora_editor_v2 import LoRAKnowledgeEditorV2, KnowledgeEditConfigV2
from knowledge_editing.heuristics import HEURISTICS


def main():
    parser = argparse.ArgumentParser(description="Test improved knowledge editing")
    parser.add_argument(
        "--heuristic",
        default="modular_multiplication",
        choices=list(HEURISTICS.keys()),
        help="Heuristic to test"
    )
    parser.add_argument(
        "--output-dir",
        default="./test_v2_output",
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="agentica-org/DeepScaleR-1.5B-Preview",
        help="Base model"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer examples, faster training"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("TESTING IMPROVED KNOWLEDGE EDITING (V2)")
    print("="*80)
    print(f"Heuristic: {args.heuristic}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print("="*80 + "\n")

    # Step 1: Generate synthetic data
    print("STEP 1: Generating Synthetic Data")
    print("-" * 80)

    synthetic_path = output_dir / "synthetic_data.json"
    generator = SyntheticDocumentGeneratorV2([args.heuristic])

    if args.quick:
        n_examples = 20  # Quick test
    else:
        n_examples = 50  # Full version

    generator.save_document(
        str(synthetic_path),
        num_examples_per_heuristic=n_examples,
        format="training",
        use_difficulty_levels=True,
        augment=True,
    )

    print(f"\n✓ Synthetic data saved to: {synthetic_path}\n")

    # Step 2: Configure improved LoRA editing
    print("STEP 2: Configuring Improved LoRA Editor")
    print("-" * 80)

    if args.quick:
        # Quick settings for testing
        config = KnowledgeEditConfigV2(
            model_name=args.model,
            synthetic_data_path=str(synthetic_path),
            output_dir=str(output_dir / "edited_model"),
            lora_r=16,  # Smaller for speed
            lora_alpha=32,
            target_layers=list(range(15, 18)),  # Fewer layers for speed
            num_train_epochs=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            warmup_steps=20,
            use_preservation_loss=True,
            preservation_alpha=0.3,
            gradient_checkpointing=True,
        )
    else:
        # Full settings
        config = KnowledgeEditConfigV2(
            model_name=args.model,
            synthetic_data_path=str(synthetic_path),
            output_dir=str(output_dir / "edited_model"),
            lora_r=32,
            lora_alpha=64,
            target_layers=list(range(12, 20)),
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            warmup_steps=50,
            use_preservation_loss=True,
            preservation_alpha=0.3,
            gradient_checkpointing=True,
        )

    print("\nConfiguration:")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Target layers: {config.target_layers}")
    print(f"  Target modules: {config.target_modules}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Preservation loss: {config.use_preservation_loss}")
    print(f"  Preservation alpha: {config.preservation_alpha}")

    # Step 3: Train
    print("\n" + "="*80)
    print("STEP 3: Training with Improved Method")
    print("="*80 + "\n")

    editor = LoRAKnowledgeEditorV2(config)

    try:
        editor.train()
        print("\n✓ Training completed successfully!")

        # Save merged model
        merged_path = output_dir / "edited_model_merged"
        editor.merge_and_save(str(merged_path))
        print(f"✓ Merged model saved to: {merged_path}")

        # Save config for reference
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "heuristic": args.heuristic,
                "model": args.model,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "target_layers": config.target_layers,
                "target_modules": config.target_modules,
                "learning_rate": config.learning_rate,
                "num_train_epochs": config.num_train_epochs,
                "preservation_loss": config.use_preservation_loss,
                "preservation_alpha": config.preservation_alpha,
                "n_examples": n_examples,
            }, f, indent=2)
        print(f"✓ Config saved to: {config_path}")

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Serve the edited model:")
        print(f"   vllm serve {merged_path} --port 8000")
        print("\n2. Run evaluation to compare before/after:")
        print(f"   python eval/sample.py --model {merged_path} --output traces_after_v2.json")
        print("\n3. Analyze results:")
        print("   python knowledge_editing/analyze_all_experiments.py \\")
        print("     --results results/")
        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
