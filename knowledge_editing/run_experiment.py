#!/usr/bin/env python3
"""
Main experiment orchestration for knowledge editing.

Implements the complete experimental pipeline from the paper:
1. Identify unstable AIME problems
2. Generate synthetic heuristic examples
3. Apply knowledge editing (LoRA or in-context)
4. Evaluate before/after stability and performance
5. Generate visualizations and reports
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_editing.heuristics import SyntheticDocumentGenerator, HEURISTICS
from knowledge_editing.stability_metrics import (
    identify_unstable_problems,
    compute_stability_comparison,
    analyze_traces_file,
)
from knowledge_editing.lora_editor import LoRAKnowledgeEditor, KnowledgeEditConfig
from knowledge_editing.depth_sensitivity import (
    DepthSensitivityAnalyzer,
    load_traces_by_depth,
)
from eval.sample import generate_cot_traces_async


class KnowledgeEditingExperiment:
    """Orchestrates the complete knowledge editing experiment."""

    def __init__(
        self,
        base_model: str,
        traces_before_path: str,
        output_dir: str,
        heuristics: Optional[List[str]] = None,
        n_problems: int = 20,
        n_rollouts: int = 50,
        edit_method: str = "lora",  # "lora" or "in_context"
    ):
        """
        Initialize experiment.

        Args:
            base_model: Path to base model
            traces_before_path: Path to pre-computed traces for baseline
            output_dir: Output directory for all results
            heuristics: List of heuristic names to edit (None = all)
            n_problems: Number of unstable problems to use
            n_rollouts: Number of rollouts per problem
            edit_method: "lora" or "in_context"
        """
        self.base_model = base_model
        self.traces_before_path = traces_before_path
        self.output_dir = Path(output_dir)
        self.heuristics = heuristics
        self.n_problems = n_problems
        self.n_rollouts = n_rollouts
        self.edit_method = edit_method

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "synthetic_data").mkdir(exist_ok=True)
        (self.output_dir / "edited_models").mkdir(exist_ok=True)
        (self.output_dir / "traces").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

    def step1_identify_unstable_problems(self) -> List[int]:
        """Step 1: Identify unstable problems suitable for editing."""
        print("\n" + "=" * 70)
        print("STEP 1: Identifying Unstable Problems")
        print("=" * 70)

        unstable_indices = identify_unstable_problems(
            self.traces_before_path,
            min_entropy=1.0,
            max_top1_share=0.5,
            require_some_correct=True,
        )

        # Limit to n_problems
        unstable_indices = unstable_indices[: self.n_problems]

        print(f"Found {len(unstable_indices)} unstable problems: {unstable_indices}")

        # Save
        output_path = self.output_dir / "unstable_problems.json"
        with open(output_path, "w") as f:
            json.dump({"unstable_indices": unstable_indices}, f, indent=2)
        print(f"Saved to: {output_path}")

        return unstable_indices

    def step2_generate_synthetic_data(self) -> str:
        """Step 2: Generate synthetic heuristic training examples."""
        print("\n" + "=" * 70)
        print("STEP 2: Generating Synthetic Heuristic Examples")
        print("=" * 70)

        generator = SyntheticDocumentGenerator(self.heuristics)
        output_path = self.output_dir / "synthetic_data" / "heuristics.json"

        generator.save_document(
            str(output_path),
            num_examples_per_heuristic=5,
            format="training",
        )

        return str(output_path)

    def step3_apply_knowledge_editing(self, synthetic_data_path: str) -> str:
        """Step 3: Apply knowledge editing via LoRA or in-context."""
        print("\n" + "=" * 70)
        print(f"STEP 3: Applying Knowledge Editing ({self.edit_method})")
        print("=" * 70)

        if self.edit_method == "lora":
            # LoRA fine-tuning
            config = KnowledgeEditConfig(
                model_name=self.base_model,
                synthetic_data_path=synthetic_data_path,
                output_dir=str(self.output_dir / "edited_models" / "lora"),
                num_train_epochs=3,
                lora_r=8,
                lora_alpha=16,
                per_device_train_batch_size=4,
                learning_rate=2e-4,
            )

            editor = LoRAKnowledgeEditor(config)
            editor.train()

            # Merge and save
            merged_path = str(self.output_dir / "edited_models" / "lora_merged")
            editor.merge_and_save(merged_path)

            return merged_path

        elif self.edit_method == "in_context":
            # For in-context, we just return the synthetic data path
            # The evaluation will inject it at inference time
            print("In-context method: will inject examples during evaluation")
            return synthetic_data_path

        else:
            raise ValueError(f"Unknown edit method: {self.edit_method}")

    async def step4_evaluate_after_editing(
        self,
        edited_model_or_data: str,
        unstable_indices: List[int],
    ) -> str:
        """Step 4: Generate traces after editing."""
        print("\n" + "=" * 70)
        print("STEP 4: Evaluating After Editing")
        print("=" * 70)

        # Load the original dataset
        from datasets import load_dataset, concatenate_datasets

        aime_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        aime_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        aime = concatenate_datasets([aime_i, aime_ii])

        # Filter to unstable problems
        unstable_problems = [aime[i] for i in unstable_indices]

        output_path = str(self.output_dir / "traces" / "after_editing.json")

        # TODO: This needs to be integrated with the actual evaluation
        # For now, we'll assume the model is already served via vLLM
        print(f"NOTE: Please ensure the edited model is served via vLLM")
        print(f"Then run eval/sample.py with the edited model")
        print(f"Output should go to: {output_path}")

        return output_path

    def step5_compute_metrics(
        self,
        traces_before_path: str,
        traces_after_path: str,
        unstable_indices: List[int],
    ) -> Dict[str, Any]:
        """Step 5: Compute and compare all metrics."""
        print("\n" + "=" * 70)
        print("STEP 5: Computing Stability Metrics")
        print("=" * 70)

        # Load traces
        with open(traces_before_path, "r") as f:
            traces_before = json.load(f)
        with open(traces_after_path, "r") as f:
            traces_after = json.load(f)

        # Filter to unstable problems
        traces_before = [traces_before[i] for i in unstable_indices]
        traces_after = [traces_after[i] for i in unstable_indices]

        # Compute per-problem comparisons
        results = []
        for i, (before, after) in enumerate(zip(traces_before, traces_after)):
            comparison = compute_stability_comparison(
                before_answers=before.get("final_answers", []),
                after_answers=after.get("final_answers", []),
                ground_truth=before.get("ground_truth"),
            )

            comparison["question_idx"] = unstable_indices[i]
            comparison["question"] = before.get("question", "")
            results.append(comparison)

        # Aggregate statistics
        aggregate = {
            "n_problems": len(results),
            "avg_entropy_reduction": sum(
                r["improvements"]["entropy_reduction"] for r in results
            )
            / len(results),
            "avg_top1_share_increase": sum(
                r["improvements"]["top1_share_increase"] for r in results
            )
            / len(results),
            "avg_correctness_increase": sum(
                r["improvements"]["correctness_increase"] for r in results
            )
            / len(results),
            "n_top1_correct_before": sum(
                r["improvements"]["top1_correct_before"] for r in results
            ),
            "n_top1_correct_after": sum(
                r["improvements"]["top1_correct_after"] for r in results
            ),
        }

        output = {
            "aggregate": aggregate,
            "per_problem": results,
        }

        # Save
        output_path = self.output_dir / "results" / "stability_comparison.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nAggregate Results:")
        print(f"  Average entropy reduction: {aggregate['avg_entropy_reduction']:.3f}")
        print(f"  Average top-1 share increase: {aggregate['avg_top1_share_increase']:.3f}")
        print(f"  Average correctness increase: {aggregate['avg_correctness_increase']:.3f}")
        print(f"  Top-1 correct: {aggregate['n_top1_correct_before']} → {aggregate['n_top1_correct_after']}")
        print(f"\nSaved to: {output_path}")

        return output

    def step6_latent_stability_analysis(
        self,
        traces_path: str,
        model_path: str,
        unstable_indices: List[int],
        name: str = "before",
    ):
        """Step 6: Analyze latent stability (requires model inference)."""
        print("\n" + "=" * 70)
        print(f"STEP 6: Latent Stability Analysis ({name})")
        print("=" * 70)

        # This uses the existing probe/latent_stability.py script
        from probe.latent_stability import analyze_question, load_model

        print(f"Loading model: {model_path}")
        tok, model = load_model(model_path)

        # Load traces
        with open(traces_path, "r") as f:
            traces = json.load(f)

        # Analyze each unstable problem
        latent_results = []
        for idx in unstable_indices[:5]:  # Limit to 5 for speed
            print(f"\nAnalyzing question {idx}...")
            rec = traces[idx]

            stats = analyze_question(
                rec,
                tok,
                model,
                kind="intermediate",
                half_window=1,
                layer_stride=1,
                layer_offset=1,
                max_samples=50,
            )

            latent_results.append({"question_idx": idx, "stats": stats})

        # Save
        output_path = (
            self.output_dir / "results" / f"latent_stability_{name}.json"
        )
        with open(output_path, "w") as f:
            json.dump(latent_results, f, indent=2)

        print(f"Saved to: {output_path}")

    def step7_generate_report(self):
        """Step 7: Generate final report and visualizations."""
        print("\n" + "=" * 70)
        print("STEP 7: Generating Report")
        print("=" * 70)

        # Create a summary report
        report = {
            "experiment_config": {
                "base_model": self.base_model,
                "edit_method": self.edit_method,
                "n_problems": self.n_problems,
                "n_rollouts": self.n_rollouts,
                "heuristics": self.heuristics or "all",
            },
            "results_files": {
                "unstable_problems": "unstable_problems.json",
                "synthetic_data": "synthetic_data/heuristics.json",
                "stability_comparison": "results/stability_comparison.json",
            },
        }

        output_path = self.output_dir / "experiment_report.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nExperiment complete!")
        print(f"Report saved to: {output_path}")
        print(f"\nAll outputs in: {self.output_dir}")

    def run(self):
        """Run the complete experiment pipeline."""
        print("\n" + "=" * 70)
        print("KNOWLEDGE EDITING EXPERIMENT")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")

        # Step 1: Identify unstable problems
        unstable_indices = self.step1_identify_unstable_problems()

        # Step 2: Generate synthetic data
        synthetic_data_path = self.step2_generate_synthetic_data()

        # Step 3: Apply knowledge editing
        edited_model_or_data = self.step3_apply_knowledge_editing(synthetic_data_path)

        # Step 4: Evaluate (requires manual vLLM serving for now)
        print("\n" + "=" * 70)
        print("STEP 4: Post-Editing Evaluation")
        print("=" * 70)
        print("Please run the following to generate post-editing traces:")
        print(f"1. Serve the edited model: {edited_model_or_data}")
        print(f"2. Run: python eval/sample.py --model <edited_model> --samples {self.n_rollouts} --output {self.output_dir}/traces/after_editing.json")
        print("3. Then run this script again with --skip-to-step5")

        # Save state for resumption
        state = {
            "unstable_indices": unstable_indices,
            "synthetic_data_path": synthetic_data_path,
            "edited_model_or_data": edited_model_or_data,
        }
        state_path = self.output_dir / "experiment_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        print(f"\nExperiment state saved to: {state_path}")

    def run_from_step5(self):
        """Resume experiment from step 5 (after traces are generated)."""
        # Load state
        state_path = self.output_dir / "experiment_state.json"
        with open(state_path, "r") as f:
            state = json.load(f)

        unstable_indices = state["unstable_indices"]

        # Step 5: Compute metrics
        traces_after_path = str(self.output_dir / "traces" / "after_editing.json")
        metrics = self.step5_compute_metrics(
            self.traces_before_path,
            traces_after_path,
            unstable_indices,
        )

        # Step 7: Generate report
        self.step7_generate_report()


def main():
    parser = argparse.ArgumentParser(description="Run knowledge editing experiment")
    parser.add_argument(
        "--base-model",
        default="agentica-org/DeepScaleR-1.5B-Preview",
        help="Base model name or path",
    )
    parser.add_argument(
        "--traces-before",
        required=True,
        help="Path to baseline traces JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="./experiments/knowledge_editing",
        help="Output directory",
    )
    parser.add_argument(
        "--heuristics",
        nargs="+",
        choices=list(HEURISTICS.keys()),
        help="Specific heuristics to edit (default: all)",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=20,
        help="Number of unstable problems to use",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=50,
        help="Number of rollouts per problem",
    )
    parser.add_argument(
        "--edit-method",
        choices=["lora", "in_context"],
        default="lora",
        help="Knowledge editing method",
    )
    parser.add_argument(
        "--skip-to-step5",
        action="store_true",
        help="Skip to step 5 (assumes traces already generated)",
    )

    args = parser.parse_args()

    # Create experiment
    experiment = KnowledgeEditingExperiment(
        base_model=args.base_model,
        traces_before_path=args.traces_before,
        output_dir=args.output_dir,
        heuristics=args.heuristics,
        n_problems=args.n_problems,
        n_rollouts=args.n_rollouts,
        edit_method=args.edit_method,
    )

    # Run
    if args.skip_to_step5:
        experiment.run_from_step5()
    else:
        experiment.run()


if __name__ == "__main__":
    main()
