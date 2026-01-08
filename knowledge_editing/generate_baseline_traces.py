#!/usr/bin/env python3
"""
Generate baseline traces for knowledge editing experiments.

This script generates the initial traces needed for the knowledge editing pipeline.
It samples from a base model (no editing) to establish baseline stability metrics.

Usage:
    python generate_baseline_traces.py \\
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --dataset aime \\
        --output baseline_traces.json \\
        --num-problems 50 \\
        --num-rollouts 50
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.sample import generate_cot_traces_async
from datasets import load_dataset, concatenate_datasets


def generate_control_prompt(target_length: int = 1000) -> str:
    """
    Generate a control prompt with irrelevant content for fair comparison.

    The control prompt contains generic math advice that doesn't help with
    specific problem-solving strategies, but matches the length of in-context
    learning prompts.

    Args:
        target_length: Approximate target length in characters

    Returns:
        Control prompt string
    """
    control_text = """Here are some general problem-solving guidelines:

Example 1:
Problem: When solving mathematical problems, remember to read carefully.
Solution:
Step 1: Read the problem statement carefully.
Step 2: Identify what is being asked.
Step 3: Write down relevant information.
Step 4: Consider different approaches.
Step 5: Execute your chosen approach step by step.
Step 6: Check your work.

Example 2:
Problem: Organization is key to mathematical problem-solving success.
Solution:
Step 1: Create a clear workspace.
Step 2: Write neatly so you can follow your work.
Step 3: Label variables and quantities clearly.
Step 4: Show all intermediate steps.
Step 5: Double-check calculations.
Step 6: Verify your final answer makes sense.

Example 3:
Problem: Developing good mathematical problem-solving habits takes practice.
Solution:
Step 1: Start with simpler problems to build confidence.
Step 2: Don't give up when facing difficult problems.
Step 3: Learn from mistakes by reviewing incorrect solutions.
Step 4: Practice regularly to maintain skills.
Step 5: Seek help when stuck on a concept.
Step 6: Apply multiple strategies when one doesn't work.

"""

    # Repeat or truncate to match target length
    if len(control_text) < target_length:
        repeats = (target_length // len(control_text)) + 1
        control_text = (control_text * repeats)[:target_length]
    else:
        control_text = control_text[:target_length]

    return control_text


def load_math_dataset(dataset_name: str, split: str = "test"):
    """
    Load a mathematical reasoning dataset.

    Args:
        dataset_name: Name of dataset ("aime", "amc", or HF dataset path)
        split: Dataset split to load

    Returns:
        Dataset object
    """
    if dataset_name.lower() == "aime":
        print("Loading AIME 2025 dataset...")
        aime_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split=split)
        aime_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split=split)
        dataset = concatenate_datasets([aime_i, aime_ii])
        print(f"Loaded {len(dataset)} AIME problems")

    elif dataset_name.lower() == "amc":
        print("Loading AMC dataset...")
        dataset = load_dataset("AI-MO/aimo-validation-amc", split=split)
        print(f"Loaded {len(dataset)} AMC problems")

    else:
        # Assume it's a HuggingFace dataset path
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(dataset)} problems")

    return dataset


async def generate_baseline_traces_async(
    model: str,
    dataset_name: str,
    output_path: str,
    num_problems: int = 50,
    num_rollouts: int = 50,
    start_idx: int = 0,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 20480,
    batch_size: int = 1,
    max_concurrent: int = 1,
    control_prompt: Optional[str] = None,
):
    """
    Generate baseline traces for knowledge editing experiments.

    Args:
        model: Model name or path
        dataset_name: Name of dataset to use
        output_path: Where to save traces
        num_problems: Number of problems to evaluate
        num_rollouts: Number of rollouts per problem
        start_idx: Starting index in dataset
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        max_concurrent: Max concurrent requests
        control_prompt: Optional control prompt to match edited condition's prompt length.
                       For fair comparison with in-context learning, should be
                       same length as the edited prompts but with irrelevant content.
    """
    print("\n" + "=" * 70)
    print("BASELINE TRACE GENERATION")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_path}")
    print(f"Problems: {num_problems} (starting from index {start_idx})")
    print(f"Rollouts per problem: {num_rollouts}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 70)

    # Load dataset
    dataset = load_math_dataset(dataset_name)

    # Compute end index
    end_idx = min(start_idx + num_problems, len(dataset))
    actual_problems = end_idx - start_idx

    print(f"\nGenerating traces for problems [{start_idx}:{end_idx}] ({actual_problems} problems)")

    if control_prompt:
        print(f"⚠ Using control prompt (length: {len(control_prompt)} chars)")
        print("  This is for fair comparison with in-context learning experiments.")
    else:
        print(f"⚠ No control prompt specified - baseline will have shorter prompts than edited condition!")
        print("  For fair comparison with in-context learning, consider using --control-prompt")

    print(f"NOTE: Make sure model is served at http://localhost:8000")
    print(f"To start server: vllm serve {model} --port 8000\n")

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate traces
    await generate_cot_traces_async(
        dataset_split=dataset,
        output_path=output_path,
        start_idx=start_idx,
        end_idx=end_idx,
        password=None,
        instruction=control_prompt,  # Use control prompt to match edited condition
        batch_size=batch_size,
        max_concurrent_requests=max_concurrent,
        samples_per_question=num_rollouts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        model=model,
    )

    print(f"\n✓ Baseline traces saved to: {output_path}")
    print(f"  Total problems: {actual_problems}")
    print(f"  Rollouts per problem: {num_rollouts}")
    print(f"  Total traces: {actual_problems * num_rollouts}")

    # Print summary statistics
    print("\nComputing summary statistics...")
    with open(output_path, 'r') as f:
        traces = json.load(f)

    if isinstance(traces, list) and len(traces) > 0:
        # Compute basic stats
        num_with_answers = sum(1 for t in traces if t.get("final_answers"))
        print(f"  Problems with answers: {num_with_answers}/{len(traces)}")

        # Check answer entropy (if we have the stability metrics module)
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from stability_metrics import analyze_traces_file

            print("\nRunning stability analysis...")
            stats = analyze_traces_file(output_path)
            print(f"  Mean entropy: {stats['mean_entropy']:.3f}")
            print(f"  Mean top-1 share: {stats['mean_top1_share']:.2%}")
            print(f"  Mean correctness: {stats['mean_correctness']:.2%}")
        except Exception as e:
            print(f"  (Could not compute detailed stats: {e})")

    print("\n" + "=" * 70)
    print("BASELINE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review traces: {output_path}")
    print(f"2. Run knowledge editing experiment:")
    print(f"   python run_experiment.py \\")
    print(f"     --traces-before {output_path} \\")
    print(f"     --output-dir ./experiments/my_experiment \\")
    print(f"     --heuristics modular_multiplication \\")
    print(f"     --edit-method lora")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline traces for knowledge editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 problems with 50 rollouts each
  python generate_baseline_traces.py \\
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
    --dataset aime \\
    --output baseline_traces.json \\
    --num-problems 50 \\
    --num-rollouts 50

  # Quick test with 5 problems
  python generate_baseline_traces.py \\
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
    --dataset aime \\
    --output test_traces.json \\
    --num-problems 5 \\
    --num-rollouts 10

  # Generate for specific problem range
  python generate_baseline_traces.py \\
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
    --dataset aime \\
    --output baseline_traces.json \\
    --start-idx 10 \\
    --num-problems 20
        """
    )

    # Required arguments
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or path (must be available via vLLM at localhost:8000)"
    )
    parser.add_argument(
        "--dataset",
        default="aime",
        help="Dataset name: 'aime', 'amc', or HuggingFace dataset path"
    )
    parser.add_argument(
        "--output",
        default="baseline_traces.json",
        help="Output file path for traces"
    )

    # Experiment parameters
    parser.add_argument(
        "--num-problems",
        type=int,
        default=50,
        help="Number of problems to evaluate (default: 50)"
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=50,
        help="Number of rollouts per problem (default: 50)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset (default: 0)"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20480,
        help="Maximum tokens to generate (default: 20480)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Max concurrent requests (default: 1)"
    )
    parser.add_argument(
        "--control-prompt",
        type=str,
        default=None,
        help="Control prompt for fair comparison with in-context learning. "
             "Should match the length of edited condition's prompt. "
             "If not provided, baseline will have no extra instructions."
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_problems <= 0:
        print("ERROR: --num-problems must be positive")
        sys.exit(1)

    if args.num_rollouts <= 0:
        print("ERROR: --num-rollouts must be positive")
        sys.exit(1)

    # Check if output file already exists
    if os.path.exists(args.output):
        response = input(f"\nWARNING: {args.output} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Run generation
    try:
        asyncio.run(generate_baseline_traces_async(
            model=args.model,
            dataset_name=args.dataset,
            output_path=args.output,
            num_problems=args.num_problems,
            num_rollouts=args.num_rollouts,
            start_idx=args.start_idx,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            control_prompt=args.control_prompt,
        ))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
