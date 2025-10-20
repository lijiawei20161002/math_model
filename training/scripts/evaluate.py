#!/usr/bin/env python3
"""
Evaluation script for mathematical reasoning models.

Usage:
    python evaluate.py --model_path <path> --data_path <path> --output_path <path>
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.integration import (
    MathRewardFunction,
    MathDataset,
    format_problem_prompt,
    extract_metrics_from_responses,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def generate_solutions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    batch_size: int = 8,
    device: str = "cuda",
) -> List[str]:
    """
    Generate solutions for a list of problems.

    Args:
        model: The language model
        tokenizer: The tokenizer
        problems: List of problem prompts
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        num_return_sequences: Number of solutions per problem
        batch_size: Batch size for generation
        device: Device to use

    Returns:
        List of generated solutions
    """
    model.eval()
    all_solutions = []

    # Process in batches
    for i in tqdm(range(0, len(problems), batch_size), desc="Generating solutions"):
        batch_problems = problems[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_problems,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        solutions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove the prompt from solutions
        for j, problem in enumerate(batch_problems):
            for k in range(num_return_sequences):
                idx = j * num_return_sequences + k
                solution = solutions[idx]
                # Strip the prompt
                if solution.startswith(problem):
                    solution = solution[len(problem):].strip()
                all_solutions.append(solution)

    return all_solutions


def evaluate_model(
    model_path: str,
    data_path: str,
    output_path: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    batch_size: int = 8,
    device: str = "cuda",
    few_shot_examples: List[Dict[str, str]] = None,
):
    """
    Evaluate a model on mathematical reasoning problems.

    Args:
        model_path: Path to model checkpoint
        data_path: Path to evaluation data
        output_path: Path to save results
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        num_return_sequences: Number of solutions per problem
        batch_size: Batch size for generation
        device: Device to use
        few_shot_examples: Optional few-shot examples
    """
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading evaluation data from {data_path}")
    dataset = MathDataset(data_path, tokenizer)

    # Prepare prompts
    logger.info("Formatting prompts")
    prompts = []
    ground_truths = []

    for i in range(len(dataset)):
        example = dataset[i]
        prompt = format_problem_prompt(
            example['prompt'],
            few_shot_examples=few_shot_examples,
        )
        prompts.append(prompt)
        ground_truths.append(example['answer'])

    # Generate solutions
    logger.info(f"Generating solutions (num_return_sequences={num_return_sequences})")
    solutions = generate_solutions(
        model=model,
        tokenizer=tokenizer,
        problems=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        batch_size=batch_size,
        device=device,
    )

    # For multiple solutions per problem, we need to replicate ground truths
    if num_return_sequences > 1:
        ground_truths_expanded = []
        for truth in ground_truths:
            ground_truths_expanded.extend([truth] * num_return_sequences)
        ground_truths = ground_truths_expanded

    # Evaluate
    logger.info("Evaluating solutions")
    reward_fn = MathRewardFunction()
    metrics = extract_metrics_from_responses(solutions, ground_truths, reward_fn)

    logger.info(f"Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"  Answer Rate: {metrics['answer_rate']:.2%}")
    logger.info(f"  Correct: {metrics['num_correct']}/{metrics['num_total']}")

    # Save results
    results = {
        "model_path": model_path,
        "data_path": data_path,
        "metrics": metrics,
        "config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": num_return_sequences,
        },
        "predictions": []
    }

    # Save individual predictions
    for i in range(0, len(solutions), num_return_sequences):
        problem_solutions = solutions[i:i + num_return_sequences]
        results["predictions"].append({
            "prompt": prompts[i // num_return_sequences],
            "ground_truth": ground_truths[i],
            "solutions": problem_solutions,
            "predicted_answers": [reward_fn.extract_answer(sol) for sol in problem_solutions],
        })

    logger.info(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate mathematical reasoning model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to evaluation data (JSON or JSONL)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 for greedy decoding)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of solutions to generate per problem",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--few_shot_examples",
        type=str,
        default=None,
        help="Path to few-shot examples (JSON)",
    )

    args = parser.parse_args()

    # Load few-shot examples if provided
    few_shot_examples = None
    if args.few_shot_examples:
        with open(args.few_shot_examples, 'r') as f:
            few_shot_examples = json.load(f)

    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        batch_size=args.batch_size,
        device=args.device,
        few_shot_examples=few_shot_examples,
    )


if __name__ == "__main__":
    main()
