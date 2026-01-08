#!/usr/bin/env python3
"""
Training script for mathematical reasoning with RL algorithms.

Usage:
    python train.py --config configs/ppo_config.yaml --model_path <path_to_model>
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.trainer.rl_trainer import RLTrainer, RLTrainerConfig
from training.algorithms.ppo import PPOConfig
from training.algorithms.m2po import M2POConfig
from training.algorithms.grpo import GRPOConfig
from training.integration import MathRewardFunction, MathDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> RLTrainerConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return RLTrainerConfig(**config_dict)


def load_model(model_path: str, device: str = "cuda"):
    """Load model from checkpoint or HuggingFace"""
    logger.info(f"Loading model from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    return model


def load_tokenizer(model_path: str):
    """Load tokenizer"""
    logger.info(f"Loading tokenizer from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_reward_function(config: RLTrainerConfig):
    """
    Create reward function for mathematical reasoning.

    Uses MathRewardFunction from training.integration for outcome-based rewards.
    Ground truths must be provided in the dataset.
    """
    # Create the reward function with outcome-based rewards
    reward_func = MathRewardFunction(
        use_outcome_reward=True,
        use_process_reward=False,
        correct_reward=1.0,
        incorrect_reward=0.0,
    )

    # Wrapper to match the expected signature
    # Note: ground_truths must be extracted from the batch data in the trainer
    def reward_fn(prompts, responses, ground_truths=None):
        """
        Compute rewards for responses.

        Args:
            prompts: List of prompt strings
            responses: List of response strings
            ground_truths: List of correct answers (required)

        Returns:
            rewards: List of reward values (one per response)
        """
        if ground_truths is None:
            logger.warning("No ground truths provided, returning zero rewards")
            return [0.0] * len(responses)

        return reward_func(prompts, responses, ground_truths)

    return reward_fn


def create_dataset(data_path: str, split: str = "train", tokenizer=None):
    """
    Create dataset for training.

    Uses MathDataset from training.integration which supports JSON/JSONL formats
    with fields: 'problem'/'question', 'answer', and optionally 'solution'.

    Args:
        data_path: Path to dataset (JSON or JSONL file)
        split: "train" or "eval" (can be used to select different files)
        tokenizer: Optional tokenizer for preprocessing

    Returns:
        dataset: MathDataset object
    """
    # If split is specified and data_path is a directory, construct the full path
    if os.path.isdir(data_path):
        # Try common naming patterns
        for ext in ['.jsonl', '.json']:
            candidate = os.path.join(data_path, f"{split}{ext}")
            if os.path.exists(candidate):
                data_path = candidate
                break

    logger.info(f"Loading {split} dataset from {data_path}")

    return MathDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=2048,
    )


def main():
    parser = argparse.ArgumentParser(description="Train mathematical reasoning model with RL")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model or HuggingFace model ID",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data",
    )
    parser.add_argument(
        "--reference_model_path",
        type=str,
        default=None,
        help="Path to reference model (if different from model_path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.local_rank >= 0:
        config.local_rank = args.local_rank

    logger.info(f"Training configuration:")
    logger.info(f"  Algorithm: {config.algorithm}")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Device: {config.device}")

    # Load model and tokenizer
    model = load_model(args.model_path, config.device)
    tokenizer = load_tokenizer(args.model_path)

    # Load reference model (if using KL penalty)
    reference_model = None
    if config.use_reference_model:
        ref_path = args.reference_model_path or args.model_path
        logger.info(f"Loading reference model from {ref_path}")
        reference_model = load_model(ref_path, config.device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

    # Create reward function
    reward_fn = create_reward_function(config)

    # Load datasets
    train_dataset = None
    eval_dataset = None

    if args.data_path:
        logger.info(f"Loading training data from {args.data_path}")
        train_dataset = create_dataset(args.data_path, split="train", tokenizer=tokenizer)
        eval_dataset = create_dataset(args.data_path, split="eval", tokenizer=tokenizer)
    else:
        logger.warning("No data path provided - using placeholder dataset")
        # Create placeholder dataset for demonstration
        class PlaceholderDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {"prompt": "Solve: 2 + 2 = "}

        train_dataset = PlaceholderDataset(size=100)
        eval_dataset = PlaceholderDataset(size=20)

    # Create trainer
    trainer = RLTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reference_model=reference_model,
    )

    # Save config
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, "training_config.json"))

    # Start training
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
