#!/usr/bin/env python3
"""
Simple example script for training with M2PO algorithm.

This demonstrates how to set up and run M2PO training for mathematical reasoning.
"""

import os
import sys
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.rl_trainer import RLTrainer, RLTrainerConfig
from algorithms.m2po import M2POConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_reward_fn():
    """
    Create a reward function for mathematical reasoning.

    In practice, this would evaluate the correctness of mathematical solutions.
    """
    def reward_fn(prompts, responses, ground_truths=None):
        """
        Compute rewards for generated responses.

        Args:
            prompts: List of problem prompts
            responses: List of generated solutions
            ground_truths: Optional list of correct answers

        Returns:
            List of reward values
        """
        rewards = []
        for i, response in enumerate(responses):
            # Simple example: reward based on response length and completion
            # In practice, this would check mathematical correctness
            reward = 0.0

            # Basic heuristics (replace with actual math evaluation)
            if len(response.strip()) > 10:
                reward += 0.5
            if "=" in response:
                reward += 0.2
            if response.strip().endswith("."):
                reward += 0.1

            # If ground truth available, check correctness
            if ground_truths and i < len(ground_truths):
                # This is where you'd implement actual answer verification
                # For now, just a placeholder
                pass

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    return reward_fn


def create_simple_dataset():
    """
    Create a simple synthetic dataset for demonstration.

    In practice, load from actual math problem datasets like GSM8K, MATH, etc.
    """
    # Simple arithmetic problems
    problems = [
        "What is 15 + 27?",
        "Calculate 8 * 9.",
        "What is 100 - 43?",
        "Solve: 2x + 5 = 13",
        "What is the area of a rectangle with width 5 and length 8?",
    ]

    answers = [
        "42",
        "72",
        "57",
        "x = 4",
        "40",
    ]

    # Create a simple dataset dictionary
    dataset = [
        {"prompt": p, "answer": a}
        for p, a in zip(problems, answers)
    ]

    return dataset


def main():
    """Main training function."""

    # Configuration
    model_name = "gpt2"  # Use a small model for testing, replace with math-tuned model
    output_dir = "./outputs/m2po_example"

    logger.info("Loading model and tokenizer...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    # Create dataset
    logger.info("Creating dataset...")
    train_dataset = create_simple_dataset()
    eval_dataset = create_simple_dataset()[:2]  # Use subset for eval

    # Create reward function
    logger.info("Creating reward function...")
    reward_fn = create_reward_fn()

    # Configure M2PO
    m2po_config = {
        "m2_budget": 0.01,  # M2 (KL^2) budget per harmful token
        "miniclip_low": 0.3,
        "miniclip_high": 0.5,
        "loss_agg_mode": "token-mean",
    }

    # Configure trainer
    config = RLTrainerConfig(
        algorithm="m2po",
        m2po_config=m2po_config,

        # Training parameters
        num_train_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=10,

        # RL-specific
        rollout_batch_size=4,
        ppo_epochs=2,
        mini_batch_size=2,

        # Generation
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,

        # Output
        output_dir=output_dir,
        logging_steps=1,
        eval_steps=5,
        save_steps=10,

        # Reference model (for KL penalty)
        use_reference_model=True,

        # Rewards
        use_outcome_reward=True,
    )

    logger.info("Initializing trainer...")

    # Create trainer
    trainer = RLTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting M2PO training...")
    logger.info(f"Algorithm: {config.algorithm}")
    logger.info(f"M2 Budget: {m2po_config['m2_budget']}")
    logger.info(f"Output directory: {output_dir}")

    # Train
    try:
        trainer.train()
        logger.info("Training completed successfully!")

        # Save final model
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_checkpoint(final_model_path)
        logger.info(f"Model saved to {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
