#!/usr/bin/env python3
"""
Compare PPO, M2PO, and GRPO algorithms on mathematical reasoning tasks.

This script demonstrates the differences between the three algorithms
and helps you choose which one to use for your use case.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.rl_trainer import RLTrainerConfig

# Algorithm configurations
ALGORITHM_CONFIGS = {
    "ppo": {
        "algorithm": "ppo",
        "ppo_config": {
            "clip_ratio": 0.2,
            "value_clip": 0.2,
            "entropy_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 1.0,
            "use_gae": True,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "normalize_advantages": True,
        },
        "description": """
        PPO (Proximal Policy Optimization):
        - Standard RL algorithm with clipped surrogate objective
        - Uses GAE (Generalized Advantage Estimation) for advantage computation
        - Requires value function for advantage estimation
        - Good baseline, well-established and stable

        Best for:
        - General-purpose RL training
        - When you have a good value function
        - Stable, predictable training
        """,
    },

    "m2po": {
        "algorithm": "m2po",
        "m2po_config": {
            "m2_budget": 0.01,  # M2 (KL^2) budget per harmful token
            "miniclip_low": 0.3,
            "miniclip_high": 0.5,
            "loss_agg_mode": "token-mean",
        },
        "description": """
        M2PO (Mathematical Measure Optimization for Policy Optimization):
        - Uses second-order KL constraint (M2/KL^2 budget)
        - Adaptively computes per-token clipping bounds
        - Focuses on controlling "harmful" tokens that hurt the objective
        - More sophisticated than PPO's fixed clipping

        Best for:
        - Mathematical reasoning tasks
        - When you want adaptive, data-driven clipping
        - Controlling policy divergence more precisely
        - Reducing harmful updates while allowing helpful ones
        """,
    },

    "grpo": {
        "algorithm": "grpo",
        "grpo_config": {
            "clip_ratio": 0.2,
            "value_clip": 0.2,
            "entropy_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 1.0,
            "advantage_method": "rloo",  # "rloo", "reinforce_pp", "direct"
            "advantage_epsilon": 1e-6,
            "whitening": True,
        },
        "description": """
        GRPO (Group Relative Policy Optimization):
        - Computes advantages by comparing responses within the same group/prompt
        - Uses RLOO (Reinforce Leave One Out) or similar group-based baselines
        - Reduces variance by using group statistics
        - Doesn't require a separate value function

        Best for:
        - When you can generate multiple responses per prompt
        - Reducing variance without a value function
        - Batch evaluation scenarios
        - Mathematical problems with multiple solution attempts
        """,
    },
}


def print_algorithm_comparison():
    """Print a detailed comparison of the three algorithms."""

    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON: PPO vs M2PO vs GRPO")
    print("=" * 80 + "\n")

    for algo_name, info in ALGORITHM_CONFIGS.items():
        print(f"\n{'─' * 80}")
        print(f"  {algo_name.upper()}")
        print(f"{'─' * 80}")
        print(info["description"])
        print(f"\nConfiguration:")
        config_key = f"{algo_name}_config"
        for key, value in info[config_key].items():
            print(f"  • {key}: {value}")
        print()

    print("\n" + "=" * 80)
    print("KEY DIFFERENCES")
    print("=" * 80 + "\n")

    print("""
    1. Clipping Strategy:
       • PPO: Fixed clip ratio (e.g., 0.2)
       • M2PO: Adaptive clipping based on M2 budget (KL^2 constraint)
       • GRPO: Fixed clip ratio, but group-relative advantages

    2. Advantage Computation:
       • PPO: Uses GAE with value function
       • M2PO: Can use various methods (typically GAE)
       • GRPO: Group-relative (RLOO, Reinforce++), no value function needed

    3. Computational Requirements:
       • PPO: Moderate (needs value function)
       • M2PO: Moderate (needs value function + M2 computation)
       • GRPO: Lower (no value function, but needs multiple samples per prompt)

    4. Variance Reduction:
       • PPO: GAE with value function
       • M2PO: GAE + adaptive clipping
       • GRPO: Group statistics (leave-one-out)

    5. Best Use Case:
       • PPO: General RL, stable baseline
       • M2PO: Math reasoning, adaptive control, precision tuning
       • GRPO: Batch generation, multiple attempts, no value function
    """)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80 + "\n")

    print("""
    Start with M2PO if:
    ✓ You're working on mathematical reasoning
    ✓ You want state-of-the-art performance
    ✓ You can tune the M2 budget hyperparameter

    Use GRPO if:
    ✓ You can generate multiple responses per prompt efficiently
    ✓ You want to avoid training a value function
    ✓ You have batch evaluation capabilities

    Fall back to PPO if:
    ✓ You want a well-tested, stable baseline
    ✓ You're doing initial experimentation
    ✓ You need predictable, documented behavior
    """)

    print("=" * 80 + "\n")


def create_config_for_algorithm(algorithm: str, base_config: Dict[str, Any]) -> RLTrainerConfig:
    """
    Create a trainer configuration for the specified algorithm.

    Args:
        algorithm: One of "ppo", "m2po", "grpo"
        base_config: Base configuration dict with common settings

    Returns:
        RLTrainerConfig configured for the algorithm
    """
    if algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(ALGORITHM_CONFIGS.keys())}")

    algo_info = ALGORITHM_CONFIGS[algorithm]
    config_dict = {**base_config, **algo_info}

    # Remove description field
    config_dict.pop("description", None)

    return RLTrainerConfig(**config_dict)


def main():
    """Main function to display algorithm comparison."""
    print_algorithm_comparison()

    # Example: Create configs for each algorithm
    print("\nExample: Creating configurations for all algorithms...\n")

    base_config = {
        "num_train_epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "output_dir": "./outputs",
        "rollout_batch_size": 8,
        "ppo_epochs": 4,
    }

    for algo in ["ppo", "m2po", "grpo"]:
        try:
            config = create_config_for_algorithm(algo, base_config)
            print(f"✓ {algo.upper()} config created successfully")
        except Exception as e:
            print(f"✗ Failed to create {algo.upper()} config: {e}")


if __name__ == "__main__":
    main()
