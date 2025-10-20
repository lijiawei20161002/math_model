#!/usr/bin/env python3
"""
Distributed training script for mathematical reasoning with RL algorithms.

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=8 train_distributed.py --config configs/ppo_config.yaml --model_path <path>

    # Multiple nodes
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<addr> --master_port=<port> \\
        train_distributed.py --config configs/ppo_config.yaml --model_path <path>
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.trainer.rl_trainer import RLTrainer, RLTrainerConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    else:
        local_rank = -1
        world_size = 1
        rank = 0

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        # Set up logging only on main process
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)

    return local_rank, world_size, rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> RLTrainerConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return RLTrainerConfig(**config_dict)


def load_model(model_path: str, local_rank: int = -1):
    """Load model from checkpoint or HuggingFace"""
    logger.info(f"Loading model from {model_path}")

    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    model = model.to(device)

    # Wrap with DDP if distributed
    if local_rank != -1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    return model


def load_tokenizer(model_path: str):
    """Load tokenizer"""
    logger.info(f"Loading tokenizer from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_reward_function(config: RLTrainerConfig):
    """
    Create reward function for mathematical reasoning.
    """

    def reward_fn(prompts, responses):
        """
        Compute rewards for responses.
        """
        # TODO: Implement actual reward computation
        import random
        return [random.random() for _ in responses]

    return reward_fn


def create_dataset(data_path: str, split: str = "train"):
    """
    Create dataset for training.
    """
    # TODO: Implement dataset loading

    class MathDataset(torch.utils.data.Dataset):
        def __init__(self, data_path, split):
            self.data = []
            logger.warning("Using placeholder dataset - implement actual data loading")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {"prompt": self.data[idx]}

    return MathDataset(data_path, split)


class DistributedRLTrainer(RLTrainer):
    """
    Distributed version of RLTrainer.
    """

    def __init__(self, *args, world_size: int = 1, rank: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = world_size
        self.rank = rank
        self.is_main_process = (rank == 0)

    def _get_train_dataloader(self):
        """Create distributed training dataloader"""
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )

    def _get_eval_dataloader(self):
        """Create distributed evaluation dataloader"""
        sampler = DistributedSampler(
            self.eval_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        from torch.utils.data import DataLoader
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )

    def save_checkpoint(self, output_dir: str):
        """Save checkpoint only on main process"""
        if self.is_main_process:
            # Unwrap DDP model
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            super().save_checkpoint(output_dir)

        # Wait for main process to finish saving
        if dist.is_initialized():
            dist.barrier()

    def log_metrics(self, metrics, step):
        """Log metrics only on main process"""
        if self.is_main_process:
            super().log_metrics(metrics, step)


def main():
    parser = argparse.ArgumentParser(description="Distributed RL training for math reasoning")

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
        help="Path to reference model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override with command-line arguments
        if args.output_dir:
            config.output_dir = args.output_dir
        config.local_rank = local_rank
        config.device = f"cuda:{local_rank}" if local_rank != -1 else "cuda"

        if rank == 0:
            logger.info(f"Distributed training configuration:")
            logger.info(f"  World size: {world_size}")
            logger.info(f"  Rank: {rank}")
            logger.info(f"  Local rank: {local_rank}")
            logger.info(f"  Algorithm: {config.algorithm}")
            logger.info(f"  Model: {args.model_path}")
            logger.info(f"  Output: {config.output_dir}")

        # Load model and tokenizer
        model = load_model(args.model_path, local_rank)
        tokenizer = load_tokenizer(args.model_path)

        # Load reference model
        reference_model = None
        if config.use_reference_model:
            ref_path = args.reference_model_path or args.model_path
            if rank == 0:
                logger.info(f"Loading reference model from {ref_path}")
            reference_model = load_model(ref_path, local_rank)
            reference_model.eval()
            for param in reference_model.parameters():
                param.requires_grad = False

        # Create reward function
        reward_fn = create_reward_function(config)

        # Load datasets
        if args.data_path:
            if rank == 0:
                logger.info(f"Loading data from {args.data_path}")
            train_dataset = create_dataset(args.data_path, split="train")
            eval_dataset = create_dataset(args.data_path, split="eval")
        else:
            if rank == 0:
                logger.warning("Using placeholder dataset")

            class PlaceholderDataset(torch.utils.data.Dataset):
                def __init__(self, size=100):
                    self.size = size

                def __len__(self):
                    return self.size

                def __getitem__(self, idx):
                    return {"prompt": "Solve: 2 + 2 = "}

            train_dataset = PlaceholderDataset(size=100)
            eval_dataset = PlaceholderDataset(size=20)

        # Create distributed trainer
        trainer = DistributedRLTrainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reference_model=reference_model,
            world_size=world_size,
            rank=rank,
        )

        # Save config on main process
        if rank == 0:
            os.makedirs(config.output_dir, exist_ok=True)
            config.save(os.path.join(config.output_dir, "training_config.json"))

        # Synchronize before training
        if dist.is_initialized():
            dist.barrier()

        # Start training
        if rank == 0:
            logger.info("Starting distributed training...")
        trainer.train()

        if rank == 0:
            logger.info("Training completed!")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
