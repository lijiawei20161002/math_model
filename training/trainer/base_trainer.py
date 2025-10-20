"""
Base trainer class for mathematical reasoning models
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Base configuration for training"""
    # Model
    model_name: str = "math_model"
    model_path: Optional[str] = None

    # Training
    num_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Optimization
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate schedule
    lr_scheduler: str = "linear"  # "linear", "cosine", "constant"

    # Logging and checkpointing
    output_dir: str = "./outputs"
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    log_to_tensorboard: bool = True

    # Evaluation
    eval_accumulation_steps: Optional[int] = None
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank: int = -1

    # Random seed
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainerConfig":
        """Load config from dictionary"""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: str) -> "TrainerConfig":
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseTrainer(ABC):
    """
    Base trainer class that handles common training logic.
    Subclasses should implement compute_loss and optionally override other methods.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        data_collator: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        # Set random seed
        self._set_seed(config.seed)

        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Setup logging
        self.writer = None
        if config.log_to_tensorboard:
            log_dir = os.path.join(config.output_dir, "logs")
            self.writer = SummaryWriter(log_dir=log_dir)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None

        # Optimizer and scheduler (to be initialized in train())
        self.optimizer = None
        self.lr_scheduler = None

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        if self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        return optimizer

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        """Create learning rate scheduler"""
        from torch.optim.lr_scheduler import LambdaLR
        import math

        if self.config.lr_scheduler == "linear":
            def lr_lambda(current_step: int):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step) / float(max(1, num_training_steps - self.config.warmup_steps))
                )
        elif self.config.lr_scheduler == "cosine":
            def lr_lambda(current_step: int):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                progress = float(current_step - self.config.warmup_steps) / float(max(1, num_training_steps - self.config.warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif self.config.lr_scheduler == "constant":
            def lr_lambda(current_step: int):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return 1.0
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")

        return LambdaLR(optimizer, lr_lambda)

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a batch. Must be implemented by subclasses.

        Args:
            model: The model
            batch: A batch of data

        Returns:
            loss: The loss tensor
        """
        pass

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: A batch of data

        Returns:
            metrics: Dictionary of metrics
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        loss = self.compute_loss(self.model, batch)

        # Backward pass
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}

        self.model.eval()
        eval_dataloader = self._get_eval_dataloader()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                loss = self.compute_loss(self.model, batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"eval_loss": avg_loss}

    def _get_train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def _get_eval_dataloader(self) -> DataLoader:
        """Create evaluation dataloader"""
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)

        # Save config
        config_path = os.path.join(output_dir, "trainer_config.json")
        self.config.save(config_path)

        # Save training state
        state_path = os.path.join(output_dir, "trainer_state.json")
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, input_dir: str):
        """Load model checkpoint"""
        # Load model
        model_path = os.path.join(input_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {model_path}")

        # Load training state
        state_path = os.path.join(input_dir, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_metric = state["best_metric"]
            logger.info(f"Training state loaded from {state_path}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to tensorboard"""
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def train(self):
        """Main training loop"""
        if self.train_dataset is None:
            raise ValueError("Training dataset is required")

        # Create dataloaders
        train_dataloader = self._get_train_dataloader()

        # Calculate total training steps
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.config.num_epochs

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler(self.optimizer, num_training_steps)

        logger.info(f"***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.config.num_epochs}")
        logger.info(f"  Batch size = {self.config.train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {num_training_steps}")

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_iterator = train_dataloader

            for step, batch in enumerate(epoch_iterator):
                # Training step
                metrics = self.training_step(batch)

                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                        self.log_metrics(metrics, self.global_step)
                        logger.info(f"Step {self.global_step}: {metrics}")

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self.log_metrics(eval_metrics, self.global_step)
                        logger.info(f"Evaluation at step {self.global_step}: {eval_metrics}")

                        # Save best model
                        metric_value = eval_metrics.get(self.config.metric_for_best_model)
                        if metric_value is not None:
                            is_best = (
                                self.best_metric is None or
                                (self.config.greater_is_better and metric_value > self.best_metric) or
                                (not self.config.greater_is_better and metric_value < self.best_metric)
                            )
                            if is_best:
                                self.best_metric = metric_value
                                best_model_dir = os.path.join(self.config.output_dir, "best_model")
                                self.save_checkpoint(best_model_dir)

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                        self.save_checkpoint(checkpoint_dir)

                        # Clean up old checkpoints
                        self._rotate_checkpoints()

        # Save final model
        final_model_dir = os.path.join(self.config.output_dir, "final_model")
        self.save_checkpoint(final_model_dir)

        if self.writer is not None:
            self.writer.close()

        logger.info("Training completed!")

    def _rotate_checkpoints(self):
        """Keep only the last N checkpoints"""
        if self.config.save_total_limit is None or self.config.save_total_limit <= 0:
            return

        # Find all checkpoints
        checkpoints = []
        for item in os.listdir(self.config.output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(self.config.output_dir, item)
                if os.path.isdir(checkpoint_path):
                    try:
                        step = int(item.split("-")[-1])
                        checkpoints.append((step, checkpoint_path))
                    except ValueError:
                        continue

        # Sort by step
        checkpoints.sort(key=lambda x: x[0])

        # Remove old checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            _, checkpoint_to_remove = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {checkpoint_to_remove}")
            import shutil
            shutil.rmtree(checkpoint_to_remove)
