"""
RL Trainer for mathematical reasoning with PPO, M2PO, GRPO, etc.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer, TrainerConfig
from ..algorithms.ppo import compute_ppo_loss, PPOConfig
from ..algorithms.m2po import compute_m2po_loss, M2POConfig
from ..algorithms.grpo import compute_grpo_loss, GRPOConfig

logger = logging.getLogger(__name__)


@dataclass
class RLTrainerConfig(TrainerConfig):
    """Configuration for RL training"""

    # Algorithm selection
    algorithm: str = "ppo"  # Options: "ppo", "m2po", "grpo"

    # PPO config
    ppo_config: Optional[Dict[str, Any]] = None

    # M2PO config
    m2po_config: Optional[Dict[str, Any]] = None

    # GRPO config
    grpo_config: Optional[Dict[str, Any]] = None

    # Generation config
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1

    # RL-specific training
    rollout_batch_size: int = 32
    ppo_epochs: int = 4
    mini_batch_size: int = 8

    # Reference model (for KL penalty)
    use_reference_model: bool = True
    reference_model_path: Optional[str] = None

    # Reward computation
    reward_model_path: Optional[str] = None
    use_outcome_reward: bool = True
    use_process_reward: bool = False

    # Value function
    use_value_function: bool = True
    value_model_path: Optional[str] = None

    def __post_init__(self):
        # Create algorithm configs if not provided
        if self.ppo_config is None:
            self.ppo_config = {}
        if self.m2po_config is None:
            self.m2po_config = {}
        if self.grpo_config is None:
            self.grpo_config = {}


class RLTrainer(BaseTrainer):
    """
    Trainer for RL algorithms (PPO, M2PO, GRPO) on mathematical reasoning tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RLTrainerConfig,
        tokenizer: Any,
        reward_fn: Any,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        reference_model: Optional[nn.Module] = None,
        value_model: Optional[nn.Module] = None,
    ):
        super().__init__(model, config, train_dataset, eval_dataset)

        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.reference_model = reference_model
        self.value_model = value_model

        # Load reference model if path provided
        if config.use_reference_model and reference_model is None and config.reference_model_path:
            logger.info(f"Loading reference model from {config.reference_model_path}")
            self.reference_model = self._load_model(config.reference_model_path)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

        # Load value model if using value function
        if config.use_value_function and value_model is None and config.value_model_path:
            logger.info(f"Loading value model from {config.value_model_path}")
            self.value_model = self._load_model(config.value_model_path)

        # Create algorithm config
        self.algo_config = self._create_algorithm_config()

    def _load_model(self, model_path: str) -> nn.Module:
        """Load a model from checkpoint"""
        # This is a placeholder - implement based on your model architecture
        raise NotImplementedError("Model loading needs to be implemented based on your architecture")

    def _create_algorithm_config(self):
        """Create the appropriate algorithm configuration"""
        if self.config.algorithm == "ppo":
            return PPOConfig(**self.config.ppo_config)
        elif self.config.algorithm == "m2po":
            return M2POConfig(**self.config.m2po_config)
        elif self.config.algorithm == "grpo":
            return GRPOConfig(**self.config.grpo_config)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def generate_responses(
        self,
        prompts: List[str],
        num_return_sequences: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate responses for prompts.

        Args:
            prompts: List of prompt strings
            num_return_sequences: Number of responses per prompt

        Returns:
            input_ids: Token IDs including prompt and response
            attention_mask: Attention mask
            response_mask: Mask indicating which tokens are part of the response
        """
        self.model.eval()

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        prompt_length = prompt_encodings.input_ids.shape[1]

        # Generate responses
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=prompt_encodings.input_ids,
                attention_mask=prompt_encodings.attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Create attention mask
        attention_mask = (output_ids != self.tokenizer.pad_token_id).long()

        # Create response mask (1 for generated tokens, 0 for prompt tokens)
        response_mask = torch.zeros_like(output_ids)
        response_mask[:, prompt_length:] = 1
        response_mask = response_mask * attention_mask

        return output_ids, attention_mask, response_mask

    def compute_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for tokens.

        Args:
            model: The model to use
            input_ids: Token IDs, shape (bs, seq_len)
            attention_mask: Attention mask, shape (bs, seq_len)
            response_mask: Response mask, shape (bs, seq_len)

        Returns:
            logprobs: Log probabilities, shape (bs, seq_len)
        """
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs of actual tokens
        logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Pad to match input length
        logprobs = F.pad(logprobs, (1, 0), value=0.0)

        # Mask padding tokens
        logprobs = logprobs * response_mask

        return logprobs

    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute value estimates.

        Args:
            input_ids: Token IDs, shape (bs, seq_len)
            attention_mask: Attention mask, shape (bs, seq_len)

        Returns:
            values: Value estimates, shape (bs, seq_len) or None
        """
        if not self.config.use_value_function or self.value_model is None:
            return None

        with torch.no_grad():
            outputs = self.value_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Assume value model has a value_head that outputs values
            if hasattr(outputs, 'value'):
                values = outputs.value.squeeze(-1)
            else:
                # If no value head, use last hidden state
                values = outputs.last_hidden_state.mean(dim=-1)

        return values

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rewards for responses.

        Args:
            prompts: List of prompt strings
            responses: List of response strings
            response_mask: Response mask, shape (bs, seq_len)

        Returns:
            rewards: Per-token rewards, shape (bs, seq_len)
        """
        # Use the provided reward function
        # This should return rewards aligned with tokens
        rewards = self.reward_fn(prompts, responses)

        # Convert to tensor if needed
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Ensure rewards match response_mask shape
        if rewards.dim() == 1:
            # Outcome reward: broadcast to all tokens
            rewards = rewards.unsqueeze(-1).expand_as(response_mask)

        # Apply mask
        rewards = rewards * response_mask

        return rewards

    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute RL loss for a batch.

        Args:
            model: The policy model
            batch: Batch containing:
                - input_ids: shape (bs, seq_len)
                - attention_mask: shape (bs, seq_len)
                - response_mask: shape (bs, seq_len)
                - old_logprobs: shape (bs, seq_len)
                - old_values: shape (bs, seq_len) or None
                - rewards: shape (bs, seq_len)
                - prompt_indices: shape (bs,) - for GRPO

        Returns:
            loss: The total loss
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        old_logprobs = batch["old_logprobs"]
        old_values = batch.get("old_values")
        rewards = batch["rewards"]

        # Compute current log probs
        logprobs = self.compute_logprobs(model, input_ids, attention_mask, response_mask)

        # Compute reference log probs (for KL penalty)
        if self.reference_model is not None:
            with torch.no_grad():
                ref_logprobs = self.compute_logprobs(
                    self.reference_model, input_ids, attention_mask, response_mask
                )
        else:
            ref_logprobs = torch.zeros_like(logprobs)

        # Compute current values
        if self.config.use_value_function:
            values = self.compute_values(input_ids, attention_mask)
        else:
            values = None

        # Compute algorithm-specific loss
        if self.config.algorithm == "ppo":
            loss_dict = compute_ppo_loss(
                logprobs=logprobs,
                ref_logprobs=ref_logprobs,
                values=values,
                old_logprobs=old_logprobs,
                old_values=old_values,
                rewards=rewards,
                response_mask=response_mask,
                config=self.algo_config,
            )
        elif self.config.algorithm == "m2po":
            loss_dict = compute_m2po_loss(
                logprobs=logprobs,
                ref_logprobs=ref_logprobs,
                values=values,
                old_logprobs=old_logprobs,
                old_values=old_values,
                rewards=rewards,
                response_mask=response_mask,
                config=self.algo_config,
            )
        elif self.config.algorithm == "grpo":
            prompt_indices = batch.get("prompt_indices", np.zeros(input_ids.shape[0]))
            loss_dict = compute_grpo_loss(
                logprobs=logprobs,
                ref_logprobs=ref_logprobs,
                values=values,
                old_logprobs=old_logprobs,
                old_values=old_values,
                rewards=rewards,
                response_mask=response_mask,
                prompt_indices=prompt_indices,
                config=self.algo_config,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        # Log additional metrics
        if self.global_step % self.config.logging_steps == 0:
            for key, value in loss_dict.items():
                if key != "loss/total" and isinstance(value, torch.Tensor):
                    self.log_metrics({key: value.item()}, self.global_step)

        return loss_dict["loss/total"]

    def train(self):
        """
        Main RL training loop with experience collection and PPO-style updates.
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset is required")

        # Create optimizer and scheduler
        num_training_steps = (
            len(self.train_dataset) // self.config.rollout_batch_size
            * self.config.num_epochs
            * self.config.ppo_epochs
        )

        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler(self.optimizer, num_training_steps)

        logger.info("***** Running RL training *****")
        logger.info(f"  Algorithm = {self.config.algorithm}")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.config.num_epochs}")
        logger.info(f"  Rollout batch size = {self.config.rollout_batch_size}")
        logger.info(f"  PPO epochs = {self.config.ppo_epochs}")
        logger.info(f"  Mini batch size = {self.config.mini_batch_size}")

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # Collect experience
            experience_buffer = self.collect_experience()

            # PPO-style updates
            for ppo_epoch in range(self.config.ppo_epochs):
                # Shuffle experience
                indices = torch.randperm(len(experience_buffer["input_ids"]))

                # Mini-batch updates
                for i in range(0, len(indices), self.config.mini_batch_size):
                    mini_batch_indices = indices[i:i + self.config.mini_batch_size]

                    # Extract mini-batch
                    mini_batch = {
                        key: value[mini_batch_indices]
                        for key, value in experience_buffer.items()
                        if isinstance(value, torch.Tensor)
                    }

                    # Training step
                    metrics = self.training_step(mini_batch)

                    # Update weights
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
                        logger.info(f"Evaluation: {eval_metrics}")

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        checkpoint_dir = f"{self.config.output_dir}/checkpoint-{self.global_step}"
                        self.save_checkpoint(checkpoint_dir)
                        self._rotate_checkpoints()

        # Save final model
        final_dir = f"{self.config.output_dir}/final_model"
        self.save_checkpoint(final_dir)

        if self.writer is not None:
            self.writer.close()

        logger.info("Training completed!")

    def collect_experience(self) -> Dict[str, torch.Tensor]:
        """
        Collect experience by generating responses and computing rewards.

        Returns:
            experience_buffer: Dictionary containing:
                - input_ids
                - attention_mask
                - response_mask
                - old_logprobs
                - old_values
                - rewards
                - prompt_indices (for GRPO)
        """
        self.model.eval()

        all_input_ids = []
        all_attention_masks = []
        all_response_masks = []
        all_old_logprobs = []
        all_old_values = []
        all_rewards = []
        all_prompt_indices = []

        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.rollout_batch_size,
            shuffle=True,
        )

        for batch_idx, batch in enumerate(dataloader):
            prompts = batch["prompt"] if isinstance(batch, dict) else batch

            # Generate responses
            input_ids, attention_mask, response_mask = self.generate_responses(
                prompts,
                num_return_sequences=self.config.num_return_sequences,
            )

            # Compute log probs
            with torch.no_grad():
                old_logprobs = self.compute_logprobs(
                    self.model, input_ids, attention_mask, response_mask
                )

                # Compute values
                old_values = self.compute_values(input_ids, attention_mask)

            # Decode responses for reward computation
            responses = self.tokenizer.batch_decode(
                input_ids * response_mask.long(),
                skip_special_tokens=True,
            )

            # Compute rewards
            rewards = self.compute_rewards(prompts, responses, response_mask)

            # Store experience
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_response_masks.append(response_mask)
            all_old_logprobs.append(old_logprobs)
            if old_values is not None:
                all_old_values.append(old_values)
            all_rewards.append(rewards)

            # For GRPO: track which prompt each response belongs to
            if self.config.algorithm == "grpo":
                prompt_indices = np.repeat(
                    np.arange(len(prompts)),
                    self.config.num_return_sequences
                )
                all_prompt_indices.append(torch.tensor(prompt_indices))

        # Concatenate all experiences
        experience_buffer = {
            "input_ids": torch.cat(all_input_ids, dim=0),
            "attention_mask": torch.cat(all_attention_masks, dim=0),
            "response_mask": torch.cat(all_response_masks, dim=0),
            "old_logprobs": torch.cat(all_old_logprobs, dim=0),
            "rewards": torch.cat(all_rewards, dim=0),
        }

        if all_old_values:
            experience_buffer["old_values"] = torch.cat(all_old_values, dim=0)

        if all_prompt_indices:
            experience_buffer["prompt_indices"] = torch.cat(all_prompt_indices, dim=0).numpy()

        logger.info(f"Collected {len(experience_buffer['input_ids'])} experiences")

        return experience_buffer
