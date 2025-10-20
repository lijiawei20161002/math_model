"""
GRPO (Group Relative Policy Optimization) Algorithm

GRPO computes advantages by comparing each response against other responses
from the same prompt (group), using RLOO (Reinforce Leave One Out) or similar
group-based baselines.

Reference: Based on implementations from the M2PO codebase.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .utils import mask_pad, compute_advantages, compute_rewards


@dataclass
class GRPOConfig:
    """Configuration for GRPO algorithm"""
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # GRPO-specific parameters
    advantage_method: str = "rloo"  # Options: "rloo", "reinforce_pp", "direct"
    advantage_epsilon: float = 1e-6
    whitening: bool = True

    # KL penalty
    kl_coef: float = 0.0
    target_kl: Optional[float] = None

    def __post_init__(self):
        valid_methods = {"rloo", "reinforce_pp", "direct"}
        if self.advantage_method not in valid_methods:
            raise ValueError(f"advantage_method must be one of {valid_methods}")


def compute_rloo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_indices: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RLOO (Reinforce Leave One Out) advantage.

    For each response, the baseline is the mean of all OTHER responses from
    the same prompt. This reduces variance while maintaining unbiasedness.

    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        prompt_indices: shape (bs,) - which prompt each response belongs to
        epsilon: small value for numerical stability

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    # Sum rewards across tokens to get total reward per response
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # Group responses by prompt
    id2scores = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        batch_size = scores.shape[0]

        # Collect all scores for each prompt
        for i in range(batch_size):
            prompt_id = prompt_indices[i]
            id2scores[prompt_id].append(scores[i])

        # Compute mean score for each prompt
        for prompt_id in id2scores:
            if len(id2scores[prompt_id]) == 1:
                # Only one response for this prompt - use zero baseline
                id2mean[prompt_id] = torch.tensor(0.0, device=scores.device)
            elif len(id2scores[prompt_id]) > 1:
                id2mean[prompt_id] = torch.mean(torch.stack(id2scores[prompt_id]))
            else:
                raise ValueError(f"No scores for prompt index: {prompt_id}")

        # Compute RLOO advantage: score * n/(n-1) - mean * n/(n-1)
        # This is the leave-one-out estimator
        for i in range(batch_size):
            prompt_id = prompt_indices[i]
            n_responses = len(id2scores[prompt_id])

            if n_responses > 1:
                # RLOO formula: (score - baseline) * n/(n-1) where baseline excludes current
                # Equivalent to: score * n/(n-1) - group_mean * n/(n-1)
                scores[i] = (scores[i] * n_responses - id2mean[prompt_id] * n_responses) / (n_responses - 1)

        # Broadcast to all tokens
        advantages = scores.unsqueeze(-1) * response_mask  # (bs, response_length)

    return advantages, advantages


def compute_reinforce_pp_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_indices: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Reinforce++ baseline advantage.

    Uses the mean of all responses (including current) from the same prompt
    as the baseline, then applies whitening.

    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        prompt_indices: shape (bs,) - which prompt each response belongs to
        epsilon: small value for numerical stability

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    id2scores = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        batch_size = scores.shape[0]

        # Collect all scores for each prompt
        for i in range(batch_size):
            prompt_id = prompt_indices[i]
            id2scores[prompt_id].append(scores[i])

        # Compute mean score for each prompt
        for prompt_id in id2scores:
            if len(id2scores[prompt_id]) == 1:
                id2mean[prompt_id] = torch.tensor(0.0, device=scores.device)
            elif len(id2scores[prompt_id]) > 1:
                id2mean[prompt_id] = torch.mean(torch.stack(id2scores[prompt_id]))
            else:
                raise ValueError(f"No scores for prompt index: {prompt_id}")

        # Subtract baseline from each score
        for i in range(batch_size):
            prompt_id = prompt_indices[i]
            scores[i] = scores[i] - id2mean[prompt_id]

        # Broadcast to all tokens and apply whitening
        advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

        # Whitening: normalize advantages to have zero mean and unit variance
        advantages = whiten_advantages(advantages, response_mask) * response_mask

    return advantages, advantages


def compute_direct_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_indices: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute direct advantage where advantage = reward directly.
    No baseline subtraction or group comparisons.

    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        prompt_indices: shape (bs,) - which prompt each response belongs to (unused)
        epsilon: small value for numerical stability (unused)

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    advantages = token_level_rewards * response_mask
    return advantages, advantages


def whiten_advantages(advantages: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages to zero mean and unit variance.

    Args:
        advantages: shape (bs, seq_len)
        mask: shape (bs, seq_len)
        epsilon: small value for numerical stability

    Returns:
        whitened advantages: shape (bs, seq_len)
    """
    # Compute mean and std over valid (masked) tokens
    valid_advantages = advantages * mask
    sum_advantages = valid_advantages.sum()
    count = mask.sum()

    if count > 0:
        mean = sum_advantages / count
        variance = ((valid_advantages - mean) ** 2 * mask).sum() / count
        std = torch.sqrt(variance + epsilon)
        whitened = (advantages - mean) / (std + epsilon)
    else:
        whitened = advantages

    return whitened


def compute_grpo_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    values: Optional[torch.Tensor],
    old_logprobs: torch.Tensor,
    old_values: Optional[torch.Tensor],
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_indices: np.ndarray,
    config: GRPOConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute GRPO loss with group-relative advantages.

    Args:
        logprobs: Current policy log probs, shape (bs, response_length)
        ref_logprobs: Reference policy log probs, shape (bs, response_length)
        values: Current value estimates, shape (bs, response_length) or None
        old_logprobs: Old policy log probs, shape (bs, response_length)
        old_values: Old value estimates, shape (bs, response_length) or None
        rewards: Per-token rewards, shape (bs, response_length)
        response_mask: Mask for valid tokens, shape (bs, response_length)
        prompt_indices: Prompt ID for each response, shape (bs,)
        config: GRPOConfig object

    Returns:
        Dictionary with loss components and statistics
    """
    # Select advantage computation method
    if config.advantage_method == "rloo":
        advantages, returns = compute_rloo_advantage(
            rewards, response_mask, prompt_indices, config.advantage_epsilon
        )
    elif config.advantage_method == "reinforce_pp":
        advantages, returns = compute_reinforce_pp_advantage(
            rewards, response_mask, prompt_indices, config.advantage_epsilon
        )
    elif config.advantage_method == "direct":
        advantages, returns = compute_direct_advantage(
            rewards, response_mask, prompt_indices, config.advantage_epsilon
        )
    else:
        raise ValueError(f"Unknown advantage method: {config.advantage_method}")

    # Optionally apply whitening
    if config.whitening and config.advantage_method != "reinforce_pp":
        advantages = whiten_advantages(advantages, response_mask) * response_mask

    # Compute policy ratio
    log_ratio = logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    # Clipped surrogate loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
    pg_loss = torch.max(pg_loss1, pg_loss2)
    pg_loss = mask_pad(pg_loss, response_mask).sum() / response_mask.sum()

    # Value loss (if using critic)
    if values is not None and old_values is not None:
        values_clipped = old_values + torch.clamp(
            values - old_values, -config.value_clip, config.value_clip
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
        vf_loss = mask_pad(vf_loss, response_mask).sum() / response_mask.sum()
    else:
        vf_loss = torch.tensor(0.0, device=logprobs.device)

    # Entropy bonus
    entropy = -(logprobs * torch.exp(logprobs))
    entropy_loss = -mask_pad(entropy, response_mask).sum() / response_mask.sum()

    # KL penalty (if enabled)
    if config.kl_coef > 0:
        kl_div = logprobs - ref_logprobs
        kl_loss = mask_pad(kl_div, response_mask).sum() / response_mask.sum()
    else:
        kl_loss = torch.tensor(0.0, device=logprobs.device)

    # Total loss
    total_loss = (
        pg_loss
        + config.vf_coef * vf_loss
        + config.entropy_coef * entropy_loss
        + config.kl_coef * kl_loss
    )

    # Compute statistics
    with torch.no_grad():
        clipfrac = ((ratio - 1.0).abs() > config.clip_ratio).float()
        clipfrac = mask_pad(clipfrac, response_mask).sum() / response_mask.sum()

        approx_kl = mask_pad(log_ratio, response_mask).sum() / response_mask.sum()

        # Explained variance (if using critic)
        if values is not None:
            returns_var = returns.var()
            explained_var = 1 - ((returns - values) ** 2).mean() / (returns_var + 1e-8)
        else:
            explained_var = torch.tensor(0.0)

    return {
        "loss/total": total_loss,
        "loss/policy": pg_loss,
        "loss/value": vf_loss,
        "loss/entropy": entropy_loss,
        "loss/kl": kl_loss,
        "metrics/ratio_mean": ratio.mean(),
        "metrics/ratio_max": ratio.max(),
        "metrics/ratio_min": ratio.min(),
        "metrics/clipfrac": clipfrac,
        "metrics/approx_kl": approx_kl,
        "metrics/explained_var": explained_var,
        "metrics/advantages_mean": advantages.mean(),
        "metrics/advantages_std": advantages.std(),
        "metrics/returns_mean": returns.mean(),
        "metrics/returns_std": returns.std(),
    }
