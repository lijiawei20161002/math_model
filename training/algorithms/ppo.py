"""
PPO (Proximal Policy Optimization) Algorithm

Standard PPO implementation with clipped surrogate objective,
value function, and entropy regularization.

Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .utils import mask_pad, compute_advantages, compute_rewards


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm"""
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0

    # GAE parameters
    use_gae: bool = True
    gamma: float = 1.0
    gae_lambda: float = 0.95

    # KL penalty
    kl_coef: float = 0.0
    target_kl: Optional[float] = None

    # Advantage normalization
    normalize_advantages: bool = True


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Per-token rewards, shape (bs, seq_len)
        values: Value estimates, shape (bs, seq_len)
        next_values: Next token value estimates, shape (bs, seq_len)
        mask: Valid token mask, shape (bs, seq_len)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: shape (bs, seq_len)
        returns: shape (bs, seq_len)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    # Compute advantages backwards through time
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = torch.zeros_like(values[:, t])
            next_mask = torch.zeros_like(mask[:, t])
        else:
            next_value = next_values[:, t]
            next_mask = mask[:, t + 1]

        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        last_gae = delta + gamma * gae_lambda * next_mask * last_gae
        advantages[:, t] = last_gae * mask[:, t]

    returns = advantages + values
    return advantages, returns


def compute_simple_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute simple n-step returns without GAE.

    Args:
        rewards: Per-token rewards, shape (bs, seq_len)
        values: Value estimates, shape (bs, seq_len)
        mask: Valid token mask, shape (bs, seq_len)
        gamma: Discount factor

    Returns:
        advantages: shape (bs, seq_len)
        returns: shape (bs, seq_len)
    """
    batch_size, seq_len = rewards.shape
    returns = torch.zeros_like(rewards)
    last_return = 0

    # Compute returns backwards through time
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_mask = 0
        else:
            next_mask = mask[:, t + 1]

        returns[:, t] = rewards[:, t] + gamma * last_return * next_mask
        last_return = returns[:, t]
        returns[:, t] = returns[:, t] * mask[:, t]

    advantages = returns - values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages to zero mean and unit variance.

    Args:
        advantages: shape (bs, seq_len)
        mask: shape (bs, seq_len)
        epsilon: small value for numerical stability

    Returns:
        normalized advantages: shape (bs, seq_len)
    """
    masked_advantages = advantages * mask
    count = mask.sum()

    if count > 0:
        mean = masked_advantages.sum() / count
        variance = ((masked_advantages - mean) ** 2 * mask).sum() / count
        std = torch.sqrt(variance + epsilon)
        normalized = (advantages - mean) / (std + epsilon)
    else:
        normalized = advantages

    return normalized * mask


def compute_ppo_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    values: Optional[torch.Tensor],
    old_logprobs: torch.Tensor,
    old_values: Optional[torch.Tensor],
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: PPOConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute PPO loss with clipped surrogate objective.

    Args:
        logprobs: Current policy log probs, shape (bs, response_length)
        ref_logprobs: Reference policy log probs, shape (bs, response_length)
        values: Current value estimates, shape (bs, response_length) or None
        old_logprobs: Old policy log probs, shape (bs, response_length)
        old_values: Old value estimates, shape (bs, response_length) or None
        rewards: Per-token rewards, shape (bs, response_length)
        response_mask: Mask for valid tokens, shape (bs, response_length)
        config: PPOConfig object

    Returns:
        Dictionary with loss components and statistics
    """
    # Compute advantages
    if config.use_gae and values is not None and old_values is not None:
        # Shift values to get next_values
        next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
        advantages, returns = compute_gae_advantages(
            rewards, old_values, next_values, response_mask,
            config.gamma, config.gae_lambda
        )
    else:
        # Simple return computation
        if values is not None and old_values is not None:
            advantages, returns = compute_simple_advantages(
                rewards, old_values, response_mask, config.gamma
            )
        else:
            # No critic - use rewards directly
            advantages = rewards * response_mask
            returns = advantages

    # Normalize advantages
    if config.normalize_advantages:
        advantages = normalize_advantages(advantages, response_mask)

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
            if returns_var > 1e-8:
                explained_var = 1 - ((returns - values) ** 2 * response_mask).sum() / (returns_var * response_mask.sum())
            else:
                explained_var = torch.tensor(0.0)
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
