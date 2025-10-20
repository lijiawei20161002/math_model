"""
Utility functions for RL algorithms
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def mask_pad(tensor: torch.Tensor, mask: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """
    Apply mask to tensor and replace masked values with pad_value.

    Args:
        tensor: Input tensor to mask
        mask: Boolean mask (True for valid positions)
        pad_value: Value to use for masked positions

    Returns:
        Masked tensor
    """
    return torch.where(mask, tensor, torch.tensor(pad_value, dtype=tensor.dtype, device=tensor.device))


def compute_rewards(
    scores: torch.Tensor,
    kl_divergence: torch.Tensor,
    kl_coeff: float = 0.1,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute final rewards with KL penalty.

    Args:
        scores: Raw scores from reward model [batch_size]
        kl_divergence: KL divergence from reference policy [batch_size, seq_len]
        kl_coeff: Coefficient for KL penalty
        mask: Optional attention mask [batch_size, seq_len]

    Returns:
        Final rewards [batch_size, seq_len]
    """
    # Sum KL divergence over sequence
    if mask is not None:
        kl_sum = (kl_divergence * mask).sum(dim=-1)  # [batch_size]
    else:
        kl_sum = kl_divergence.sum(dim=-1)  # [batch_size]

    # Compute reward: score - kl_coeff * kl
    rewards = scores - kl_coeff * kl_sum  # [batch_size]

    return rewards


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards at each timestep [batch_size, seq_len]
        values: Value estimates at each timestep [batch_size, seq_len]
        mask: Attention mask [batch_size, seq_len]
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: Advantage estimates [batch_size, seq_len]
        returns: Discounted returns [batch_size, seq_len]
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=device)

    # Compute advantages backward through time
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_values = torch.zeros(batch_size, device=device)
            next_mask = torch.zeros(batch_size, device=device)
        else:
            next_values = values[:, t + 1]
            next_mask = mask[:, t + 1]

        # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_values * next_mask - values[:, t]

        # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
        last_gae = delta + gamma * lam * last_gae * next_mask
        advantages[:, t] = last_gae

    # Returns are advantages + values
    returns = advantages + values

    # Apply mask
    advantages = mask_pad(advantages, mask.bool(), 0.0)
    returns = mask_pad(returns, mask.bool(), 0.0)

    return advantages, returns


def compute_log_probs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute log probabilities of tokens given logits.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        tokens: Target tokens [batch_size, seq_len]
        mask: Optional attention mask [batch_size, seq_len]

    Returns:
        Log probabilities [batch_size, seq_len]
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]

    # Gather log probs for actual tokens
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=tokens.unsqueeze(-1)
    ).squeeze(-1)  # [batch_size, seq_len]

    # Apply mask if provided
    if mask is not None:
        token_log_probs = mask_pad(token_log_probs, mask.bool(), 0.0)

    return token_log_probs


def compute_kl_divergence(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference distributions.

    Args:
        policy_logits: Logits from current policy [batch_size, seq_len, vocab_size]
        reference_logits: Logits from reference policy [batch_size, seq_len, vocab_size]
        mask: Optional attention mask [batch_size, seq_len]

    Returns:
        KL divergence per token [batch_size, seq_len]
    """
    # Compute log probabilities
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)

    # KL(policy || reference) = sum_i p_i * (log p_i - log q_i)
    kl = torch.sum(
        torch.exp(policy_log_probs) * (policy_log_probs - reference_log_probs),
        dim=-1
    )  # [batch_size, seq_len]

    # Apply mask if provided
    if mask is not None:
        kl = mask_pad(kl, mask.bool(), 0.0)

    return kl


def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize advantages across batch.

    Args:
        advantages: Advantage estimates [batch_size, seq_len]
        mask: Attention mask [batch_size, seq_len]
        eps: Small constant for numerical stability

    Returns:
        Normalized advantages [batch_size, seq_len]
    """
    # Compute mean and std over valid tokens
    valid_advantages = advantages * mask
    num_valid = mask.sum()

    if num_valid > 0:
        mean = valid_advantages.sum() / num_valid
        var = ((valid_advantages - mean * mask) ** 2).sum() / num_valid
        std = torch.sqrt(var + eps)

        normalized = (advantages - mean) / std
        normalized = mask_pad(normalized, mask.bool(), 0.0)
    else:
        normalized = advantages

    return normalized
