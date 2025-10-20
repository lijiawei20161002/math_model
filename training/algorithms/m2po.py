"""
M2PO (Mathematical Measure Optimization for Policy Optimization) Algorithm

This implements the M2PO algorithm which uses a second-order KL constraint (M2/KL^2 budget)
to adaptively clip the policy gradient updates. The algorithm dynamically computes per-token
clipping bounds based on the harmful tokens that would exceed the budget.

Reference: Based on the implementation from the M2PO paper and codebase.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F


@dataclass
class M2POConfig:
    """Configuration for M2PO algorithm."""
    m2_budget: float = 0.01  # M2 (KL^2) budget per harmful token
    miniclip_low: float = 0.3  # Minimum clipping for ratio < 1
    miniclip_high: float = 0.5  # Minimum clipping for ratio > 1
    loss_agg_mode: str = "token-mean"  # How to aggregate loss: "token-mean", "seq-mean-token-sum", etc.


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute mean over masked elements."""
    return (tensor * mask).sum() / (mask.sum() + eps)


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: Loss matrix of shape (batch_size, seq_len)
        loss_mask: Mask of shape (batch_size, seq_len)
        loss_agg_mode: Aggregation mode: "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"

    Returns:
        Aggregated scalar loss
    """
    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@torch.no_grad()
def get_ratio_stats(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    bins: tuple = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
    eps: float = 1e-12,
    tol: float = 1e-6
) -> Dict[str, float]:
    """
    Compute statistics about the policy ratio distribution.

    Args:
        ratio: Policy ratio exp(log_new - log_old)
        advantages: Advantage values
        response_mask: Mask for valid tokens
        log_prob: New policy log probabilities
        old_log_prob: Old policy log probabilities
        bins: Bin edges for histogram
        eps: Small epsilon for numerical stability
        tol: Tolerance for equality checks

    Returns:
        Dictionary of statistics
    """
    mask = response_mask.bool()
    finite = torch.isfinite(ratio)
    mask = mask & finite

    edges = torch.tensor(bins, device=ratio.device, dtype=ratio.dtype)
    bin_idx = torch.bucketize(ratio, edges, right=True)

    def compute_for(cond: torch.Tensor, prefix: str):
        m = mask & cond
        counts = torch.zeros(len(bins) + 2, device=ratio.device, dtype=torch.float32)
        stats = {}

        if m.any():
            eq1_mask = (torch.abs(ratio - 1.0) <= tol) & m
            not_eq1_mask = m & (~eq1_mask)

            if not_eq1_mask.any():
                idx = bin_idx[not_eq1_mask].reshape(-1).long()
                shift = (idx >= 4).long()
                idx = idx + shift
                counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

            counts[4] = eq1_mask.float().sum()

            total = counts.sum()
            if total > eps:
                fracs = counts / total
            else:
                fracs = counts

            # Store bin fractions
            bin_names = ['inf_0.2', '0.2_0.5', '0.5_0.8', '0.8_1.0', 'eq_1.0',
                        '1.0_1.2', '1.2_1.5', '1.5_2.0', 'gt_2.0']
            for i, name in enumerate(bin_names):
                stats[f'{prefix}/{name}'] = fracs[i].item()

            # Average ratio
            stats[f'{prefix}/avg'] = ratio[m].mean().item()
        else:
            stats[f'{prefix}/avg'] = 1.0

        return stats

    all_stats = {}

    # Compute for positive advantages
    pos_adv = advantages[:, 0] > eps
    all_stats.update(compute_for(pos_adv, 'ratio_pos'))

    # Compute for negative advantages
    neg_adv = advantages[:, 0] < -eps
    all_stats.update(compute_for(neg_adv, 'ratio_neg'))

    # Compute for all non-zero advantages
    nonzero_adv = torch.abs(advantages[:, 0]) > eps
    all_stats.update(compute_for(nonzero_adv, 'ratio_nonzero'))

    return all_stats


def _solve_tau_from_sorted_delta2(
    sorted_delta2: torch.Tensor,
    target_sum: float
) -> Tuple[float, float]:
    """
    Find threshold τ such that capping each delta^2 at τ yields sum = target_sum.

    Args:
        sorted_delta2: Sorted delta^2 values in ascending order
        target_sum: Target sum for M2 budget

    Returns:
        tau: Threshold value (square root of the found delta^2)
        M2_after: Average M2 after clipping
    """
    if target_sum <= 1e-12:
        return 0.0, 0.0

    csum = torch.cumsum(sorted_delta2, dim=0)
    n = sorted_delta2.numel()

    for k in range(n):
        left_sum = float(csum[k].item())
        rest = n - k - 1
        m2 = sorted_delta2[k].item() - 1e-12

        if m2 * rest + left_sum >= target_sum - 1e-12:
            if k == 0:
                return 0.0, csum[-1].item() / n
            else:
                M2_after = (sorted_delta2[k-1].item() * (rest + 1) + float(csum[k-1].item())) / n
                return float((sorted_delta2[k-1].item() - 1e-12) ** 0.5), M2_after

    return 100000.0, sorted_delta2.mean().item()


def _get_trust_region_tokens_delta_sq(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract squared KL divergences (delta^2) for harmful tokens only.

    Harmful tokens are those where the policy update would hurt the objective:
    - Positive advantage with ratio > 1 (policy becoming more confident on good action)
    - Negative advantage with ratio < 1 (policy becoming more confident on bad action)

    Args:
        old_log_prob: Log probabilities from old policy [batch_size, seq_len]
        log_prob: Log probabilities from new policy [batch_size, seq_len]
        advantages: Advantage estimates [batch_size, seq_len]
        response_mask: Mask for valid tokens [batch_size, seq_len]

    Returns:
        Concatenated delta^2 values for harmful tokens
    """
    mask = response_mask.bool()
    adv_example = advantages[:, 0]
    pos_adv_mask = adv_example > 1e-12
    neg_adv_mask = adv_example < -1e-12

    delta = old_log_prob - log_prob  # Δ = log p_old - log p_new
    ratio = torch.exp(-delta)  # r = exp(log_new - log_old)

    pos_adv_response_mask = mask[pos_adv_mask]
    neg_adv_response_mask = mask[neg_adv_mask]

    pos_adv_ratio = ratio[pos_adv_mask]
    neg_adv_ratio = ratio[neg_adv_mask]

    pos_adv_r_gt_1_mask = pos_adv_ratio > 1.0 + 1e-12
    neg_adv_r_lt_1_mask = neg_adv_ratio < 1.0 - 1e-12

    delta_sq = delta.pow(2)
    pos_adv_harm_tokens_delta_sq = delta_sq[pos_adv_mask][pos_adv_r_gt_1_mask & pos_adv_response_mask]
    neg_adv_harm_tokens_delta_sq = delta_sq[neg_adv_mask][neg_adv_r_lt_1_mask & neg_adv_response_mask]

    tr_tokens_delta_sq = torch.cat([pos_adv_harm_tokens_delta_sq, neg_adv_harm_tokens_delta_sq])

    return tr_tokens_delta_sq


def kpo_clip_harmful_tokens(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    m2_budget: float
) -> Tuple[float, float, float, float]:
    """
    Compute global clipping bounds (clip_low, clip_high) under an M2 budget.

    This function:
    1. Identifies harmful tokens where the update would exceed the budget
    2. Sorts them by delta^2 = (log p_old - log p_new)^2
    3. Finds threshold τ to cap |delta| such that overall M2 <= m2_budget
    4. Maps τ to ratio bounds: clip_low = exp(-τ), clip_high = exp(+τ)

    Args:
        old_log_prob: Old policy log probabilities [batch_size, seq_len]
        log_prob: New policy log probabilities [batch_size, seq_len]
        advantages: Advantage estimates [batch_size, seq_len]
        response_mask: Mask for valid tokens [batch_size, seq_len]
        m2_budget: M2 (KL^2) budget per token

    Returns:
        clip_low: Lower ratio bound for (adv<0 & r<1)
        clip_high: Upper ratio bound for (adv>0 & r>1)
        M2_now: Current M2 before clipping
        M2_after: Expected M2 after clipping
    """
    tr_tokens_delta_sq = _get_trust_region_tokens_delta_sq(
        old_log_prob, log_prob, advantages, response_mask
    )
    token_num = tr_tokens_delta_sq.numel()

    if token_num == 0:  # No harmful tokens
        return 0.0, 100000.0, 0.0, 0.0

    target_total = m2_budget * float(token_num)
    M2_now = float(tr_tokens_delta_sq.sum().detach().item() / token_num)

    if M2_now <= m2_budget + 1e-12:
        # Already within budget
        return 0.0, 100000.0, M2_now, M2_now

    print(f"[M2PO] Current M2: {M2_now:.6f}, Budget: {m2_budget:.6f}")

    sorted_delta2, _ = torch.sort(tr_tokens_delta_sq)
    tau, M2_after = _solve_tau_from_sorted_delta2(sorted_delta2, target_total)

    # Map |Δ|<=τ to ratio bounds
    clip_low = float(torch.exp(torch.tensor(-tau)).item())
    clip_high = float(torch.exp(torch.tensor(+tau)).item())

    return clip_low, clip_high, M2_now, M2_after


def compute_m2po_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: M2POConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute M2PO policy loss with adaptive clipping under M2 budget.

    This is the core M2PO algorithm that:
    1. Computes dynamic clipping bounds based on the M2 budget
    2. Applies per-token clipping with these bounds
    3. Computes the clipped policy gradient loss

    Args:
        old_log_prob: Log probabilities from old policy [batch_size, seq_len]
        log_prob: Log probabilities from current policy [batch_size, seq_len]
        advantages: Advantage estimates [batch_size, seq_len]
        response_mask: Mask for valid response tokens [batch_size, seq_len]
        config: M2PO configuration

    Returns:
        pg_loss: Policy gradient loss (scalar)
        pg_clipfrac: Fraction of tokens that were clipped
        ppo_kl: Average KL divergence
        kl_penalty: KL penalty term (currently 0 for M2PO)
        stats: Dictionary of statistics including M2 values and ratio distributions
    """
    # Get adaptive clipping bounds
    clip_low, clip_high, M2_data, M2_after = kpo_clip_harmful_tokens(
        old_log_prob, log_prob, advantages, response_mask, config.m2_budget
    )

    # Convert from policy bounds to ratio bounds
    # clip_low/high from kpo_clip are exp(-tau) and exp(+tau)
    # We need to convert to actual clipping ranges: 1-clip_low and clip_high-1
    clip_low = 1 - clip_low
    clip_high = clip_high - 1

    # Apply minimum clipping
    if config.miniclip_low is not None and clip_low < config.miniclip_low:
        clip_low = config.miniclip_low
    if config.miniclip_high is not None and clip_high < config.miniclip_high:
        clip_high = config.miniclip_high

    print(f"[M2PO] Clip range: [{1-clip_low:.4f}, {1+clip_high:.4f}]")

    # Compute policy ratio
    ratio = torch.exp(log_prob - old_log_prob)
    ppo_kl = masked_mean(-(log_prob - old_log_prob), response_mask)

    # Get ratio statistics
    ratio_stats = get_ratio_stats(ratio, advantages, response_mask, log_prob, old_log_prob)

    # Compute clipped policy gradient loss
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clip_pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Fraction of tokens clipped
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # Aggregate loss
    pg_loss = agg_loss(loss_mat=clip_pg_losses, loss_mask=response_mask,
                      loss_agg_mode=config.loss_agg_mode)

    # Add M2PO-specific stats
    ratio_stats.update({
        "m2po/clip_low": clip_low,
        "m2po/clip_high": clip_high,
        "m2po/M2": M2_data,
        "m2po/M2_after": M2_after,
        "m2po/M2_budget": config.m2_budget,
    })

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0, device=pg_loss.device), ratio_stats


# Backward compatibility alias
compute_m2po_policy_loss = compute_m2po_loss
