"""Masked objectives with openpi's t=1 noise, t=0 data flow convention."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_predicate_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Labels: 1 supported, 0 refuted, -1 unknown. Unknown is never negative."""
    if logits.shape != labels.shape:
        raise ValueError("Logits and labels must have matching shapes")
    if not torch.all((labels == -1) | (labels == 0) | (labels == 1)):
        raise ValueError("Predicate labels must be -1, 0, or 1")
    known = labels != -1
    safe_logits = torch.where(known, logits, torch.zeros_like(logits))
    targets = torch.where(known, labels, torch.zeros_like(labels)).to(logits.dtype)
    values = F.binary_cross_entropy_with_logits(safe_logits, targets, reduction="none")
    return torch.where(known, values, torch.zeros_like(values)).sum() / known.sum().clamp_min(1)


def flow_matching_loss(
    velocity: Tensor, actions: Tensor, noise: Tensor, valid_steps: Tensor
) -> Tensor:
    """Mask episode padding. Target noise - actions matches decreasing-time sampling."""
    if velocity.ndim != 3 or velocity.shape != actions.shape or actions.shape != noise.shape:
        raise ValueError("velocity, actions, and noise must have the same [B,H,A] shape")
    if valid_steps.shape != velocity.shape[:2] or valid_steps.dtype != torch.bool:
        raise ValueError("valid_steps must be boolean [B,H]")
    mask = valid_steps.unsqueeze(-1).expand_as(velocity)
    v, a, n = (torch.where(mask, x, torch.zeros_like(x)) for x in (velocity, actions, noise))
    return ((v - (n - a)) ** 2).sum() / mask.sum().clamp_min(1)
