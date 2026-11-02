"""
src/training/losses.py
Loss functions for fault diagnosis and domain adaptation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import numpy as np


def build_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    Parameters
    ----------
    labels:
        Integer label array (0 to num_classes-1).
    num_classes:
        Total number of classes.
    device:
        Target torch device.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(num_classes,)``.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes     # normalise
    return torch.tensor(weights, dtype=torch.float32, device=device)


def cross_entropy_loss(
    class_weighted: bool = True,
    labels: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> nn.CrossEntropyLoss:
    """Return a CrossEntropyLoss, optionally with class weighting."""
    if class_weighted and labels is not None and num_classes is not None:
        weight = build_class_weights(labels, num_classes, device or torch.device("cpu"))
        return nn.CrossEntropyLoss(weight=weight)
    return nn.CrossEntropyLoss()
