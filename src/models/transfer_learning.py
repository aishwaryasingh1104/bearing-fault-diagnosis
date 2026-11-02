"""
src/models/transfer_learning.py
Hierarchical Transfer Learning module for cross-condition bearing fault diagnosis.

Design rationale
----------------
This module implements a provisional hierarchical transfer-learning baseline
inspired by domain-adaptation approaches commonly used in bearing fault
diagnosis under variable operating conditions.

Implementation status
---------------------
This is labelled a **Provisional Transfer Learning Baseline** because the
specific architecture of the second reference paper (if provided) has not been
fully verified at the time of implementation.  When that paper becomes
available, update:
    * ``HierarchicalTransferModel`` – adjust feature extractor, heads, and
      domain-adaptation loss to match the paper exactly.
    * Document the change and re-label the module as "Paper 2 Reproduction".

Core components
---------------
1. **Feature extractor** – shared backbone (AttentionCNN or BaselineCNN)
   pre-trained on source domain.
2. **Source classifier head** – linear head for source-domain labels.
3. **Target classifier head** – linear head for target-domain labels
   (may differ when label sets differ across conditions).
4. **Domain discriminator** – adversarial MLP that pushes source and target
   feature distributions together (DANN-style, Ganin et al. 2016).
5. **Hierarchical levels** – Level 1: fault type classification.
               Level 2: fault severity (added only when data supports it).

Transfer-learning protocol
--------------------------
Stage 1 (source pre-training):
    Train backbone + source head on source-domain data.
    Freeze backbone after convergence.

Stage 2 (adaptation):
    Unfreeze backbone.
    Train with:
        L_total = L_source_cls + λ * L_domain_adv
    where L_domain_adv uses gradient reversal to align source and target
    feature distributions.

Stage 3 (target fine-tuning):
    Fine-tune backbone + target head on available target training data.

References
----------
Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks.
JMLR, 17(59), 1–35.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

from .attention_cnn import AttentionCNN, build_attention_cnn
from .baseline_cnn import BaselineCNN, build_baseline_cnn

logger = logging.getLogger("bfd.transfer")


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for DANN-style domain adaptation)
# ---------------------------------------------------------------------------

class GradientReversal(Function):
    """Reverses the gradient sign during backpropagation.

    Forward pass: identity.
    Backward pass: multiply gradient by ``-alpha``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (alpha,) = ctx.saved_tensors
        return grad_output.neg() * alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


# ---------------------------------------------------------------------------
# Domain discriminator
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """Binary MLP that predicts whether a feature belongs to source or target.

    Used with gradient reversal for adversarial domain adaptation.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),   # source=0, target=1
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        x_rev = grad_reverse(x, alpha)
        return self.net(x_rev)


# ---------------------------------------------------------------------------
# Main hierarchical transfer model
# ---------------------------------------------------------------------------

class HierarchicalTransferModel(nn.Module):
    """Provisional hierarchical transfer-learning model.

    Parameters
    ----------
    backbone_cfg:
        Configuration dict passed to :func:`build_attention_cnn`.
    n_source_classes:
        Number of classes in the source domain.
    n_target_classes:
        Number of classes in the target domain.  May differ from source.
    feature_dim:
        Dimension of the shared feature vector (= last conv channel count).
    adaptation_weight:
        λ – weight for the domain-adversarial loss term.
    use_adversarial:
        When ``True`` (default), enable the gradient-reversal domain discriminator.
    """

    def __init__(
        self,
        backbone_cfg: Dict,
        n_source_classes: int,
        n_target_classes: int,
        feature_dim: int = 128,
        adaptation_weight: float = 0.3,
        use_adversarial: bool = True,
    ):
        super().__init__()
        # Backbone: shared feature extractor
        self.backbone = build_attention_cnn(backbone_cfg)

        # Level-1 classification heads (fault type)
        self.source_head = nn.Linear(feature_dim, n_source_classes)
        self.target_head = nn.Linear(feature_dim, n_target_classes)

        # Domain discriminator
        self.domain_disc = DomainDiscriminator(feature_dim, hidden_dim=feature_dim)
        self.use_adversarial = use_adversarial
        self.adaptation_weight = adaptation_weight

        self.feature_dim = feature_dim

    # ------------------------------------------------------------------
    # Training forward passes
    # ------------------------------------------------------------------

    def forward_source(self, x: torch.Tensor) -> torch.Tensor:
        """Return source-domain classification logits."""
        features = self.backbone.extract_features(x)
        return self.source_head(features)

    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """Return target-domain classification logits."""
        features = self.backbone.extract_features(x)
        return self.target_head(features)

    def forward_domain(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Return domain-discriminator logits (0=source, 1=target)."""
        features = self.backbone.extract_features(x)
        return self.domain_disc(features, alpha)

    def compute_adaptation_loss(
        self,
        src_batch: torch.Tensor,
        tgt_batch: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Compute the domain adversarial alignment loss.

        The discriminator is trained to distinguish source from target
        while the backbone is trained to confuse it (via gradient reversal).

        Returns
        -------
        torch.Tensor
            Scalar cross-entropy loss for the domain discriminator.
        """
        src_feat = self.backbone.extract_features(src_batch)
        tgt_feat = self.backbone.extract_features(tgt_batch)
        all_feat = torch.cat([src_feat, tgt_feat], dim=0)

        src_labels = torch.zeros(len(src_feat), dtype=torch.long, device=src_feat.device)
        tgt_labels = torch.ones(len(tgt_feat), dtype=torch.long, device=tgt_feat.device)
        domain_labels = torch.cat([src_labels, tgt_labels])

        # BUG-07 fix: pass all_feat directly — DomainDiscriminator.forward()
        # already applies grad_reverse internally.  The old code called
        # grad_reverse TWICE, cancelling the reversal effect.
        domain_logits = self.domain_disc(all_feat, alpha)
        return nn.functional.cross_entropy(domain_logits, domain_labels)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, domain: str = "target") -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input tensor ``(B, C, L)``.
        domain:
            ``'source'`` or ``'target'`` – selects which classifier head to use.
        """
        if domain == "source":
            return self.forward_source(x)
        return self.forward_target(x)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (Stage 1 → Stage 2 transition)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("Backbone unfrozen.")

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_transfer_model(
    cfg: Dict,
    n_source_classes: int,
    n_target_classes: int,
) -> HierarchicalTransferModel:
    """Build a :class:`HierarchicalTransferModel` from a configuration dict."""
    backbone_cfg = cfg.get("backbone", {
        "in_channels": 1,
        "num_classes": n_source_classes,
        "conv_channels": [32, 64, 128],
        "kernel_sizes": [7, 5, 3],
        "attention_heads": 4,
        "attention_dropout": 0.1,
        "dropout": 0.3,
        "fc_units": [256, 128],
    })
    feature_dim = backbone_cfg.get("conv_channels", [32, 64, 128])[-1]

    return HierarchicalTransferModel(
        backbone_cfg       = backbone_cfg,
        n_source_classes   = n_source_classes,
        n_target_classes   = n_target_classes,
        feature_dim        = feature_dim,
        adaptation_weight  = cfg.get("adaptation_weight", 0.3),
        use_adversarial    = cfg.get("use_adversarial", True),
    )
