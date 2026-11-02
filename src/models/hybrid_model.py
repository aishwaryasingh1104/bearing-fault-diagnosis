"""
src/models/hybrid_model.py
End-to-end Hybrid Pipeline: EWT → Self-Attention CNN → (Transfer Learning).

Pipeline
--------
1. Input: raw vibration signal window  (B, 1, L)
2. EWT decomposition (in numpy, applied before batching OR as a batch layer)
3. Self-Attention CNN classification or feature extraction
4. Optional: hierarchical transfer learning head

Two operating modes
-------------------
Mode A (inference with EWT layer):
    The ``HybridModel`` wraps an ``EWTLayer`` (approximated as a frequency-
    domain bandpass filter in torch for GPU compatibility).  This enables
    end-to-end backpropagation through the EWT approximation.

Mode B (pre-computed EWT features):
    EWT is applied offline in preprocessing.  The CNN receives the pre-
    computed multi-channel array (B, n_components, L) directly.

The default (and recommended) mode is Mode B, which uses the verified
scipy-based EWT implementation and passes multi-channel arrays to the CNN.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .attention_cnn import AttentionCNN, build_attention_cnn
from .transfer_learning import HierarchicalTransferModel, build_transfer_model

logger = logging.getLogger("bfd.hybrid")


class HybridModel(nn.Module):
    """EWT + Self-Attention CNN pipeline (Mode B: pre-computed EWT features).

    This model expects inputs that have already been decomposed by EWT
    (see ``src/signal_processing/ewt.py``).

    Input tensor shape: ``(B, n_ewt_components, window_length)``
    Output:             ``(B, num_classes)`` logits

    Parameters
    ----------
    backbone_cfg:
        Configuration for the :class:`AttentionCNN` backbone.
        ``in_channels`` should equal the number of EWT components.
    use_transfer:
        When ``True``, the model routes features through a
        :class:`HierarchicalTransferModel` for cross-domain scenarios.
    transfer_cfg:
        Transfer-learning configuration.  Required when
        ``use_transfer=True``.
    n_source_classes:
        Number of source-domain classes (transfer mode only).
    n_target_classes:
        Number of target-domain classes (transfer mode only).
    """

    def __init__(
        self,
        backbone_cfg: Dict,
        use_transfer: bool = False,
        transfer_cfg: Optional[Dict] = None,
        n_source_classes: int = 4,
        n_target_classes: int = 4,
    ):
        super().__init__()
        self.use_transfer = use_transfer

        if use_transfer:
            self.model = build_transfer_model(
                cfg              = transfer_cfg or {},
                n_source_classes = n_source_classes,
                n_target_classes = n_target_classes,
            )
        else:
            self.model = build_attention_cnn(backbone_cfg)

    def forward(self, x: torch.Tensor, domain: str = "target") -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, n_ewt_components, window_length)``
        domain:
            Only used in transfer mode.  ``'source'`` or ``'target'``.
        """
        if self.use_transfer:
            return self.model.forward(x, domain=domain)
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature vector before the classification head."""
        if self.use_transfer:
            return self.model.backbone.extract_features(x)
        return self.model.extract_features(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_hybrid_model(
    cfg: Dict,
    use_transfer: bool = False,
    n_source_classes: int = 4,
    n_target_classes: int = 4,
) -> HybridModel:
    """Construct a :class:`HybridModel` from a config dict."""
    backbone_cfg  = cfg.get("ewt_attention_cnn", cfg.get("attention_cnn", {}))
    transfer_cfg  = cfg.get("transfer", {})
    return HybridModel(
        backbone_cfg     = backbone_cfg,
        use_transfer     = use_transfer,
        transfer_cfg     = transfer_cfg,
        n_source_classes = n_source_classes,
        n_target_classes = n_target_classes,
    )
