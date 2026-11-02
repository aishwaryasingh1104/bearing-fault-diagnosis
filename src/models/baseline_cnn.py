"""
src/models/baseline_cnn.py
Configurable 1-D Baseline CNN for vibration-signal fault classification.

Architecture
------------
Input: (batch, in_channels, window_length)
  → [Conv1d → BatchNorm → ReLU → MaxPool] × n_blocks
  → AdaptiveAvgPool
  → Flatten
  → [Linear → ReLU → Dropout] × n_fc_layers
  → Linear (logits, num_classes)

The model returns raw logits; use nn.CrossEntropyLoss for training.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Single convolutional block: Conv → BN → ReLU → MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2   # same-size padding
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """Configurable 1-D CNN for bearing fault classification.

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for a single raw signal; or equal to
        the number of EWT components).
    num_classes:
        Number of fault categories.
    conv_channels:
        Output channel counts for each convolutional block.
    kernel_sizes:
        Kernel sizes for each block.  Must have the same length as
        *conv_channels*.
    dropout:
        Dropout probability applied after each conv block and FC layer.
    fc_units:
        Number of hidden units in each fully-connected layer.
    pool_size:
        Max-pooling stride applied after each conv block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
        fc_units: Optional[List[int]] = None,
        pool_size: int = 2,
    ):
        super().__init__()
        conv_channels = conv_channels or [32, 64, 128]
        kernel_sizes  = kernel_sizes  or [7, 5, 3]
        fc_units      = fc_units      or [256, 128]

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have equal length.")

        # --- Build convolutional blocks ---
        blocks: List[nn.Module] = []
        ch_in = in_channels
        for ch_out, k in zip(conv_channels, kernel_sizes):
            blocks.append(ConvBlock1D(ch_in, ch_out, kernel_size=k, pool_size=pool_size, dropout=dropout))
            ch_in = ch_out
        self.conv_blocks = nn.Sequential(*blocks)

        # Global average pooling → fixed feature size regardless of input length
        self.gap = nn.AdaptiveAvgPool1d(1)

        # --- Fully-connected classifier ---
        fc_layers: List[nn.Module] = []
        fc_in = conv_channels[-1]
        for fc_out in fc_units:
            fc_layers += [
                nn.Linear(fc_in, fc_out),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            fc_in = fc_out
        fc_layers.append(nn.Linear(fc_in, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, in_channels, window_length)``

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``
        """
        features = self.conv_blocks(x)        # (B, C, L')
        pooled   = self.gap(features).squeeze(-1)   # (B, C)
        logits   = self.classifier(pooled)    # (B, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the GAP feature vector (before classifier)."""
        features = self.conv_blocks(x)
        return self.gap(features).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_baseline_cnn(cfg: dict) -> BaselineCNN:
    """Construct a :class:`BaselineCNN` from a configuration dict."""
    return BaselineCNN(
        in_channels    = cfg.get("in_channels", 1),
        num_classes    = cfg.get("num_classes", 4),
        conv_channels  = cfg.get("conv_channels", [32, 64, 128]),
        kernel_sizes   = cfg.get("kernel_sizes", [7, 5, 3]),
        dropout        = cfg.get("dropout", 0.3),
        fc_units       = cfg.get("fc_units", [256, 128]),
    )
