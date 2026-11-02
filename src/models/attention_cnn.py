"""
src/models/attention_cnn.py
Self-Attention Enhanced 1-D CNN for bearing fault classification.

Architecture overview
---------------------
Input: (B, C_in, L)
  → [Conv1d → BN → ReLU → MaxPool] × n_conv_blocks       ← local feature extraction
  → MultiheadSelfAttention over the temporal dimension     ← relational feature learning
  → AdaptiveAvgPool → Flatten
  → [Linear → ReLU → Dropout] × n_fc_layers
  → Linear (logits)

Attention mechanism
-------------------
After convolutional feature extraction, the feature map has shape
(B, C, L').  Each position in L' is treated as a "token" with C-dimensional
embedding.  We transpose to (B, L', C) and apply nn.MultiheadAttention.
The attention output is transposed back to (B, C, L') and added to the
residual connection before pooling.

This design is a research-inspired reproduction based on the concept of
position-wise self-attention applied to 1-D CNN feature maps for vibration
fault diagnosis.  It is not a verbatim copy of any single paper.

If the user's reference Paper 1 specifies a different attention placement or
formulation, update the ``AttentionModule`` accordingly and document the
change here.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .baseline_cnn import ConvBlock1D


# ---------------------------------------------------------------------------
# Self-Attention module
# ---------------------------------------------------------------------------

class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the temporal dimension of a feature map.

    Parameters
    ----------
    embed_dim:
        Number of feature channels (= CNN output channels).
    num_heads:
        Number of attention heads.  Must divide *embed_dim*.
    dropout:
        Attention dropout probability.

    Input
    -----
    x : (B, C, L)   – CNN feature map

    Output
    ------
    out : (B, C, L) – attended feature map (residual added)
    attn_weights : (B, L, L) – averaged attention weights across heads
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Ensure embed_dim is divisible by num_heads
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # expects (B, L, C)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, L)
        B, C, L = x.shape
        # Rearrange to (B, L, C) for MultiheadAttention
        x_t = x.permute(0, 2, 1)                         # (B, L, C)
        attn_out, attn_weights = self.attn(x_t, x_t, x_t)  # self-attention
        # Residual + LayerNorm
        out = self.norm(x_t + self.dropout(attn_out))     # (B, L, C)
        out = out.permute(0, 2, 1)                        # (B, C, L)
        return out, attn_weights


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AttentionCNN(nn.Module):
    """Self-Attention enhanced 1-D CNN for bearing fault classification.

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for raw signal, or n_ewt_components).
    num_classes:
        Number of fault labels.
    conv_channels:
        Channel counts per conv block.
    kernel_sizes:
        Kernel sizes per conv block.
    attention_heads:
        Number of self-attention heads.
    attention_dropout:
        Dropout inside the attention module.
    dropout:
        Global dropout rate for conv blocks and FC layers.
    fc_units:
        Hidden units per fully-connected layer.
    pool_size:
        MaxPool stride.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        dropout: float = 0.3,
        fc_units: Optional[List[int]] = None,
        pool_size: int = 2,
    ):
        super().__init__()
        conv_channels = conv_channels or [32, 64, 128]
        kernel_sizes  = kernel_sizes  or [7, 5, 3]
        fc_units      = fc_units      or [256, 128]

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must be the same length.")

        # --- Convolutional feature extractor ---
        blocks: List[nn.Module] = []
        ch_in = in_channels
        for ch_out, k in zip(conv_channels, kernel_sizes):
            blocks.append(
                ConvBlock1D(ch_in, ch_out, kernel_size=k, pool_size=pool_size, dropout=dropout)
            )
            ch_in = ch_out
        self.conv_blocks = nn.Sequential(*blocks)

        # --- Self-Attention module ---
        self.attention = SelfAttention1D(
            embed_dim  = conv_channels[-1],
            num_heads  = attention_heads,
            dropout    = attention_dropout,
        )

        # --- Global Average Pooling ---
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
            Shape ``(B, in_channels, L)``

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``
        """
        # Local feature extraction
        features = self.conv_blocks(x)               # (B, C, L')
        # Relational feature learning
        features, _ = self.attention(features)        # (B, C, L')
        # Pooling
        pooled = self.gap(features).squeeze(-1)      # (B, C)
        return self.classifier(pooled)               # (B, num_classes)

    def forward_with_attention(self, x: torch.Tensor):
        """Return logits AND attention weights for visualization."""
        features = self.conv_blocks(x)
        features, attn_weights = self.attention(features)
        pooled  = self.gap(features).squeeze(-1)
        logits  = self.classifier(pooled)
        return logits, attn_weights

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature vector (before classifier) for transfer learning."""
        features = self.conv_blocks(x)
        features, _ = self.attention(features)
        return self.gap(features).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_attention_cnn(cfg: dict) -> AttentionCNN:
    """Build an :class:`AttentionCNN` from a configuration dict."""
    return AttentionCNN(
        in_channels       = cfg.get("in_channels", 1),
        num_classes       = cfg.get("num_classes", 4),
        conv_channels     = cfg.get("conv_channels", [32, 64, 128]),
        kernel_sizes      = cfg.get("kernel_sizes", [7, 5, 3]),
        attention_heads   = cfg.get("attention_heads", 4),
        attention_dropout = cfg.get("attention_dropout", 0.1),
        dropout           = cfg.get("dropout", 0.3),
        fc_units          = cfg.get("fc_units", [256, 128]),
    )
