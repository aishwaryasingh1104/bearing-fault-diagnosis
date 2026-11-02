"""
tests/test_models.py
Unit and integration tests for all model architectures.

Covers:
- Input/output tensor shapes
- Forward pass (no crash)
- Backward pass (gradients flow)
- Feature extraction
- Checkpoint save/load roundtrip
- Transfer learning model instantiation
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN, build_baseline_cnn
from src.models.attention_cnn import AttentionCNN, build_attention_cnn
from src.models.transfer_learning import HierarchicalTransferModel, build_transfer_model
from src.models.hybrid_model import HybridModel, build_hybrid_model


DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helper: single batch of synthetic input
# ---------------------------------------------------------------------------

def _rand_batch(B=4, C=1, L=2048) -> torch.Tensor:
    return torch.randn(B, C, L)


# ---------------------------------------------------------------------------
# BaselineCNN
# ---------------------------------------------------------------------------

class TestBaselineCNN:
    def test_output_shape(self):
        model = BaselineCNN(in_channels=1, num_classes=4)
        x = _rand_batch(4, 1, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 4), f"Expected (4,4) got {out.shape}"

    def test_backward_pass(self):
        model = BaselineCNN(in_channels=1, num_classes=4)
        x = _rand_batch(4, 1, 2048)
        y = torch.randint(0, 4, (4,))
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        # Check that at least one parameter has a gradient
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                       for p in model.parameters())
        assert has_grad, "No gradient in baseline CNN"

    def test_feature_extraction_shape(self):
        model = BaselineCNN(in_channels=1, num_classes=4,
                            conv_channels=[32, 64, 128])
        x = _rand_batch(4, 1, 2048)
        with torch.no_grad():
            feat = model.extract_features(x)
        assert feat.shape == (4, 128)

    def test_ewt_input(self):
        model = BaselineCNN(in_channels=3, num_classes=4)
        x = _rand_batch(4, 3, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 4)

    def test_n_parameters_positive(self):
        model = BaselineCNN()
        assert model.n_parameters() > 0

    def test_build_from_cfg(self):
        cfg = {"in_channels": 1, "num_classes": 3, "conv_channels": [16, 32],
               "kernel_sizes": [5, 3], "dropout": 0.2, "fc_units": [64]}
        model = build_baseline_cnn(cfg)
        x = _rand_batch(2, 1, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# AttentionCNN
# ---------------------------------------------------------------------------

class TestAttentionCNN:
    def test_output_shape(self):
        model = AttentionCNN(in_channels=1, num_classes=4)
        x = _rand_batch(4, 1, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 4)

    def test_attention_output_shape(self):
        model = AttentionCNN(in_channels=1, num_classes=4)
        x = _rand_batch(2, 1, 2048)
        with torch.no_grad():
            logits, attn = model.forward_with_attention(x)
        assert logits.shape == (2, 4)
        assert attn.ndim == 3   # (B, L', L')

    def test_backward_pass(self):
        model = AttentionCNN(in_channels=1, num_classes=4)
        x = _rand_batch(4, 1, 2048)
        y = torch.randint(0, 4, (4,))
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                       for p in model.parameters())
        assert has_grad

    def test_ewt_input(self):
        model = AttentionCNN(in_channels=3, num_classes=4)
        x = _rand_batch(4, 3, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 4)

    def test_feature_extraction_shape(self):
        model = AttentionCNN(in_channels=1, num_classes=4)
        x = _rand_batch(3, 1, 2048)
        with torch.no_grad():
            feat = model.extract_features(x)
        assert feat.shape[0] == 3

    def test_attention_heads_divisibility(self):
        """Should auto-reduce heads if embed_dim not divisible."""
        model = AttentionCNN(in_channels=1, num_classes=4,
                             conv_channels=[32, 64, 100],  # 100 % 4 != 0
                             attention_heads=4)
        x = _rand_batch(2, 1, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# HierarchicalTransferModel
# ---------------------------------------------------------------------------

class TestTransferModel:
    def _make_model(self, n_src=4, n_tgt=4):
        backbone_cfg = {
            "in_channels": 1, "num_classes": n_src,
            "conv_channels": [16, 32, 64], "kernel_sizes": [5, 3, 3],
            "attention_heads": 4, "attention_dropout": 0.0,
            "dropout": 0.0, "fc_units": [64, 32],
        }
        return HierarchicalTransferModel(
            backbone_cfg=backbone_cfg,
            n_source_classes=n_src,
            n_target_classes=n_tgt,
            feature_dim=64,
        )

    def test_source_head_shape(self):
        model = self._make_model(4, 3)
        x = _rand_batch(4, 1, 2048)
        with torch.no_grad():
            out = model.forward_source(x)
        assert out.shape == (4, 4)

    def test_target_head_shape(self):
        model = self._make_model(4, 3)
        x = _rand_batch(4, 1, 2048)
        with torch.no_grad():
            out = model.forward_target(x)
        assert out.shape == (4, 3)

    def test_domain_discriminator_shape(self):
        model = self._make_model()
        x = _rand_batch(4, 1, 2048)
        with torch.no_grad():
            out = model.forward_domain(x, alpha=1.0)
        assert out.shape == (4, 2)   # binary: source vs target

    def test_freeze_unfreeze(self):
        model = self._make_model()
        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_adaptation_loss_scalar(self):
        model = self._make_model()
        src = _rand_batch(4, 1, 2048)
        tgt = _rand_batch(4, 1, 2048)
        loss = model.compute_adaptation_loss(src, tgt, alpha=1.0)
        assert loss.ndim == 0   # scalar tensor
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# HybridModel
# ---------------------------------------------------------------------------

class TestHybridModel:
    def test_without_transfer(self):
        cfg = {
            "in_channels": 3, "num_classes": 4,
            "conv_channels": [16, 32, 64], "kernel_sizes": [5, 3, 3],
            "attention_heads": 4, "attention_dropout": 0.0,
            "dropout": 0.0, "fc_units": [64],
        }
        model = HybridModel(backbone_cfg=cfg, use_transfer=False)
        x = _rand_batch(4, 3, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 4)

    def test_with_transfer(self):
        backbone_cfg = {
            "in_channels": 3, "num_classes": 4,
            "conv_channels": [16, 32, 64], "kernel_sizes": [5, 3, 3],
            "attention_heads": 4, "attention_dropout": 0.0,
            "dropout": 0.0, "fc_units": [64, 32],
        }
        transfer_cfg = {"backbone": backbone_cfg, "adaptation_weight": 0.3}
        model = HybridModel(backbone_cfg=backbone_cfg, use_transfer=True,
                            transfer_cfg=transfer_cfg,
                            n_source_classes=4, n_target_classes=4)
        x = _rand_batch(2, 3, 2048)
        with torch.no_grad():
            out = model(x, domain="target")
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    def test_save_and_load(self, tmp_path):
        from src.training.checkpointing import save_checkpoint, load_checkpoint

        model = BaselineCNN(in_channels=1, num_classes=4)
        optimizer = torch.optim.Adam(model.parameters())

        # Run one forward/backward to set some non-zero weights
        x, y = _rand_batch(2, 1, 2048), torch.randint(0, 4, (2,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()

        ckpt_path = save_checkpoint(
            model=model, optimizer=optimizer, epoch=1,
            metrics={"accuracy": 0.5}, config={},
            checkpoint_dir=tmp_path, tag="test",
        )
        assert ckpt_path.exists()

        # Load into fresh model
        new_model = BaselineCNN(in_channels=1, num_classes=4)
        payload = load_checkpoint(ckpt_path, new_model, device=torch.device("cpu"))

        assert payload["epoch"] == 1
        assert payload["metrics"]["accuracy"] == 0.5

        # Weights must match
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), new_model.state_dict().items()):
            assert torch.allclose(v1, v2), f"Weight mismatch at {k1}"

    def test_load_missing_checkpoint_raises(self, tmp_path):
        from src.training.checkpointing import load_checkpoint
        model = BaselineCNN()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt", model)
