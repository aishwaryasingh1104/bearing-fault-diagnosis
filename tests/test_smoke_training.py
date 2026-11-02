"""
tests/test_smoke_training.py
CPU smoke test: verifies the entire training loop runs end-to-end
on tiny synthetic data without crashing.

IMPORTANT: This test uses clearly labelled SYNTHETIC data.
Results from this test have no scientific meaning.
"""
from __future__ import annotations

import numpy as np
import torch

from src.utils.reproducibility import set_seed
from src.models.baseline_cnn import BaselineCNN
from src.models.attention_cnn import AttentionCNN
from src.training.trainer import Trainer


def _make_tiny_data(n_train=40, n_val=12, n_classes=4, window_len=512, seed=0):
    """Generate clearly synthetic data for smoke testing."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, window_len)).astype(np.float32)
    X_val   = rng.standard_normal((n_val,   window_len)).astype(np.float32)
    y_train = rng.integers(0, n_classes, n_train, dtype=np.int64)
    y_val   = rng.integers(0, n_classes, n_val,   dtype=np.int64)
    return X_train, y_train, X_val, y_val


class TestBaselineCNNSmoke:
    """Smoke-test: Baseline CNN trains on tiny synthetic data."""

    def test_training_completes(self, tmp_path):
        set_seed(42)
        device = torch.device("cpu")
        X_tr, y_tr, X_v, y_v = _make_tiny_data(window_len=512)

        model = BaselineCNN(in_channels=1, num_classes=4,
                            conv_channels=[8, 16], kernel_sizes=[5, 3],
                            fc_units=[32])

        cfg = {
            "training": {
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adam",
                "scheduler": "none",
                "early_stopping_patience": 5,
                "checkpoint_dir": str(tmp_path),
                "class_weighted_loss": True,
                "scheduler_T_max": 3,
            }
        }
        trainer = Trainer(model=model, device=device, config=cfg,
                          experiment_name="smoke_baseline")
        result = trainer.train(X_tr, y_tr, X_v, y_v, num_classes=4, multichannel=False)

        assert result.status == "completed"
        assert len(result.train_losses) > 0
        assert result.checkpoint_path is not None
        # Just check training ran; accuracy on random data should be random
        bm = result.best_metrics
        assert 0.0 <= bm.get("accuracy", 0.0) <= 1.0


class TestAttentionCNNSmoke:
    """Smoke-test: Attention CNN with EWT multi-channel input."""

    def test_training_completes_multichannel(self, tmp_path):
        set_seed(42)
        device = torch.device("cpu")

        rng = np.random.default_rng(0)
        X_tr = rng.standard_normal((40, 3, 512)).astype(np.float32)
        y_tr = rng.integers(0, 4, 40, dtype=np.int64)
        X_v  = rng.standard_normal((12, 3, 512)).astype(np.float32)
        y_v  = rng.integers(0, 4, 12, dtype=np.int64)

        model = AttentionCNN(in_channels=3, num_classes=4,
                             conv_channels=[8, 16], kernel_sizes=[5, 3],
                             fc_units=[32])

        cfg = {
            "training": {
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adam",
                "scheduler": "none",
                "early_stopping_patience": 5,
                "checkpoint_dir": str(tmp_path),
                "class_weighted_loss": True,
                "scheduler_T_max": 3,
            }
        }
        trainer = Trainer(model=model, device=device, config=cfg,
                          experiment_name="smoke_attention")
        result = trainer.train(X_tr, y_tr, X_v, y_v, num_classes=4, multichannel=True)

        assert result.status == "completed"
        assert len(result.train_losses) > 0
        bm = result.best_metrics
        assert 0.0 <= bm.get("accuracy", 0.0) <= 1.0
