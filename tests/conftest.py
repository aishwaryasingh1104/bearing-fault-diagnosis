"""
tests/conftest.py
Shared pytest fixtures for the bearing fault diagnosis test suite.

All fixtures generate clearly labelled synthetic data.
No genuine bearing data is used in unit tests.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def synthetic_signal_1d() -> np.ndarray:
    """1-D synthetic vibration signal — 4096 samples at 12 kHz (simulated)."""
    rng = np.random.default_rng(0)
    t = np.arange(4096) / 12_000
    sig = (
        0.5 * np.sin(2 * np.pi * 100 * t)          # carrier
        + 0.8 * np.sin(2 * np.pi * 162 * t)        # fault frequency
        + 0.1 * rng.standard_normal(4096)           # noise
    )
    return sig.astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_windows() -> tuple:
    """Synthetic (N=50, L=2048) windows with binary labels."""
    rng = np.random.default_rng(1)
    windows = rng.standard_normal((50, 2048)).astype(np.float32)
    labels  = rng.integers(0, 4, 50, dtype=np.int64)
    return windows, labels


@pytest.fixture(scope="session")
def synthetic_ewt_windows() -> tuple:
    """Synthetic 3-channel EWT-style windows (N=50, C=3, L=2048)."""
    rng = np.random.default_rng(2)
    windows = rng.standard_normal((50, 3, 2048)).astype(np.float32)
    labels  = rng.integers(0, 4, 50, dtype=np.int64)
    return windows, labels


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")
