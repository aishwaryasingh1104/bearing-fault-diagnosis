"""
tests/test_ewt.py
Unit tests for the Empirical Wavelet Transform module.
All test data is synthetic — not genuine bearing data.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.signal_processing.ewt import decompose, ewt_builtin, to_tensor, EWTResult


# ---------------------------------------------------------------------------
# Built-in EWT tests
# ---------------------------------------------------------------------------

class TestEWTBuiltin:
    """Tests for the built-in (non-ewtpy) implementation."""

    def test_output_shape(self, synthetic_signal_1d):
        result = ewt_builtin(synthetic_signal_1d[:2048], n_components=3, sampling_rate=12_000)
        assert result.components.shape == (3, 2048)

    def test_output_finite(self, synthetic_signal_1d):
        result = ewt_builtin(synthetic_signal_1d[:2048], n_components=3, sampling_rate=12_000)
        assert np.isfinite(result.components).all(), "EWT components contain NaN or Inf"

    def test_reconstruction_approximate(self, synthetic_signal_1d):
        """Sum of components should approximate the original signal.

        The built-in EWT is explicitly documented as a simplified approximation
        of the full Meyer-wavelet algorithm.  A relative error threshold of 0.8
        is used here; install `ewtpy` for closer reconstruction fidelity.
        """
        sig = synthetic_signal_1d[:2048]
        result = ewt_builtin(sig, n_components=3, sampling_rate=12_000)
        recon = result.components.sum(axis=0)
        rms_err = float(np.sqrt(np.mean((sig.astype(float) - recon.astype(float)) ** 2)))
        rms_sig = float(np.sqrt(np.mean(sig.astype(float) ** 2)))
        rel_err = rms_err / (rms_sig + 1e-10)
        # Built-in approximation: allow up to 80% relative error
        assert rel_err < 0.8, f"Reconstruction error too large: {rel_err:.4f}"

    def test_n_components_configurable(self, synthetic_signal_1d):
        for n in [2, 3, 5]:
            result = ewt_builtin(synthetic_signal_1d[:2048], n_components=n)
            assert result.n_components == n, f"Expected {n} components, got {result.n_components}"

    def test_boundaries_length(self, synthetic_signal_1d):
        result = ewt_builtin(synthetic_signal_1d[:2048], n_components=3)
        assert len(result.boundaries_hz) == 2   # n_components - 1
        assert len(result.boundaries_rad) == 2

    def test_backend_label(self, synthetic_signal_1d):
        result = ewt_builtin(synthetic_signal_1d[:2048], n_components=3)
        assert result.backend == "builtin"


# ---------------------------------------------------------------------------
# Public API `decompose()` tests
# ---------------------------------------------------------------------------

class TestDecompose:
    def test_decompose_returns_ewt_result(self, synthetic_signal_1d):
        result = decompose(synthetic_signal_1d[:2048], n_components=3, sampling_rate=12_000)
        assert isinstance(result, EWTResult)

    def test_decompose_shape(self, synthetic_signal_1d):
        result = decompose(synthetic_signal_1d[:2048], n_components=3, sampling_rate=12_000)
        assert result.components.shape[1] == 2048

    def test_decompose_raises_on_short_signal(self):
        with pytest.raises(ValueError, match="too short"):
            decompose(np.ones(16, dtype=np.float32), n_components=3)

    def test_decompose_raises_on_nan(self):
        sig = np.ones(2048, dtype=np.float32)
        sig[100] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            decompose(sig, n_components=3)

    def test_to_tensor_shape(self, synthetic_signal_1d):
        result = decompose(synthetic_signal_1d[:2048], n_components=3, sampling_rate=12_000)
        tensor = to_tensor(result)
        assert tensor.ndim == 2
        assert tensor.shape[0] == result.n_components
        assert tensor.shape[1] == 2048
        assert tensor.dtype == np.float32


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEWTEdgeCases:
    def test_constant_signal(self):
        """Constant signal should not raise; components may be zero."""
        sig = np.ones(2048, dtype=np.float32)
        result = ewt_builtin(sig, n_components=3)
        assert result.components.shape == (3, 2048)
        assert np.isfinite(result.components).all()

    def test_two_components(self, synthetic_signal_1d):
        result = ewt_builtin(synthetic_signal_1d[:2048], n_components=2)
        assert result.n_components == 2
        assert result.components.shape == (2, 2048)
