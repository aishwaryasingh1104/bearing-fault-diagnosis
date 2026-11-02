"""
tests/test_preprocessing.py
Unit tests for signal preprocessing: segmentation, normalization, leakage checks.
All test data is clearly synthetic — not genuine bearing data.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import (
    Normalizer,
    PreprocessingConfig,
    check_leakage,
    segment_signal,
    validate_signal,
    preprocess_recordings,
)


# ---------------------------------------------------------------------------
# Segmentation tests
# ---------------------------------------------------------------------------

class TestSegmentation:
    def test_output_shape_no_overlap(self):
        sig = np.ones(4096, dtype=np.float32)
        windows = segment_signal(sig, window_length=1024, overlap=0.0)
        assert windows.shape == (4, 1024)

    def test_output_shape_with_overlap(self):
        sig = np.ones(4096, dtype=np.float32)
        windows = segment_signal(sig, window_length=1024, overlap=0.5)
        # step = 512; starts = [0, 512, 1024, 1536, 2048, 2560, 3072] → 7 windows
        assert windows.shape[0] >= 7
        assert windows.shape[1] == 1024

    def test_raises_if_signal_too_short(self):
        sig = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="shorter than window_length"):
            segment_signal(sig, window_length=1024, overlap=0.0)

    def test_windows_are_contiguous_slices(self):
        sig = np.arange(4096, dtype=np.float32)
        windows = segment_signal(sig, window_length=512, overlap=0.0)
        # First window should start at 0
        np.testing.assert_array_equal(windows[0], sig[:512])


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_zscore_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 2048)).astype(np.float32)
        norm = Normalizer("zscore").fit(data)
        out = norm.transform(data)
        assert abs(out.mean()) < 0.1
        assert abs(out.std() - 1.0) < 0.1

    def test_minmax_range(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 2048)).astype(np.float32)
        norm = Normalizer("minmax").fit(data)
        out = norm.transform(data)
        assert out.min() >= -1e-6
        assert out.max() <= 1 + 1e-6

    def test_none_passthrough(self):
        data = np.ones((10, 512), dtype=np.float32) * 5.0
        norm = Normalizer("none").fit(data)
        out = norm.transform(data)
        np.testing.assert_array_equal(out, data)

    def test_transform_before_fit_raises(self):
        norm = Normalizer("zscore")
        with pytest.raises(RuntimeError, match="fitted"):
            norm.transform(np.ones((5, 512), dtype=np.float32))

    def test_fit_only_on_train_not_test(self):
        """Verify fitted params differ when fitting only on a subset.

        Both N(0,σ_train) and N(0,σ_test) have mean≈0, so a mean-shift test
        is not meaningful here.  Instead we verify that the *variance* of the
        transformed test set is NOT 1.0, which happens when the normalizer was
        fit on a different distribution (train std≈2, test std≈10).
        """
        rng = np.random.default_rng(42)
        train = rng.standard_normal((80, 2048)).astype(np.float32) * 2.0    # std≈2
        test  = rng.standard_normal((20, 2048)).astype(np.float32) * 10.0   # std≈10
        norm = Normalizer("zscore").fit(train)
        test_out = norm.transform(test)
        # If normalizer had been re-fit on test data the std would be ≈1.
        # Since it was fit on train (std≈2), test_out has std≈5, not ≈1.
        test_std = float(test_out.std())
        assert abs(test_std - 1.0) > 1.5, (
            f"Expected test std far from 1.0 (normalizer fit on train); got {test_std:.4f}"
        )


# ---------------------------------------------------------------------------
# Signal validation tests
# ---------------------------------------------------------------------------

class TestValidateSignal:
    def test_valid_signal(self):
        sig = np.ones(1024, dtype=np.float32)
        valid, msg = validate_signal(sig)
        assert valid

    def test_empty_signal(self):
        valid, msg = validate_signal(np.array([]))
        assert not valid
        assert "empty" in msg.lower()

    def test_nan_signal(self):
        sig = np.ones(1024, dtype=np.float32)
        sig[10] = np.nan
        valid, msg = validate_signal(sig)
        assert not valid
        assert "non-finite" in msg.lower()

    def test_inf_signal(self):
        sig = np.ones(1024, dtype=np.float32)
        sig[100] = np.inf
        valid, msg = validate_signal(sig)
        assert not valid

    def test_short_signal(self):
        sig = np.ones(10, dtype=np.float32)
        valid, msg = validate_signal(sig, min_samples=256)
        assert not valid
        assert "short" in msg.lower()


# ---------------------------------------------------------------------------
# Leakage check tests
# ---------------------------------------------------------------------------

class TestLeakageCheck:
    def test_clean_split(self):
        train_ids = np.array(["rec_1", "rec_2", "rec_3"])
        test_ids  = np.array(["rec_4", "rec_5"])
        clean, shared = check_leakage(train_ids, test_ids)
        assert clean
        assert shared == []

    def test_leakage_detected(self):
        train_ids = np.array(["rec_1", "rec_2", "rec_3"])
        test_ids  = np.array(["rec_2", "rec_4"])
        clean, shared = check_leakage(train_ids, test_ids)
        assert not clean
        assert "rec_2" in shared


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestPreprocessingPipeline:
    def test_full_pipeline(self):
        """Smoke test: signal list → windows → normalized arrays."""
        rng = np.random.default_rng(0)
        recordings_signals = [
            (rng.standard_normal(8192).astype(np.float32), i % 4, f"rec_{i}")
            for i in range(10)
        ]
        cfg = PreprocessingConfig(window_length=2048, overlap=0.5, normalization="zscore")
        windows, labels, rec_ids, normalizer = preprocess_recordings(
            recordings_signals, cfg, is_train=True
        )
        assert windows.ndim == 2
        assert windows.shape[1] == 2048
        assert len(labels) == len(windows)
        assert len(rec_ids) == len(windows)
        assert np.isfinite(windows).all()
