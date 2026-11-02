"""
src/data/preprocessing.py
Vibration-signal segmentation and normalization pipeline.

Design principles
-----------------
* Normalization parameters (mean, std, min, max) are computed **only** from
  the training split and then applied to validation and test splits.
* Overlapping windows are created **after** the recording-level split to
  prevent any window from a single recording appearing in both train and test.
* All split manifests are saved as JSON so experiments are reproducible.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bfd.preprocessing")


# ---------------------------------------------------------------------------
# Window normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """Fit-on-train, apply-to-all signal normalizer.

    Parameters
    ----------
    method:
        ``'zscore'`` (zero mean, unit variance), ``'minmax'`` (scale to
        [0, 1]), or ``'none'`` (no normalization).
    """

    def __init__(self, method: str = "zscore"):
        self.method = method.lower()
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, signals: np.ndarray) -> "Normalizer":
        """Compute normalization statistics from training data.

        Parameters
        ----------
        signals:
            Array of shape ``(N, L)`` where N is the number of windows
            and L is the window length.
        """
        if self.method == "none":
            self._fitted = True
            return self
        flat = signals.ravel().astype(np.float64)
        if self.method == "zscore":
            self._mean = float(flat.mean())
            self._std = float(flat.std())
            if self._std < 1e-10:
                logger.warning("Signal std is nearly zero (%.2e); using std=1", self._std)
                self._std = 1.0
        elif self.method == "minmax":
            self._min = float(flat.min())
            self._max = float(flat.max())
            if (self._max - self._min) < 1e-10:
                logger.warning("Signal range is nearly zero; minmax will produce zeros")
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        self._fitted = True
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Apply fitted normalization to *signals*.

        Parameters
        ----------
        signals:
            Array of shape ``(N, L)`` (float32 recommended).

        Returns
        -------
        np.ndarray
            Normalized array, same shape as input.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before calling transform().")
        signals = signals.astype(np.float32)
        if self.method == "none":
            return signals
        if self.method == "zscore":
            return (signals - self._mean) / self._std
        if self.method == "minmax":
            rng = self._max - self._min
            if rng < 1e-10:
                return np.zeros_like(signals)
            return (signals - self._min) / rng
        return signals

    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        return self.fit(signals).transform(signals)

    def params(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "mean": self._mean,
            "std": self._std,
            "min": self._min,
            "max": self._max,
        }


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_signal(
    signal: np.ndarray,
    window_length: int = 2048,
    overlap: float = 0.5,
) -> np.ndarray:
    """Divide *signal* into fixed-length overlapping windows.

    Parameters
    ----------
    signal:
        1-D vibration signal array.
    window_length:
        Number of samples per window.
    overlap:
        Fractional overlap between consecutive windows (0 ≤ overlap < 1).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_windows, window_length)``.

    Raises
    ------
    ValueError
        When the signal is shorter than *window_length*.
    """
    signal = np.asarray(signal, dtype=np.float32).ravel()
    if signal.size < window_length:
        raise ValueError(
            f"Signal length ({signal.size}) is shorter than window_length ({window_length})."
        )
    step = max(1, int(window_length * (1.0 - overlap)))
    starts = np.arange(0, signal.size - window_length + 1, step)
    windows = np.stack([signal[s : s + window_length] for s in starts])
    return windows


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Configurable parameters for the preprocessing pipeline."""
    window_length: int = 2048
    overlap: float = 0.5
    normalization: str = "zscore"   # zscore | minmax | none
    min_windows_per_class: int = 10


def preprocess_recordings(
    recordings_signals: List[Tuple[np.ndarray, int, str]],
    cfg: PreprocessingConfig,
    is_train: bool = True,
    normalizer: Optional[Normalizer] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Normalizer]:
    """
    Segment and normalize a collection of (signal, label_index, recording_id) tuples.

    Parameters
    ----------
    recordings_signals:
        List of ``(signal_array, label_index, recording_id)`` tuples.
    cfg:
        Preprocessing configuration.
    is_train:
        If ``True``, fit the normalizer on this data.  If ``False``, an
        existing *normalizer* must be provided.
    normalizer:
        A pre-fitted :class:`Normalizer`.  Ignored when ``is_train=True``.

    Returns
    -------
    (windows, labels, rec_ids, normalizer)
        * windows  – ``(N, window_length)`` float32 array
        * labels   – ``(N,)`` int64 array
        * rec_ids  – ``(N,)`` string array (recording identifier per window)
        * normalizer – fitted :class:`Normalizer` instance
    """
    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []
    all_rec_ids: List[str] = []

    for signal, label_idx, rec_id in recordings_signals:
        try:
            windows = segment_signal(signal, cfg.window_length, cfg.overlap)
        except ValueError as exc:
            logger.warning("Skipping recording %s: %s", rec_id, exc)
            continue
        all_windows.append(windows)
        all_labels.extend([label_idx] * len(windows))
        all_rec_ids.extend([rec_id] * len(windows))

    if not all_windows:
        raise ValueError("No valid windows produced from the provided recordings.")

    windows_arr = np.concatenate(all_windows, axis=0).astype(np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)
    rec_ids_arr = np.array(all_rec_ids)

    # Normalization
    if normalizer is None:
        normalizer = Normalizer(method=cfg.normalization)

    if is_train:
        windows_arr = normalizer.fit_transform(windows_arr)
    else:
        windows_arr = normalizer.transform(windows_arr)

    logger.info(
        "Preprocessed %d windows from %d recordings (is_train=%s)",
        len(windows_arr), len(recordings_signals), is_train,
    )
    return windows_arr, labels_arr, rec_ids_arr, normalizer


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

def check_leakage(
    train_rec_ids: np.ndarray,
    test_rec_ids: np.ndarray,
) -> Tuple[bool, List[str]]:
    """Verify that no recording appears in both train and test sets.

    Returns
    -------
    (clean, shared_ids)
        ``clean`` is ``True`` when there is no leakage.
    """
    train_set = set(train_rec_ids.tolist())
    test_set = set(test_rec_ids.tolist())
    shared = sorted(train_set & test_set)
    return len(shared) == 0, shared


def validate_signal(signal: np.ndarray, min_samples: int = 256) -> Tuple[bool, str]:
    """Sanity-check a raw vibration signal.

    Returns
    -------
    (valid, message)
    """
    if signal is None or len(signal) == 0:
        return False, "Signal is empty."
    if not np.issubdtype(signal.dtype, np.floating):
        try:
            signal = signal.astype(np.float32)
        except Exception:
            return False, "Signal cannot be converted to float."
    if not np.isfinite(signal).all():
        n_bad = int(np.sum(~np.isfinite(signal)))
        return False, f"Signal contains {n_bad} non-finite values (NaN/Inf)."
    if len(signal) < min_samples:
        return False, f"Signal too short: {len(signal)} < {min_samples} samples."
    return True, "OK"
