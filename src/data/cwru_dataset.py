"""
src/data/cwru_dataset.py
Dataset adapter for the Case Western Reserve University (CWRU) bearing dataset.

Official page: https://engineering.case.edu/bearingdatacenter/download-data-file

File format:
    MATLAB .mat files with variable names DE_time (drive-end), FE_time (fan-end),
    BA_time (base-acceleration), and an RPM scalar.

Label taxonomy used here
    normal          – healthy bearing
    inner_race      – inner-race fault (IR)
    outer_race      – outer-race fault (OR), any clock position
    ball            – rolling-element / ball fault (B)

Fault diameter codes in filenames: 007, 014, 021, 028 (inches × 1000)
Load codes in filenames: 0, 1, 2, 3 (hp)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_dataset import BaseBearingDataset, RecordingMeta

logger = logging.getLogger("bfd.cwru")

# ---------------------------------------------------------------------------
# CWRU filename-to-label heuristics
# ---------------------------------------------------------------------------

# Known prefixes that appear in CWRU filenames
_LABEL_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)(^|[_\s-])normal",         "normal"),
    (r"(?i)(^|[_\s-])IR\d?",          "inner_race"),
    (r"(?i)(^|[_\s-])OR\d?",          "outer_race"),
    (r"(?i)(^|[_\s-])B\d",            "ball"),    # matches B007, B014, B021, B028
]

_FAULT_SEVERITY_RE = re.compile(r"(\d{3})", re.IGNORECASE)
# Match load suffix: _0hp, _1HP, _0, _1, _2, _3 (CWRU uses both conventions)
_LOAD_RE = re.compile(r"[_-](\d)(?:hp|HP)?(?=[_.]|$)", re.IGNORECASE)


def _infer_label(stem: str) -> str:
    """Infer fault category from a filename stem (case-insensitive)."""
    for pattern, label in _LABEL_PATTERNS:
        # (?i) is already inside the pattern strings, so no extra flag needed
        if re.search(pattern, stem):
            return label
    return "unknown"


def _infer_severity(stem: str) -> str:
    """Infer fault diameter from filename (e.g., '007' → '0.007 inch')."""
    match = _FAULT_SEVERITY_RE.search(stem)
    if match:
        code = match.group(1)
        try:
            return f"{int(code) / 1000:.3f}_inch"
        except ValueError:
            return code
    return ""


def _infer_load(stem: str) -> Optional[float]:
    """Infer motor load in horsepower from filename (0–3 hp)."""
    match = _LOAD_RE.search(stem)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Loader helper – handles both legacy .mat (scipy) and v7.3 HDF5 mats
# ---------------------------------------------------------------------------

def _load_mat(path: Path, key: str) -> Optional[np.ndarray]:
    """Try to load a named variable from a MATLAB .mat file.

    Attempts scipy.io first (MATLAB ≤ 7.2).  Falls back to h5py for
    MATLAB ≥ 7.3 (HDF5-backed) files.

    Returns
    -------
    np.ndarray or None
    """
    # --- scipy (v5/v6/v7) ---
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        if key in mat:
            arr = np.array(mat[key], dtype=np.float32).ravel()
            if arr.size > 0:
                return arr
    except Exception:
        pass  # fall through to h5py

    # --- h5py (v7.3 HDF5) ---
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            if key in f:
                arr = np.array(f[key], dtype=np.float32).ravel()
                if arr.size > 0:
                    return arr
    except Exception:
        pass

    return None


def _available_keys(path: Path) -> List[str]:
    """Return all variable names inside a .mat file."""
    keys: List[str] = []
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat if not k.startswith("_")]
    except Exception:
        pass
    if not keys:
        try:
            import h5py
            with h5py.File(str(path), "r") as f:
                keys = list(f.keys())
        except Exception:
            pass
    return keys


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

class CWRUDataset(BaseBearingDataset):
    """CWRU bearing dataset adapter.

    Parameters
    ----------
    root:
        Path to the top-level CWRU directory that contains the .mat files
        (or subdirectories containing them).
    sampling_rate:
        Nominal sampling rate.  CWRU files were recorded at 12 000 Hz or
        48 000 Hz depending on sensor placement.  The adapter reads this
        from the filename convention when possible.
    channel:
        Which vibration channel to load.  One of ``'DE'`` (drive-end),
        ``'FE'`` (fan-end), or ``'BA'`` (base-acceleration).
    """

    CHANNEL_KEYS = {"DE": "DE_time", "FE": "FE_time", "BA": "BA_time"}

    def __init__(
        self,
        root: str | Path,
        sampling_rate: int = 12_000,
        channel: str = "DE",
        cfg: Optional[Dict] = None,
    ):
        super().__init__(root, sampling_rate, cfg)
        self.channel = channel.upper()
        self.channel_key = self.CHANNEL_KEYS.get(self.channel, "DE_time")

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def discover_recordings(self) -> List[RecordingMeta]:
        """Walk ``self.root`` and register every .mat file found."""
        recordings: List[RecordingMeta] = []

        mat_files = sorted(self.root.rglob("*.mat"))
        if not mat_files:
            logger.warning("No .mat files found under %s", self.root)
            return recordings

        for path in mat_files:
            stem = path.stem
            label = _infer_label(stem)
            if label == "unknown":
                logger.debug("Cannot infer label for %s – skipping", path.name)
                continue

            # Determine available signal length without loading all data
            available_keys = _available_keys(path)
            signal_key = None
            for key in [self.channel_key, "DE_time", "FE_time"]:
                if key in available_keys:
                    signal_key = key
                    break

            if signal_key is None:
                logger.warning("No usable time-series key in %s (%s) – skipping", path.name, available_keys)
                continue

            arr = _load_mat(path, signal_key)
            if arr is None:
                continue

            load_hp = _infer_load(stem)
            meta = RecordingMeta(
                recording_id=stem,
                file_path=str(path),
                label=label,
                label_index=-1,            # filled after label_map is built
                sampling_rate=self._infer_fs(path, signal_key),
                n_samples=len(arr),
                bearing_location=self.channel,
                fault_type=label,
                fault_severity=_infer_severity(stem),
                speed_rpm=self._hp_to_rpm(load_hp),
                load_Nm=load_hp,
                dataset="CWRU",
                operating_condition=f"load_{int(load_hp) if load_hp is not None else 'unknown'}hp",
                extra={"mat_key": signal_key, "available_keys": available_keys},
            )
            recordings.append(meta)

        # Assign integer labels
        labels_sorted = sorted(set(r.label for r in recordings))
        label_map = {lab: idx for idx, lab in enumerate(labels_sorted)}
        for r in recordings:
            r.label_index = label_map[r.label]

        logger.info("CWRU: found %d recordings, %d classes under %s", len(recordings), len(label_map), self.root)
        return recordings

    def load_signal(self, meta: RecordingMeta) -> np.ndarray:
        """Load the vibration signal described by *meta*."""
        path = Path(meta.file_path)
        mat_key = meta.extra.get("mat_key", self.channel_key)
        arr = _load_mat(path, mat_key)
        if arr is None:
            raise ValueError(f"Could not load signal from {path} using key '{mat_key}'")
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_fs(self, path: Path, key: str) -> int:
        """Guess sampling frequency from filename or channel key."""
        name = path.stem.lower()
        # CWRU 48 kHz files contain 'b' prefix for base-acceleration
        if "48k" in name or key == "BA_time":
            return 48_000
        return self.sampling_rate

    @staticmethod
    def _hp_to_rpm(load_hp: Optional[float]) -> Optional[float]:
        """Approximate motor speed from motor load (CWRU documentation)."""
        if load_hp is None:
            return None
        _map = {0.0: 1797.0, 1.0: 1772.0, 2.0: 1750.0, 3.0: 1730.0}
        return _map.get(float(load_hp))

    def get_operating_conditions(self) -> List[str]:
        """Return unique operating-condition labels."""
        return sorted(set(r.operating_condition for r in self.get_recordings()))
