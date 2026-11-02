"""
src/data/paderborn_dataset.py
Dataset adapter for the Paderborn University KAt-DataCenter bearing dataset.

Official page: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download

Directory layout (as distributed by Paderborn):
    <root>/
        K001/   ← bearing condition label (healthy / damaged)
            N09_M07_F10/   ← operating condition (speed_torque_radialload)
                *.mat       ← MATLAB data files (one per measurement)
        KA01/ ...
        KI01/ ...
        ...

Bearing-code taxonomy (from Paderborn documentation):
    K001–K006     →  healthy (undamaged)
    KA01–KA09     →  artificial outer-race damage
    KI01–KI09     →  artificial inner-race damage
    KB23–KB27     →  real (accelerated) damage (mixed)
    KA16,KA22,… →  real outer damage
    KI16,KI18,… →  real inner damage

Operating-condition folder names encode:
    N<speed_rpm/100>_M<torque_Nm×10>_F<radial_load_N/100>
    e.g. N09_M07_F10 → 900 rpm, 0.7 Nm, 1000 N
         N15_M01_F10 → 1500 rpm, 0.1 Nm, 1000 N
         N15_M07_F04 → 1500 rpm, 0.7 Nm,  400 N

Vibration channel key inside the .mat file: typically 'vibration'
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_dataset import BaseBearingDataset, RecordingMeta

logger = logging.getLogger("bfd.paderborn")

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

_HEALTH_MAP: Dict[str, str] = {
    "K001": "healthy", "K002": "healthy", "K003": "healthy",
    "K004": "healthy", "K005": "healthy", "K006": "healthy",
}

def _bearing_code_to_label(code: str) -> str:
    """Map a Paderborn bearing code to a fault-type label."""
    code_upper = code.upper()
    if code_upper in _HEALTH_MAP:
        return "healthy"
    if code_upper.startswith("KA"):
        return "outer_race"
    if code_upper.startswith("KI"):
        return "inner_race"
    if code_upper.startswith("KB"):
        return "mixed_damage"
    return "unknown"


# ---------------------------------------------------------------------------
# Operating-condition parser
# ---------------------------------------------------------------------------

_OC_RE = re.compile(r"N(\d+)_M(\d+)_F(\d+)", re.IGNORECASE)


def _parse_operating_condition(folder_name: str) -> Dict[str, Optional[float]]:
    """Extract speed (rpm), torque (Nm), radial load (N) from folder name."""
    m = _OC_RE.search(folder_name)
    if m:
        speed_code, torque_code, load_code = m.groups()
        return {
            "speed_rpm": float(speed_code) * 100,
            "torque_Nm": float(torque_code) / 10,
            "radial_load_N": float(load_code) * 100,
        }
    return {"speed_rpm": None, "torque_Nm": None, "radial_load_N": None}


# ---------------------------------------------------------------------------
# .mat loader (same approach as CWRU adapter)
# ---------------------------------------------------------------------------

def _load_mat_variable(path: Path, preferred_keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    """Load the first matching variable from a .mat file."""
    # Try scipy
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        for key in preferred_keys:
            if key in mat:
                arr = np.array(mat[key], dtype=np.float32).ravel()
                if arr.size > 0:
                    return arr
        # Fallback: first non-meta key with a large array
        for key, val in mat.items():
            if key.startswith("_"):
                continue
            try:
                arr = np.array(val, dtype=np.float32).ravel()
                if arr.size > 100:
                    return arr
            except Exception:
                continue
    except Exception:
        pass

    # Try h5py
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            for key in preferred_keys:
                if key in f:
                    arr = np.array(f[key], dtype=np.float32).ravel()
                    if arr.size > 0:
                        return arr
            # Fallback
            for key in f.keys():
                arr = np.array(f[key], dtype=np.float32).ravel()
                if arr.size > 100:
                    return arr
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

class PaderbornDataset(BaseBearingDataset):
    """Paderborn University KAt bearing dataset adapter.

    Expected directory layout::

        <root>/
            <bearing_code>/
                <operating_condition>/
                    *.mat

    Parameters
    ----------
    root:
        Path to the top-level Paderborn dataset directory.
    sampling_rate:
        Nominal sampling rate (64 000 Hz per Paderborn documentation).
    signal_keys:
        Tuple of MATLAB variable names to try when loading vibration data.
    """

    def __init__(
        self,
        root: str | Path,
        sampling_rate: int = 64_000,
        signal_keys: Tuple[str, ...] = ("vibration", "Vibration", "vibration_1", "X"),
        cfg: Optional[Dict] = None,
    ):
        super().__init__(root, sampling_rate, cfg)
        self.signal_keys = signal_keys

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def discover_recordings(self) -> List[RecordingMeta]:
        recordings: List[RecordingMeta] = []

        if not self.root.exists():
            logger.warning("Paderborn root not found: %s", self.root)
            return recordings

        # Walk: <bearing_code> / <operating_condition> / *.mat
        for bearing_dir in sorted(self.root.iterdir()):
            if not bearing_dir.is_dir():
                continue
            bearing_code = bearing_dir.name
            label = _bearing_code_to_label(bearing_code)
            if label == "unknown":
                logger.debug("Skipping unknown bearing code: %s", bearing_code)
                continue

            for oc_dir in sorted(bearing_dir.iterdir()):
                if not oc_dir.is_dir():
                    continue
                oc_name = oc_dir.name
                oc_meta = _parse_operating_condition(oc_name)

                for mat_file in sorted(oc_dir.glob("*.mat")):
                    arr = _load_mat_variable(mat_file, self.signal_keys)
                    if arr is None:
                        logger.debug("Could not load signal from %s", mat_file)
                        continue

                    rec_id = f"{bearing_code}__{oc_name}__{mat_file.stem}"
                    meta = RecordingMeta(
                        recording_id=rec_id,
                        file_path=str(mat_file),
                        label=label,
                        label_index=-1,
                        sampling_rate=self.sampling_rate,
                        n_samples=len(arr),
                        bearing_location="",
                        fault_type=label,
                        fault_severity=bearing_code,
                        speed_rpm=oc_meta["speed_rpm"],
                        load_Nm=oc_meta["torque_Nm"],
                        radial_load_N=oc_meta["radial_load_N"],
                        dataset="Paderborn",
                        operating_condition=oc_name,
                        extra={"bearing_code": bearing_code},
                    )
                    recordings.append(meta)

        # Assign integer labels
        labels_sorted = sorted(set(r.label for r in recordings))
        label_map = {lab: idx for idx, lab in enumerate(labels_sorted)}
        for r in recordings:
            r.label_index = label_map[r.label]

        logger.info(
            "Paderborn: found %d recordings, %d classes, %d operating conditions",
            len(recordings),
            len(label_map),
            len(set(r.operating_condition for r in recordings)),
        )
        return recordings

    def load_signal(self, meta: RecordingMeta) -> np.ndarray:
        path = Path(meta.file_path)
        arr = _load_mat_variable(path, self.signal_keys)
        if arr is None:
            raise ValueError(f"Cannot load signal from {path}")
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Transfer-learning helpers
    # ------------------------------------------------------------------

    def get_operating_conditions(self) -> List[str]:
        """Return unique operating-condition labels."""
        return sorted(set(r.operating_condition for r in self.get_recordings()))

    def filter_by_condition(self, condition: str) -> List[RecordingMeta]:
        """Return recordings that match a specific operating condition."""
        return [r for r in self.get_recordings() if r.operating_condition == condition]

    def get_condition_label_map(self, condition: str) -> Dict[str, int]:
        """Build label-to-index map for a specific operating condition."""
        recs = self.filter_by_condition(condition)
        labels_sorted = sorted(set(r.label for r in recs))
        return {lab: idx for idx, lab in enumerate(labels_sorted)}
