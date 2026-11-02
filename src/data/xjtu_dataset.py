"""
src/data/xjtu_dataset.py
Optional dataset adapter for the XJTU-SY bearing degradation dataset.

Official page: https://biaowang.tech/xjtu-sy-bearing-datasets/

NOTE: This dataset is designed for run-to-failure (degradation tracking)
experiments rather than fault classification.  It is kept separate from the
main CWRU / Paderborn classification experiments.

Directory layout (as distributed):
    <root>/
        35Hz12kN/
            Bearing1_1/
                acc_00001.csv
                acc_00002.csv
                ...
            Bearing1_2/ ...
        37.5Hz11kN/ ...
        40Hz10kN/  ...

Each CSV file contains two columns: horizontal and vertical acceleration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base_dataset import BaseBearingDataset, RecordingMeta

logger = logging.getLogger("bfd.xjtu")

# Map condition folder names to structured metadata
_CONDITION_MAP = {
    "35Hz12kN":  {"speed_rpm": 2100, "radial_load_N": 12_000},
    "37.5Hz11kN": {"speed_rpm": 2250, "radial_load_N": 11_000},
    "40Hz10kN":  {"speed_rpm": 2400, "radial_load_N": 10_000},
}


class XJTUSYDataset(BaseBearingDataset):
    """XJTU-SY run-to-failure bearing dataset adapter (optional).

    This dataset does not have ground-truth class labels per window.
    Labels here are derived from the bearing number and condition:
    bearings 1-3 per condition exhibit different fault evolutions.

    **Warning:** Do not mix XJTU-SY labels directly with CWRU or Paderborn
    labels in the same experiment without an explicit mapping justification.
    """

    SAMPLING_RATE = 25_600   # Hz, per XJTU documentation

    def __init__(self, root: str | Path, channel: str = "horizontal", cfg: Optional[Dict] = None):
        super().__init__(root, self.SAMPLING_RATE, cfg)
        self.channel = channel  # 'horizontal' or 'vertical'

    def discover_recordings(self) -> List[RecordingMeta]:
        """Walk dataset and treat each CSV as one recording snapshot."""
        recordings: List[RecordingMeta] = []

        if not self.root.exists():
            logger.warning("XJTU-SY root not found: %s", self.root)
            return recordings

        for cond_dir in sorted(self.root.iterdir()):
            if not cond_dir.is_dir():
                continue
            cond_name = cond_dir.name
            cond_meta = _CONDITION_MAP.get(cond_name, {})

            for bearing_dir in sorted(cond_dir.iterdir()):
                if not bearing_dir.is_dir():
                    continue
                bearing_id = bearing_dir.name   # e.g. Bearing1_1

                csv_files = sorted(bearing_dir.glob("acc_*.csv"))
                for csv_file in csv_files:
                    # Each file = one 1.28-second snippet at 25.6 kHz → 32768 samples
                    label = f"{cond_name}_{bearing_id}"
                    meta = RecordingMeta(
                        recording_id=f"{bearing_id}_{csv_file.stem}",
                        file_path=str(csv_file),
                        label=label,
                        label_index=-1,
                        sampling_rate=self.SAMPLING_RATE,
                        n_samples=32_768,          # approximate
                        bearing_location=self.channel,
                        fault_type="degradation",
                        fault_severity="unknown",
                        speed_rpm=cond_meta.get("speed_rpm"),
                        radial_load_N=cond_meta.get("radial_load_N"),
                        dataset="XJTU-SY",
                        operating_condition=cond_name,
                        extra={"bearing_id": bearing_id, "condition": cond_name},
                    )
                    recordings.append(meta)

        labels_sorted = sorted(set(r.label for r in recordings))
        label_map = {lab: idx for idx, lab in enumerate(labels_sorted)}
        for r in recordings:
            r.label_index = label_map[r.label]

        logger.info("XJTU-SY: found %d CSV snapshots across %d bearings", len(recordings), len(label_map))
        return recordings

    def load_signal(self, meta: RecordingMeta) -> np.ndarray:
        path = Path(meta.file_path)
        df = pd.read_csv(path, header=None, names=["horizontal", "vertical"])
        col = "horizontal" if self.channel == "horizontal" else "vertical"
        return df[col].to_numpy(dtype=np.float32)
