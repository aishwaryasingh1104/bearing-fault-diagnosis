"""
src/data/base_dataset.py
Abstract base class for all bearing-dataset adapters.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bfd.data")


# ---------------------------------------------------------------------------
# Recording descriptor
# ---------------------------------------------------------------------------

@dataclass
class RecordingMeta:
    """Metadata for a single vibration recording."""
    recording_id: str
    file_path: str
    label: str
    label_index: int
    sampling_rate: int
    n_samples: int
    bearing_location: str = ""
    fault_type: str = ""
    fault_severity: str = ""
    speed_rpm: Optional[float] = None
    load_Nm: Optional[float] = None
    radial_load_N: Optional[float] = None
    dataset: str = ""
    operating_condition: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseBearingDataset(ABC):
    """Abstract interface that every dataset adapter must implement.

    Sub-classes must implement:
        - :meth:`discover_recordings`
        - :meth:`load_signal`

    And optionally override:
        - :meth:`validate`
        - :meth:`label_map` (property)
    """

    def __init__(self, root: str | Path, sampling_rate: int, cfg: Optional[Dict] = None):
        self.root = Path(root)
        self.sampling_rate = sampling_rate
        self.cfg = cfg or {}
        self._recordings: List[RecordingMeta] = []
        self._label_map: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def discover_recordings(self) -> List[RecordingMeta]:
        """Scan ``self.root`` and return a list of :class:`RecordingMeta`.

        This method must:
        - Read actual files present on disk.
        - Assign verified labels (not fabricated).
        - Set ``sampling_rate`` and ``n_samples`` from the real file.
        - Never fabricate metadata.
        """

    @abstractmethod
    def load_signal(self, meta: RecordingMeta) -> np.ndarray:
        """Load the raw vibration signal for a recording.

        Parameters
        ----------
        meta:
            The metadata descriptor returned by :meth:`discover_recordings`.

        Returns
        -------
        np.ndarray
            1-D float32 array of vibration samples.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def validate(self) -> Tuple[bool, List[str]]:
        """Check whether ``self.root`` exists and contains expected files.

        Returns
        -------
        (valid, messages)
        """
        messages: List[str] = []
        if not self.root.exists():
            messages.append(f"Data root not found: {self.root}")
            return False, messages
        recordings = self.discover_recordings()
        if not recordings:
            messages.append(f"No valid recordings found under: {self.root}")
            return False, messages
        messages.append(f"Found {len(recordings)} recordings across {len(set(r.label for r in recordings))} classes.")
        return True, messages

    def get_recordings(self) -> List[RecordingMeta]:
        """Return cached recording list, discovering if needed."""
        if not self._recordings:
            self._recordings = self.discover_recordings()
        return self._recordings

    def get_label_map(self) -> Dict[str, int]:
        """Return label-to-integer mapping derived from discovered recordings."""
        if not self._label_map:
            recordings = self.get_recordings()
            labels = sorted(set(r.label for r in recordings))
            self._label_map = {lab: idx for idx, lab in enumerate(labels)}
        return self._label_map

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for display in the dashboard."""
        recs = self.get_recordings()
        from collections import Counter
        label_counts = Counter(r.label for r in recs)
        return {
            "dataset": self.__class__.__name__,
            "root": str(self.root),
            "n_recordings": len(recs),
            "n_classes": len(label_counts),
            "class_distribution": dict(label_counts),
            "sampling_rate": self.sampling_rate,
        }

    def save_manifest(self, output_path: str | Path) -> None:
        """Serialize the recording list to a JSON manifest file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recs = self.get_recordings()
        manifest = {
            "dataset": self.__class__.__name__,
            "root": str(self.root),
            "sampling_rate": self.sampling_rate,
            "n_recordings": len(recs),
            "label_map": self.get_label_map(),
            "recordings": [r.to_dict() for r in recs],
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info("Manifest saved to %s", output_path)

    def __len__(self) -> int:
        return len(self.get_recordings())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(root={self.root!s}, "
            f"recordings={len(self.get_recordings())})"
        )
