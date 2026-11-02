"""
src/data/splitting.py
Recording-level data splitting to prevent window-level leakage.

Strategy
--------
1. Group all recordings by recording_id.
2. Shuffle the recording IDs (with a fixed seed).
3. Assign whole recordings to train / val / test splits.
4. Only AFTER the split, create overlapping windows within each split.

This prevents windows from the same physical recording appearing in
both the training and test sets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_dataset import RecordingMeta

logger = logging.getLogger("bfd.splitting")


def recording_level_split(
    recordings: List[RecordingMeta],
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[RecordingMeta], List[RecordingMeta], List[RecordingMeta]]:
    """Split recordings into train, validation, and test groups.

    Splitting is performed at the **recording** level, not at the window
    level, to prevent data leakage.

    Parameters
    ----------
    recordings:
        Full list of :class:`RecordingMeta` objects.
    test_fraction:
        Fraction of recordings assigned to the test set.
    val_fraction:
        Fraction of recordings assigned to the validation set (computed
        from the remaining data after test split).
    seed:
        NumPy random seed for reproducibility.
    stratify:
        When ``True``, attempt to keep label proportions balanced across
        splits (stratified recording-level split).

    Returns
    -------
    (train, val, test)
    """
    rng = np.random.default_rng(seed)

    if stratify:
        # Group by label first
        from collections import defaultdict
        label_groups: Dict[str, List[RecordingMeta]] = defaultdict(list)
        for rec in recordings:
            label_groups[rec.label].append(rec)

        train_recs, val_recs, test_recs = [], [], []
        for label, recs in label_groups.items():
            idx = rng.permutation(len(recs))
            recs_shuffled = [recs[i] for i in idx]
            n_test = max(1, int(len(recs_shuffled) * test_fraction))
            n_val  = max(1, int(len(recs_shuffled) * val_fraction))
            test_recs.extend(recs_shuffled[:n_test])
            val_recs.extend(recs_shuffled[n_test:n_test + n_val])
            train_recs.extend(recs_shuffled[n_test + n_val:])
    else:
        idx = rng.permutation(len(recordings))
        shuffled = [recordings[i] for i in idx]
        n_test = max(1, int(len(shuffled) * test_fraction))
        n_val  = max(1, int(len(shuffled) * val_fraction))
        test_recs  = shuffled[:n_test]
        val_recs   = shuffled[n_test:n_test + n_val]
        train_recs = shuffled[n_test + n_val:]

    logger.info(
        "Recording split – train: %d, val: %d, test: %d",
        len(train_recs), len(val_recs), len(test_recs),
    )
    return train_recs, val_recs, test_recs


def save_split_manifest(
    train: List[RecordingMeta],
    val: List[RecordingMeta],
    test: List[RecordingMeta],
    output_path: str | Path,
    seed: int = 42,
    extra_meta: Optional[Dict] = None,
) -> None:
    """Persist the recording-level split as a JSON manifest.

    Parameters
    ----------
    output_path:
        File path where the manifest will be written.
    extra_meta:
        Any additional metadata to embed in the manifest (e.g. config hash).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": seed,
        "n_train_recordings": len(train),
        "n_val_recordings":   len(val),
        "n_test_recordings":  len(test),
        "extra": extra_meta or {},
        "train": [r.recording_id for r in train],
        "val":   [r.recording_id for r in val],
        "test":  [r.recording_id for r in test],
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Split manifest saved to %s", output_path)


def load_split_manifest(manifest_path: str | Path) -> Dict:
    """Load a previously saved split manifest."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_manifest_split(
    recordings: List[RecordingMeta],
    manifest: Dict,
) -> Tuple[List[RecordingMeta], List[RecordingMeta], List[RecordingMeta]]:
    """Re-apply a saved manifest to a list of recordings.

    Recordings not found in any split (e.g. if files were added later)
    are logged and discarded.
    """
    train_ids = set(manifest["train"])
    val_ids   = set(manifest["val"])
    test_ids  = set(manifest["test"])

    train, val, test = [], [], []
    for rec in recordings:
        if rec.recording_id in train_ids:
            train.append(rec)
        elif rec.recording_id in val_ids:
            val.append(rec)
        elif rec.recording_id in test_ids:
            test.append(rec)
        else:
            logger.debug("Recording %s not found in manifest – skipped", rec.recording_id)

    logger.info("Applied manifest – train: %d, val: %d, test: %d", len(train), len(val), len(test))
    return train, val, test
