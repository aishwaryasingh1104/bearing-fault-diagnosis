"""
tests/test_splits.py
Unit tests for recording-level data splitting.
"""
from __future__ import annotations

import pytest

from src.data.base_dataset import RecordingMeta
from src.data.splitting import recording_level_split, save_split_manifest, load_split_manifest, apply_manifest_split


def _make_recordings(n: int, n_classes: int = 4):
    labels = ["normal", "inner_race", "outer_race", "ball"]
    return [
        RecordingMeta(
            recording_id=f"rec_{i:04d}",
            file_path=f"/fake/rec_{i}.mat",
            label=labels[i % n_classes],
            label_index=i % n_classes,
            sampling_rate=12_000,
            n_samples=48_000,
        )
        for i in range(n)
    ]


class TestRecordingLevelSplit:
    def test_no_overlap(self):
        recs = _make_recordings(40)
        train, val, test = recording_level_split(recs, test_fraction=0.2, val_fraction=0.15, seed=42)

        train_ids = {r.recording_id for r in train}
        val_ids   = {r.recording_id for r in val}
        test_ids  = {r.recording_id for r in test}

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids  & test_ids) == 0

    def test_covers_all_recordings(self):
        recs = _make_recordings(40)
        train, val, test = recording_level_split(recs, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == 40

    def test_reproducible_with_same_seed(self):
        recs = _make_recordings(40)
        t1, v1, te1 = recording_level_split(recs, seed=0)
        t2, v2, te2 = recording_level_split(recs, seed=0)
        assert [r.recording_id for r in t1] == [r.recording_id for r in t2]
        assert [r.recording_id for r in te1] == [r.recording_id for r in te2]

    def test_different_seeds_differ(self):
        recs = _make_recordings(40)
        t1, _, _ = recording_level_split(recs, seed=0)
        t2, _, _ = recording_level_split(recs, seed=999)
        assert [r.recording_id for r in t1] != [r.recording_id for r in t2]

    def test_stratified_split_has_all_classes_in_train(self):
        recs = _make_recordings(40)
        train, val, test = recording_level_split(recs, seed=42, stratify=True)
        train_labels = {r.label for r in train}
        assert len(train_labels) == 4   # all classes present in train


class TestManifestRoundtrip:
    def test_save_and_load(self, tmp_path):
        recs = _make_recordings(20)
        train, val, test = recording_level_split(recs, seed=42)
        manifest_path = tmp_path / "split.json"
        save_split_manifest(train, val, test, manifest_path, seed=42)

        manifest = load_split_manifest(manifest_path)
        assert "train" in manifest
        assert "val" in manifest
        assert "test" in manifest
        assert manifest["seed"] == 42

    def test_apply_manifest_reproduces_split(self, tmp_path):
        recs = _make_recordings(20)
        train, val, test = recording_level_split(recs, seed=42)
        manifest_path = tmp_path / "split.json"
        save_split_manifest(train, val, test, manifest_path, seed=42)

        manifest = load_split_manifest(manifest_path)
        t2, v2, te2 = apply_manifest_split(recs, manifest)

        assert sorted(r.recording_id for r in train) == sorted(r.recording_id for r in t2)
        assert sorted(r.recording_id for r in test)  == sorted(r.recording_id for r in te2)
