"""
tests/test_metrics.py
Unit tests for evaluation metrics.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, metrics_to_dataframe


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        metrics = compute_metrics(y, y, num_classes=4)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["macro_precision"] == pytest.approx(1.0)
        assert metrics["macro_recall"] == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        metrics = compute_metrics(y_true, y_pred, num_classes=2)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_confusion_matrix_shape(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(y, y, num_classes=3)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert len(cm[0]) == 3

    def test_empty_input_returns_error(self):
        metrics = compute_metrics(np.array([]), np.array([]), num_classes=4)
        assert "error" in metrics

    def test_per_class_lengths(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, 100)
        metrics = compute_metrics(y, y, num_classes=4)
        assert len(metrics["per_class_f1"]) == 4
        assert len(metrics["per_class_precision"]) == 4
        assert len(metrics["per_class_recall"]) == 4

    def test_class_names_embedded(self):
        y = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y, y, num_classes=2, class_names=["normal", "fault"])
        assert "class_names" in metrics
        assert metrics["class_names"] == ["normal", "fault"]

    def test_metrics_to_dataframe(self):
        y = np.arange(4)
        metrics = compute_metrics(y, y, num_classes=4, class_names=["a", "b", "c", "d"])
        df = metrics_to_dataframe(metrics)
        assert len(df) == 4
        assert "Class" in df.columns
        assert "F1" in df.columns
