"""
src/evaluation/metrics.py
Classification metrics computation utilities.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float | dict]:
    """Compute all classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    num_classes:
        Total number of classes (used to set ``labels`` in sklearn calls).
    class_names:
        Optional list of class name strings for per-class reporting.

    Returns
    -------
    dict with keys:
        accuracy, macro_precision, macro_recall, macro_f1,
        per_class_precision, per_class_recall, per_class_f1,
        confusion_matrix (list of lists)
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"error": "Empty predictions/labels"}

    labels = list(range(num_classes)) if num_classes else None

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    result: Dict = {
        "accuracy":         round(acc, 6),
        "macro_precision":  round(float(macro_prec), 6),
        "macro_recall":     round(float(macro_rec), 6),
        "macro_f1":         round(float(macro_f1), 6),
        "per_class_precision": [round(float(v), 6) for v in prec],
        "per_class_recall":    [round(float(v), 6) for v in rec],
        "per_class_f1":        [round(float(v), 6) for v in f1],
        "per_class_support":   [int(v) for v in support],
        "confusion_matrix":    cm,
    }

    if class_names:
        result["class_names"] = class_names

    return result


def metrics_to_dataframe(metrics: Dict) -> "pandas.DataFrame":
    """Convert per-class metrics to a pandas DataFrame."""
    import pandas as pd

    n = len(metrics.get("per_class_f1", []))
    names = metrics.get("class_names", [f"class_{i}" for i in range(n)])
    data = {
        "Class": names,
        "Precision": metrics.get("per_class_precision", [None] * n),
        "Recall":    metrics.get("per_class_recall",    [None] * n),
        "F1":        metrics.get("per_class_f1",        [None] * n),
        "Support":   metrics.get("per_class_support",   [None] * n),
    }
    return pd.DataFrame(data)
