"""
src/evaluation/confusion_matrix.py
Confusion matrix utilities and plotly-based visualisation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go


def plot_confusion_matrix(
    cm: List[List[int]] | np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> go.Figure:
    """Return a Plotly heatmap of the confusion matrix.

    Parameters
    ----------
    cm:
        2-D confusion matrix (rows = true, cols = predicted).
    class_names:
        Display names for each class.
    title:
        Figure title.
    normalize:
        When ``True``, display row-normalized values (recall per class).
    """
    cm_arr = np.array(cm, dtype=float)
    n = cm_arr.shape[0]
    names = class_names or [f"Class {i}" for i in range(n)]

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_arr / row_sums
        hover_fmt = ".2%"
    else:
        cm_display = cm_arr
        hover_fmt = "d"

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_display,
            x=names,
            y=names,
            colorscale="Blues",
            showscale=True,
            text=cm_display,
            texttemplate=f"%{{text:{hover_fmt}}}",
            textfont={"size": 12},
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis_autorange="reversed",
        height=450,
        margin=dict(l=80, r=40, t=60, b=80),
    )
    return fig
