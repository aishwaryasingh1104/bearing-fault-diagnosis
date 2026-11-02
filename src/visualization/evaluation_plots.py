"""
src/visualization/evaluation_plots.py
Evaluation charts: per-class bar plots, F1 radar, precision-recall charts.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_per_class_metrics(
    metrics: Dict,
    class_names: Optional[List[str]] = None,
) -> go.Figure:
    """Grouped bar chart of per-class precision, recall, and F1."""
    n = len(metrics.get("per_class_f1", []))
    names = class_names or metrics.get("class_names", [f"class_{i}" for i in range(n)])

    fig = go.Figure()
    for metric_key, color, display_name in [
        ("per_class_precision", "#4C78A8", "Precision"),
        ("per_class_recall",    "#F58518", "Recall"),
        ("per_class_f1",        "#54A24B", "F1"),
    ]:
        values = metrics.get(metric_key, [0.0] * n)
        fig.add_trace(go.Bar(
            name=display_name,
            x=names,
            y=values,
            marker_color=color,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Per-Class Metrics",
        xaxis_title="Class",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.15]),
        height=380,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=20, t=60, b=80),
    )
    return fig


def plot_metric_radar(
    models_metrics: Dict[str, Dict],
) -> go.Figure:
    """Radar (spider) chart comparing metrics across multiple models."""
    radar_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    theta = ["Accuracy", "Precision", "Recall", "F1"]

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (model_name, metrics) in enumerate(models_metrics.items()):
        r = [metrics.get(k, 0.0) for k in radar_keys]
        r.append(r[0])   # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta + [theta[0]],
            fill="toself",
            name=model_name,
            line_color=colors[i % len(colors)],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Radar",
        height=400,
    )
    return fig
