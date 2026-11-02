"""
src/visualization/training_plots.py
Plotly training curves and learning-rate plots.
"""
from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    best_epoch: int = 0,
    experiment_name: str = "",
) -> go.Figure:
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode="lines+markers",
                             name="Train Loss", line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses,   mode="lines+markers",
                             name="Val Loss",   line=dict(color="#F58518", dash="dash")))
    if best_epoch > 0 and best_epoch <= len(val_losses):
        fig.add_vline(x=best_epoch, line_dash="dot", line_color="#54A24B",
                      annotation_text=f"Best epoch {best_epoch}",
                      annotation_position="top right")
    fig.update_layout(
        title=f"Loss Curves — {experiment_name}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=350,
        legend=dict(x=0.7, y=0.9),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def plot_val_accuracy_curve(
    val_metrics: List[Dict],
    experiment_name: str = "",
) -> go.Figure:
    epochs = list(range(1, len(val_metrics) + 1))
    acc    = [m.get("accuracy",  0.0) for m in val_metrics]
    macro_f1 = [m.get("macro_f1", 0.0) for m in val_metrics]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=acc,      mode="lines+markers", name="Val Accuracy",
                             line=dict(color="#54A24B")))
    fig.add_trace(go.Scatter(x=epochs, y=macro_f1, mode="lines+markers", name="Val Macro F1",
                             line=dict(color="#E45756", dash="dash")))
    fig.update_layout(
        title=f"Validation Metrics — {experiment_name}",
        xaxis_title="Epoch",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        height=300,
        legend=dict(x=0.01, y=0.05),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig
