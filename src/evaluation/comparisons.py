"""
src/evaluation/comparisons.py
Multi-experiment comparison utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def load_experiment_results(results_dir: str | Path) -> List[Dict]:
    """Load all experiment result JSON files from a directory."""
    results_dir = Path(results_dir)
    results = []
    for path in sorted(results_dir.glob("*_result.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                results.append(json.load(fh))
        except Exception:
            pass
    return results


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Flatten a list of experiment result dicts into a pandas DataFrame."""
    rows = []
    for res in results:
        bm = res.get("best_metrics", {})
        rows.append({
            "Experiment": res.get("experiment_name", "?"),
            "Status":     res.get("status", "?"),
            "Best Epoch": res.get("best_epoch", "-"),
            "Val Loss":   round(res.get("best_val_loss", float("nan")), 4),
            "Accuracy":   round(bm.get("accuracy", float("nan")), 4),
            "Macro F1":   round(bm.get("macro_f1", float("nan")), 4),
            "Macro Prec": round(bm.get("macro_precision", float("nan")), 4),
            "Macro Rec":  round(bm.get("macro_recall", float("nan")), 4),
            "Train Time (s)": round(res.get("training_time_s", 0), 1),
            "Checkpoint": res.get("checkpoint_path", ""),
        })
    return pd.DataFrame(rows)


def plot_metric_comparison(df: pd.DataFrame, metric: str = "Accuracy") -> go.Figure:
    """Bar chart comparing a metric across experiments."""
    fig = px.bar(
        df[df["Status"] == "completed"],
        x="Experiment",
        y=metric,
        color="Experiment",
        title=f"Model Comparison – {metric}",
        text=metric,
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(showlegend=False, height=400)
    return fig


def plot_training_curves(result: Dict) -> go.Figure:
    """Line plot of train and val loss over epochs for one experiment."""
    train_losses = result.get("train_losses", [])
    val_losses   = result.get("val_losses", [])
    epochs = list(range(1, len(train_losses) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode="lines", name="Train Loss",
                             line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses,   mode="lines", name="Val Loss",
                             line=dict(color="#F58518", dash="dash")))
    best_epoch = result.get("best_epoch")
    if best_epoch and 0 < best_epoch <= len(val_losses):
        fig.add_vline(x=best_epoch, line_dash="dot", line_color="green",
                      annotation_text=f"Best epoch {best_epoch}")
    fig.update_layout(
        title=f"Training Curves – {result.get('experiment_name', '')}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=350,
    )
    return fig
