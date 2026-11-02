"""
src/visualization/signal_plots.py
Plotly and Matplotlib plots for vibration signals and EWT components.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_raw_signal(
    signal: np.ndarray,
    sampling_rate: float = 12_000.0,
    title: str = "Raw Vibration Signal",
    max_samples: int = 4096,
) -> go.Figure:
    """Plot a raw vibration signal in the time domain.

    Parameters
    ----------
    signal:
        1-D signal array.
    sampling_rate:
        Sampling frequency in Hz (used to compute time axis in seconds).
    title:
        Plot title.
    max_samples:
        Downsample signal for display if longer (preserves shape, not decimated).
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    if len(sig) > max_samples:
        idx = np.linspace(0, len(sig) - 1, max_samples, dtype=int)
        sig = sig[idx]

    t = np.arange(len(sig)) / sampling_rate

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=sig, mode="lines",
        line=dict(color="#4C78A8", width=1),
        name="Vibration",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=280,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def plot_fft(
    signal: np.ndarray,
    sampling_rate: float = 12_000.0,
    title: str = "Frequency Spectrum",
    max_freq_hz: Optional[float] = None,
) -> go.Figure:
    """Plot the single-sided amplitude spectrum of a signal."""
    sig = np.asarray(signal, dtype=np.float64).ravel()
    N = len(sig)
    freqs = np.fft.rfftfreq(N, d=1.0 / sampling_rate)
    magnitudes = np.abs(np.fft.rfft(sig)) * 2 / N

    if max_freq_hz:
        mask = freqs <= max_freq_hz
        freqs = freqs[mask]
        magnitudes = magnitudes[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=magnitudes, mode="lines",
        line=dict(color="#F58518", width=1),
        name="Amplitude",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=280,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def plot_ewt_components(
    signal: np.ndarray,
    components: np.ndarray,
    sampling_rate: float = 12_000.0,
    boundaries_hz: Optional[List[float]] = None,
) -> go.Figure:
    """Plot original signal and each EWT component in a stacked sub-plot.

    Parameters
    ----------
    signal:
        Original 1-D signal.
    components:
        EWT output of shape ``(n_components, window_length)``.
    sampling_rate:
        Used to compute the time axis.
    boundaries_hz:
        Detected spectral boundaries (shown as text annotations).
    """
    n_comp = components.shape[0]
    n_rows = n_comp + 1   # original + each component

    subplot_titles = ["Original Signal"] + [f"EWT Component {i+1}" for i in range(n_comp)]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.04)

    t = np.arange(len(signal)) / sampling_rate

    # Original signal
    fig.add_trace(
        go.Scatter(x=t, y=signal.astype(float), mode="lines",
                   line=dict(color="#4C78A8", width=1), name="Original"),
        row=1, col=1,
    )

    colors = ["#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for i, comp in enumerate(components):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(x=t, y=comp.astype(float), mode="lines",
                       line=dict(color=color, width=1),
                       name=f"Component {i+1}"),
            row=i + 2, col=1,
        )

    # Annotate boundaries
    if boundaries_hz:
        bound_str = " | ".join(f"{b:.1f} Hz" for b in boundaries_hz)
        fig.add_annotation(
            text=f"Spectral boundaries: {bound_str}",
            xref="paper", yref="paper",
            x=0.5, y=1.0, showarrow=False,
            font=dict(size=10, color="grey"),
        )

    fig.update_layout(
        height=120 * n_rows + 80,
        showlegend=False,
        margin=dict(l=60, r=20, t=80, b=50),
    )
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    return fig


def plot_window_comparison(
    raw_window: np.ndarray,
    normalized_window: np.ndarray,
    sampling_rate: float = 12_000.0,
) -> go.Figure:
    """Side-by-side comparison of raw vs normalized window."""
    t = np.arange(len(raw_window)) / sampling_rate

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Raw Window", "Normalized Window"])
    fig.add_trace(go.Scatter(x=t, y=raw_window.astype(float), mode="lines",
                             line=dict(color="#4C78A8"), name="Raw"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=normalized_window.astype(float), mode="lines",
                             line=dict(color="#F58518"), name="Normalized"),
                  row=1, col=2)
    fig.update_layout(height=280, showlegend=False, margin=dict(l=60, r=20, t=60, b=50))
    return fig
