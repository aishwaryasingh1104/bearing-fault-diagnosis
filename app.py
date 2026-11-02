"""
app.py – Streamlit dashboard for Bearing Fault Diagnosis
An Intelligent Bearing Fault Diagnosis Framework Using Self-Attention Enhanced
CNN and Hierarchical Transfer Learning under Variable Working Conditions.

Structure
---------
1. Imports and page config
2. Custom CSS
3. All helper functions (defined BEFORE the page-router so the elif chain is valid)
4. Sidebar navigation
5. Config / device info loading
6. Page-router: if/elif chain — NO function definitions inside
"""
from __future__ import annotations

import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project src is importable
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bearing Fault Diagnosis | Research Framework",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #1a1d2e 100%);
    border-right: 1px solid #2d3561;
}
[data-testid="stSidebar"] * { color: #e0e4f0 !important; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1e2337 0%, #252a42 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
}
.metric-card h4 {
    color: #7c85b5;
    margin: 0 0 4px;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-card .value { color: #a8d8ea; font-size: 1.6rem; font-weight: 700; }

/* Status badges */
.badge-ok   { background: #1a472a; color: #6fcf97; border-radius: 8px; padding: 2px 10px; font-size: 0.8rem; }
.badge-warn { background: #472c1a; color: #f2994a; border-radius: 8px; padding: 2px 10px; font-size: 0.8rem; }
.badge-err  { background: #471a1a; color: #eb5757; border-radius: 8px; padding: 2px 10px; font-size: 0.8rem; }

/* Headings */
h1 { color: #a8d8ea !important; }
h2 { color: #cdd3f0 !important; }
h3 { color: #b0b8e0 !important; }

/* Step blocks */
.step-block {
    border-left: 3px solid #4a6fa5;
    padding-left: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS  (must be defined BEFORE the if/elif page-router)
# =============================================================================

# Colour palette used across bar charts – cycled for any number of classes
_BAR_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756",
                "#72B7B2", "#B279A2", "#FF9DA7", "#9D755D"]


def _cycle_colors(n: int):
    """Return a list of *n* colours by cycling the project palette."""
    return [_BAR_PALETTE[i % len(_BAR_PALETTE)] for i in range(n)]


def _make_synthetic_signal(
    fault_type: str, n_samples: int = 4096, fs: int = 12_000
) -> np.ndarray:
    """Generate a labelled synthetic vibration signal for pipeline testing.
    This signal is **not** genuine bearing data.

    Fault frequencies are approximate CWRU-style characteristic frequencies
    for a 6205-2RS JEM SKF bearing at 1797 rpm and are clearly documented.
    """
    rng = np.random.default_rng(0)
    t = np.arange(n_samples) / fs

    # Carrier (shaft rotation harmonic) + broadband noise
    sig = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.1 * rng.standard_normal(n_samples)

    # Approximate CWRU characteristic fault frequencies (Hz) at 1797 rpm
    fault_freqs = {
        "inner_race": 162.2,
        "outer_race": 107.4,
        "ball":       141.2,
    }
    if fault_type in fault_freqs:
        ff = fault_freqs[fault_type]
        # Amplitude modulation at shaft frequency (≈29.9 Hz at 1797 rpm)
        sig += 0.8 * np.sin(2 * np.pi * ff * t) * (1 + 0.3 * np.sin(2 * np.pi * 29.9 * t))

    return sig.astype(np.float32)


def _run_training_page(model_type, n_classes, epochs, batch_size, lr, seed, use_synthetic):
    """Execute one training run and stream progress + results to the Streamlit page."""
    from src.utils.reproducibility import set_seed, get_device
    from src.training.trainer import Trainer

    set_seed(int(seed))
    device = get_device()
    st.info(f"Device: `{device}`")

    # Build model
    use_ewt       = "EWT" in model_type
    use_attention = "Attention" in model_type or "Self-Attention" in model_type

    model_cfg = {
        "in_channels":       3 if use_ewt else 1,
        "num_classes":       int(n_classes),
        "conv_channels":     [32, 64, 128],
        "kernel_sizes":      [7, 5, 3],
        "attention_heads":   4,
        "attention_dropout": 0.1,
        "dropout":           0.3,
        "fc_units":          [256, 128],
    }

    if use_attention:
        from src.models.attention_cnn import build_attention_cnn
        model = build_attention_cnn(model_cfg)
    else:
        from src.models.baseline_cnn import build_baseline_cnn
        model = build_baseline_cnn(model_cfg)

    st.success(f"Model: `{model.__class__.__name__}` — {model.n_parameters():,} parameters")

    # Generate or load data
    with st.spinner("Preparing data…"):
        if use_synthetic:
            rng = np.random.default_rng(int(seed))
            if use_ewt:
                X_train = rng.standard_normal((200, 3, 2048)).astype(np.float32)
                X_val   = rng.standard_normal(( 60, 3, 2048)).astype(np.float32)
            else:
                X_train = rng.standard_normal((200, 2048)).astype(np.float32)
                X_val   = rng.standard_normal(( 60, 2048)).astype(np.float32)
            y_train = rng.integers(0, int(n_classes), 200, dtype=np.int64)
            y_val   = rng.integers(0, int(n_classes),  60, dtype=np.int64)
            st.warning("⚠️ Using **synthetic random data** for demonstration. "
                       "Results have no scientific meaning.")
        else:
            st.error("Real dataset training: load dataset first via the Dataset Explorer page.")
            return

    # Training config — all values driven by UI sliders (no magic numbers here)
    train_cfg = {
        "training": {
            "epochs":                   int(epochs),
            "batch_size":               int(batch_size),
            "learning_rate":            float(lr),
            "weight_decay":             1e-4,
            "optimizer":                "adam",
            "scheduler":                "cosine",
            "scheduler_T_max":          int(epochs),
            "early_stopping_patience":  max(3, int(epochs) // 4),
            "checkpoint_dir":           "artifacts/checkpoints",
            "class_weighted_loss":      True,
            "grad_clip_norm":           5.0,
        }
    }

    # Live progress
    progress_bar    = st.progress(0.0, text="Training…")
    loss_container  = st.empty()
    train_losses: list = []
    val_losses:   list = []

    def on_progress(epoch, total, train_loss, val_loss, metrics):
        pct = epoch / total
        progress_bar.progress(
            pct,
            text=f"Epoch {epoch}/{total} | train={train_loss:.4f} | val={val_loss:.4f}",
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if len(train_losses) > 1:
            from src.visualization.training_plots import plot_loss_curves
            fig = plot_loss_curves(train_losses, val_losses,
                                   best_epoch=0, experiment_name=model_type)
            loss_container.plotly_chart(fig, use_container_width=True)

    trainer = Trainer(
        model=model, device=device, config=train_cfg,
        experiment_name=model_type, progress_callback=on_progress,
    )
    result = trainer.train(X_train, y_train, X_val, y_val,
                           num_classes=int(n_classes), multichannel=use_ewt)

    progress_bar.progress(1.0, text="✅ Training complete!")

    # Summary
    st.markdown("### Training Results")
    bm = result.best_metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Epoch",   result.best_epoch)
    c2.metric("Val Accuracy", f"{bm.get('accuracy', 0):.4f}")
    c3.metric("Macro F1",     f"{bm.get('macro_f1', 0):.4f}")
    c4.metric("Train Time",   f"{result.training_time_s:.1f}s")

    if result.checkpoint_path:
        st.success(f"Best checkpoint saved: `{result.checkpoint_path}`")

    result.save("artifacts/results")
    st.caption("Result saved to `artifacts/results/`")


def _run_transfer_page(n_src_cls, n_tgt_cls, adapt_weight, freeze_epochs,
                       epochs, seed, src_label, tgt_label):
    """Transfer learning demo with synthetic data."""
    # BUG-01 fix: imports at top of function — before any usage of torch objects
    import torch
    import torch.nn as nn

    from src.utils.reproducibility import set_seed, get_device
    from src.models.transfer_learning import build_transfer_model

    set_seed(seed)
    device = get_device()
    st.info(f"Device: `{device}` — synthetic data mode (no dataset required for demo)")

    backbone_cfg = {
        "in_channels":       1,
        "num_classes":       n_src_cls,
        "conv_channels":     [32, 64, 128],
        "kernel_sizes":      [7, 5, 3],
        "attention_heads":   4,
        "attention_dropout": 0.1,
        "dropout":           0.3,
        "fc_units":          [256, 128],
    }
    transfer_cfg = {
        "backbone":           backbone_cfg,
        "adaptation_weight":  adapt_weight,
        "use_adversarial":    True,
    }

    model = build_transfer_model(transfer_cfg, n_src_cls, n_tgt_cls).to(device)
    st.success(f"Model instantiated — {model.n_parameters():,} parameters")
    st.warning("⚠️ Using synthetic random data. Results have no scientific meaning.")

    rng  = np.random.default_rng(seed)
    X_src = torch.tensor(rng.standard_normal((200, 1, 2048)).astype(np.float32)).to(device)
    y_src = torch.tensor(rng.integers(0, n_src_cls, 200, dtype=np.int64)).to(device)
    X_tgt = torch.tensor(rng.standard_normal((100, 1, 2048)).astype(np.float32)).to(device)
    y_tgt = torch.tensor(rng.integers(0, n_tgt_cls, 100, dtype=np.int64)).to(device)

    optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    src_criterion = nn.CrossEntropyLoss()

    progress = st.progress(0.0, text="Pre-training on source…")
    src_accs, tgt_accs = [], []

    model.freeze_backbone()
    for ep in range(1, epochs + 1):
        if ep == freeze_epochs + 1:
            model.unfreeze_backbone()
            st.info(f"Epoch {ep}: Backbone unfrozen — starting domain adaptation.")

        model.train()
        optimizer.zero_grad()

        src_logits = model.forward_source(X_src)
        loss_src   = src_criterion(src_logits, y_src)

        alpha    = min(1.0, ep / epochs)
        loss_dom = model.compute_adaptation_loss(X_src, X_tgt, alpha=alpha) * adapt_weight

        (loss_src + loss_dom).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            src_acc = (model.forward_source(X_src).argmax(1) == y_src).float().mean().item()
            tgt_acc = (model.forward_target(X_tgt).argmax(1) == y_tgt).float().mean().item()
        src_accs.append(src_acc)
        tgt_accs.append(tgt_acc)

        progress.progress(
            ep / epochs,
            text=f"Epoch {ep}/{epochs} | src_acc={src_acc:.3f} | tgt_acc={tgt_acc:.3f}",
        )

    progress.progress(1.0, text="✅ Done")

    eps = list(range(1, epochs + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=src_accs, mode="lines+markers",
                             name=f"Source ({src_label})", line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=eps, y=tgt_accs, mode="lines+markers",
                             name=f"Target ({tgt_label})",
                             line=dict(color="#F58518", dash="dash")))
    if freeze_epochs > 0:
        fig.add_vline(x=freeze_epochs, line_dash="dot", line_color="grey",
                      annotation_text="Backbone unfrozen")
    fig.update_layout(
        title="Transfer Learning: Source vs Target Accuracy",
        xaxis_title="Epoch", yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.1]), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric(f"Final Source Acc ({src_label})", f"{src_accs[-1]:.4f}")
    c2.metric(f"Final Target Acc ({tgt_label})", f"{tgt_accs[-1]:.4f}")


def _run_diagnosis(ckpt_path, uploaded, input_fs, use_ewt, n_ewt_comp,
                   window_length: int = 2048):
    """Run inference on a single signal file or a synthetic fallback."""
    import torch
    from src.training.checkpointing import load_checkpoint
    from src.data.preprocessing import validate_signal, segment_signal, Normalizer
    from src.signal_processing.ewt import decompose
    from src.visualization.signal_plots import plot_raw_signal

    try:
        # ── Load signal ──────────────────────────────────────────────────────
        if uploaded is None:
            sig = _make_synthetic_signal("inner_race", 8192, int(input_fs))
            st.warning("No file uploaded — using synthetic inner-race signal for demonstration.")
        else:
            raw = uploaded.read()
            if uploaded.name.endswith(".mat"):
                import scipy.io as sio
                mat = sio.loadmat(io.BytesIO(raw), squeeze_me=True)
                key = next((k for k in mat if not k.startswith("_")), None)
                if key is None:
                    st.error("No usable variable in .mat file.")
                    return
                sig = np.array(mat[key], dtype=np.float32).ravel()
            else:
                sig = pd.read_csv(io.StringIO(raw.decode()), header=None).iloc[:, 0].to_numpy(np.float32)

        valid, msg = validate_signal(sig)
        if not valid:
            st.error(f"Signal validation failed: {msg}")
            return

        st.plotly_chart(
            plot_raw_signal(sig, sampling_rate=float(input_fs), title="Input Signal"),
            use_container_width=True,
        )

        # ── Preprocess ───────────────────────────────────────────────────────
        # BUG-04 fix: window_length is a parameter, not hardcoded to 2048
        norm   = Normalizer("zscore").fit(sig.reshape(1, -1))
        window = segment_signal(
            norm.transform(sig.reshape(1, -1)).ravel(),
            window_length=window_length,
            overlap=0,
        )[0]

        if use_ewt:
            ewt_result = decompose(window, n_components=n_ewt_comp,
                                   sampling_rate=float(input_fs))
            inp = torch.tensor(ewt_result.components).unsqueeze(0)   # (1, C, L)
        else:
            inp = torch.tensor(window).unsqueeze(0).unsqueeze(0)     # (1, 1, L)

        # ── Build model from checkpoint ──────────────────────────────────────
        # BUG-05 fix: read num_classes from checkpoint label_map, not hardcoded=4
        _peek         = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _label_map_pk = _peek.get("label_map", {})
        n_cls         = len(_label_map_pk) if _label_map_pk else 4

        if use_ewt:
            from src.models.attention_cnn import AttentionCNN
            dummy_model = AttentionCNN(in_channels=n_ewt_comp, num_classes=n_cls)
        else:
            from src.models.baseline_cnn import BaselineCNN
            dummy_model = BaselineCNN(in_channels=1, num_classes=n_cls)

        payload      = load_checkpoint(ckpt_path, dummy_model, strict=False)
        label_map    = payload.get("label_map", {})
        idx_to_label = (
            {v: k for k, v in label_map.items()} if label_map
            else {0: "normal", 1: "inner_race", 2: "outer_race", 3: "ball"}
        )

        dummy_model.eval()
        with torch.no_grad():
            logits   = dummy_model(inp)
            probs    = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_idx = int(probs.argmax())

        pred_label  = idx_to_label.get(pred_idx, f"class_{pred_idx}")
        confidence  = float(probs.max())

        st.markdown(f"### Prediction: **{pred_label}**")
        st.markdown(f"Confidence: **{confidence:.2%}**")

        # BUG-03 fix: collapsed duplicate if/else that produced the same expression
        labels_list = [idx_to_label.get(i, f"class_{i}") for i in range(len(probs))]

        fig = go.Figure(go.Bar(
            x=labels_list, y=probs.tolist(),
            marker_color=["#54A24B" if i == pred_idx else "#4C78A8"
                          for i in range(len(probs))],
            text=[f"{p:.3f}" for p in probs], textposition="outside",
        ))
        fig.update_layout(title="Class Probability",
                          yaxis=dict(range=[0, 1.2]), height=320)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(f"Diagnosis failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
PAGES = {
    "🏠  Project Overview":  "overview",
    "📂  Dataset Explorer":  "dataset",
    "🌊  Signal Processing": "signal",
    "🧠  Model Training":    "training",
    "🔄  Transfer Learning": "transfer",
    "📊  Model Comparison":  "comparison",
    "🔍  Fault Diagnosis":   "diagnosis",
    "📋  Reports & History": "reports",
}

with st.sidebar:
    st.markdown("## ⚙️ BFD Framework")
    st.markdown("---")
    page_label = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]
    st.markdown("---")
    cfg_path = st.text_input(
        "Config file", value="configs/default.yaml",
        help="YAML experiment configuration",
    )
    st.markdown("---")
    st.caption("v1.0.0 · Research Framework")

# =============================================================================
# CONFIG AND DEVICE
# =============================================================================

@st.cache_data(show_spinner=False)
def _load_cfg(path: str) -> dict:
    try:
        from src.utils.config import load_config
        return load_config(path)
    except Exception as exc:
        return {"_error": str(exc)}


@st.cache_data(show_spinner=False)
def _get_device_info() -> dict:
    from src.utils.reproducibility import device_info
    return device_info()


cfg      = _load_cfg(cfg_path)
dev_info = _get_device_info()

# =============================================================================
# PAGE 1: PROJECT OVERVIEW
# =============================================================================
if page == "overview":
    st.title("⚙️ Intelligent Bearing Fault Diagnosis Framework")
    st.markdown("""
    > **An Intelligent Bearing Fault Diagnosis Framework Using Self-Attention Enhanced CNN
    and Hierarchical Transfer Learning under Variable Working Conditions**
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h4>Models</h4>'
                    '<div class="value">4</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>Datasets</h4>'
                    '<div class="value">3</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>Experiments</h4>'
                    '<div class="value">A – F</div></div>', unsafe_allow_html=True)
    with col4:
        cuda_str = "✅ CUDA" if dev_info.get("cuda_available") else "CPU only"
        st.markdown(f'<div class="metric-card"><h4>Device</h4>'
                    f'<div class="value">{cuda_str}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab_obj, tab_method, tab_arch, tab_status = st.tabs(
        ["Objective", "Methodology", "Architecture", "Status"]
    )

    with tab_obj:
        st.markdown("""
        ### Research Objective
        Develop an intelligent condition-monitoring system that analyses bearing vibration signals
        and classifies bearing health conditions under **variable operating conditions**.

        ### Research Question
        > *Can combining Empirical Wavelet Transform (EWT) signal decomposition, self-attention-based
        feature learning, and hierarchical transfer learning improve bearing fault diagnosis when the
        operating speed or load changes?*

        This is treated as a **research hypothesis** — the hybrid model is not assumed superior
        before experiments are conducted.

        ### Fault Categories
        | Code | Description |
        |------|-------------|
        | `normal` | Healthy bearing |
        | `inner_race` | Fault on the inner race |
        | `outer_race` | Fault on the outer race |
        | `ball` | Rolling-element fault |
        """)

    with tab_method:
        st.markdown("""
        ### Three-Component Methodology

        **Component 1 — EWT Signal Decomposition**
        - Empirical Wavelet Transform (Gilles, 2013)
        - Adaptive frequency-band detection from the signal spectrum
        - Decomposes raw vibration into empirical modes capturing different fault-frequency bands
        - Configurable number of components (default: 3)

        **Component 2 — Self-Attention CNN**
        - 1-D CNN for local temporal feature extraction
        - Multi-head self-attention for relational feature learning across positions
        - Global Average Pooling reduces length-dependence
        - Dropout and batch normalisation for regularisation

        **Component 3 — Hierarchical Transfer Learning**
        - Domain-adversarial adaptation (DANN-style, Ganin et al. 2016)
        - Gradient reversal layer aligns source and target feature distributions
        - Separate classification heads for source and target domains
        - Hierarchical levels: Level 1 = fault type, Level 2 = fault severity
        - Provisional baseline — to be updated when reference Paper 2 details are verified
        """)

    with tab_arch:
        st.markdown("""
        ### End-to-End Pipeline
        ```
        Raw Vibration Signal
               │
               ▼
        ┌──────────────────┐
        │   Preprocessing  │  Segmentation (L=2048) → Normalization
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  EWT Decomp.     │  3-component empirical wavelet decomposition
        └────────┬─────────┘    (B, 3, 2048)
                 │
                 ▼
        ┌──────────────────────────────────┐
        │   Self-Attention 1D CNN          │
        │   Conv Blocks → Attention → GAP  │
        └────────┬─────────────────────────┘
                 │ feature vector (B, 128)
                 ▼
        ┌──────────────────────────────────┐
        │  Hierarchical Transfer Learning  │
        │  Source Head / Target Head       │
        │  Domain Discriminator (GRL)      │
        └────────┬─────────────────────────┘
                 │
                 ▼
          Fault Classification
          (normal | inner_race | outer_race | ball)
        ```

        ### Tensor Shapes
        | Stage | Shape |
        |-------|-------|
        | Raw input | `(B, 1, 2048)` |
        | EWT output | `(B, 3, 2048)` |
        | After Conv Blocks | `(B, 128, L')` |
        | After Attention | `(B, 128, L')` |
        | After GAP | `(B, 128)` |
        | Logits | `(B, num_classes)` |
        """)

    with tab_status:
        st.markdown("### Implementation Status")
        items = [
            ("✅", "Project structure and configuration"),
            ("✅", "CWRU dataset adapter (MATLAB .mat loading)"),
            ("✅", "Paderborn dataset adapter (operating-condition parsing)"),
            ("✅", "XJTU-SY adapter (optional, degradation experiments)"),
            ("✅", "Signal preprocessing (windowing, normalization)"),
            ("✅", "Recording-level data splitting (leakage prevention)"),
            ("✅", "Empirical Wavelet Transform module"),
            ("✅", "Baseline 1D CNN"),
            ("✅", "Self-Attention 1D CNN"),
            ("✅", "Hierarchical Transfer Learning (provisional DANN baseline)"),
            ("✅", "Hybrid EWT + Attention CNN + Transfer Learning pipeline"),
            ("✅", "Training loop with early stopping & checkpointing"),
            ("✅", "Evaluation metrics (accuracy, macro F1, confusion matrix)"),
            ("✅", "Streamlit dashboard (8 pages)"),
            ("✅", "Automated test suite"),
            ("⏳", "Real dataset experiments (requires data download)"),
            ("⏳", "Reference Paper 2 architecture verification"),
        ]
        for icon, text in items:
            st.markdown(f"{icon} {text}")
        st.markdown("---")
        st.markdown("### Device Information")
        for k, v in dev_info.items():
            st.markdown(f"- **{k}**: `{v}`")


# =============================================================================
# PAGE 2: DATASET EXPLORER
# =============================================================================
elif page == "dataset":
    st.title("📂 Dataset Explorer")

    ds_name   = st.selectbox("Select Dataset", ["CWRU", "Paderborn", "XJTU-SY"])
    data_root = st.text_input(
        "Dataset root directory",
        value=f"data/raw/{ds_name}",
        help="Absolute or relative path to the downloaded dataset files.",
    )

    if st.button("🔍 Validate & Load Dataset", type="primary"):
        with st.spinner("Scanning for recordings…"):
            try:
                if ds_name == "CWRU":
                    from src.data.cwru_dataset import CWRUDataset
                    ds = CWRUDataset(
                        root=data_root,
                        sampling_rate=cfg.get("data", {}).get("cwru", {}).get(
                            "sampling_rate", 12_000
                        ),
                    )
                elif ds_name == "Paderborn":
                    from src.data.paderborn_dataset import PaderbornDataset
                    ds = PaderbornDataset(root=data_root)
                else:
                    from src.data.xjtu_dataset import XJTUSYDataset
                    ds = XJTUSYDataset(root=data_root)

                valid, messages = ds.validate()
                for msg in messages:
                    if valid:
                        st.success(msg)
                    else:
                        st.error(msg)

                if valid:
                    recordings = ds.get_recordings()
                    summary    = ds.summary()

                    st.markdown("### Dataset Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Recordings", summary["n_recordings"])
                    c2.metric("Classes",    summary["n_classes"])
                    c3.metric("Sampling Rate", f"{summary['sampling_rate']} Hz")
                    c4.metric("Total Windows (est.)", "Requires preprocessing")

                    st.markdown("### Class Distribution")
                    dist = summary["class_distribution"]
                    # BUG-06 fix: palette is cycled so >4 classes never crash
                    fig = go.Figure(go.Bar(
                        x=list(dist.keys()),
                        y=list(dist.values()),
                        marker_color=_cycle_colors(len(dist)),
                        text=list(dist.values()),
                        textposition="outside",
                    ))
                    fig.update_layout(title="Recordings per Class", height=320,
                                      xaxis_title="Fault Class", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### Recording Metadata")
                    df = pd.DataFrame([r.to_dict() for r in recordings])
                    cols_show = ["recording_id", "label", "n_samples", "sampling_rate",
                                 "operating_condition", "fault_severity", "speed_rpm"]
                    cols_show = [c for c in cols_show if c in df.columns]
                    st.dataframe(df[cols_show], use_container_width=True)

                    # Signal preview
                    st.markdown("### Signal Preview")
                    sel_idx = st.number_input(
                        "Recording index", min_value=0,
                        max_value=len(recordings) - 1, value=0,
                    )
                    if st.button("▶ Load and plot signal"):
                        sig  = ds.load_signal(recordings[int(sel_idx)])
                        meta = recordings[int(sel_idx)]
                        from src.visualization.signal_plots import plot_raw_signal, plot_fft
                        st.plotly_chart(
                            plot_raw_signal(sig, sampling_rate=meta.sampling_rate,
                                            title=f"{meta.recording_id} – {meta.label}"),
                            use_container_width=True,
                        )
                        st.plotly_chart(
                            plot_fft(sig, sampling_rate=meta.sampling_rate),
                            use_container_width=True,
                        )

            except Exception as exc:
                st.error(f"Error loading dataset: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    st.markdown("---")
    with st.expander("📥 Dataset Download Instructions"):
        st.markdown("""
        ### CWRU Bearing Dataset
        1. Visit: https://engineering.case.edu/bearingdatacenter/download-data-file
        2. Download MATLAB `.mat` files for the desired loads (0–3 hp).
        3. Place them in `data/raw/CWRU/` (any subdirectory structure is supported).
        4. Recommended files: Normal baseline + 0.007 inch fault diameter (inner, outer, ball) at each load.

        ### Paderborn University Dataset
        1. Visit: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
        2. Request access and download.
        3. Extract so that the structure is:
           ```
           data/raw/Paderborn/
               K001/N09_M07_F10/*.mat
               KA01/N09_M07_F10/*.mat
               ...
           ```

        ### XJTU-SY (optional)
        1. Visit: https://biaowang.tech/xjtu-sy-bearing-datasets/
        2. Download and extract so that:
           ```
           data/raw/XJTU-SY/
               35Hz12kN/Bearing1_1/acc_00001.csv
               ...
           ```
        """)


# =============================================================================
# PAGE 3: SIGNAL PROCESSING
# =============================================================================
elif page == "signal":
    st.title("🌊 Signal Processing")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Configuration")
        win_len     = st.slider("Window Length", 512, 8192, 2048, step=256)
        overlap     = st.slider("Overlap (%)", 0, 90, 50) / 100.0
        norm_method = st.selectbox("Normalization", ["zscore", "minmax", "none"])
        n_ewt       = st.slider("EWT Components", 2, 8, 3)
        fs          = st.number_input("Sampling Rate (Hz)", value=12_000, min_value=1000)

        st.markdown("### Generate Synthetic Test Signal")
        st.caption("Synthetic signals are clearly labelled and used only for pipeline verification.")
        fault_type = st.selectbox("Simulated fault",
                                  ["normal", "inner_race", "outer_race", "ball"])
        if st.button("▶ Generate & Analyse"):
            st.session_state["synth_signal"] = _make_synthetic_signal(fault_type, win_len, int(fs))
            st.session_state["synth_fs"]     = fs
            st.session_state["synth_label"]  = fault_type

    with col_right:
        if "synth_signal" in st.session_state:
            sig    = st.session_state["synth_signal"]
            fs_val = st.session_state["synth_fs"]
            label  = st.session_state["synth_label"]

            st.info(f"⚠️ **Synthetic signal** — `{label}` — for pipeline testing only.  "
                    "Not genuine bearing data.")

            from src.visualization.signal_plots import (
                plot_raw_signal, plot_fft, plot_ewt_components, plot_window_comparison
            )
            from src.data.preprocessing import segment_signal, Normalizer
            from src.signal_processing.ewt import decompose

            st.plotly_chart(
                plot_raw_signal(sig, sampling_rate=fs_val, title=f"[SYNTHETIC] {label}"),
                use_container_width=True,
            )

            try:
                windows = segment_signal(sig, win_len, overlap)
                st.success(f"Segmentation: {len(windows)} windows of length {win_len}")

                norm  = Normalizer(method=norm_method).fit(windows)
                w_norm = norm.transform(windows)

                st.plotly_chart(
                    plot_window_comparison(windows[0], w_norm[0], sampling_rate=fs_val),
                    use_container_width=True,
                )

                st.markdown("#### EWT Decomposition")
                with st.spinner("Running EWT…"):
                    try:
                        result = decompose(windows[0], n_components=n_ewt,
                                           sampling_rate=float(fs_val))
                        st.success(
                            f"EWT complete — backend: `{result.backend}` | "
                            f"components: {result.n_components} | "
                            f"reconstruction error: {result.reconstruction_error:.4f}"
                        )
                        st.plotly_chart(
                            plot_ewt_components(windows[0], result.components,
                                                sampling_rate=float(fs_val),
                                                boundaries_hz=result.boundaries_hz),
                            use_container_width=True,
                        )
                        st.plotly_chart(
                            plot_fft(sig, sampling_rate=float(fs_val)),
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.error(f"EWT failed: {exc}")
            except Exception as exc:
                st.error(f"Segmentation failed: {exc}")
        else:
            st.info("👈  Configure and press **Generate & Analyse** to see signal processing results.")


# =============================================================================
# PAGE 4: MODEL TRAINING
# =============================================================================
elif page == "training":
    st.title("🧠 Model Training")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Model Configuration")
        model_type = st.selectbox("Model", [
            "Baseline 1D CNN (raw)",
            "EWT + CNN",
            "Self-Attention CNN (raw)",
            "EWT + Self-Attention CNN",
        ])
        n_classes  = st.number_input("Num Classes", 2, 10, 4)
        epochs     = st.slider("Epochs", 1, 100, 10)
        batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
        lr         = st.select_slider(
            "Learning Rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
        )
        seed = st.number_input("Random Seed", value=42)

        st.markdown("### Data Source")
        use_synthetic = st.checkbox(
            "Use synthetic fixtures (no dataset required)", value=True
        )
        if not use_synthetic:
            st.text_input("Dataset path", value="data/raw/CWRU")

        train_btn = st.button("🚀 Start Training", type="primary")

    with col2:
        if train_btn:
            _run_training_page(model_type, n_classes, epochs, batch_size, lr,
                               seed, use_synthetic)
        else:
            st.info("👈  Configure and press **Start Training** to run an experiment.")
            ckpt_dir = Path("artifacts/checkpoints")
            if ckpt_dir.exists():
                from src.training.checkpointing import list_checkpoints
                ckpts = list_checkpoints(ckpt_dir)
                if ckpts:
                    st.markdown("### Existing Checkpoints")
                    for ckpt in ckpts[:5]:
                        st.markdown(f"- `{ckpt.name}` ({ckpt.stat().st_size // 1024} KB)")


# =============================================================================
# PAGE 5: TRANSFER LEARNING
# =============================================================================
elif page == "transfer":
    st.title("🔄 Transfer Learning")

    st.markdown("""
    ### Cross-Condition Transfer Learning
    Train a model on a **source operating condition** and evaluate its adaptation to a
    **target operating condition** (different speed, torque, or load).

    > ⚠️ **Implementation Status:** Provisional DANN-style baseline.
    > The exact architecture of the second reference paper will be incorporated
    > when its full details are verified.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Source Domain")
        src_dataset = st.selectbox("Source dataset",
                                   ["CWRU (load=0hp)", "Paderborn N09_M07_F10"])
        n_src_cls   = st.number_input("Source classes", 2, 8, 4, key="src_cls")

    with col2:
        st.markdown("#### Target Domain")
        tgt_dataset = st.selectbox("Target dataset",
                                   ["CWRU (load=2hp)", "Paderborn N15_M07_F04"])
        n_tgt_cls   = st.number_input("Target classes", 2, 8, 4, key="tgt_cls")

    st.markdown("---")
    adapt_weight  = st.slider("Adaptation weight λ", 0.0, 1.0, 0.3)
    freeze_epochs = st.slider("Backbone freeze epochs", 0, 20, 5)
    epochs_tl     = st.slider("Total epochs", 1, 50, 15)
    seed_tl       = st.number_input("Seed", value=42, key="tl_seed")

    if st.button("🚀 Run Transfer Learning Experiment", type="primary"):
        _run_transfer_page(
            n_src_cls=int(n_src_cls), n_tgt_cls=int(n_tgt_cls),
            adapt_weight=adapt_weight, freeze_epochs=int(freeze_epochs),
            epochs=int(epochs_tl), seed=int(seed_tl),
            src_label=src_dataset, tgt_label=tgt_dataset,
        )

    st.markdown("---")
    with st.expander("📋 Transfer Learning Protocol"):
        st.markdown("""
        **Stage 1 – Source Pre-training**
        - Train backbone + source head on source-domain training data.
        - Backbone is frozen after `freeze_epochs` epochs.

        **Stage 2 – Domain Adaptation**
        - Unfreeze backbone.
        - Simultaneously minimise:
          - `L_cls` (source classification cross-entropy)
          - `λ × L_domain` (adversarial domain alignment via gradient reversal)

        **Stage 3 – Target Fine-tuning**
        - Fine-tune target head on available target training data.

        **Evaluation Protocol**
        - Report source-domain and target-domain accuracy separately.
        - Target test recordings are held completely separate during training.
        """)


# =============================================================================
# PAGE 6: MODEL COMPARISON
# =============================================================================
elif page == "comparison":
    st.title("📊 Model Comparison")

    results_dir = Path("artifacts/results")
    from src.evaluation.comparisons import (
        load_experiment_results, results_to_dataframe,
        plot_metric_comparison, plot_training_curves,
    )
    results = load_experiment_results(results_dir)

    if not results:
        st.info("No completed experiments found yet.  Train models on the **Model Training** page.")
        st.markdown("### Expected Experiments")
        exp_table = {
            "Exp": ["A", "B", "C", "D", "E", "F"],
            "Name": [
                "Baseline 1D CNN (raw)",
                "EWT + CNN",
                "Self-Attention CNN (raw)",
                "EWT + Self-Attention CNN",
                "Transfer Learning (no EWT)",
                "Hybrid (EWT + Attention + Transfer)",
            ],
            "Status": ["⏳ Pending"] * 6,
        }
        st.dataframe(pd.DataFrame(exp_table), use_container_width=True)
    else:
        df = results_to_dataframe(results)
        st.markdown("### Experiment Results Table")
        st.dataframe(df, use_container_width=True)

        metric_choice = st.selectbox(
            "Compare by", ["Accuracy", "Macro F1", "Macro Prec", "Macro Rec"]
        )
        if len(df[df["Status"] == "completed"]) > 0:
            st.plotly_chart(
                plot_metric_comparison(df, metric=metric_choice),
                use_container_width=True,
            )

        st.markdown("### Training Curves")
        exp_names = [r.get("experiment_name", "?") for r in results]
        sel_exp    = st.selectbox("Select experiment", exp_names)
        sel_result = next(r for r in results if r.get("experiment_name") == sel_exp)
        st.plotly_chart(plot_training_curves(sel_result), use_container_width=True)

        st.markdown("### Confusion Matrix")
        cm_data = sel_result.get("best_metrics", {}).get("confusion_matrix")
        if cm_data:
            from src.evaluation.confusion_matrix import plot_confusion_matrix
            st.plotly_chart(
                plot_confusion_matrix(cm_data, title=f"Confusion Matrix – {sel_exp}"),
                use_container_width=True,
            )

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("📥 Export CSV", csv_buf.getvalue(),
                           file_name="experiment_results.csv", mime="text/csv")


# =============================================================================
# PAGE 7: FAULT DIAGNOSIS
# =============================================================================
elif page == "diagnosis":
    st.title("🔍 Fault Diagnosis")

    st.markdown("""
    > ⚠️ **Research Disclaimer:** Predictions are experimental research outputs from a student
    project.  They are **not** a substitute for qualified industrial bearing inspection.

    Upload a `.mat` or `.csv` vibration file, or use a pre-loaded signal for inference.
    """)

    ckpt_dir = Path("artifacts/checkpoints")
    from src.training.checkpointing import list_checkpoints
    checkpoints = list_checkpoints(ckpt_dir)

    if not checkpoints:
        st.warning("No model checkpoints found.  Please train a model first (Model Training page).")
    else:
        ckpt_choice = st.selectbox("Select checkpoint", [c.name for c in checkpoints])
        ckpt_path   = ckpt_dir / ckpt_choice

        uploaded    = st.file_uploader("Upload vibration file (.mat or .csv)", type=["mat", "csv"])
        input_fs    = st.number_input("Sampling rate (Hz)", value=12_000)
        use_ewt     = st.checkbox("Apply EWT preprocessing (if model requires multi-channel input)")
        n_ewt_inf   = st.slider("EWT components", 2, 8, 3) if use_ewt else 1
        # BUG-04 fix: window length is now exposed to the user
        diag_win    = st.number_input("Window length (samples)", value=2048,
                                      min_value=256, max_value=8192, step=256)

        if st.button("🔍 Run Diagnosis", type="primary"):
            _run_diagnosis(ckpt_path, uploaded, input_fs, use_ewt, n_ewt_inf,
                           window_length=int(diag_win))


# =============================================================================
# PAGE 8: REPORTS & EXPERIMENT HISTORY
# =============================================================================
elif page == "reports":
    st.title("📋 Reports & Experiment History")

    results_dir = Path("artifacts/results")
    from src.evaluation.comparisons import load_experiment_results, results_to_dataframe

    results = load_experiment_results(results_dir)

    if not results:
        st.info("No completed experiments yet.")
    else:
        df = results_to_dataframe(results)
        st.markdown("### Experiment Summary")
        st.dataframe(df, use_container_width=True)

        st.markdown("### Reproducibility Summary")
        for res in results:
            with st.expander(
                f"📌 {res.get('experiment_name', '?')} — {res.get('status', '?')}"
            ):
                cfg_r     = res.get("config", {})
                train_cfg = cfg_r.get("training", cfg_r)
                st.json({
                    "seed":         train_cfg.get("seed", "N/A"),
                    "epochs":       train_cfg.get("epochs"),
                    "batch_size":   train_cfg.get("batch_size"),
                    "lr":           train_cfg.get("learning_rate"),
                    "optimizer":    train_cfg.get("optimizer"),
                    "scheduler":    train_cfg.get("scheduler"),
                    "best_epoch":   res.get("best_epoch"),
                    "checkpoint":   res.get("checkpoint_path"),
                    "warnings":     res.get("warnings"),
                })

        st.markdown("### Export")
        col1, col2 = st.columns(2)
        with col1:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("📥 Download CSV", csv_buf.getvalue(),
                               file_name="results.csv", mime="text/csv")
        with col2:
            json_str = json.dumps(list(results), indent=2, default=str)
            st.download_button("📥 Download JSON", json_str,
                               file_name="results.json", mime="application/json")
