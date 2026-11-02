# Bearing Fault Diagnosis Framework

**An Intelligent Bearing Fault Diagnosis Framework Using Self-Attention Enhanced CNN and Hierarchical Transfer Learning under Variable Working Conditions**

---

## 1. Project Overview

This repository implements a complete research framework for intelligent bearing fault diagnosis from vibration signals. It combines three methodological components:

1. **Empirical Wavelet Transform (EWT)** – adaptive frequency-band decomposition of raw vibration signals.
2. **Self-Attention Enhanced 1-D CNN** – convolutional feature extraction augmented with multi-head self-attention for fault-feature learning.
3. **Hierarchical Transfer Learning** – domain-adversarial adaptation to generalise fault classification across variable operating conditions.

The framework is designed as an honest research pipeline: it does not assume the hybrid model is superior before running experiments, and it clearly distinguishes verified implementations from provisional baselines.

---

## 2. Research Motivation

Industrial bearings operate under varying speeds and loads. A model trained at one operating condition typically degrades when deployed at another. This project asks:

> *Can combining EWT signal decomposition, attention-based feature learning, and hierarchical transfer learning improve bearing fault diagnosis when operating conditions change?*

---

## 3. Architecture

```
Raw Vibration Signal  (B, 1, 2048)
        │
        ▼
┌──────────────────────┐
│   Preprocessing      │  Segmentation → Normalization
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│   EWT Decomposition  │  (B, 3, 2048)
└────────┬─────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Self-Attention 1D CNN               │
│  Conv Blocks → Attention → GAP       │  (B, 128)
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Hierarchical Transfer Learning      │
│  Source Head | Target Head           │
│  Domain Discriminator (GRL)          │
└────────┬────────────────────────────┘
         │
         ▼
  Fault Classification
  normal | inner_race | outer_race | ball
```

---

## 4. Datasets

### A. CWRU Bearing Dataset (Primary Benchmark)
- **URL**: https://engineering.case.edu/bearingdatacenter/download-data-file
- **Format**: MATLAB `.mat` files
- **Sampling rate**: 12 kHz (or 48 kHz)
- **Fault classes**: `normal`, `inner_race`, `outer_race`, `ball`
- **Download**: Manual download required. Place files in `data/raw/CWRU/`

### B. Paderborn University KAt Dataset (Transfer Learning)
- **URL**: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
- **Format**: MATLAB `.mat` files per operating condition
- **Sampling rate**: 64 kHz
- **Required structure**:
  ```
  data/raw/Paderborn/
      K001/N09_M07_F10/*.mat
      KA01/N09_M07_F10/*.mat
      ...
  ```

### C. XJTU-SY (Optional, Degradation)
- **URL**: https://biaowang.tech/xjtu-sy-bearing-datasets/
- **Format**: CSV files (`acc_XXXXX.csv`)
- **Note**: Run-to-failure dataset. Not used in main classification experiments unless label mapping is explicitly justified.

---

## 5. Environment Setup

### Requirements
- Python 3.11+
- Windows / macOS / Linux

### Installation

```bash
# 1. Clone or download the project
cd bearing-fault-diagnosis

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. Running the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 7. Running Experiments

### Experiment A: Baseline 1D CNN

In the **Model Training** page:
- Select `Baseline 1D CNN (raw)`
- Enable `Use synthetic fixtures` for a smoke test
- Click `Start Training`

For real data, point the path to your CWRU download.

### Experiment B: EWT + CNN

- Select `EWT + CNN`
- Ensure `n_components = 3` in the Signal Processing page
- The model expects `(B, 3, 2048)` inputs

### Experiment C: Self-Attention CNN

- Select `Self-Attention CNN (raw)`
- Same data pipeline as baseline

### Experiment D: EWT + Self-Attention CNN (Proposed Hybrid)

- Select `EWT + Self-Attention CNN`
- The EWT decomposition is applied as preprocessing

### Experiment E & F: Transfer Learning

In the **Transfer Learning** page:
- Select source and target operating conditions
- Configure adaptation weight λ
- Click `Run Transfer Learning Experiment`

---

## 8. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run only smoke test
pytest tests/test_smoke_training.py -v

# Run EWT tests
pytest tests/test_ewt.py -v
```

---

## 9. Comparing Models

Navigate to the **Model Comparison** page after training multiple models. The page shows:
- Accuracy and macro-F1 bar charts
- Confusion matrices
- Training curves
- Exportable CSV/JSON results

---

## 10. Exporting Results

From the **Reports & History** page:
- Download CSV or JSON result tables
- Each row includes experiment name, best epoch, accuracy, F1, training time, and checkpoint path

---

## 11. Known Limitations

| Limitation | Status |
|------------|--------|
| Reference Paper 2 architecture | Provisional DANN baseline — update when paper details are verified |
| CWRU label inference from filename | Heuristic regex — verify against CWRU documentation |
| EWT built-in vs ewtpy | Built-in is an approximation; install `ewtpy` for closer fidelity |
| GPU training | Supported (CUDA auto-detected); not tested on MPS (Apple Silicon) |
| Paderborn label taxonomy | Supports K001–K006, KA01–KA09, KI01–KI09; KB codes labelled `mixed_damage` |
| XJTU-SY labels | No ground-truth fault-type labels per window; not used in main experiments |

---

## 12. Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
→ Run from the `bearing-fault-diagnosis/` directory, or add it to `PYTHONPATH`.

**`.mat` files fail to load**
→ Try `pip install mat73` for HDF5-backed MATLAB v7.3 files.

**CUDA out of memory**
→ Reduce `batch_size` in the training page.

**EWT produces poor reconstruction**
→ Install `ewtpy` for a more faithful EWT implementation (`pip install ewtpy`).

**Streamlit resets training on every interaction**
→ Use the explicit `Start Training` button; state is managed via `st.session_state`.

---

## 13. Project Structure

```
bearing-fault-diagnosis/
├── app.py                     # Streamlit dashboard (8 pages)
├── requirements.txt
├── README.md
├── configs/
│   └── default.yaml           # Experiment configuration
├── data/
│   ├── raw/                   # Downloaded datasets (not committed)
│   ├── processed/             # Preprocessed windows (not committed)
│   └── manifests/             # Split manifests (committed)
├── src/
│   ├── data/                  # Dataset adapters and preprocessing
│   ├── signal_processing/     # EWT module
│   ├── models/                # CNN, Attention CNN, Transfer, Hybrid
│   ├── training/              # Trainer, losses, checkpointing
│   ├── evaluation/            # Metrics, confusion matrix, comparisons
│   ├── visualization/         # Plotly signal, training, evaluation plots
│   └── utils/                 # Config, reproducibility, logging
├── tests/                     # pytest test suite
├── artifacts/
│   ├── checkpoints/           # Model checkpoints
│   ├── results/               # Experiment result JSON files
│   └── figures/               # Exported plots
└── docs/
    ├── methodology.md
    ├── dataset_guide.md
    ├── weekly_roadmap.md
    └── experiment_protocol.md
```
