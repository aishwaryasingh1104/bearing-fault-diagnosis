# Weekly Roadmap: 10-Week Research Project

**Project:** Intelligent Bearing Fault Diagnosis Using EWT, Self-Attention CNN, and Hierarchical Transfer Learning

---

## Week 1: Project Kickoff and Literature Review

**Deliverables**
- Complete project setup (virtual environment, dependencies, directory structure)
- Literature review notes on EWT, 1-D CNNs, attention mechanisms, transfer learning for fault diagnosis
- Draft research hypothesis and experimental plan
- Review reference papers and document verified architecture details

**Supervisor Talking Points**
- Explain the three-component methodology and why each component is chosen
- Present the experimental comparison plan (Experiments A–F)
- Identify any dataset access requirements early

---

## Week 2: Dataset Acquisition and Validation

**Deliverables**
- Download CWRU dataset and verify loading with the adapter (`src/data/cwru_dataset.py`)
- Request access to Paderborn dataset
- Run `pytest tests/test_splits.py` and `tests/test_preprocessing.py`
- Generate dataset summary tables and class-distribution plots
- Save a data manifest (`data/manifests/cwru_manifest.json`)

**Supervisor Talking Points**
- Show the class distribution across CWRU loads (0–3 hp)
- Demonstrate recording-level split with leakage prevention
- Present the data manifest as a reproducibility artifact

---

## Week 3: Signal Preprocessing and EWT

**Deliverables**
- Implement and validate signal windowing (L=2048, 50% overlap)
- Implement z-score normalization (fit on train only)
- Run `pytest tests/test_ewt.py` — all tests passing
- Generate EWT decomposition visualizations for representative signals
- Document reconstruction error and spectral boundaries

**Supervisor Talking Points**
- Demonstrate the EWT decomposition in the Signal Processing page
- Show the 3-component breakdown and what each captures
- Explain the difference between built-in EWT and the `ewtpy` reference library

---

## Week 4: Baseline CNN Training (Experiment A)

**Deliverables**
- Train Baseline 1D CNN on CWRU data (Experiment A)
- Record: accuracy, macro F1, confusion matrix, training time
- Save checkpoint and split manifest
- Run `pytest tests/test_smoke_training.py`

**Supervisor Talking Points**
- Present training and validation loss curves
- Show confusion matrix and per-class F1 for each fault type
- Identify any class imbalance issues and explain the class-weighted loss

---

## Week 5: EWT-Preprocessed CNN (Experiment B)

**Deliverables**
- Run Experiment B: EWT + Baseline CNN
- Compare against Experiment A using the Model Comparison page
- Document any performance change (positive or negative — be honest)
- Analyse which EWT components capture fault-relevant features

**Supervisor Talking Points**
- Present the A vs B comparison honestly
- Discuss why EWT may or may not improve over raw signal input
- Show spectral boundary visualizations

---

## Week 6: Self-Attention CNN (Experiments C & D)

**Deliverables**
- Train Self-Attention CNN on raw signals (Experiment C)
- Train EWT + Self-Attention CNN (Experiment D)
- Compare all four experiments (A–D) in the Model Comparison page
- Document attention weights for representative fault signals

**Supervisor Talking Points**
- Explain the self-attention mechanism and what it learns
- Present the 4-model comparison table
- Discuss computational cost vs accuracy trade-offs

---

## Week 7: Transfer Learning Setup (Experiment E)

**Deliverables**
- Paderborn dataset loaded and validated
- Source condition (e.g. N09_M07_F10) and target condition (e.g. N15_M07_F04) selected
- Experiment E: Transfer learning without EWT (frozen backbone + target head fine-tuning)
- Report source-domain and target-domain accuracy separately

**Supervisor Talking Points**
- Explain the domain shift between operating conditions
- Show source vs target accuracy drop without adaptation
- Present the DANN-style gradient reversal mechanism

---

## Week 8: Hybrid Model and Full Pipeline (Experiment F)

**Deliverables**
- Experiment F: EWT + Attention CNN + Hierarchical Transfer Learning
- Complete all 6 experiments with consistent evaluation protocol
- Generate the full comparison table (Experiments A–F)
- Ablation: measure contribution of each component

**Supervisor Talking Points**
- Present the full 6-model comparison
- Discuss which components contribute most to cross-condition performance
- Acknowledge where the hybrid did or did not outperform baselines

---

## Week 9: Analysis and Fault Diagnosis Demo

**Deliverables**
- Final model selected based on validation performance
- Fault Diagnosis page demo with a real CWRU recording
- Generate and export all plots (confusion matrices, training curves, radar chart)
- Write the experiment protocol document (`docs/experiment_protocol.md`)

**Supervisor Talking Points**
- Run a live fault diagnosis demonstration
- Show the prediction probability bars for each class
- Present the reproducibility summary (seed, manifest, checkpoint path)

---

## Week 10: Documentation and Presentation

**Deliverables**
- Complete README with all sections filled
- Methodology document (`docs/methodology.md`)
- Weekly progress report submitted
- Project slides prepared for final presentation
- All tests passing (`pytest tests/ -v`)
- Final checkpoint archived

**Supervisor Talking Points**
- Summarise the research question and whether the hypothesis was supported
- Present limitations honestly (provisional transfer architecture, data constraints)
- Outline future work (severity classification, online adaptation, real-time deployment)

---

## Presentation Outline (15 minutes)

1. **Introduction** (2 min): Problem statement, research question, why bearing health matters
2. **Methodology** (4 min): EWT, self-attention CNN, transfer learning — one slide each
3. **Datasets** (1 min): CWRU and Paderborn, operating conditions, class distribution
4. **Results** (5 min): A–F comparison table, confusion matrices, training curves
5. **Demo** (2 min): Live fault diagnosis in the Streamlit dashboard
6. **Conclusions** (1 min): Key findings, limitations, next steps
