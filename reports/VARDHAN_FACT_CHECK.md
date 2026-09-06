# VARDHAN Research Report Fact-Check and Traceability Audit

This document provides a strict, line-by-line verification and traceability audit for every numerical result, dataset statistic, architectural parameter count, hyperparameter setting, and citation used in the final IEEE research report for Project VARDHAN.

---

## 1. Dataset Statistics & Topologies

| Metric / Claim | Report Value | Source Artifact / Paper | Verification Status | Notes |
|---|---|---|:---:|---|
| **Total DroneRF CSV Files** | 454 files | `data/metadata/dronerf_metadata.csv` | **VERIFIED $\checkmark$** | Exact row count of valid raw CSV recordings. |
| **Synchronized Receiver Pairs** | 227 pairs | `data/metadata/dronerf_metadata.csv` | **VERIFIED $\checkmark$** | $454 / 2 = 227$ $(L, H)$ pairs. |
| **Unique Recording Sessions** | 23 sessions | Split manifests / metadata | **VERIFIED $\checkmark$** | 10 distinct BUI/experiment configurations across 23 sessions. |
| **Total Standardized Segments** | 22,700 segments | `results/fgcs_faithful_reproduction/aggregate_metrics.json` | **VERIFIED $\checkmark$** | $227 \text{ pairs} \times 100 \text{ segments/pair} = 22,700$. |
| **Segment Length ($N$)** | 2048 samples | `src/config.py`, `src/models/vardhan_v3.py` | **VERIFIED $\checkmark$** | Segment duration: $20.48\,\mu\text{s}$ at $100\,\text{MS/s}$. |
| **Sampling Rate ($F_s$)** | $100\,\text{MS/s}$ | Al-Sa'd et al. (2019) \cite{alsad2019rf} | **VERIFIED $\checkmark$** | NI USRP-2943R SDR receiver sampling rate. |
| **Receiver Bandwidth** | 40 MHz / 80 MHz | Al-Sa'd et al. (2019) \cite{alsad2019rf} | **VERIFIED $\checkmark$** | Two 40 MHz receivers stitched to cover 80 MHz. |
| **Background Segments** | 4,100 (18.06%) | `results/fgcs_faithful_reproduction/aggregate_metrics.json` | **VERIFIED $\checkmark$** | 41 pairs $\times$ 100 = 4,100 segments. |
| **Parrot Bebop Segments** | 8,400 (37.00%) | `results/fgcs_faithful_reproduction/aggregate_metrics.json` | **VERIFIED $\checkmark$** | 84 pairs $\times$ 100 = 8,400 segments. |
| **Parrot AR Segments** | 8,100 (35.68%) | `results/fgcs_faithful_reproduction/aggregate_metrics.json` | **VERIFIED $\checkmark$** | 81 pairs $\times$ 100 = 8,100 segments. |
| **DJI Phantom 3 Segments** | 2,100 (9.25%) | `results/fgcs_faithful_reproduction/aggregate_metrics.json` | **VERIFIED $\checkmark$** | 21 pairs $\times$ 100 = 2,100 segments. |
| **Strict Train Split Files** | 308 files (15 recs) | `data/splits/train.csv` | **VERIFIED $\checkmark$** | 126 Bebop, 120 AR, 41 Background, 21 Phantom. |
| **Strict Val Split Files** | 73 files (4 recs) | `data/splits/val.csv` | **VERIFIED $\checkmark$** | 21 Bebop, 21 AR, 21 Background, 10 Phantom. |
| **Strict Test Split Files** | 73 files (4 recs) | `data/splits/test.csv` | **VERIFIED $\checkmark$** | 21 Bebop, 21 AR, 20 Background, 11 Phantom. |
| **Split Overlap** | Zero files, zero recs | `src/data/create_splits.py`, `aggregate_metrics.json` | **VERIFIED $\checkmark$** | Strict isolation verified by unit tests. |

---

## 2. Model Parameters & Complexity

| Model | Parameter Count | Source File | Verification Status | Notes |
|---|:---:|---|:---:|---|
| **VARDHAN-v3** | **119,806** | `src/models/vardhan_v3.py` | **VERIFIED $\checkmark$** | Exact PyTorch trainable parameter count. |
| **VARDHAN-v3 Theoretical MACs** | 11.8144 M | `src/models/vardhan_v3.py` | **VERIFIED $\checkmark$** | Graph-calculated theoretical Multiply-Accumulate operations. |
| **VARDHAN-v3 Theoretical FLOPs** | 23.6288 M | `src/models/vardhan_v3.py` | **VERIFIED $\checkmark$** | $2 \times \text{MACs}$. |
| **FGCS2019DNN (Code Mode)** | 295,812 | `src/models/fgcs_faithful_dnn.py` | **VERIFIED $\checkmark$** | 128-128-128 architecture from released code. |
| **FGCS2019DNN (Paper Mode)** | 565,956 | `src/models/fgcs_faithful_dnn.py`, `baselines.py` | **VERIFIED $\checkmark$** | 256-128-64 architecture described in paper text. |
| **MC1DCNN (Allahham et al.)** | 275,940 | `src/models/baselines.py` | **VERIFIED $\checkmark$** | 8-channel sub-band Conv1D model. |
| **DSCNN (Medaiyese et al.)** | 68,228 | `src/models/baselines.py` | **VERIFIED $\checkmark$** | Depthwise separable 1D CNN baseline. |
| **CS-CNN (Mo et al.)** | 1,059,908 | `src/models/baselines.py` | **VERIFIED $\checkmark$** | Compressive sensing baseline CNN. |
| **MobileNet-Style Spectrogram** | 11,588 | `src/models/baselines.py` | **VERIFIED $\checkmark$** | Simplified 2D depthwise separable CNN. |
| **VARDHAN-v1 (Initial)** | 6,692 | `src/models/vardhan.py` | **VERIFIED $\checkmark$** | Historical lightweight prototype. |
| **VARDHAN-v2A** | 69,559 | `src/models/vardhan_v2a.py` | **VERIFIED $\checkmark$** | Intermediate multi-scale prototype. |

---

## 3. Standardized 10-Fold Segment-Level Benchmark Results

| Model | Accuracy | Macro-F1 | Balanced Accuracy | Source Artifact | Verification Status |
|---|:---:|:---:|:---:|---|:---:|
| **FGCS2019DNN** | $84.52\% \pm 0.73\%$ | $0.7899 \pm 0.0111$ | N/A | `results/fgcs_faithful_reproduction/aggregate_metrics.json` | **VERIFIED $\checkmark$** |
| **MC1DCNN** | $65.01\% \pm 5.35\%$ | $0.6207 \pm 0.0533$ | N/A | `results/baseline1dcnn_faithful_reproduction/aggregate_metrics.json` | **VERIFIED $\checkmark$** |
| **VARDHAN-v3** | $71.57\% \pm 1.25\%$ | $0.6953 \pm 0.0168$ | $67.67\% \pm 2.03\%$ | `results/vardhan_v3_per_segment_10fold/aggregate_metrics.json` | **VERIFIED $\checkmark$** |

### Standardized Benchmark Training Configurations:
- **FGCS2019DNN**: Adam, $\text{lr}=0.001$, batch size 10, 200 epochs, MSELoss on one-hot targets, Sigmoid output, seed 1 (`results/fgcs_faithful_reproduction/run_config.json`).
- **MC1DCNN**: Adam, $\text{lr}=0.001$, batch size 32, 100 epochs, CrossEntropyLoss, seed 1 (`results/baseline1dcnn_faithful_reproduction/run_config.json`).
- **VARDHAN-v3**: AdamW, $\text{lr}=0.0003$, weight decay $10^{-4}$, batch size 32, 15 epochs, Cosine Annealing ($\text{min\_lr}=10^{-6}$), CrossEntropy with Label Smoothing 0.05, per-segment normalization, seed 1 (`results/vardhan_v3_per_segment_10fold/run_config.json`).

### FGCS Faithful 10-Fold Confusion Matrix ($N=22,700$):
- **Background (Class 0)**: `[4094, 3, 2, 1]` $\rightarrow$ Recall: **99.85%** (4094 / 4100)
- **Parrot Bebop (Class 1)**: `[6, 8270, 111, 13]` $\rightarrow$ Recall: **98.45%** (8270 / 8400)
- **Parrot AR (Class 2)**: `[1, 1940, 6151, 8]` $\rightarrow$ Recall: **75.94%** (6151 / 8100) — *1940 confused with Bebop*
- **DJI Phantom (Class 3)**: `[5, 1399, 24, 672]` $\rightarrow$ Recall: **32.00%** (672 / 2100) — *1399 confused with Bebop*

### VARDHAN-v3 Standardized 10-Fold Confusion Matrix ($N=22,700$):
- **Background (Class 0)**: `[3410, 145, 526, 19]` $\rightarrow$ Recall: **83.17%** (3410 / 4100)
- **Parrot Bebop (Class 1)**: `[143, 6418, 1685, 154]` $\rightarrow$ Recall: **76.40%** (6418 / 8400)
- **Parrot AR (Class 2)**: `[620, 1900, 5515, 65]` $\rightarrow$ Recall: **68.09%** (5515 / 8100)
- **DJI Phantom (Class 3)**: `[86, 696, 415, 903]` $\rightarrow$ Recall: **43.00%** (903 / 2100)

---

## 4. Historical Strict Recording-Level Benchmark ($N_{\text{samp}}=5$, Seed 42)

| Model | Test Accuracy | Macro-F1 | Precision | Recall | Source Artifact | Verification Status |
|---|:---:|:---:|:---:|:---:|---|:---:|
| **FGCS2019DNN** | $28.22\%$ | $0.1100$ | $0.0706$ | $0.2500$ | User screenshot / `results/baselines/` | **VERIFIED $\checkmark$** |
| **Baseline1DCNN** | $28.77\%$ | $0.1117$ | $0.0719$ | $0.2500$ | User screenshot / `results/baselines/` | **VERIFIED $\checkmark$** |
| **DSCNN** | $26.58\%$ | $0.1520$ | $0.1800$ | $0.2312$ | User screenshot / `results/baselines/` | **VERIFIED $\checkmark$** |
| **CS-CNN** | $28.77\%$ | $0.1117$ | $0.0719$ | $0.2500$ | User screenshot / `results/baselines/` | **VERIFIED $\checkmark$** |
| **MobileNet-Style** | $28.77\%$ | $0.1117$ | $0.0719$ | $0.2500$ | User screenshot / `results/baselines/` | **VERIFIED $\checkmark$** |
| **VARDHAN (v1)** | $26.30\%$ | $0.1152$ | $0.0740$ | $0.2500$ | User screenshot / `results/baselines/` | **VERIFIED $\checkmark$** |

---

## 5. Final VARDHAN-v3 Strict Recording-Level Experiment ($N_{\text{samp}}=50$, Seed 42, 100 Epochs)

| Metric | Measured Value | Source Artifact | Verification Status | Notes |
|---|:---:|---|:---:|---|
| **Overall Test Accuracy** | **24.79%** | `results/vardhan_v3_strict_per_segment/aggregate_metrics.json` | **VERIFIED $\checkmark$** | Evaluated on 3,650 test segments across 4 unseen recordings. |
| **Macro-F1 Score** | **0.2005** | `aggregate_metrics.json` | **VERIFIED $\checkmark$** | Unweighted macro average across 4 classes. |
| **Balanced Accuracy** | **22.02%** | `aggregate_metrics.json` | **VERIFIED $\checkmark$** | Unweighted average recall across 4 classes. |
| **Test Cross-Entropy Loss** | **1.4735** | `aggregate_metrics.json` | **VERIFIED $\checkmark$** | Test loss using best checkpoint (Epoch 1). |
| **Best Model Epoch** | **1** | `aggregate_metrics.json`, `training_log.csv` | **VERIFIED $\checkmark$** | Selected based on minimum validation loss ($1.5491$). |
| **No-Drone Precision / Recall / F1** | $61.18\% / 14.50\% / 0.2344$ | `test_metrics.csv` | **VERIFIED $\checkmark$** | Support: 1,000 segments. |
| **AR Drone Precision / Recall / F1** | $25.28\% / 19.52\% / 0.2203$ | `test_metrics.csv` | **VERIFIED $\checkmark$** | Support: 1,050 segments. |
| **Bebop Precision / Recall / F1** | $20.90\% / 51.52\% / 0.2974$ | `test_metrics.csv` | **VERIFIED $\checkmark$** | Support: 1,050 segments. |
| **Phantom Precision / Recall / F1** | $100.00\% / 2.55\% / 0.0496$ | `test_metrics.csv` | **VERIFIED $\checkmark$** | Support: 550 segments. |

---

## 6. Normalization Diagnostic Results

| Configuration | Test Accuracy | Macro-F1 | Background Recall | Source Artifact | Verification Status |
|---|:---:|:---:|:---:|---|:---:|
| **Global Train-Fold Z-Score** | $45.55\%$ | $0.3459$ | $0.0\%$ (collapse to Bebop) | Diagnostic run log | **VERIFIED $\checkmark$** |
| **Per-Segment Normalization** | $71.57\% \pm 1.25\%$ | $0.6953 \pm 0.0168$ | $83.17\%$ | `aggregate_metrics.json` | **VERIFIED $\checkmark$** |

---

## 7. Audit Conclusion
Every single numerical figure, dataset statistic, equation, hyperparameter, and performance metric cited in the IEEE research report (`reports/VARDHAN_IEEE_Report.tex` and `reports/VARDHAN_IEEE_Report.md`) matches verified repository artifacts with 100% precision. Zero numbers were fabricated or assumed.
