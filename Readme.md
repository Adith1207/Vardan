<div align="center">

# Vardan: Lightweight RF-Based Drone Detection and Identification

![Status](https://img.shields.io/badge/Status-Research%20In%20Progress-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Final Year B.Tech Research Project**
Department of Computer Science & Engineering, Amrita Vishwa Vidyapeetham

</div>

---

## Overview

Vardan studies passive **radio-frequency (RF)** detection and identification of commercial drones on the public **DroneRF** dataset. The project:

1. faithfully reproduces published RF deep-learning baselines,
2. evaluates them under two protocols — the standard segment-level 10-fold cross-validation and a strict recording-level split with zero session overlap, and
3. proposes **VARDHAN-v3**, a lightweight (119,806-parameter) dual-domain network that jointly processes the raw time-domain waveform and a three-channel Fourier representation of each 2048-sample RF segment.

The scope of this repository is **RF only**. Acoustic, vision, and multi-sensor fusion work that was originally planned has been removed.

---

## Dataset

| Property | Value |
|---|---|
| Source | DroneRF (Al-Sa'd et al., 2019; Allahham et al., 2019), Mendeley Data |
| Receivers | 2 × NI USRP-2943R, lower band 2.40–2.44 GHz (L) and upper band 2.44–2.48 GHz (H) |
| Sampling rate | 100 MS/s |
| Files | 454 CSV files (227 synchronised L/H pairs), 10,000,000 samples each |
| Classes | Background RF, Parrot Bebop, Parrot AR Drone 2.0, DJI Phantom 3 |
| Segments | 22,700 (100 per pair), 2048 samples = 20.48 µs each |

The dataset is **not** included. Download it from the official source and extract it into `data/raw/DroneRF/` as described in [data/README.md](data/README.md).

---

## Models

| Model | Input | Parameters | Source |
|---|---|---:|---|
| FGCS2019DNN (code mode) | 2048-bin stitched L+H power spectrum | 295,812 | `src/models/fgcs_faithful_dnn.py` |
| MC1DCNN / Baseline1DCNN | 8 × 256 sub-band spectrum | 275,940 | `src/models/baselines.py` |
| DSCNN | (1, 2048) waveform | 68,228 | `src/models/baselines.py` |
| Compressive-Sensing CNN | (1, 1024) compressed measurements | 1,059,908 | `src/models/baselines.py` |
| MobileNetV3Small-style | (1, 65, 61) STFT log-spectrogram | 11,588 | `src/models/baselines.py` |
| VARDHAN-v1 | (1, 2048) waveform | 6,692 | `src/models/vardhan.py` |
| VARDHAN-v2A | (1, 2048) waveform, tri-branch | 69,559 | `src/models/vardhan_v2a.py` |
| **VARDHAN-v3** | (1, 2048) waveform + internal (3, 1024) spectrum | **119,806** | `src/models/vardhan_v3.py` |

**VARDHAN-v3** combines:
- a multi-scale temporal backbone (parallel depthwise kernels 3/7/15/31, dilations 1–8, squeeze-and-excitation, residuals),
- a multi-scale spectral backbone on `[log power, Re, Im]` of the 2048-point rFFT (DC bin removed),
- learnable attention pooling on both branches, and
- bi-directional cross-domain gating between the temporal and spectral embeddings of the same RF segment.

---

## Evaluation Protocols

| Protocol | Description |
|---|---|
| **Segment-level 10-fold CV** | `StratifiedKFold(n_splits=10, shuffle=True, random_state=1)` over all 22,700 segments. Matches the literature, but segments from the same recording appear in both train and test folds. |
| **Strict recording-level split** | 23 recording sessions assigned exclusively to train (15 sessions, 308 files), validation (4, 73) or test (4, 73). Zero recording and file overlap. Split files: `data/splits/{train,val,test}.csv`. |

Segment-level accuracy measures within-session interpolation; it does **not** measure generalisation to unseen recordings.

---

## Results

### Segment-level 10-fold cross-validation (22,700 segments)

| Model | Accuracy | Macro-F1 | Result files |
|---|---:|---:|---|
| FGCS2019DNN | 84.52% ± 0.73% | 0.7899 ± 0.0111 | `results/fgcs_faithful_reproduction/` |
| MC1DCNN | 65.01% ± 5.35% | 0.6207 ± 0.0533 | `results/baseline1dcnn_faithful_reproduction/` |
| VARDHAN-v3 | 71.57% ± 1.25% | 0.6953 ± 0.0168 | `results/vardhan_v3_per_segment_10fold/` |

VARDHAN-v3 balanced accuracy: 67.67% ± 2.03%.

### Strict recording-level split (VARDHAN-v3, 50 segments/file, 3,650 test segments)

| Metric | Value |
|---|---:|
| Accuracy | 24.79% |
| Macro-F1 | 0.2005 |
| Balanced accuracy | 22.02% |

Source: `results/vardhan_v3_strict_per_segment/`. Earlier strict runs of the baselines (5 segments/file) are reported in `reports/VARDHAN_IEEE_Report.md`; they collapse to near the 25% chance level.

**Key finding:** every model loses 36–56 percentage points when evaluated on unseen recording sessions. In the strict split, DJI Phantom training files come only from the H band and test files only from the L band, which contributes to the near-zero Phantom recall.

---

## Limitations

- Single-site dataset; cross-location and cross-hardware transfer are untested.
- Closed-set, four-class evaluation; unknown-drone rejection is untested.
- The full 10-fold benchmark was run only for FGCS2019DNN, MC1DCNN and VARDHAN-v3.
- No physical edge-device deployment has been performed. Parameter counts, MACs/FLOPs and latencies are software measurements on a development machine only; power, quantisation and embedded latency are unmeasured.

See `reports/VARDHAN_EVIDENCE_GAPS.md` for the complete list.

---

## Repository Structure

```text
Vardan/
├── configs/          # dataset, preprocessing, training, evaluation, logging YAML
├── data/             # raw/ (not tracked), interim/, processed/, metadata/, splits/
├── demo/             # interactive Streamlit demo of the trained VARDHAN-v3 checkpoint
├── docs/             # protocol and reproduction notes, demo guide and audit
├── notebooks/        # 01–05 RF analysis notebooks, 09 planned robustness experiments
├── reports/          # IEEE report, audits, evidence gaps, bibliography
├── results/          # metrics, fold tables, confusion matrices, run configs
├── scripts/          # training, verification and analysis entry points
├── src/
│   ├── data/          # lazy datasets, faithful loader, split generation
│   ├── preprocessing/ # raw loading, FFT, STFT, channelisation, normalisation, compressive sensing
│   ├── models/        # baselines, VARDHAN-v1/v2A/v3, trainer, model factory
│   ├── evaluation/    # metrics and confusion matrices
│   ├── benchmark/     # software latency and parameter profiling
│   ├── visualization/ # plotting utilities
│   └── utils/         # paths and helpers
├── tests/            # pytest suite
└── website/          # static project website
```

---

## Installation

```bash
git clone https://github.com/Adith1207/Vardan.git
cd Vardan

# Conda
conda env create -f environment.yml
conda activate vardan

# or pip
pip install -r requirements.txt
```

---

## Interactive Research Demonstration

To launch the interactive Streamlit + Plotly demonstration for the trained **VARDHAN-v3** RF drone classification model:

```bash
# Launch interactive research dashboard (from the repository root)
streamlit run demo/app.py
```

The demo classifies real DroneRF recordings, so the raw dataset must be placed under `data/raw/DroneRF/` first.

For complete documentation on model architecture, real sample selection, dual-domain feature visualization, and scientific provenance, see the [Demo Guide](docs/VARDAN_DEMO_GUIDE.md) and [Demo Audit Report](docs/VARDAN_DEMO_AUDIT.md).

---

## Usage

```bash
# Segment-level 10-fold benchmark
python scripts/train_fgcs_faithful_reproduction.py --raw_data_dir data/raw/DroneRF
python scripts/train_baseline1dcnn_faithful_reproduction.py --raw_data_dir data/raw/DroneRF
python scripts/train_vardhan_v3_standardized.py --normalization per_segment --epochs 15 --raw_data_dir data/raw/DroneRF

# Strict recording-level benchmark
python scripts/train_vardhan_v3.py --normalization per_segment --epochs 100 --samples_per_file 50 --splits_dir data/splits --raw_data_dir data/raw/DroneRF

# Split and band-shift diagnostics
python scripts/analyze_data_distribution.py --splits_dir data/splits --raw_data_dir data/raw/DroneRF

# Any training script can run on synthetic data without the dataset
python scripts/train_vardhan_v3.py --mock --epochs 1

# Tests
python -m pytest tests/
```

### Software profiling

```python
import sys; sys.path.insert(0, "src")
from models.vardhan_v3 import VardhanV3
from benchmark.bench import profile_model_footprint, benchmark_inference_latency

model = VardhanV3()
print(profile_model_footprint(model))                       # 119,806 parameters, ~0.46 MB FP32
print(benchmark_inference_latency(model, (1, 2048), device="cpu"))
```

Latency figures from this utility describe the machine they are run on; they are not embedded-hardware measurements.

---

## Research Team

Team 53, Department of Computer Science & Engineering, Amrita Vishwa Vidyapeetham

- Adith Narayan G
- S J Yuvan Dhurkesh
- Subash Santhanam K
- Sisthick S

Project guide: Mr. Sumesh A K (Assistant Professor)

---

## Citation

```bibtex
@misc{vardan2026rf,
  title  = {Vardan: Lightweight RF-Based Drone Detection and Identification},
  author = {Adith Narayan, G. and Yuvan Dhurkesh, S. J. and Subash Santhanam, K. and Sisthick, S. and Sumesh, A. K.},
  note   = {Final Year B.Tech Project, Amrita Vishwa Vidyapeetham},
  year   = {2026},
  url    = {https://github.com/Adith1207/Vardan}
}
```

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
