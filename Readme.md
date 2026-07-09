<div align="center">

# 🛰️ Portable Indigenous Multi-Modal Counter-UAS System (Vardan)

### AI-Powered Lightweight Multi-Sensor Drone Detection Framework for Edge Devices

![Status](https://img.shields.io/badge/Status-Research%20In%20Progress-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![TinyML](https://img.shields.io/badge/TinyML-Edge%20AI-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

### 🎓 Final Year B.Tech Research Project

**Department of Computer Science & Engineering**  
**Amrita Vishwa Vidyapeetham**

*"Towards an Indigenous, Portable, and Energy-Efficient Counter-UAS Detection System using Multi-Modal Artificial Intelligence."*

</div>

---

## 📖 Project Overview

The rapid increase in the use of **Unmanned Aerial Vehicles (UAVs)** has introduced security threats through unauthorized drone operations. This project develops **Vardan**, a lightweight, portable, multi-modal counter-UAS framework. By fusing inputs from **Radio Frequency (RF)**, **Acoustic waveforms**, and **Computer Vision feeds**, Vardan increases detection robustness under noise while remaining computationally optimized for microcontroller and embedded-device deployment.

### System Architecture Diagram
```text
                    Portable Counter-UAS System
                  ┌───────────────────────────────┐
                  │      Incoming Drone           │
                  └──────────────┬────────────────┘
                                 │
         ┌───────────────────────┼────────────────────────┐
         │                       │                        │
         ▼                       ▼                        ▼
    RF Detection            Acoustic Detection      Vision Detection
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼
                    Adaptive Sensor Fusion Engine
                                 │
                                 ▼
                    Drone / No Drone Classification
                                 │
                                 ▼
                        Threat Level & Alert System
```

---

## 🎯 Research Goals

1. **Robust RF Signature Classification**: Detect drone signals and identify flight modes under varying Signal-to-Noise Ratios (SNRs).
2. **Acoustic Signature Detection**: Implement lightweight, audio-spectrogram classifiers to detect drone rotors in the acoustic near-field.
3. **Optimized Computer Vision**: Deploy highly optimized, low-latency object classifiers/detectors for visual identification.
4. **Adaptive Late Fusion**: Fuse probabilities dynamically using entropy-based confidence weighting to manage sensor degradation.
5. **Edge & TinyML Benchmarking**: Benchmark latency, parameter sizes, RAM, and Flash memory consumption to target ARM Cortex microcontrollers.

---

## 📂 Repository Structure

The project directory tree is organized for research scale and reproducibility:

```text
Vardan/
├── pyproject.toml              # Build system, tool configurations, and metadata
├── requirements.txt            # Python dependencies (PyTorch, Librosa, etc.)
├── environment.yml             # Conda environment definition
├── LICENSE                     # MIT License
├── configs/                    # Hyperparameters and options (No hardcoding)
│   ├── dataset.yaml            # Dataset details, sampling rates, splits
│   ├── preprocessing.yaml      # Signal windows, FFT size, wavelets
│   ├── training.yaml           # Learning rates, scheduler, epochs
│   ├── evaluation.yaml         # Decision thresholds, TinyML constraints
│   └── logging.yaml            # Rotating log configurations, ML tracking
├── data/                       # Data separation (Raw data is read-only)
│   ├── raw/DroneRF/            # Landing directory for official raw data
│   ├── interim/                # Temporary files during preprocessing
│   └── processed/              # Split/transformed datasets (FFT, spectrograms)
├── notebooks/                  # Single-responsibility research notebooks
│   ├── 01_DroneRF_Dataset_Exploration.ipynb
│   ├── 02_RF_Signal_Processing.ipynb
│   ├── 03_Baseline_Model_Reproduction.ipynb
│   ├── 04_Benchmarking.ipynb
│   ├── 05_Vardhan_RF_Module.ipynb
│   ├── 06_Acoustic_Module.ipynb
│   ├── 07_Vision_Module.ipynb
│   ├── 08_Multimodal_Fusion.ipynb
│   └── 09_Experiments.ipynb
├── src/                        # Modular python package
│   ├── data/                   # Data loader helpers
│   ├── preprocessing/          # Signal transforms (STFT, wavelets)
│   ├── features/               # Descriptive feature extraction
│   ├── visualization/          # Spectrogram and signal plotters
│   ├── models/                 # Baselines and VardhanRFNet architecture
│   ├── evaluation/             # Metrics calculation (Precision, Recall, F1)
│   ├── benchmark/              # Latency & memory profiling utilities
│   ├── fusion/                 # Adaptive late-fusion algorithms
│   └── utils/                  # Paths resolver and helpers (set_seed)
├── models/                     # Weights storage directory
│   ├── baselines/              # Trained weights for baseline models
│   ├── vardhan/                # Trained weights for Vardan models
│   └── checkpoints/            # Epoch-wise train checkpoints
├── experiments/                # Running tracking logs and results
│   ├── configs/                # Saved configs of individual runs
│   ├── logs/                   # System outputs / training logs
│   ├── runs/                   # Local logs / TensorBoard outputs
│   └── reports/                # Experiment summaries
├── results/                    # Research output exports (LaTeX/JSON)
│   ├── tables/                 # Text/LaTeX formatted results tables
│   ├── plots/                  # Static evaluation graphs
│   ├── metrics/                # Raw accuracy scores in JSON
│   └── confusion_matrices/     # Confusion matrix arrays in CSV/JSON
└── reports/                    # Literature notes and paper draft assets
```

---

## ⚙️ Installation

We support package management through `conda` or standard `pip`. 

### Method 1: Conda (Recommended for cross-platform stability)
```bash
conda env create -f environment.yml
conda activate vardan
```

### Method 2: Pip
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📡 Dataset Configuration

> [!IMPORTANT]
> **The DroneRF dataset is NOT included in this repository due to licensing restrictions and file size limits.**
> Users should download the raw data separately from the official source and extract the files into the `data/raw/DroneRF` folder.

Follow the guidelines inside [data/README.md](file:///c:/Users/subsa/Desktop/DRONE/Vardan/data/README.md) to set up raw signals correctly. Path loading must utilize [src/utils/paths.py](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/utils/paths.py) to resolve locations dynamically.

---

## 🔄 Research Workflow

To replicate research results:
1. Parse raw signals and explore data configurations using [01_DroneRF_Dataset_Exploration.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/01_DroneRF_Dataset_Exploration.ipynb).
2. Experiment with wavelets and STFT in [02_RF_Signal_Processing.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/02_RF_Signal_Processing.ipynb).
3. Replicate baseline results using [03_Baseline_Model_Reproduction.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/03_Baseline_Model_Reproduction.ipynb).
4. Run edge latency benchmarking with [04_Benchmarking.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/04_Benchmarking.ipynb).
5. Compare performance of the proposed Vardan model in [05_Vardhan_RF_Module.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/05_Vardhan_RF_Module.ipynb).

Ensure reproducible seeds are set via `set_seed` in [src/utils/helpers.py](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/utils/helpers.py).

---

## 📅 Milestones & Progress

| Phase | Description | Status |
|--------|-------------|--------|
| Phase 1 | Repository Scaffolding & Path Resolution | 🟢 Completed |
| Phase 2 | RF Signal Preprocessing & Baselines | 🟡 In Progress |
| Phase 3 | Acoustic Signal Classifier | ⏳ Planned |
| Phase 4 | Vision Object Detector | ⏳ Planned |
| Phase 5 | Adaptive Multimodal Sensor Fusion | ⏳ Planned |
| Phase 6 | Edge TinyML Optimization & Deployment | ⏳ Planned |

---

## 👨‍💻 Research Team

* **Department of Computer Science & Engineering, Amrita Vishwa Vidyapeetham**
* **Team 53**:
  - Adith Narayan G
  - S J Yuvan Dhurkesh
  - Subash Santhanam K
  - Sisthick S
* **Project Guide**:
  - Mr. Sumesh A K (Assistant Professor)

---

## 📑 Citation Placeholder

When publishing or referencing this project, please cite:

```bibtex
@article{vardan2026multimodal,
  title={Towards an Indigenous, Portable and Energy-Efficient Counter-UAS Detection System using Adaptive Multi-Modal Artificial Intelligence},
  author={Adith Narayan, G. and Yuvan Dhurkesh, S. J. and Subash Santhanam, K. and Sisthick, S. and Sumesh, A. K.},
  journal={arXiv preprint (Submission in Progress)},
  year={2026}
}
```

---

## 📜 License

This repository is released under the **MIT License**. See [LICENSE](file:///c:/Users/subsa/Desktop/DRONE/Vardan/LICENSE) for details.
