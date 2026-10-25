# Vardan Research Notebooks

This directory contains Jupyter notebooks used for incremental, reproducible research.

## Notebook Philosophy

1.  **Single Scientific Focus**: Every notebook is dedicated to answering exactly one research question.
2.  **No Code Duplication**: Notebooks should contain analysis, plotting, and visualization. All reusable functions, data loaders, signal processing pipelines, and model architectures must live in the `src/` directory.
3.  **Reproducible & Config-Driven**: Do not hardcode parameters. Load configurations from the `configs/` directory and use the `src/utils/paths.py` utility for locating files.

## Notebook Checklist

*   [ ] **[01_DroneRF_Dataset_Exploration.ipynb](01_DroneRF_Dataset_Exploration.ipynb)**: *What does the DroneRF dataset contain?*
    *   Examines dataset structure, label distribution, file formats, and class balance.
*   [ ] **[02_DroneRF_Signal_Characterization.ipynb](02_DroneRF_Signal_Characterization.ipynb)**: *How should RF signals be represented?*
    *   Explores time-domain, FFT, STFT, stationarity, autocorrelation, and amplitude-distribution properties.
*   [ ] **[03_Baseline_Model_Reproduction.ipynb](03_Baseline_Model_Reproduction.ipynb)**: *How do published baseline models perform?*
    *   Replicates performance of 1D CNNs, DS-CNNs, and MobileNetV3 small.
*   [ ] **[04_Benchmarking.ipynb](04_Benchmarking.ipynb)**: *Which baseline is strongest?*
    *   Compares models in accuracy, size, and software-measured inference latency.
*   [ ] **[05_Vardhan_RF_Module.ipynb](05_Vardhan_RF_Module.ipynb)**: *Can the Vardhan RF module outperform them?*
    *   Tests the custom proposed RF neural network architecture.
*   [ ] **[09_Experiments.ipynb](09_Experiments.ipynb)**: *How robust is the RF classifier to channel noise and interference?*
    *   Planned RF robustness experiments (placeholder; no results yet).

Notebooks 06&ndash;08 (acoustic, vision, and multimodal fusion placeholders) were removed when the project scope was narrowed to RF-only research.
