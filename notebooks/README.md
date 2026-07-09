# Vardan Research Notebooks

This directory contains Jupyter notebooks used for incremental, reproducible research.

## Notebook Philosophy

1.  **Single Scientific Focus**: Every notebook is dedicated to answering exactly one research question.
2.  **No Code Duplication**: Notebooks should contain analysis, plotting, and visualization. All reusable functions, data loaders, signal processing pipelines, and model architectures must live in the `src/` directory.
3.  **Reproducible & Config-Driven**: Do not hardcode parameters. Load configurations from the `configs/` directory and use the `src/utils/paths.py` utility for locating files.

## Notebook Checklist

*   [ ] **[01_DroneRF_Dataset_Exploration.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/01_DroneRF_Dataset_Exploration.ipynb)**: *What does the DroneRF dataset contain?*
    *   Examines dataset structure, label distribution, file formats, and class balance.
*   [ ] **[02_RF_Signal_Processing.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/02_RF_Signal_Processing.ipynb)**: *How should RF signals be represented?*
    *   Explores FFT, STFT, Wavelet transforms, and normalization techniques.
*   [ ] **[03_Baseline_Model_Reproduction.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/03_Baseline_Model_Reproduction.ipynb)**: *How do published baseline models perform?*
    *   Replicates performance of 1D CNNs, DS-CNNs, and MobileNetV3 small.
*   [ ] **[04_Benchmarking.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/04_Benchmarking.ipynb)**: *Which baseline is strongest?*
    *   Compares models in accuracy, size, and inference latency on CPU/Edge.
*   [ ] **[05_Vardhan_RF_Module.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/05_Vardhan_RF_Module.ipynb)**: *Can the Vardhan RF module outperform them?*
    *   Tests the custom proposed RF neural network architecture.
*   [ ] **[06_Acoustic_Module.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/06_Acoustic_Module.ipynb)**: *How can acoustic signals supplement detection?*
    *   Evaluates audio spectrogram features for near-field detection.
*   [ ] **[07_Vision_Module.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/07_Vision_Module.ipynb)**: *How can vision-based detection be optimized for micro-UAVs?*
    *   Benchmarks lightweight object detection/classification on camera feeds.
*   [ ] **[08_Multimodal_Fusion.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/08_Multimodal_Fusion.ipynb)**: *How should modalities be fused?*
    *   Investigates early, late, and adaptive sensor fusion algorithms.
*   [ ] **[09_Experiments.ipynb](file:///c:/Users/subsa/Desktop/DRONE/Vardan/notebooks/09_Experiments.ipynb)**: *What are the system's limits under environmental noise?*
    *   Systematic evaluation of fusion model robustness under multi-sensor noise.
