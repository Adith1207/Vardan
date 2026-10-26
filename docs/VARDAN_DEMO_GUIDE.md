# VARDAN Research Demo Guide

**Document**: `docs/VARDAN_DEMO_GUIDE.md`
**Application Entry Point**: `demo/app.py`
**Framework**: Streamlit + Plotly
**Status**: Verified & Interactive

---

## 1. Overview

The **VARDAN Interactive Research Demo** is a presentation-ready application designed for thesis evaluations, faculty reviews, and conference presentations. It allows users to interactively inspect real RF signals from the public DroneRF benchmark dataset, run live inference with the trained **VARDHAN-v3** model on CPU or GPU, inspect intermediate dual-domain features, and explore the generalization findings of the research paper.

---

## 2. Model & Checkpoint Specifications

- **Model Architecture**: `VARDHAN-v3` (Multi-Scale Dual-Domain RF-Net)
- **Trainable Parameters**: Exactly **119,806** (0.457 MB model size)
- **Default Checkpoint Path**:
  `models/checkpoints/vardhan_v3_per_segment_15ep/checkpoint_fold_1.pt`
- **Alternative Checkpoint Path**:
  `models/checkpoints/vardhan_v3_per_segment_15ep/final_model_fold_1.pt`
- **Training Origin**: Fold 1 of 10-Fold Stratified Cross-Validation on DroneRF (Standardized benchmark).
- **Stored Metrics in Checkpoint**:
  - Test Accuracy: **71.54%**
  - Test Macro-F1: **0.7027**
  - Test Balanced Accuracy: **68.38%**
  - Best Epoch: **5**

---

## 3. Verified Input Representation & Preprocessing Pipeline

### 3.1 Input Tensor at Model Boundary
- **Tensor Shape**: `(B, 1, 2048)` (Batch size = 1 for demo inference)
- **Data Type**: `torch.float32`
- **Physical Meaning**: 2,048 contiguous raw time-domain RF voltage samples acquired at a sampling rate of **40 MS/s** ($f_s = 40\text{ MHz}$).
- **Observation Window**: $2048 / 40\text{ MHz} = \mathbf{51.2\text{ microseconds}}$.

### 3.2 Exact Preprocessing Sequence
1. **Sample Slicing**: Slices 2,048 samples of the L-band recording at the user-selected segment offset ($\text{offset} \times 100,000$).
2. **Per-Segment Normalization**: Standardizes each 2048-sample waveform independently to zero-mean and unit-variance:
   $$\tilde{x} = \frac{x - \mu}{\sigma + 10^{-8}}$$
   *(No global dataset statistics are needed, so there is no leakage.)*
3. **Internal Spectral Extraction (`extract_spectral_representation`)**:
   Inside `VardhanV3.forward()`, the model detrends the input and applies `torch.fft.rfft(..., n=2048)` to generate a 3-channel spectral tensor `(B, 3, 1024)`:
   - Channel 0: Log Power Spectrum: $\log(1 + 1000 \cdot |X|^2 / 2048)$
   - Channel 1: Normalized Real Fourier Component: $\text{Re}(X) / \sqrt{2048}$
   - Channel 2: Normalized Imaginary Fourier Component: $\text{Im}(X) / \sqrt{2048}$
4. **Dual-Domain Feature Fusion**:
   - 4 Multi-Scale Temporal Blocks (TCN) $\to$ 80-d attention-pooled features.
   - 4 Multi-Scale Spectral Blocks (FFT) $\to$ 80-d attention-pooled features.
   - Bi-directional cross-domain gating $\to$ 320-d fused representation.
   - Classification head $\to$ 4 logits.

---

## 4. Verified Class Mapping

The demo follows the faithful FGCS training pipeline label index mapping:

| Class Index | Class Name | Description |
| :---: | :--- | :--- |
| **0** | `Background RF activities` | Environmental RF noise and background wireless activity without drone transmission |
| **1** | `Bebop drone` | Parrot Bebop quadcopter telemetry / control link in 2.4 GHz band |
| **2** | `AR drone` | Parrot AR.Drone 2.0 Wi-Fi communication and video transmission link |
| **3** | `Phantom drone` | DJI Phantom 3 remote controller / Lightbridge RF transmission |

> [!CAUTION]
> Do not use `src/constants.py` for decoding predictions of this checkpoint. `src/constants.py` uses index 1 for AR Drone and index 2 for Bebop Drone, which would swap predictions. The demo binds to the verified training mapping in `demo/config.py`.

---

## 5. Dataset Directory Structure

The raw DroneRF dataset is not stored in Git. The application discovers paired DroneRF recordings from either uncompressed CSV files or packed `.rar` archives under `data/raw/DroneRF/`:

```text
data/raw/
└── DroneRF/
    ├── DroneRF.zip
    └── DroneRF/
        ├── Background RF activites/
        │   ├── RF Data_00000_H1.rar
        │   └── ...
        ├── Bepop drone/
        │   ├── RF Data_10000_H.rar
        │   └── ...
        ├── AR drone/
        │   ├── RF Data_10100_H.rar
        │   └── ...
        └── Phantom drone/
            ├── RF Data_11000_H.rar
            └── ...
```

Extracted CSV files (for example `data/raw/DroneRF/unzipped_data/.../10100L_0.csv`) are found as well. Reading `.rar` archives relies on the `tar` that ships with Windows 10/11 (libarchive / bsdtar).

---

## 6. How to Launch and Use the Demo

### 6.1 Prerequisites & Installation
Install the project requirements, which include `streamlit`, `plotly` and `numba`:
```bash
pip install -r requirements.txt
```
`numba` matters for the demo: without it, reading one 10M-sample recording takes over a minute instead of well under a second.

### 6.2 Launch Command
Run the Streamlit application from the repository root:
```bash
streamlit run demo/app.py
```
*(The browser will automatically open to `http://localhost:8501`.)* Running from the repository root also picks up `.streamlit/config.toml`, which applies the dark theme the demo is styled for.

### 6.3 Interactive Usage Walkthrough
1. **Sidebar — Checkpoint Selection**: Confirm that `checkpoint_fold_1.pt` is selected. Expand **Model Checkpoint Details** to review the parameter count and fold metrics.
2. **Sidebar — Sample Selection**:
   - Filter by drone class or select **All Classes**.
   - Choose any genuine DroneRF recording pair from the dropdown.
   - Adjust the **Segment Offset Window** (0 to 99) to evaluate different 51.2 µs temporal slices of the recording.
3. **Tab 1: Live Signal & Prediction**:
   - Inspect the KPI cards displaying Ground Truth, Predicted Class, Confidence (%), and measured inference latency.
   - Inspect the **Normalized Time-Domain Voltage** plot (Plotly interactive zoom/hover).
   - Inspect the **Real-Valued Power Spectral Density** plot (0 to 20 MHz baseband).
   - Inspect the **Ranked Posterior Probabilities** horizontal bar chart.
   - Expand the **Dual-Domain Internal Representation** to view the 3 spectral channels.
4. **Tab 2: Inference Pipeline**: Trace the exact transformation steps and tensor shapes.
5. **Tab 3: Model Architecture**: Review the parameter breakdown for each module (119,806 params total).
6. **Tab 4: Scientific Provenance**: Review the generalization comparison (71.57% segment-level 10-fold mean vs. 24.79% strict recording-level) and view the interactive 10-fold confusion matrix.

---

## 7. How to Run Automated Demo Tests

To verify that the demo modules, checkpoint loading, normalization, and sample discovery function correctly:
```bash
pytest tests/test_demo.py -v
```
The suite has 8 tests. Two of them read real DroneRF recordings and are skipped when `data/raw/DroneRF/` is empty; the other 6 run on any fresh clone.

To run the complete repository test suite:
```bash
pytest tests/
```

---

## 8. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **`No valid checkpoint found`** | Checkpoint not present in the expected folder. | Ensure `checkpoint_fold_1.pt` is present in `models/checkpoints/vardhan_v3_per_segment_15ep/`. |
| **`FileNotFoundError: No DroneRF dataset archives or CSVs found`** | Raw dataset missing from `data/raw/DroneRF/`. | Place the DroneRF `.rar` archives in `data/raw/DroneRF/DroneRF/` or extracted CSVs under `data/raw/DroneRF/`. |
| **Slow initial startup (10–15 s)** | Scanning RAR archives on first launch. | The demo caches the discovery manifest. Moving the segment slider within the same recording reuses the cached signal. |
| **Each new recording takes a minute or more to load** | `numba` is not installed. | `pip install numba` (included in `requirements.txt`). |
| **Title text is invisible / charts look washed out** | Streamlit started in the light theme. | Launch from the repository root so `.streamlit/config.toml` is used, or switch to the dark theme in Streamlit's settings menu. |
| **`CUDA out of memory`** | GPU memory exhausted by background jobs. | The model is small and runs well on CPU. Construct `VardanInferenceEngine(device="cpu")` to force CPU. |

---

## 9. Scientific Limitations & What the Demo Does Not Claim

1. **Not a Multi-Modal System**: The demo operates exclusively on RF signal data. It does not use acoustic, vision, or external sensor fusion.
2. **Not a Flashed Microcontroller**: The reported latency and parameter footprint (119.8k params, 0.46 MB) are software profiling results on the machine running the demo, not physical microcontroller measurements. Latency ranged from ~7 ms to ~70 ms per window across the machines we tested.
3. **Generalization Scope**: The interactive checkpoint is trained on Fold 1 of the standardized segment-level split (71.54% accuracy). As reported in the paper, strict evaluation on completely unseen recordings drops to 24.79%, consistent with recording-specific domain shift.
