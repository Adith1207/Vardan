# VARDAN Research Demo Audit & Verification Report

**Document**: `docs/VARDAN_DEMO_AUDIT.md`
**Date**: October 2026
**Status**: VERIFIED & REPRODUCIBLE
**Model Under Audit**: VARDHAN-v3 (Multi-Scale Dual-Domain RF-Net)
**Primary Checkpoint**: `models/checkpoints/vardhan_v3_per_segment_15ep/checkpoint_fold_1.pt`
**Alternative Checkpoint**: `models/checkpoints/vardhan_v3_per_segment_15ep/final_model_fold_1.pt`

---

## 1. Executive Summary & Verification Verdict

An executable audit of the VARDAN repository was performed to verify model loading, input representations, preprocessing, dataset pairing, normalization, and class mappings before building the interactive demo (`demo/app.py`).

| Audit Area | Implementation Path | Verified Truth | Status |
| :--- | :--- | :--- | :---: |
| **Model Class** | `src/models/vardhan_v3.py` (`class VardhanV3`) | Exactly 119,806 trainable parameters. Dual-domain architecture with a multi-scale TCN time stem and a deterministic rFFT spectral stem. | **PASS** |
| **Trained Checkpoint** | `models/checkpoints/vardhan_v3_per_segment_15ep/checkpoint_fold_1.pt` | All 147 state-dict tensors match with `strict=True`. Trained on Fold 1 with `per_segment` normalization. Best epoch = 5. | **PASS** |
| **Input Representation** | `src/models/vardhan_v3.py` (`forward()`) | **Raw time-domain waveform tensor of shape `(B, 1, 2048)`**. The model performs its own internal FFT via `extract_spectral_representation()`. It does **not** receive pre-computed spectra. | **PASS** |
| **Class Mapping** | `src/preprocessing/fgcs_faithful.py` & checkpoint | `0`: Background RF activities, `1`: Bebop drone, `2`: AR drone, `3`: Phantom drone. (Differs from generic `src/constants.py`.) | **PASS** |
| **Normalization** | Training runner & checkpoint | **`per_segment` standardization**: `x_norm = (x - mean(x)) / (std(x) + 1e-8)`. No global statistics required. | **PASS** |
| **Fallback Analysis** | `fgcs_faithful_loader.py` & training runner | Random data is only generated when `mock=True` (used by the test suite). Real-data loading raises `FileNotFoundError` instead of substituting random signals. | **PASS** |
| **Real Sample Loading** | `data/raw/DroneRF/DroneRF/*.rar` | 227 synchronized recording pairs discovered on the development machine with the full dataset. Real signals are extracted through the `tar` stream and classified live. | **PASS** |

---

## 2. Checkpoint Forensic Verification

### 2.1 File Characteristics & Integrity
Two checkpoint files exist in `models/checkpoints/vardhan_v3_per_segment_15ep/` and are tracked in Git:
- **Best Validation Model**: `checkpoint_fold_1.pt` (529,043 bytes / ~517 KB)
- **Final Epoch Model**: `final_model_fold_1.pt` (536,492 bytes / ~524 KB)

### 2.2 Metadata Inspection
Inspection of `checkpoint_fold_1.pt` yielded the following stored keys and values:
```json
{
  "fold": 1,
  "best_epoch": 5,
  "normalization": "per_segment",
  "norm_mean": null,
  "norm_std": null,
  "class_weighted": false,
  "test_loss": 0.7788078407884169,
  "test_accuracy": 0.7154185022026431,
  "test_macro_f1": 0.7027018041393609,
  "test_balanced_accuracy": 0.6837720996257582,
  "train_class_counts": {
    "0": 3690,
    "1": 7560,
    "2": 7290,
    "3": 1890
  }
}
```

`final_model_fold_1.pt` stores the weights after the last epoch (`final_epoch`: 15, `final_test_accuracy`: 0.6731).

### 2.3 Layer Matching & Parameter Count
- Initializing `VardhanV3(num_classes=4)` creates an architecture with **119,806 trainable parameters**.
- Calling `model.load_state_dict(checkpoint["model_state_dict"], strict=True)` matches all 147 weight and bias tensors with **zero missing keys and zero unexpected keys**.
- The checkpoint contains only tensors and plain metadata, so it loads with `torch.load(..., weights_only=True)`.
- Forward pass with input shape `(1, 1, 2048)` returns logits of shape `(1, 4)`. Softmax yields valid probability distributions that sum to 1.0.
- Single-window CPU inference measured **~7.17 ms** on the development machine and roughly 40–70 ms on a slower laptop. Latency depends on the machine; the demo reports the time it measures live.

---

## 3. Input Representation & Preprocessing Pipeline Resolution

### 3.1 The Potential Discrepancy Resolved
- **Literature Baseline Preprocessing (`FGCS2019DNN`)**: In `src/preprocessing/fgcs_faithful.py`, the Al-Sa'd baseline reproduction takes paired Low-band (L) and High-band (H) files, detrends them, computes 2048-point FFTs, normalizes the boundary, and concatenates them into a 2048-dimensional power spectrum.
- **VARDHAN-v3 Verified Preprocessing**: VARDHAN-v3 does **not** take a power spectrum as input. In `scripts/train_vardhan_v3_standardized.py` (`materialize_standardized_waveforms`, lines 205–217):
  1. It reads the raw continuous time-domain voltage samples from the Low-band (L) recording (`read_raw_signal_10m`).
  2. It extracts 2048-sample slices directly from the time-domain waveform at regular segment offsets (`seg_i * 100,000`).
  3. It applies independent `per_segment` standardization:
     $$\tilde{x} = \frac{x - \mu}{\sigma + 10^{-8}}$$
  4. The normalized tensor $\tilde{x} \in \mathbb{R}^{B \times 1 \times 2048}$ is passed directly to `VardhanV3`.

The demo's `DemoDataLoader.load_real_segment()` (`demo/data_utils.py`) reproduces these steps exactly for a single window.

### 3.2 Internal Model Feature Extraction
Once inside `VardhanV3.forward(x)`:
1. **Temporal Stem**: `self.time_stem(x)` directly convolves the normalized time-domain voltage slice, followed by 4 multi-scale dilated temporal blocks.
2. **Spectral Extraction (`extract_spectral_representation`)**: The model computes an on-the-fly Real FFT:
   - Per-sample detrending: $x_{\text{detrend}} = x - \text{mean}(x)$
   - $X = \text{rfft}(x_{\text{detrend}}, n=2048)$
   - Discards the DC bin ($k=0$), retaining 1024 positive bins.
   - Stacks 3 channels: Log Power Spectrum, Normalized Real Fourier Component, Normalized Imaginary Fourier Component $\to (B, 3, 1024)$.
3. **Cross-Domain Gated Fusion**: Temporal features (80-d) and spectral features (80-d) cross-modulate through sigmoid gating and concatenate to 320-d before classification into 4 logits.

---

## 4. Class Mapping Verification

The repository contains two different class mappings:
1. **Generic / Canonical Scaffold (`src/constants.py`)**:
   - `0`: `no_drone`
   - `1`: `ar_drone`
   - `2`: `bebop_drone`
   - `3`: `phantom_drone`
2. **Faithful FGCS Training Pipeline (`src/preprocessing/fgcs_faithful.py` & Checkpoint)**:
   - `0`: `Background RF activities` (BUI `00000`)
   - `1`: `Bebop drone` (BUI `10000`, `10001`, `10010`, `10011`)
   - `2`: `AR drone` (BUI `10100`, `10101`, `10110`, `10111`)
   - `3`: `Phantom drone` (BUI `11000`)

**Verdict**: The trained checkpoint was produced by `scripts/train_vardhan_v3_standardized.py`, which binds to the FGCS ordering (`0: Background, 1: Bebop, 2: AR, 3: Phantom`). Decoding with `src/constants.py` would swap Bebop and AR Drone predictions. The demo **must use the FGCS class mapping**, which `demo/config.py` does.

---

## 5. Fallback & Synthetic Data Audit

1. **`src/data/fgcs_faithful_loader.py`**:
   - Lines 297–300: `power_spectrum = rng.rand(2048).astype(np.float32)` is guarded behind `if self.mock:`.
   - `mock=False` is the default.
   - For real data, it invokes `read_raw_signal_10m()`, which raises `FileNotFoundError` if the signal cannot be read. It never silently substitutes random noise.
2. **`scripts/train_vardhan_v3_standardized.py`**:
   - Lines 181–186: Random generation is guarded behind `if mock:`.
3. **`src/preprocessing/signals.py`**:
   - `compute_stft()` previously returned random placeholder output; it has since been replaced with a real windowed-FFT implementation. It is not used by VARDHAN-v3 or its training pipeline either way.
4. **Metadata fallback in pair discovery**:
   - When raw files are missing, `discover_and_pair_dronerf_files()` falls back to `data/metadata/dronerf_metadata.csv`, which lists all 227 pairs with paths that may not exist locally. `DemoDataLoader` therefore keeps only pairs whose L-band file is actually on disk, and reports a clear error when none are.
5. **Demo Implementation Rule**:
   - In the demo, any loading error is shown as a visible error in the UI. No mock or fallback branches are reachable from real-data inference paths.

---

## 6. Scientific Provenance & Evaluation Context

- **Checkpoint Protocol**: The available checkpoint `checkpoint_fold_1.pt` represents **Fold 1 of a 10-Fold Stratified Cross-Validation benchmark** across 22,700 segments (with segment-level splitting).
- **Test Metrics on Fold 1**:
  - Test Accuracy: **71.54%**
  - Test Macro-F1: **0.7027**
  - Test Balanced Accuracy: **68.38%**
  - Test Loss: **0.7788**
- **Aggregate 10-Fold Metrics** (`results/vardhan_v3_per_segment_10fold/`): 71.57% ± 1.25% accuracy, 0.6953 ± 0.0168 macro-F1.
- **Generalization Limitation**:
  - Under segment-level CV, segments from the same physical recording may be shared across train and test folds, yielding optimistic scores (~71.5%).
  - Under strict recording-level evaluation (completely unseen recordings), VARDHAN-v3 accuracy drops to **24.79%** (`results/vardhan_v3_strict_per_segment/`). The historical strict benchmark (5 windows per recording) gave 28.22% for FGCS2019DNN and 28.77% for MC1DCNN.
  - The demo interface displays this distinction so evaluators understand the research context.
