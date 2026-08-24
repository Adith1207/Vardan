# Baseline Training & Evaluation Protocol

This document establishes the common, fair, and reproducible experimental protocol for training and evaluating baseline neural network architectures on the DroneRF dataset for 4-class drone detection and classification.

Every parameter and protocol decision is tagged according to its methodological origin:
- **`PAPER-EXPLICIT`**: Directly specified in the source publication.
- **`PAPER-INFERRED`**: Implementation detail derived from paper context or standard literature defaults.
- **`OUR-DESIGN-CHOICE`**: Specific choice made in our experimental framework for standard comparison.
- **`NOT-SPECIFIED`**: Parameter omitted or unmentioned in the source publication.

---

## 1. Dataset Protocol

| Parameter | Specification | Origin / Tag |
| :--- | :--- | :--- |
| **Source Dataset** | DroneRF Dataset (Raw RF signals collected at 100 MHz) | `PAPER-EXPLICIT` |
| **Total Recordings** | 454 raw CSV files | `PAPER-EXPLICIT` |
| **Number of Classes** | 4 classes (`no_drone`, `ar_drone`, `bebop_drone`, `phantom_drone`) | `PAPER-EXPLICIT` |
| **Canonical Labels** | `0`: `no_drone`, `1`: `ar_drone`, `2`: `bebop_drone`, `3`: `phantom_drone` | `OUR-DESIGN-CHOICE` |
| **Segment Length** | 2048 samples per segment | `PAPER-EXPLICIT` |

---

## 2. Split Protocol

| Parameter | Specification | Origin / Tag |
| :--- | :--- | :--- |
| **Split Strategy** | Deterministic file-level stratified shuffle | `OUR-DESIGN-CHOICE` |
| **Split Ratio** | 70% Train (320 files), 15% Validation (67 files), 15% Test (67 files) | `PAPER-EXPLICIT` |
| **Overlap** | Zero file overlap (`set(train) ∩ set(val) ∩ set(test) = ∅`) | `OUR-DESIGN-CHOICE` |
| **Class Balance** | Every class represented in Train, Val, AND Test | `OUR-DESIGN-CHOICE` |
| **Split Files** | `data/splits/train.csv`, `val.csv`, `test.csv` | `OUR-DESIGN-CHOICE` |

---

## 3. Preprocessing Protocol

| Model | Representation | Transformation Pipeline | Tag |
| :--- | :--- | :--- | :--- |
| **`FGCS2019DNN`** | 2048-pt Power Spectrum | DC Offset Removal $\rightarrow$ 2048-pt FFT $\rightarrow$ Power Spectrum $\rightarrow$ Train Max Norm | `PAPER-EXPLICIT` |
| **`Baseline1DCNN`** | 2-channel I/Q RF Waveform | Dual I/Q channels $\rightarrow$ Train-fitted $z$-score normalization | `PAPER-EXPLICIT` |
| **`DSCNN`** | 2-channel I/Q RF Waveform | Log dynamic range compression $\rightarrow$ Train-fitted $z$-score normalization | `PAPER-INFERRED` |
| **`MobileNetV3Small`** | 2D STFT Spectrogram | STFT ($n_{\text{fft}}=1024, \text{hop}=256$) $\rightarrow$ Log dB scaling $\rightarrow$ Channel normalization | `PAPER-INFERRED` |

*Data Leakage Prevention: Normalization parameters (mean, std, max, min) are fitted strictly on `train.csv`. Validation and test splits use pre-computed training parameters without updating scaler state.*

---

## 4. Model Architecture List

1. **`FGCS2019DNN`**: 4-layer Deep MLP (`2048 → 256 → 128 → 64 → 4`).
2. **`Baseline1DCNN`**: Standard 1D Convolutional Neural Network (`Conv1D → MaxPool1D → Conv1D → MaxPool1D → FC → FC`).
3. **`DSCNN`**: Depthwise Separable 1D CNN for TinyML edge efficiency (`Conv1D → DepthwiseConv1D → PointwiseConv1D → MaxPool1D → FC`).
4. **`MobileNetV3Small`**: Lightweight 2D Spectrogram Convolutional Neural Network.

---

## 5. Input Tensor Dimensions

| Model | Input Tensor Shape | Output Logits Shape |
| :--- | :--- | :--- |
| **`FGCS2019DNN`** | `(batch_size, 2048)` | `(batch_size, 4)` |
| **`Baseline1DCNN`** | `(batch_size, 2, 2048)` | `(batch_size, 4)` |
| **`DSCNN`** | `(batch_size, 2, 2048)` | `(batch_size, 4)` |
| **`MobileNetV3Small`** | `(batch_size, 1, 65, 61)` | `(batch_size, 4)` |

---

## 6. Loss Functions

- **Loss**: `torch.nn.CrossEntropyLoss()` for 4-class multiclass classification (`PAPER-INFERRED`).
- **Output Formats**: Models return raw linear logits `(batch_size, 4)` to preserve numerical stability during loss calculation.

---

## 7. Optimizers & Hyperparameters

- **Optimizer**: Adam Optimizer (`torch.optim.Adam`) (`PAPER-EXPLICIT` for FGCS, `PAPER-INFERRED` for CNNs).
- **Learning Rate**: $lr = 0.001$ ($10^{-3}$) (`PAPER-EXPLICIT`).
- **Weight Decay**: $0.0$ (`PAPER-INFERRED`).

---

## 8. Batch Sizes

- **`FGCS2019DNN`**: Batch size = 10 (`PAPER-EXPLICIT`).
- **`Baseline1DCNN`**: Batch size = 32 (`OUR-DESIGN-CHOICE`).
- **`DSCNN`**: Batch size = 32 (`OUR-DESIGN-CHOICE`).
- **`MobileNetV3Small`**: Batch size = 32 (`OUR-DESIGN-CHOICE`).

---

## 9. Epoch Limits

- **Full Baseline Training**: 100 to 200 epochs (`PAPER-EXPLICIT`).
- **Smoke Test**: 2 tiny epochs (2 batches per epoch) for protocol verification (`OUR-DESIGN-CHOICE`).

---

## 10. Validation Protocol

- **Validation Frequency**: Evaluated at the end of every training epoch.
- **Metrics Tracked**: Validation Loss, Validation Accuracy, Validation Macro F1.
- **Early Stopping**: Optional patience = 15 epochs based on Validation Loss (`OUR-DESIGN-CHOICE`).

---

## 11. Checkpoint Protocol

- **Directory**: `checkpoints/<model_name>/`
- **Best Model Criterion**: Saved when Validation Loss reaches a new minimum.
- **Artifacts Saved**: Model `state_dict`, optimizer `state_dict`, epoch, and best validation metrics.

---

## 12. Evaluation Metrics

- **Primary Metrics**: Accuracy, Macro F1, Weighted F1 (`PAPER-EXPLICIT`).
- **Secondary Metrics**: Macro Precision, Macro Recall, Per-Class Precision, Per-Class Recall, Per-Class F1.
- **Matrix**: 4x4 Confusion Matrix (`PAPER-EXPLICIT`).

---

## 13. Computational & Profiling Metrics

- **Parameter Counts**: Total parameters & Trainable parameters (`OUR-DESIGN-CHOICE`).
- **Model Footprint**: Estimated size in MB (float32).
- **Inference Latency**: Mean & standard deviation CPU latency (ms) over 100 benchmark runs.

---

## 14. Random Seed

- **Seed**: `42` (`torch.manual_seed(42)`, `np.random.seed(42)`).

---

## 15. Reproducibility Notes & Integrity Checks

- **Zero Data Leakage**: Normalization statistics are learned strictly from `train.csv`.
- **Zero NPZ Caching**: Data is loaded lazily on demand during DataLoader iterations.
- **Zero Overlap**: File-level split guarantees zero file overlap across splits.
