# FORENSIC AUDIT REPORT: VARDHAN-v1 MODEL & STRICT RECORDING-LEVEL BENCHMARK

**Audit Target**: `VardhanRFNet` (VARDHAN-v1) Architecture, Preprocessing Pipeline, and Experimental Artifacts  
**Dataset**: DroneRF Database (454 CSV files, 23 physical recordings)  
**Evaluation Protocol**: Strict Recording-Level Partitioning (`seed=42`, Zero Recording Overlap)  
**Target Classes**: `0: Background RF activities`, `1: AR Drone`, `2: Bepop drone`, `3: Phantom drone`  
**Date**: September 3, 2026  

---

## 1. EXECUTIVE SUMMARY

An exhaustive forensic audit was conducted on the existing **VARDHAN-v1 (`VardhanRFNet`)** implementation and its historical baseline results.

### Key Audit Findings:
1. **Root Cause of ~28.77% Accuracy**:
   - The ~28.77% test accuracy obtained by VARDHAN-v1 is **100% attributable to complete majority-class collapse**.
   - On the strict recording-level test set ($N=146$ samples), the network predicted **100% of all test instances as Class 2 (`bebop_drone`)**.
   - The overall accuracy of $28.77\%$ matches the exact test set prevalence of the Bebop drone ($42 / 146 = 28.767\%$). True Macro-F1 is **$0.1117$**, and Precision/Recall for Background, AR Drone, and Phantom Drone are all **$0.0000$**.
2. **Representation Deficiencies**:
   - VARDHAN-v1 processes isolated 2048-sample slices ($0.0512\text{ ms}$) of real-valued time-domain voltage waveforms.
   - It receives **no explicit spectral information**, **no discrete Fourier transform**, **no complex I/Q representation**, and **no synchronized multi-band $(L, H)$ stitching**.
   - Normalization is a global scalar Z-score computed from training files (`(x - mean) / std`).
3. **Severe Information Bottleneck in Architecture**:
   - The two convolutional branches have narrow temporal receptive fields of **7 samples ($0.175\ \mu\text{s}$)** and **11 samples ($0.275\ \mu\text{s}$)** at $40\text{ MSps}$.
   - After `MaxPool1d(4)`, the architecture applies **`AdaptiveAvgPool1d(1)`**, which averages all 512 temporal feature vectors into a single 64-dimensional vector.
   - This completely erases all temporal sequence ordering, burst dynamics, pulse repetition intervals, and phase structures.
4. **Recording-Level Distribution Shifts**:
   - In the strict recording-level split, Phantom Drone is present **exclusively on the $H$-band ($2.44\text{–}2.48\text{ GHz}$)** in Train, but **exclusively on the $L$-band ($2.40\text{–}2.44\text{ GHz}$)** in Validation and Test.
   - Because single-file time-domain loading omits the other $40\text{ MHz}$ half of the receiver spectrum, the model is presented with out-of-distribution frequency energy at test time.

---

## 2. EXISTING RESULT ARTIFACTS

All VARDHAN-v1 result artifacts on the strict recording-level benchmark are located and verified in the repository:

| Artifact Type | File Path | File Size | Description & Contents |
| :--- | :--- | :---: | :--- |
| **Metrics Summary** | [`results/baselines/vardhan/metrics.json`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/metrics.json) | $1,120\text{ B}$ | Overall test accuracy ($0.2877$), macro-F1 ($0.1117$), per-class precision/recall/F1, parameter count ($6,852$), latency ($1.05\text{ ms}$). |
| **Confusion Matrix (CSV)** | [`results/baselines/vardhan/confusion_matrix.csv`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/confusion_matrix.csv) | $130\text{ B}$ | $4 \times 4$ raw test prediction contingency table. |
| **Confusion Matrix (JSON)**| [`results/baselines/vardhan/confusion_matrix.json`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/confusion_matrix.json) | $175\text{ B}$ | $4 \times 4$ nested list of test prediction counts. |
| **Training History (CSV)** | [`results/baselines/vardhan/history.csv`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/history.csv) | $451\text{ B}$ | Per-epoch epoch index, train loss, train accuracy, val loss, val accuracy, val macro-F1. |
| **Training History (JSON)**| [`results/baselines/vardhan/history.json`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/history.json) | $671\text{ B}$ | JSON dictionary with training/validation trajectory arrays across epochs 1–4. |
| **Run Configuration** | [`results/baselines/vardhan/run_config.json`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/run_config.json) | $747\text{ B}$ | Hyperparameters, random seed ($42$), batch size ($4$), learning rate ($1e-3$), optimizer (`Adam`), timestamp. |
| **Best Checkpoint** | [`models/checkpoints/baselines/vardhan/best.pt`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/models/checkpoints/baselines/vardhan/best.pt) | $99,183\text{ B}$ | Model weights and optimizer state from best validation epoch (`epoch=3`, `val_loss=1.4607`). |
| **Last Checkpoint** | [`models/checkpoints/baselines/vardhan/last.pt`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/models/checkpoints/baselines/vardhan/last.pt) | $99,183\text{ B}$ | Latest model and optimizer weights saved at end of run (`epoch=4`). |

---

## 3. TRAINING DYNAMICS & TRAJECTORY ANALYSIS

Extracted directly from [`results/baselines/vardhan/history.json`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/history.json):

### Trajectory Table:
| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Validation Macro-F1 | Best Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $1.2521$ | $41.40\%$ | $1.4672$ | $28.77\%$ | $0.1117$ | — |
| **2** | $1.1971$ | $42.53\%$ | $1.4629$ | $28.77\%$ | $0.1117$ | — |
| **3** | $1.2142$ | $40.58\%$ | **$1.4607$** | **$28.77\%$** | **$0.1117$** | **Best Checkpoint** |
| **4** | $1.1891$ | $41.23\%$ | $1.4741$ | $29.45\%$ | $0.1664$ | Last Checkpoint |

### Forensic Diagnosis of Training Behavior:
1. **Severe Underfitting on Valid Representations**:
   - The training loss stagnates around $\approx 1.19\text{–}1.25$ (random chance 4-class Cross-Entropy is $-\ln(0.25) \approx 1.386$).
   - Training accuracy stays capped at $\approx 40.5\%\text{–}42.5\%$, which corresponds exactly to the majority class prevalence in the training set (`bepop_drone` $126/308 = 40.91\%$).
2. **Static Validation Plateau**:
   - Validation loss remains near theoretical random chance ($1.4607\text{–}1.4741$).
   - Validation accuracy never improves past $28.77\%$ (the exact test prevalence of `bebop_drone`).
3. **Absence of Overfitting**:
   - The model is not overfitting (training loss does not drop to near 0). It fails to extract discriminative features even from the training set, indicating that raw time-domain 1D convolutions with 7–11 sample receptive fields cannot extract drone signatures.

---

## 4. TEST CONFUSION MATRIX & CLASS-LEVEL METRICS

Directly extracted from [`results/baselines/vardhan/confusion_matrix.json`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/confusion_matrix.json) and [`metrics.json`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/results/baselines/vardhan/metrics.json):

### A. Test Confusion Matrix ($N=146$ Samples)

```
                       Predicted Class
                Background    AR Drone    Bepop Drone    Phantom Drone    Total True
True Class:
  Background         0           0            40               0              40
  AR Drone           0           0            42               0              42
  Bepop Drone        0           0            42               0              42
  Phantom Drone      0           0            22               0              22
Total Pred:          0           0           146               0             146
```

### B. Per-Class Performance Breakdown

| Class Index & Name | True Support | Predicted Count | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0: Background RF activities** | $40$ | $0$ | $0.0000$ | $0.0000$ | $0.0000$ | Complete Miss (0% recall) |
| **1: AR Drone** | $42$ | $0$ | $0.0000$ | $0.0000$ | $0.0000$ | Complete Miss (0% recall) |
| **2: Bepop drone** | $42$ | $146$ | $0.2877$ | $1.0000$ | $0.4468$ | Total Collapse Target |
| **3: Phantom drone** | $22$ | $0$ | $0.0000$ | $0.0000$ | $0.0000$ | Complete Miss (0% recall) |
| **Macro Average** | **146** | **146** | **0.0719** | **0.2500** | **0.1117** | **Severely Impaired** |

- **Overall Test Accuracy**: **$28.7671\%$** ($42 / 146$)
- **Test Macro-F1**: **$0.1117$**
- **Test Loss**: **$1.4699$**

---

## 5. CLASS COLLAPSE AUDIT

### Distribution Comparison:
- **True Test Class Distribution**:
  - Class 0 (Background): $40 / 146 = \mathbf{27.40\%}$
  - Class 1 (AR Drone): $42 / 146 = \mathbf{28.77\%}$
  - Class 2 (Bepop drone): $42 / 146 = \mathbf{28.77\%}$
  - Class 3 (Phantom drone): $22 / 146 = \mathbf{15.07\%}$
- **Predicted Test Class Distribution**:
  - Class 0 (Background): $0 / 146 = \mathbf{0.00\%}$
  - Class 1 (AR Drone): $0 / 146 = \mathbf{0.00\%}$
  - Class 2 (Bepop drone): $146 / 146 = \mathbf{100.00\%}$
  - Class 3 (Phantom drone): $0 / 146 = \mathbf{0.00\%}$

### Conclusive Classification:
The ~28.77% result is **Category B: Complete Majority-Class Collapse**.  
The network outputs a constant positive bias for Class 2 (`bebop_drone`) for every input sample, failing to classify any sample into Background, AR Drone, or Phantom Drone.

---

## 6. INPUT REPRESENTATION AUDIT

Tracing the exact runtime path in [`src/data/loader.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/data/loader.py#L356-L370):

```python
elif name in ["vardhan", "vardhanrfnet", "vardhan_rf"]:
  # VARDHAN: 1D real RF waveform -> train-fitted Z-score normalized -> (1, 2048)
  norm_sig = (raw_sig - self.norm_stats["mean"]) / (
      self.norm_stats["std"] + 1e-8
  )
  out_tensor = np.expand_dims(norm_sig, axis=0).astype(np.float32)
```

### Audited Representation Properties:
1. **Raw Slicing**: Slices 2048 raw samples ($0.0512\text{ ms}$) from a single CSV file.
2. **No DC Removal**: `remove_dc` is omitted for the raw waveform loader branch (unlike FGCS/MC1DCNN which detrend the mean).
3. **Scalar Z-Score Normalization**: Scales by global scalar `mean` and `std` computed strictly from training split files (`fit_train_normalization_stats`).
4. **Single Isolated 40 MHz Band**: The slice comes from a single $L$ or $H$ file ($40\text{ MHz}$ bandwidth). There is zero receiver synchronization with the companion band.
5. **No Fourier Transform / Spectral Features**: The input contains only raw oscillatory time-domain amplitudes.
6. **No Complex Quadrature (I/Q)**: DroneRF CSVs store a single real-valued voltage column; quadrature phase information is absent.

---

## 7. ARCHITECTURE AUDIT & THEORETICAL BOTTLENECK

Tracing [`src/models/vardhan.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/models/vardhan.py):

### Layer-by-Layer Architecture & Shape Propagation:
```
Input: (B, 1, 2048)
├── Branch 1:
│   ├── Conv1d(1, 16, kernel_size=3, padding=1)           -> (B, 16, 2048)  [64 params]
│   ├── ReLU()                                            -> (B, 16, 2048)
│   └── Conv1d(16, 32, kernel_size=3, padding=2, dil=2)   -> (B, 32, 2048)  [1,568 params]
├── Branch 2:
│   ├── Conv1d(1, 16, kernel_size=7, padding=3)           -> (B, 16, 2048)  [128 params]
│   ├── ReLU()                                            -> (B, 16, 2048)
│   └── Conv1d(16, 32, kernel_size=5, padding=2)          -> (B, 32, 2048)  [2,592 params]
├── Concatenation: torch.cat([b1, b2], dim=1)             -> (B, 64, 2048)
├── Transition:
│   ├── BatchNorm1d(64)                                   -> (B, 64, 2048)  [128 params]
│   ├── ReLU()                                            -> (B, 64, 2048)
│   └── MaxPool1d(kernel_size=4)                          -> (B, 64, 512)
├── Global Pooling:
│   ├── AdaptiveAvgPool1d(1)                              -> (B, 64, 1)     [0 params]
│   └── Squeeze(-1)                                       -> (B, 64)
└── Classifier Head:
    ├── Linear(64, 32)                                    -> (B, 32)        [2,080 params]
    ├── ReLU()                                            -> (B, 32)
    ├── Dropout(p=0.3)                                    -> (B, 32)
    └── Linear(32, 4)                                     -> (B, 4)         [132 params]
Total Trainable Parameters: 6,692
```

### Theoretical Weaknesses Identified:
1. **Extreme Temporal Subsampling Bottleneck**:
   - `AdaptiveAvgPool1d(1)` averages all 512 activation vectors across time into a single 64-dimensional vector.
   - For raw RF time-domain signals, averaging oscillatory filter responses over $51.2\ \mu\text{s}$ destroys carrier frequency modulation, hop timing, burst periodicity, and waveform envelope shapes.
2. **Insufficient Receptive Field**:
   - Branch 1 receptive field: **7 samples ($0.175\ \mu\text{s}$)**.
   - Branch 2 receptive field: **11 samples ($0.275\ \mu\text{s}$)**.
   - An 11-sample window at $40\text{ MSps}$ cannot resolve frequency components narrower than $\sim 3.6\text{ MHz}$, making it mathematically incapable of discriminating fine drone sub-carrier channels.
3. **Severe Under-parameterization**:
   - Total parameters: **6,692** ($0.026\text{ MB}$).
   - While lightweight, 6.6k parameters is too constrained to simultaneously learn multi-carrier filterbanks, demodulation, and high-level flight mode classification from raw noisy time-domain waveforms.

---

## 8. RECORDING-LEVEL SPLIT & DISTRIBUTION SHIFT ANALYSIS

Auditing [`data/splits/train.csv`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/data/splits/train.csv), [`val.csv`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/data/splits/val.csv), [`test.csv`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/data/splits/test.csv):

### Recording Partition Table:
| Split | Total Files | Physical Recordings | Receiver Bands | Key Distribution Shifts |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 308 | 15 recordings | $185\ H$-band / $123\ L$-band | Phantom drone is present **ONLY on $H$-band** (`11000_H`). Zero $L$-band Phantom files. |
| **Validation** | 73 | 4 recordings | $21\ H$-band / $52\ L$-band | Phantom drone is present **ONLY on $L$-band** (`11000_L1`). Bebop mode 0 only (`10000_L`). |
| **Test** | 73 | 4 recordings | $21\ H$-band / $52\ L$-band | Phantom drone is present **ONLY on $L$-band** (`11000_L2`). AR mode 0 only (`10100_L`). |

### Impact on Raw Single-File Models:
- Because the baseline loader feeds isolated 40 MHz files without $L/H$ synchronization:
  - VARDHAN-v1 was trained on Phantom drone signals exclusively centered at $2.46\text{ GHz}$ ($H$-band).
  - At test time, VARDHAN-v1 was evaluated on Phantom drone signals centered at $2.42\text{ GHz}$ ($L$-band).
  - Without full-band 80 MHz representation or $L/H$ synchronization, the model faces a complete out-of-distribution frequency shift.

---

## 9. COMPARISON AGAINST REPRODUCED BASELINES

| Model | Input Representation | Input Shape | Domain | Parameters | 10-Fold Segment CV Accuracy | Strict Recording Split Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **FGCS Faithful DNN** | 2048-pt stitched power spectrum ($L+H$) | `(2048,)` | Frequency | 329,476 | **84.52%** | ~28.22% |
| **MC1DCNN Faithful** | 8 $\times$ 10 MHz sub-bands ($L+H$) | `(8, 256)` | Frequency | 275,940 | **65.01%** | ~28.77% |
| **DSCNN (Baseline)** | Z-score normalized raw waveform | `(1, 2048)` | Time | 16,836 | — | ~28.77% |
| **VARDHAN-v1** | Z-score normalized raw waveform | `(1, 2048)` | Time | **6,692** | — | **28.77%** |

---

## 10. RANKED LIMITATIONS OF VARDHAN-v1

Based strictly on audit evidence:

### 1. Complete Temporal Erasure via `AdaptiveAvgPool1d(1)` (Rank 1)
- **Evidence**: `feat = self.global_pool(feat).squeeze(-1)` reduces all 512 temporal feature steps to a single global average. Raw RF oscillations average to near zero or static energy levels.
- **Confidence**: **HIGH**
- **Testable Experiment**: Replace global average pooling with temporal attention, multi-head pooling, or 1D spatial flattening.

### 2. Lack of Explicit Frequency-Domain / Spectral Information (Rank 2)
- **Evidence**: Baselines with explicit 2048-point DFT power spectra achieve 84.52% on 10-fold CV, while raw waveform models collapse to majority-class prediction.
- **Confidence**: **HIGH**
- **Testable Experiment**: Feed an explicit time-frequency representation (STFT, multi-channel PSD, or learnable filterbank) into the network.

### 3. Missing Synchronized 80 MHz Receiver Pairing (Rank 3)
- **Evidence**: Train split has Phantom on $H$-band only, whereas Test split has Phantom on $L$-band only. An isolated 40 MHz receiver input cannot bridge this band gap.
- **Confidence**: **HIGH**
- **Testable Experiment**: Synchronize $(x^{(L)}, x^{(H)})$ pairs into full 80 MHz representations prior to feature extraction.

### 4. Narrow Convolutional Receptive Field (Rank 4)
- **Evidence**: Maximum receptive field is 11 samples ($0.275\ \mu\text{s}$), which cannot capture RF pulse envelope modulations ($> 10\ \mu\text{s}$).
- **Confidence**: **HIGH**
- **Testable Experiment**: Incorporate multi-scale dilated convolutions with exponentially expanding dilation ($d \in \{1, 2, 4, 8, 16\}$) or larger kernels ($k=31, 63$).

### 5. Insufficient Model Capacity (Rank 5)
- **Evidence**: Model contains only 6,692 parameters ($0.026\text{ MB}$), causing underfitting on multi-class raw RF data.
- **Confidence**: **MEDIUM**
- **Testable Experiment**: Scale model depth and channel capacity to 50k–200k parameters while retaining edge-deployable efficiency.

---

## 11. REQUIREMENTS FOR VARDHAN-v2

Based strictly on the forensic findings, VARDHAN-v2 should satisfy the following engineering requirements:

1. **Time-Frequency Representation Integration**:
   - Must incorporate explicit spectral features (e.g., STFT spectrograms or multi-channel power spectral sub-bands) rather than purely unassisted raw time-domain waveforms.
2. **Synchronized Dual-Band Support**:
   - Must support full 80 MHz coverage via synchronized $(L, H)$ receiver pairing to resolve the $L/H$ band distribution shift across recording sessions.
3. **Multi-Scale Temporal Receptive Fields**:
   - Feature extractors must span both micro-scale RF transient features ($0.1\text{–}1\ \mu\text{s}$) and macro-scale burst/packet structures ($10\text{–}50\ \mu\text{s}$).
4. **Non-Destructive Feature Aggregation**:
   - Replace simple global average pooling (`AdaptiveAvgPool1d(1)`) with attention-based pooling, temporal statistics pooling, or multi-scale convolutional flattening.
5. **Calibrated Model Capacity**:
   - Scale parameter budget to $\sim 50\text{k}\text{–}250\text{k}$ parameters to ensure sufficient representational power while preserving ultra-fast edge inference ($< 5\text{ ms}$).

---

## 12. METHODOLOGICAL SEPARATION OF EVIDENCE

- **FACT**:
  - VARDHAN-v1 achieves $28.77\%$ accuracy because 100% of test predictions collapse to Class 2 (`bebop_drone`).
  - VARDHAN-v1 has exactly 6,692 trainable parameters.
  - VARDHAN-v1 uses `AdaptiveAvgPool1d(1)` directly after `MaxPool1d(4)`.
  - In the strict recording split, Train contains Phantom on $H$-band only, while Val/Test contain Phantom on $L$-band only.
- **INFERENCE**:
  - Global average pooling on oscillatory raw time-domain waveforms destroys discriminative phase and burst timing.
  - Receptive fields of 7–11 samples are too narrow to resolve sub-MHz frequency channels from time-domain samples alone.
- **HYPOTHESIS**:
  - Incorporating multi-scale frequency-domain representations and attention pooling in VARDHAN-v2 will overcome the majority-class collapse and achieve out-of-session generalization on the strict recording-level benchmark.

---

## 13. RECOMMENDED NEXT EXPERIMENTS

1. **Pre-V2 Architecture Design & Review**:
   - Design VARDHAN-v2 satisfying the 5 core requirements above without modifying any baseline code.
2. **Feature Representation Ablation**:
   - Benchmark VARDHAN-v2 across:
     - (a) Raw synchronized waveforms with dilated multi-scale temporal convolutions.
     - (b) Multi-channel sub-band power representations.
     - (c) 2D STFT spectrogram representations.
3. **Evaluation on Both Benchmarks**:
   - Evaluate VARDHAN-v2 on both the **Strict Recording-Level Benchmark** (generalization test) and the **10-Fold Full Dataset CV** (benchmark parity).
