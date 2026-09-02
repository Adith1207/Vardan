# RF Input Representation Implementation Report

**Project**: VARDHAN (Lightweight RF Deep Learning for Counter-UAS)  
**Report**: `reports/input_representation_implementation.md`  
**Reference Specification**: `reports/final_input_representation_spec.md`  
**Date**: September 2, 2026  

---

## 1. Executive Summary

This task successfully implemented the literature-faithful and physically valid RF input representations across all five baseline architectures and the proposed VARDHAN model. The ungrounded `q = np.roll(i, 1)` synthetic pseudo-I/Q hack has been **completely eliminated** from the repository, restoring true real-valued RF physics.

All six model architectures have been verified through unit tests, shape contracts, forward passes, loss computations, and tiny mock smoke tests.

---

## 2. Files Modified

| File Path | Description of Changes |
|---|---|
| [`src/data/loader.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/data/loader.py) | Completely removed `q = np.roll(i, 1)`. Added dedicated lazy preprocessing branches for all 6 models. |
| [`src/preprocessing/compression.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/preprocessing/compression.py) | Implemented `CompressiveSensingMatrix(n_input=2048, n_compressed=1024, seed=42)` with deterministic Gaussian projection $\Phi x$. |
| [`src/preprocessing/pipeline.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/preprocessing/pipeline.py) | Integrated `CompressiveSensingMatrix`, updated `SpectrumChannelizer(channel_count=8, overlap=0.0)`, and updated `process_compressed()`. |
| [`src/models/baselines.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/models/baselines.py) | Updated `Baseline1DCNN(in_channels=8, seq_length=256)`, `DSCNN(in_channels=1, seq_length=2048)`, and created `CompressiveSensingCNN(in_channels=1, seq_length=1024)`. |
| [`src/models/vardhan.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/models/vardhan.py) | Updated `VardhanRFNet(in_channels=1, seq_length=2048)` for 1D real RF waveform input. |
| [`src/models/model_factory.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/models/model_factory.py) | Updated factory constructors and aliases for all 6 models with correct default channel counts and sequence lengths. |
| [`src/models/__init__.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/models/__init__.py) | Exported `CompressiveSensingCNN` and `CS_CNN`. |
| [`scripts/run_smoke_test.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/scripts/run_smoke_test.py) | Updated model specifications to cover all 6 finalized representations. |
| [`scripts/verify_preprocessing.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/scripts/verify_preprocessing.py) | Updated contract definitions to verify all 6 representations. |
| [`tests/test_final_input_representations.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/tests/test_final_input_representations.py) | Added comprehensive test suite verifying shapes, forward passes, determinism, zero leakage, and zero pseudo-I/Q. |
| [`tests/test_imports.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/tests/test_imports.py), [`tests/test_trainer_checkpoints.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/tests/test_trainer_checkpoints.py), [`tests/test_kaggle_path_resolution.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/tests/test_kaggle_path_resolution.py) | Updated fixtures and assertions for 1-channel waveform inputs. |

---

## 3. Exact Before/After Tensor Shapes & Preprocessing Paths

| Model Key | Model Class | Before Shape | After Shape | Exact Preprocessing Pipeline |
|---|---|:---:|:---:|---|
| **`fgcs2019dnn`** | `FGCS2019DNN` | `(B, 2048)` | `(B, 2048)` | Raw 2048 $\rightarrow$ Mean DC removal $\rightarrow$ 2048-pt FFT $\rightarrow$ $\|FFT\|^2 \rightarrow$ Max-norm |
| **`baseline1dcnn`** | `Baseline1DCNN` | `(B, 2, 2048)` | `(B, 8, 256)` | Raw 2048 $\rightarrow$ Mean DC removal $\rightarrow$ 2048-pt FFT $\rightarrow$ $\|FFT\|^2 \rightarrow$ `SpectrumChannelizer(8, overlap=0.0)` |
| **`dscnn`** | `DSCNN` (TinyML) | `(B, 2, 2048)` | `(B, 1, 2048)` | Raw 2048 $\rightarrow$ Train-fitted Z-score normalization $\rightarrow$ Expand channel |
| **`compressed_sensing`** | `CompressiveSensingCNN` | `(B, 2, 2048)` | `(B, 1, 1024)` | Raw 2048 $\rightarrow$ Compressive projection $y = \Phi x$ ($\Phi \in \mathbb{R}^{1024 \times 2048}$, seed=42) $\rightarrow$ Z-score norm |
| **`mobilenetv3small`** | `MobileNetV3Small` | `(B, 1, 65, 61)` | `(B, 1, 65, 61)` | Raw 2048 $\rightarrow$ STFT (Hann, $N_{\text{fft}}=1024$, hop=256) $\rightarrow$ $20\log_{10}(\|STFT\|)$ $\rightarrow$ Z-score norm |
| **`vardhan`** | `VardhanRFNet` | `(B, 2, 2048)` | `(B, 1, 2048)` | Raw 2048 $\rightarrow$ Train-fitted Z-score normalization $\rightarrow$ Expand channel |

---

## 4. Compressive Sensing Matrix ($\Phi$) Verification

- **Dimensions**: $\Phi \in \mathbb{R}^{1024 \times 2048}$ (Compression Ratio $CR = 0.50$).
- **Distribution**: Gaussian random distribution $\Phi_{i,j} \sim \mathcal{N}(0, 1/1024)$ with rows normalized to unit $L_2$ norm ($\|\phi_i\|_2 = 1.0$).
- **Determinism & Persistence**: Fixed deterministic seed `seed=42`. Multiple instantiations produce bitwise identical matrices (`assert np.array_equal(cs1.phi, cs2.phi)` passed).
- **Single Matrix Rule**: The same matrix $\Phi$ is shared across training, validation, testing, and real-time inference.

---

## 5. Verification & Smoke Test Results

### 5.1 Pytest Suite Execution
```bash
$ pytest
============================== 30 passed in 4.42s ==============================
```
- 30/30 unit tests, contract tests, checkpoint tests, and representation tests passed with 0 errors.

### 5.2 Tiny Smoke Test Suite (`scripts/run_smoke_test.py --mock`)
- **Dataset Loading**: 616 Train, 146 Val, 146 Test samples generated for all 6 models with 0 NaNs and 0 Infs.
- **Model Forward Passes**: All 6 models produced `(4, 4)` raw class logits.
- **Training Progression (2 Epochs, 2 Batches/Epoch)**:
  - `fgcs2019dnn`: Epoch 1/2 Loss: 1.3828 $\rightarrow$ Epoch 2/2 Loss: 1.1452 (Finite)
  - `baseline1dcnn`: Epoch 1/2 Loss: 1.3537 $\rightarrow$ Epoch 2/2 Loss: 0.9671 (Finite)
  - `dscnn`: Epoch 1/2 Loss: 0.7993 $\rightarrow$ Epoch 2/2 Loss: 0.0001 (Finite)
  - `compressed_sensing`: Epoch 1/2 Loss: 0.7341 $\rightarrow$ Epoch 2/2 Loss: 0.0015 (Finite)
  - `mobilenetv3small`: Epoch 1/2 Loss: 1.4299 $\rightarrow$ Epoch 2/2 Loss: 1.2781 (Finite)
  - `vardhan`: Epoch 1/2 Loss: 1.0722 $\rightarrow$ Epoch 2/2 Loss: 0.9381 (Finite)
- **CPU Inference Latencies**:
  - `FGCS2019DNN`: $0.052\text{ ms}$ ($565,956$ params)
  - `Baseline1DCNN`: $0.139\text{ ms}$ ($275,940$ params)
  - `DSCNN`: $0.590\text{ ms}$ ($68,228$ params)
  - `CompressiveSensing`: $0.245\text{ ms}$ ($1,059,908$ params)
  - `MobileNetV3Small`: $0.494\text{ ms}$ ($11,588$ params)
  - `VardhanRFNet`: $0.850\text{ ms}$ ($6,692$ params)

---

## 6. Strict Protocol Confirmations

1. **Pseudo-I/Q Completely Removed**: Verified by `test_no_pseudo_iq_in_dataset()` and workspace-wide regex search. Zero instances of `np.roll` exist in the data loading pipeline.
2. **Dataset Splits Unchanged**: `train.csv` (308 files), `val.csv` (73 files), and `test.csv` (73 files) with Seed 42 recording-level stratification remain strictly unchanged with zero session leakage.
3. **No Full Training Started**: Only tiny 2-epoch, 2-batch smoke tests were executed to verify numerical stability.
4. **VARDHAN Magnitude+Phase**: Kept as a future ablation; primary VARDHAN model uses 1D real RF waveforms `(B, 1, 2048)`.

---

## 7. Implementation Status Summary

```
========================================
         IMPLEMENTATION STATUS
========================================
FGCS2019DNN:         PASS  (B, 2048)
Baseline1DCNN:       PASS  (B, 8, 256)
DSCNN (TinyML):      PASS  (B, 1, 2048)
Compressed Sensing:  PASS  (B, 1, 1024)
MobileNetV3Small:    PASS  (B, 1, 65, 61)
VardhanRFNet:        PASS  (B, 1, 2048)
Pytest Suite:        PASS  (30/30 passed)
Smoke Tests:         PASS  (All 6 models)
========================================
```
