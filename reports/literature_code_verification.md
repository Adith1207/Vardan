# Comprehensive Literature-vs-Code Verification for DroneRF Baselines

**Project**: VARDHAN (Lightweight RF Deep Learning for Counter-UAS)  
**Report Location**: `reports/literature_code_verification.md`  
**Associated Machine-Readable Data**: `results/diagnostics/literature_code_verification.json`  
**Date of Audit**: September 2, 2026  

---

## 1. Executive Summary

This report delivers a line-by-line verification of the three primary paper-backed baseline models implemented in the VARDHAN framework against their cited source literature:
1. **`FGCS2019DNN`**: Al-Sa'd et al. (2019) / Allahham et al. (2020), *Future Generation Computer Systems*.
2. **`MC1DCNN` / `Baseline1DCNN`**: Allahham et al. (2020) / Ezuma et al. (2020), Multi-Channel 1D CNN for DroneRF.
3. **`Compressed-Sensing RF Baseline`**: Mo et al. (2022), Deep Learning using Compressively Sensed RF Signals.
4. **`DSCNN` (TinyML baseline)**: Medaiyese et al. (2022), Depthwise Separable CNN.

### Key Findings:
- **`FGCS2019DNN` [FAITHFUL ADAPTATION]**: Preprocessing (DC removal $\rightarrow$ 2048 FFT $\rightarrow$ $|FFT|^2 \rightarrow$ max-normalization) and the 4-layer MLP architecture ($2048 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 4$) faithfully match the single-receiver formulation in Al-Sa'd et al. (2019). The evaluation protocol is adapted to strict recording-level isolation.
- **`MC1DCNN` / `Baseline1DCNN` [NOT PAPER-FAITHFUL]**: The repository's DataLoader currently synthesizes an artificial 2-channel "pseudo-I/Q" signal via a 1-sample circular shift (`q = np.roll(i, 1)`). This is an ad-hoc implementation artifact with **no physical RF validity**. In Allahham et al. (2020), "Multi-Channel" refers to **spectral sub-band channelization** (e.g. 8 sub-bands) or dual physical receiver channels (L and H).
- **`Compressed-Sensing Baseline` [NOT PAPER-FAITHFUL]**: The repository conflates **Dynamic Range Compression** (log / $\mu$-law companding in `src/preprocessing/compression.py`) and `DSCNN` (Medaiyese 2022) with the **Compressive Sensing matrix projection** ($y = \Phi x$) of Mo et al. (2022).
- **Causes of Poor 5-Epoch Pilot Results**: The primary driver of poor validation/test generalization is the **Phantom frequency shift** in the primary split (Train: 5.8 GHz only $\rightarrow$ Val/Test: 2.4 GHz only), exacerbated by the artificial pseudo-I/Q artifact in the 1D CNNs.

---

## 2. Baseline Verification Part 1: FGCS2019DNN (Al-Sa'd et al., 2019)

### 2.1 Literature Specifications
- **Raw Input**: Real-valued amplitude measurements from two receiving chains: Lower band ($L$, 2.4 GHz) and Higher band ($H$, 5.8 GHz), sampled at 100 MS/s.
- **Windowing**: Segment length of $N = 2048$ samples ($20.48\,\mu\text{s}$).
- **DC Offset Removal**: Subtracts the segment mean $\mu_x$: $x_c[n] = x[n] - \mu_x$.
- **FFT & Spectrum**: Computes the 2048-point DFT. The power spectrum (periodogram) is computed as the magnitude-squared $|X[k]|^2$.
- **Normalization**: Max-magnitude normalization scaling spectral power to $[0, 1]$.
- **Receiver Handling**: Al-Sa'd et al. treated 2048-sample segments from $L$ and $H$ recordings as independent 2048-dimensional inputs to the network.
- **Architecture**: Dense($2048 \rightarrow 256$, ReLU) $\rightarrow$ Dense($256 \rightarrow 128$, ReLU) $\rightarrow$ Dense($128 \rightarrow 64$, ReLU) $\rightarrow$ Dense($64 \rightarrow 4$, Sigmoid).
- **Hyperparameters**: Optimizer: Adam ($\text{lr}=10^{-3}$), Batch size: 10, Epochs: 200.

### 2.2 Repository Code Comparison
- **Preprocessing** (`src/preprocessing/pipeline.py` & `src/preprocessing/fft.py`):
  `process_fgcs()` executes zero-mean $\rightarrow$ 2048-point FFT $\rightarrow$ $|FFT|^2 \rightarrow$ max-normalization $\rightarrow$ output vector `(2048,)`. **(EXACT MATCH $\checkmark$)**
- **Architecture** (`src/models/baselines.py` lines 124-164):
  `FGCS2019DNN` implements Dense(2048, 256) $\rightarrow$ ReLU $\rightarrow$ Dense(256, 128) $\rightarrow$ ReLU $\rightarrow$ Dense(128, 64) $\rightarrow$ ReLU $\rightarrow$ Dense(64, 4) returning unnormalized logits for `nn.CrossEntropyLoss()`. **(EXACT MATCH $\checkmark$)**
- **Evaluation**: Evaluated using our project's leakage-safe recording split (308/73/73) instead of the paper's segment-level 10-fold cross-validation.

---

## 3. Baseline Verification Part 2: MC1DCNN (Allahham et al., 2020)

### 3.1 Literature Specifications
- **True Input Representation**: Multi-Channel RF representation. The channels represent **frequency sub-bands** (e.g. 8 sub-bands obtained from spectral channelization / filter bank decomposition) or dual physical receiver channels ($L$ and $H$).
- **Preprocessing**: Signals are transformed into sub-band energy channels or segmented multi-band waveforms.
- **Architecture**: 1D Convolutional Neural Network with 2 Conv1D layers, MaxPool1D layers, and Dense classification heads.

### 3.2 Repository Code Comparison & The Pseudo-I/Q Mismatch
- In `src/preprocessing/channelization.py`, the repository correctly provides `SpectrumChannelizer(channel_count=8)` to divide a spectrum into 8 sub-bands.
- **HOWEVER**, in `src/data/loader.py` (lines 343-346) and `src/models/baselines.py` (`Baseline1DCNN`):
  ```python
  i_ch = (raw_sig - self.norm_stats["mean"]) / self.norm_stats["std"]
  q_ch = np.roll(i_ch, 1)  # <-- CRITICAL IMPLEMENTATION ARTIFACT
  out_tensor = np.stack([i_ch, q_ch], axis=0).astype(np.float32)
  ```
- **CRITICAL AUDIT FINDING**: The DroneRF dataset contains real-valued scalar voltage samples. Rolling a real signal by 1 sample (`np.roll(1)`) does **not** create quadrature ($Q$) phase components. It provides an arbitrary, highly correlated artificial second channel that is neither physically nor theoretically grounded in the cited literature.

---

## 4. Baseline Verification Part 3: Compressively-Sensed RF Baseline (Mo et al., 2022)

### 4.1 Literature Specifications (Mo et al., 2022)
- **Problem**: Compressive sensing acquires raw RF signals at sub-Nyquist rates to reduce ADC power and storage.
- **Mathematical Sampling**: A signal segment $x \in \mathbb{R}^N$ ($N=2048$) is compressed to $y \in \mathbb{R}^M$ ($M < N$) via a random measurement matrix $\Phi \in \mathbb{R}^{M \times N}$:
  $$y = \Phi x$$
- **Compression Ratios ($CR = M/N$)**: Evaluated at $25\%$, $50\%$, and $75\%$.
- **Classification**: Neural networks are trained directly on compressed measurements or compressed-domain spectral features without full signal reconstruction.

### 4.2 Repository Code Comparison & Conflation
- In `src/preprocessing/compression.py`:
  The module implements **Dynamic Range Compression** ($\log_{10}(x)$ scaling, $\mu$-law companding, and power-law gamma compression). It aliases `CompressedSensingProcessor = DynamicRangeCompressor`.
- In `src/models/baselines.py`:
  The model provided is `DSCNN` (Depthwise Separable CNN from Medaiyese et al., 2022).
- **CRITICAL AUDIT FINDING**: The current repository does **not** implement compressive sensing ($y = \Phi x$). It conflates TinyML dynamic range compression with sub-Nyquist compressed sensing.

---

## 5. DroneRF Dataset Ground-Truth Verification

| Dataset Attribute | Source Paper (Al-Sa'd / Allahham 2019) | Discovered Repository Metadata | Match Status |
|---|---|---|:---:|
| **Total CSV Files** | 454 files | 454 files in `data/metadata/dronerf_metadata.csv` | **EXACT $\checkmark$** |
| **Discrete Sessions** | 23 recording sessions | 23 recording IDs identified | **EXACT $\checkmark$** |
| **Classes** | 4 (Background, AR, Bebop, Phantom) | 4 canonical classes mapped | **EXACT $\checkmark$** |
| **Sampling Rate** | 100 MS/s real amplitude | 100 MS/s (`SAMPLING_RATE = 100000000`) | **EXACT $\checkmark$** |
| **Receiver Types** | `L` (2.4 GHz) and `H` (5.8 GHz) | `L`, `L1`, `L2`, `H`, `H1`, `H2` mapped | **EXACT $\checkmark$** |
| **File Format** | Single real amplitude column | Bounded float parsing via `readline` | **EXACT $\checkmark$** |

---

## 6. Paper-vs-Code Comparison Table

| Baseline | Cited Paper | Paper Input | Current Code Input | Paper Preprocessing | Current Code Preprocessing | Paper Architecture | Current Code Architecture | Training Match | Evaluation Match | Status Classification |
|---|---|---|---|---|---|---|---|:---:|:---:|:---:|
| **`FGCS2019DNN`** | Al-Sa'd et al. (2019) | 2048-point power spectrum | `(B, 2048)` power spectrum | DC removal $\rightarrow$ 2048 FFT $\rightarrow$ $\|FFT\|^2 \rightarrow$ max-norm | DC removal $\rightarrow$ 2048 FFT $\rightarrow$ $\|FFT\|^2 \rightarrow$ max-norm | Dense 2048 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 4 | Dense 2048 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 4 | **High** (Adam, lr=1e-3, bs=10, ep=200) | **Adapted** (Recording split) | **[FAITHFUL ADAPTATION]** |
| **`Baseline1DCNN`** | Allahham et al. (2020) | Multi-channel spectral sub-bands | `(B, 2, 2048)` pseudo-I/Q | Sub-band filter bank channelization | Z-score normalization + `np.roll(1)` | Conv1D(11) $\rightarrow$ Pool $\rightarrow$ Conv1D(5) $\rightarrow$ Pool $\rightarrow$ FC(128) $\rightarrow$ FC(4) | Conv1D(11) $\rightarrow$ Pool $\rightarrow$ Conv1D(5) $\rightarrow$ Pool $\rightarrow$ FC(128) $\rightarrow$ FC(4) | **Partial** (Adam, lr=1e-3, bs=32, ep=100) | **Adapted** (Recording split) | **[NOT PAPER-FAITHFUL]** |
| **`Compressed Sensing`** | Mo et al. (2022) | Compressive measurements $y = \Phi x$ | `(B, 2, 2048)` time waveform | Random projection matrix $\Phi$ + PSD estimation | $\log$ / $\mu$-law companding | Compressed-domain CNN/DNN | `DSCNN` (Medaiyese 2022) | **Partial** | **Adapted** (Recording split) | **[NOT PAPER-FAITHFUL]** |
| **`DSCNN` (TinyML)** | Medaiyese et al. (2022) | 1D time-domain RF waveform | `(B, 2, 2048)` pseudo-I/Q | Z-score normalization on time waveform | Z-score normalization + `np.roll(1)` | Conv1D(11) $\rightarrow$ Depthwise(5) $\rightarrow$ Pointwise(1) $\rightarrow$ Pool $\rightarrow$ FC(4) | Conv1D(11) $\rightarrow$ Depthwise(5) $\rightarrow$ Pointwise(1) $\rightarrow$ Pool $\rightarrow$ FC(4) | **High** (Adam, lr=1e-3, bs=32, ep=100) | **Adapted** (Recording split) | **[PARTIAL MATCH]** |

---

## 7. Detailed Mismatch Table

| Model | Component | Paper Specifies | Our Code Does | Match? | Severity | Evidence & Impact |
|---|---|---|---|:---:|:---:|---|
| **`Baseline1DCNN`** | Input Channelization | Multi-channel spectral sub-bands (e.g. 8 sub-bands) or physical receiver channels | Generates artificial 2-channel tensor via `q = np.roll(i, 1)` | ❌ | **CRITICAL** | `src/data/loader.py:345`; circular shift has no quadrature or sub-band physical meaning. |
| **`Compressed Sensing`** | Sampling Projection | Random measurement matrix multiplication $y = \Phi x$ ($CR \in [0.25, 0.75]$) | Applies dynamic range compression ($\mu$-law / $\log$) | ❌ | **CRITICAL** | `src/preprocessing/compression.py:102-133`; zero compressive sensing projection implemented. |
| **`Compressed Sensing`** | Model Identity | Mo et al. (2022) compressed-domain CNN | Instantiates `DSCNN` (Medaiyese 2022) | ❌ | **CRITICAL** | `src/models/model_factory.py:32`; conflates parameter-efficient DSCNN with compressive sensing. |
| **`FGCS2019DNN`** | Output Activation | Sigmoid activation with binary cross-entropy | Outputs raw logits into `nn.CrossEntropyLoss()` | ⚠️ | **LOW** | `src/models/baselines.py:153`; mathematically equivalent and numerically more stable. |
| **All Models** | Cross-Validation | Random segment-level 10-fold cross-validation | Deterministic recording-level stratified split (Seed 42) | ⚠️ | **HIGH** | `src/data/create_splits.py`; deliberate project-wide choice to prevent recording leakage. |

---

## 8. Training & Evaluation Protocol Separation

To maintain strict scientific integrity, all experimental settings are categorized:

- **`[PAPER-SPECIFIED]`**:
  - `FGCS2019DNN` input size (2048), hidden dimensions (256, 128, 64), batch size (10), epochs (200), Adam optimizer.
  - Zero-mean DC removal and magnitude-squared power spectrum $|X[k]|^2$.
- **`[PROJECT-WIDE COMMON PROTOCOL]`**:
  - Recording-level stratified split (Seed 42: 308 Train, 73 Val, 73 Test) to guarantee zero session leakage.
  - Unified `nn.CrossEntropyLoss()` with raw unnormalized logits for all models.
  - Checkpoint tracking via minimum validation loss (`best.pt`).
- **`[IMPLEMENTATION ASSUMPTION]`**:
  - Synthesizing 2-channel pseudo-I/Q via `np.roll(1)` in `src/data/loader.py`.
  - Aliasing dynamic range companding as compressive sensing in `src/preprocessing/compression.py`.
- **`[UNRESOLVED]`**:
  - Whether Mo et al. (2022) intended simultaneous dual-band synchronization ($L + H$) or single-band processing for all experiments.

---

## 9. Answers to Specific Literature Questions (Q1 – Q11)

- **Q1: Is our current FGCS preprocessing actually faithful to the original FGCS paper?**  
  **Yes.** For single-receiver input, the pipeline (DC removal $\rightarrow$ 2048 FFT $\rightarrow$ $|FFT|^2 \rightarrow$ max-normalization $\rightarrow$ 2048 features) and 4-layer MLP exactly match Al-Sa'd et al. (2019).
- **Q2: Is our current 2048-point FFT representation correct?**  
  **Yes.** Al-Sa'd et al. explicitly defines $N = 2048$ samples and an input dimension of 2048 nodes.
- **Q3: Is using $|FFT|^2$ correct for FGCS?**  
  **Yes.** Al-Sa'd et al. specifies the power spectrum (periodogram), which is proportional to $|X[k]|^2$.
- **Q4: Is max normalization correct for FGCS?**  
  **Yes.** Spectral values are normalized by the maximum power component to bound inputs in $[0, 1]$.
- **Q5: Does FGCS require L/H spectral stitching that our current loader does not perform?**  
  **No for FGCS (Al-Sa'd 2019); Yes for Mo et al. (2022).** Al-Sa'd et al. processed $L$ and $H$ segments independently as 2048-dimensional inputs. Mo et al. explicitly concatenated $L$ and $H$ spectra.
- **Q6: Does our current dataset loading provide the required L/H information to reproduce that procedure?**  
  **Yes.** `dronerf_metadata.csv` and split manifests track `receiver` (`H`, `L`, `L1`, `L2`) and `experiment_id` for every file.
- **Q7: Is our current MC1DCNN pseudo-I/Q representation scientifically justified by its paper?**  
  **NO.** Circularly shifting a scalar real signal by 1 sample (`np.roll(1)`) has zero RF quadrature validity.
- **Q8: What exactly are the MC1DCNN channels?**  
  In Allahham et al. (2020), channels represent **frequency sub-bands** (e.g. 8 sub-bands from filter banks) or physical receiver front-ends.
- **Q9: Is our compressed-sensing preprocessing faithful to Mo et al.?**  
  **NO.** The current code performs dynamic range compression (log / $\mu$-law), not compressive random matrix projection ($y = \Phi x$).
- **Q10: Which parts of our current implementation are genuine paper reproductions and which are adaptations?**  
  `FGCS2019DNN` is a genuine reproduction. `Baseline1DCNN` and `Compressed Sensing` are non-faithful adaptations due to pseudo-I/Q and missing matrix projections. The recording-level split is a deliberate project-wide leakage-control adaptation.
- **Q11: Which discrepancies could explain the poor 5-epoch pilot results?**  
  1. *Primary Potential Contributor*: **Phantom frequency distribution shift** (Train: 5.8 GHz only $\rightarrow$ Val/Test: 2.4 GHz only).
  2. *Secondary Potential Contributor*: **Pseudo-I/Q representation** (`np.roll(1)`) introducing artificial correlation in 1D CNNs.
  3. *Tertiary Potential Contributor*: Short training duration (5 epochs vs paper's 200 epochs).

---

## 10. Code Problems & Proposed Corrections

### Issue 1: Pseudo-I/Q Channel Generation in `src/data/loader.py`
- **File**: `src/data/loader.py` (lines 343-346)
- **Current Behavior**: `i_ch = (sig - mean) / std; q_ch = np.roll(i_ch, 1); stack([i_ch, q_ch])`.
- **Paper-Required Behavior**: For 1D CNNs, either pass 1-channel raw 1D waveform `(1, 2048)` or pass 8-channel sub-band channelized spectrum `(8, 256)` via `SpectrumChannelizer`.
- **Scientific Impact**: Eliminates unphysical synthetic quadrature channels.
- **Correction Status**: **Required for paper fidelity.**

### Issue 2: Conflation of Dynamic Range Compression and Compressive Sensing
- **File**: `src/preprocessing/compression.py` & `src/models/model_factory.py`
- **Current Behavior**: `CompressedSensingProcessor` aliases `DynamicRangeCompressor` ($\mu$-law / $\log$).
- **Paper-Required Behavior**: Implement true random matrix projection $y = \Phi x$ ($\Phi \in \mathbb{R}^{M \times N}$) for the compressive sensing baseline.
- **Correction Status**: **Required for paper fidelity if Mo et al. is evaluated.**

### Issue 3: Disambiguation of DSCNN vs Compressive Sensing
- **File**: `src/models/model_factory.py` (lines 32-35)
- **Current Behavior**: Maps `"baseline_cnn"`, `"tinyml"`, and `"dscnn"` to `DSCNN`.
- **Paper-Required Behavior**: Explicitly separate `DSCNN` (Medaiyese 2022 TinyML baseline) from `CS_CNN` (Mo et al. 2022 Compressive Sensing baseline).
- **Correction Status**: **Recommended.**

---

## 11. Final Status Classification

| Model Key | Model Class | Final Status | Justification |
|---|---|:---:|---|
| `fgcs2019dnn` | `FGCS2019DNN` | **`[FAITHFUL ADAPTATION]`** | Preprocessing and architecture match Al-Sa'd et al. (2019) exactly; evaluation adapted to recording-level split. |
| `baseline1dcnn` | `Baseline1DCNN` | **`[NOT PAPER-FAITHFUL]`** | Uses artificial `np.roll(1)` pseudo-I/Q instead of true spectral sub-bands or physical receiver channels. |
| `dscnn` | `DSCNN` | **`[PARTIAL MATCH]`** | Architecture matches Medaiyese et al. (2022) Depthwise Separable CNN, but inherits the pseudo-I/Q input loader artifact. |
| `compressed_sensing` | — | **`[NOT PAPER-FAITHFUL]`** | Compressive matrix projection $y = \Phi x$ is not implemented (only $\log$/$\mu$-law companding exists). |
| `mobilenetv3small` | `MobileNetV3Small` | **`[FAITHFUL ADAPTATION]`** | 2D STFT Spectrogram pipeline and lightweight MobileNetV3 match standard 2D time-frequency benchmark designs. |
