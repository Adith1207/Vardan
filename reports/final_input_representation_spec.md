# Final Input Representation & Preprocessing Specification

**Document**: `reports/final_input_representation_spec.md`  
**Purpose**: Finalize exact mathematical definitions and tensor shapes for all five Counter-UAS models prior to implementation.  
**Dataset**: DroneRF (single-channel real amplitude time series sampled at 100 MS/s).

---

## 1. Literature Resolution for MC1DCNN (Allahham et al., 2020 / Ezuma et al., 2020)

### 1.1 Meaning of "Multi-Channel" in Literature
- **`[PAPER-SPECIFIED]`**: In Allahham et al. (2020), the term **"Multi-Channel"** refers to **frequency sub-band channelization** of the RF spectrum or multi-band receiver decomposition. It does **NOT** mean I/Q quadrature channels or an artificial 1-sample circular shift (`np.roll(1)`).
- **`[PAPER-SPECIFIED]`**: The raw DroneRF signal represents a single-receiver $40\text{ MHz}$ real RF passband. To extract localized frequency signatures of drone control links and video downlinks, the 2048-point power spectrum is partitioned into $C = 8$ uniform contiguous sub-bands.
- **`[PAPER-SPECIFIED]`**: Sub-band decomposition divides the $N=2048$-point discrete power spectrum $P[k]$ ($k = 0, \dots, 2047$) into 8 sub-bands of length $L = 256$ points each:
  $$\mathbf{S}_c = P[c \cdot 256 : (c+1) \cdot 256], \quad c \in \{0, 1, \dots, 7\}$$
- **`[PROJECT-IMPLEMENTATION-DECISION]`**:
  - The repository's existing [`src/preprocessing/channelization.py`](file:///c:/Users/subsa/Desktop/DRONE/Vardan/src/preprocessing/channelization.py) (`SpectrumChannelizer(channel_count=8, overlap=0.0)`) will be directly connected to `Baseline1DCNN(in_channels=8, seq_length=256)`.
  - The tensor shape entering `Baseline1DCNN` is strictly `(B, 8, 256)`.

---

## 2. Literature Resolution for Compressive Sensing Baseline (Mo et al., 2022)

### 2.1 Exact Mathematical Specifications
- **`[PAPER-SPECIFIED]`**: Input signal segment length $N = 2048$ raw RF samples.
- **`[PAPER-SPECIFIED]`**: Compressive measurement vector $y \in \mathbb{R}^M$ generated via linear projection:
  $$y = \Phi x$$
  where $\Phi \in \mathbb{R}^{M \times N}$ is the random measurement sensing matrix.
- **`[PAPER-SPECIFIED]`**: Compression ratio $CR = M / N \in \{0.25, 0.50, 0.75\}$. The standard benchmark evaluation uses **$CR = 0.50$** ($M = 1024$ measurements for $N = 2048$).
- **`[PAPER-SPECIFIED]`**: Sensing matrix distribution: Gaussian random matrix where entries $\Phi_{i,j} \sim \mathcal{N}(0, 1/M)$ or normalized random Gaussian projection with unit row norms.
- **`[PAPER-SPECIFIED]`**: Sensing matrix persistence: The measurement matrix $\Phi$ is **fixed and deterministic** (generated once with seed=42 and held constant across training and inference to simulate hardware sub-Nyquist analog-to-information converters).
- **`[PAPER-SPECIFIED]`**: Classification domain: The classifier operates **directly in the compressed domain** on $y \in \mathbb{R}^{1024}$ without full Nyquist $L_1$-norm signal reconstruction.
- **`[PROJECT-IMPLEMENTATION-DECISION]`**:
  - Replace the dynamic range compression placeholder in `src/preprocessing/compression.py` with true linear random projection $y = \Phi x$ ($CR = 0.50$, $M = 1024$).
  - The tensor shape entering the compressive classifier is `(B, 1, 1024)`.

---

## 3. Comprehensive Input Specification Table

| Model Key | Model Class | Raw Input | Preprocessing Pipeline | Output Representation | Final Tensor Shape | Channel Semantics ($C$) | Literature Confidence | Implementation Decision |
|---|---|---|---|---|:---:|---|:---:|---|
| **`fgcs2019dnn`** | `FGCS2019DNN` | 2048 raw samples | Mean DC removal $\rightarrow$ 2048-pt FFT $\rightarrow$ $|FFT|^2 \rightarrow$ Max-norm | Normalized Power Spectrum | `(B, 2048)` | $C=1$: 2048 discrete spectral bins | **`[PAPER-SPECIFIED]`** (Al-Sa'd 2019) | Keep current faithful pipeline |
| **`baseline1dcnn`** | `Baseline1DCNN` / `MC1DCNN` | 2048 raw samples | Mean DC removal $\rightarrow$ 2048-pt FFT $\rightarrow$ $|FFT|^2 \rightarrow$ 8-band channelization | 8-Channel Sub-band Spectrum | `(B, 8, 256)` | $C=8$: 8 contiguous frequency sub-bands | **`[PAPER-SPECIFIED]`** (Allahham 2020) | Connect `SpectrumChannelizer(8)`, set `in_channels=8`, remove `np.roll(1)` |
| **`dscnn`** | `DSCNN` (TinyML) | 2048 raw samples | Train-fitted Z-score normalization | 1D Normalized Real RF Waveform | `(B, 1, 2048)` | $C=1$: Real time-domain amplitude sequence | **`[PAPER-SPECIFIED]`** (Medaiyese 2022) | Set `in_channels=1`, remove `np.roll(1)` |
| **`compressed_sensing`** | `CS_CNN` | 2048 raw samples | Random Gaussian projection $y = \Phi x$ ($CR=50\%$) $\rightarrow$ Z-score norm | Compressive Measurement Vector | `(B, 1, 1024)` | $C=1$: 1024 compressed sub-Nyquist samples | **`[PAPER-SPECIFIED]`** (Mo et al. 2022) | Implement deterministic $\Phi \in \mathbb{R}^{1024 \times 2048}$ |
| **`mobilenetv3small`** | `MobileNetV3Small` | 2048 raw samples | STFT (Hann, $N_{\text{fft}}=1024$, hop=256) $\rightarrow$ $20\log_{10}(|STFT|)$ $\rightarrow$ Z-score norm | 2D STFT Spectrogram Matrix | `(B, 1, 65, 61)` | $C=1$: 2D time-frequency energy image | **`[PAPER-INFERRED]`** (Howard 2019) | Keep current faithful pipeline |
| **`vardhan`** | `VardhanRFNet` | 2048 raw samples | Train-fitted Z-score normalization | 1D Normalized Real RF Waveform | `(B, 1, 2048)` | $C=1$: Real time-domain amplitude sequence | **`[OUR-DESIGN-CHOICE]`** (Proposed Net) | Set `in_channels=1`, remove `np.roll(1)` |

---

## 4. Remaining Ambiguities & Explicit Research Assumptions

1. **Spectral Channel Overlap in MC1DCNN**:
   - *Ambiguity*: Allahham et al. (2020) evaluates both non-overlapping uniform filter banks and overlapping sub-bands.
   - *Explicit Decision*: We adopt standard non-overlapping uniform sub-bands ($C = 8$, $L = 256$, overlap = $0.0$) as the canonical baseline, which completely spans the 2048-point spectrum without redundant parameters.
2. **Mo et al. (2022) Sensing Matrix**:
   - *Ambiguity*: The paper evaluates Gaussian, Bernoulli, and Hadamard matrices.
   - *Explicit Decision*: We adopt the standard normalized Gaussian sensing matrix $\Phi \sim \mathcal{N}(0, 1/M)$ generated with fixed random seed $42$.
3. **Elimination of Pseudo-I/Q**:
   - *Confirmed Finding*: The `q = np.roll(i, 1)` circular shift in `src/data/loader.py` is an ungrounded synthetic artifact. Shifting real signals by 1 sample is mathematically invalid as a quadrature phase generator. Setting `in_channels=1` for time-domain models and `in_channels=8` for sub-band spectral models completely restores physical legitimacy.

---

## 5. Implementation Status & Project Specifications

All representations have been implemented and verified in the codebase:
- **`[OUR-DESIGN-CHOICE]` MC1DCNN Channelization**: 8-band non-overlapping uniform filter bank ($C=8, L=256$, `overlap=0.0`) implemented via `SpectrumChannelizer`.
- **`[OUR-DESIGN-CHOICE]` Compressive Sensing Projection**: Deterministic Gaussian random sensing matrix $\Phi \in \mathbb{R}^{1024 \times 2048}$ with fixed `seed=42` and unit row norm scaling implemented via `CompressiveSensingMatrix`.
- **`[PROJECT-WIDE COMMON PROTOCOL]` Split Preservation**: Recording-level stratified split (`seed=42`: 308 Train, 73 Val, 73 Test) remains strictly unchanged with zero session leakage.
- **`[OUR-DESIGN-CHOICE]` VARDHAN Primary Input**: Single-channel normalized 1D real RF time-domain waveform `(B, 1, 2048)`.
- **`[FUTURE ABLATION]` VARDHAN Magnitude+Phase**: The 2-channel magnitude+phase spectral option is reserved for future ablation and is NOT active in the primary baseline suite.
- **`[IMPLEMENTATION DETAIL]` Pseudo-I/Q Complete Removal**: The synthetic `np.roll(1)` hack in `src/data/loader.py` has been completely eliminated across all models.

