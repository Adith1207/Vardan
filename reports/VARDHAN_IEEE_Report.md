# VARDHAN: A Lightweight Dual-Domain RF-Based Framework for Drone Detection and Classification

**Author**: Subash Santhanam K.  
**Affiliation**: Department of Computer Science and Engineering  
**Project**: VARDHAN (Versatile Adaptive Radio-frequency Detection and Hierarchical Analysis Network)  
**Date**: September 2026  
**Format**: IEEE Research Paper / Technical Project Report  

---

## Abstract

Radio frequency (RF) fingerprinting provides an effective, passive, non-line-of-sight modality for counter-unmanned aerial systems (counter-UAS) and drone surveillance. While deep learning models evaluated on the public DroneRF dataset report high classification accuracy under randomized segment-level cross-validation, their ability to generalize to unseen recording sessions remains a critical open question. In this paper, we present **VARDHAN** (**V**ersatile **A**daptive **R**adio-frequency **D**etection and **H**ierarchical **A**nalysis **N**etwork), a lightweight dual-domain architecture designed for resource-constrained edge sensing. VARDHAN-v3 processes raw 2048-sample RF waveform segments by concurrently extracting multi-scale temporal dynamics via parallel depthwise dilated convolutions ($k \in \{3, 7, 15, 31\}$, dilation $d \in \{1, 2, 4, 8\}$) and a three-channel complex Fourier representation comprising log-power spectrum, normalized real Fourier coefficients, and normalized imaginary Fourier coefficients. Branch representations are enhanced with Squeeze-and-Excitation channel attention, aggregated via learnable self-attention pooling, and interactively fused using bi-directional cross-domain gating with only **119,806** trainable parameters (11.81 M MACs).

To rigorously evaluate cross-recording robustness, we evaluate baseline architectures under both a standardized 10-fold segment-level benchmark (22,700 segments) and a strict recording-level partition (454 files across 23 sessions with zero session leakage). Under standardized segment-level evaluation, our faithful reproduction of FGCS2019DNN achieves $84.52\% \pm 0.73\%$ accuracy (macro-F1: $0.7899 \pm 0.0111$), MC1DCNN achieves $65.01\% \pm 5.35\%$ (macro-F1: $0.6207 \pm 0.0533$), and VARDHAN-v3 achieves $71.57\% \pm 1.25\%$ accuracy (macro-F1: $0.6953 \pm 0.0168$, balanced accuracy: $67.67\% \pm 2.03\%$). Conversely, under strict recording-level evaluation, accuracy across evaluated baselines drops to $26.30\%\text{--}28.77\%$ in historical benchmarks ($N_{\text{samp}}=5$) and $24.79\%$ for VARDHAN-v3 ($N_{\text{samp}}=50$, macro-F1: $0.2005$). The substantial gap between segment-level and recording-level performance is consistent with strong intra-recording correlation and recording-specific domain variation, highlighting the importance of recording-level evaluation and providing a reproducible benchmark for future domain-robust counter-UAS systems.

**Index Terms**—Unmanned aerial vehicles (UAVs), counter-UAS, radio frequency (RF) fingerprinting, dual-domain deep learning, temporal convolutional networks, complex spectral features, DroneRF dataset, recording leakage, edge computing.

---

## I. Introduction

The proliferation of low-altitude civilian and commercial Unmanned Aerial Vehicles (UAVs, commonly termed drones) has accelerated across various domains, including precision agriculture, infrastructure inspection, geographic surveying, and emergency disaster relief [1], [2]. However, the widespread accessibility of low-cost commercial drones poses escalating security, privacy, and safety hazards. Unauthorized drone incursions have disrupted commercial airport operations, endangered critical infrastructure, facilitated contraband smuggling across borders, and created severe airspace collision risks [3], [4]. Consequently, developing automated, reliable, and energy-efficient counter-UAS (C-UAS) detection and classification technologies is imperative for civilian and military airspace security.

Traditional drone surveillance modalities—including primary radar, electro-optical/infrared (EO/IR) cameras, and acoustic sensor arrays—exhibit fundamental physical constraints in complex operational environments [1], [3]. Optical and infrared cameras require unobstructed line-of-sight (LoS), adequate ambient lighting, and high-resolution optics, rendering them ineffective in fog, heavy precipitation, direct solar glare, or low-light night conditions [5]. Acoustic sensing is hindered by severe attenuation with distance, urban ambient acoustic noise, and aerodynamic interference from wind or vehicular traffic [6]. Active radar systems suffer from small radar cross-sections (RCS) of micro-drones, severe multi-path ground clutter, high equipment costs, and active RF emission footprints that reveal sensor locations [3].

In contrast, passive Radio Frequency (RF) sensing provides a resilient surveillance modality. Consumer UAVs rely on wireless bidirectional RF communication links between the unmanned aircraft and its Ground Control Station (GCS) or remote controller to transmit telemetric data, command-and-control (C2) packets, and real-time video downlinks (predominantly within the 2.4 GHz and 5.8 GHz industrial, scientific, and medical (ISM) bands) [2], [7]. Intercepting these passive RF emissions enables non-line-of-sight detection over long operational ranges regardless of visual obscurity, weather, or acoustic clutter, and does not require the sensing platform to intentionally transmit RF energy.

To facilitate data-driven RF classification research, Al-Sa'd et al. [2], [7] released the public **DroneRF** dataset, containing raw high-rate RF waveform recordings from three commercial UAV models (Parrot Bebop, Parrot AR Drone 2.0, DJI Phantom 3) and background RF environments across 10 flight states. Subsequent studies [2], [8]–[13] have applied Deep Neural Networks (DNN), 1D/2D Convolutional Neural Networks (CNN), and XGBoost to DroneRF, reporting impressive classification accuracies exceeding $84\%\text{--}99\%$.

However, a critical methodological limitation underlies much of the published literature: *segment-level cross-validation protocols*. In segment-level evaluation, contiguous temporal recordings are partitioned into thousands of sub-segments, and segments originating from the same physical recording session are distributed across both training and test folds. Because consecutive RF segments within a single physical session may share transmitter-, receiver-, propagation-, and environment-specific characteristics, deep networks can achieve high test accuracy by interpolating session-specific stationary signatures rather than learning invariant drone-specific modulation features. When models are subjected to *strict recording-level separation*—where test segments originate entirely from unseen recording sessions—classification performance degrades substantially.

To address these challenges, this project investigates the reproducibility of published RF baselines, analyzes the cross-recording generalization gap, and proposes **VARDHAN-v3** (**V**ersatile **A**daptive **R**adio-frequency **D**etection and **H**ierarchical **A**nalysis **N**etwork), a lightweight dual-domain architecture. VARDHAN-v3 combines multi-scale temporal convolutions ($k \in \{3, 7, 15, 31\}$) with a 3-channel complex Fourier representation preserving both phase components ($\text{Re}(X)$, $\text{Im}(X)$) and log-power spectrum.

```
+-----------------------------------------------------------------------------+
|                               VARDHAN SYSTEM                                |
|                                                                             |
|  +------------------+     +-------------------+     +--------------------+  |
|  |  USRP Receiver   | --> | 2048-Sample Slice | --> | Dual Preprocessing |  |
|  | 100 MS/s Capture |     | Detrend & Norm    |     | Waveform & rFFT    |  |
|  +------------------+     +-------------------+     +--------------------+  |
|                                                               |             |
|                  +--------------------------------------------+             |
|                  |                                            |             |
|                  v                                            v             |
|        +-------------------+                        +-------------------+   |
|        | Temporal MS-TCN   |                        | 3-Ch Spectral Net |   |
|        | k in {3,7,15,31}  |                        | k in {5,11}       |   |
|        | Dilated Residuals |                        | [LogP; Re; Im]    |   |
|        +-------------------+                        +-------------------+   |
|                  |                                            |             |
|                  v                                            v             |
|        +-------------------+                        +-------------------+   |
|        | Attention Pool hT |                        | Attention Pool hF |   |
|        +-------------------+                        +-------------------+   |
|                  |                                            |             |
|                  +---------------------+----------------------+             |
|                                        |                                    |
|                                        v                                    |
|                        +-------------------------------+                    |
|                        | Cross-Domain Gated Fusion     |                    |
|                        | gT = sig(hF), gF = sig(hT)    |                    |
|                        | Concatenated 320-D Embedding  |                    |
|                        +-------------------------------+                    |
|                                        |                                    |
|                                        v                                    |
|                        +-------------------------------+                    |
|                        | Classification Head (4 Class) |                    |
|                        | Background / Bebop / AR / DJI |                    |
|                        +-------------------------------+                    |
+-----------------------------------------------------------------------------+
```

### Core Contributions
1. **Faithful Baseline Reproduction & Benchmarking**: We faithfully reproduce the available FGCS2019DNN [2] implementation and construct a standardized MC1DCNN [8] baseline from its reported architectural details alongside VARDHAN-v3 on a standardized 22,700-segment DroneRF 10-fold benchmark, and benchmark additional baseline models (DSCNN [9], Compressive Sensing CNN [10], MobileNet-style baseline) under a strict recording-level protocol.
2. **Strict Recording-Level Protocol**: We formulate an open-source, leak-free recording-level evaluation split (454 files, 23 sessions: 308 train / 73 val / 73 test) with zero recording or file overlap.
3. **Lightweight Dual-Domain Architecture (VARDHAN-v3)**: We design a dual-branch neural network combining multi-scale dilated temporal convolutions, 3-channel complex Fourier representations, Squeeze-and-Excitation attention, attention pooling, and bi-directional cross-domain gating with exactly **119,806** parameters and 11.81 M theoretical MACs.
4. **Systematic Generalization Gap Quantification**: We present a comparative study contrasting segment-level cross-validation ($71\%\text{--}85\%$ accuracy) against unseen recording evaluation ($24\%\text{--}29\%$ accuracy), demonstrating the cross-session domain shift in RF fingerprinting.
5. **Signal Normalization Analysis**: We investigate the mathematical mechanism of global fold Z-score normalization versus per-segment normalization, identifying how high-amplitude transmitters can introduce scale disparities between classes.

---

## II. Related Work

Passive RF drone detection has emerged as a major focus within wireless security and signal processing. Table 1 summarizes primary published studies on the DroneRF dataset.

### Table 1: Summary of Representative RF-Based UAV Detection Literature on DroneRF

| Reference | Published Model | Input Representation | Preprocessing Pipeline | Reported Performance | Reproduction / Benchmark Status |
|---|---|---|---|---|---|
| **Al-Sa'd et al. (2019)** [2] | 4-Layer MLP (DNN) | 2048-pt stitched power spectrum | DC removal, 2048-pt FFT, $\|X[k]\|^2$, max-norm, MSE loss | 2-class: 99.7%<br>4-class: 84.5%<br>10-class: 46.8% | **Faithfully Reproduced (Code-Mode, 295.8k params)**:<br>$84.52\% \pm 0.73\%$ (4-class, 10-fold) |
| **Allahham et al. (2020)** [8] | Multi-Channel 1D CNN | 8 sub-bands $\times$ 256 DFT bins | 2048 DFT bins split into 8 non-overlapping 10 MHz sub-bands | 2-class: 100%<br>4-class: 94.6%<br>10-class: 87.4% | **Standardized Benchmark (275.9k params)**:<br>$65.01\% \pm 5.35\%$ (4-class, 10-fold) |
| **Mo et al. (2022)** [10] | Multi-stage DNN/CNN with CS | Compressively sensed RF ($CR \in \{0.25, 0.5, 0.75\}$) | Bernoulli MCRD sampling matrix $y = \Phi x$, stitched power spectra | 2-class: 100%<br>4-class: 99.6%<br>10-class: 99.3% | Implemented and evaluated under historical strict benchmark (1,059,908 params) |
| **Medaiyese et al. (2021)** [9] | XGBoost / DSCNN | Low-frequency RF spectrum / 1D waveform | Discrete wavelet transform and energy transient extraction | 2-class: 99.9%<br>4-class: 90.73%<br>10-class: 70.09% | Evaluated under historical strict benchmark (68,228 params) |
| **Debas et al. (2024)** [1] | Comprehensive Forensics Survey | Multi-modal forensics | Review of drone forensic models (CCAFM, DRFRF, ALFA, DroneRF) | Survey synthesis | Benchmark framing \& standardized metrics |

### A. Literature Results vs. Project Reproductions
It is essential to formally distinguish between *published literature results* and *our standardized reproductions*. Published results in [2], [8], [10] evaluated performance using randomized segment splits where segments from identical recording files were intermixed. In our faithful reproduction of FGCS2019DNN under a standardized 10-fold cross-validation protocol over all 22,700 segments of DroneRF, the model achieves $84.52\% \pm 0.73\%$ accuracy, closely matching the original paper's reported $84.5\%$. However, under our historical strict recording-level benchmark, the FGCS-based baseline (565,956 parameters) achieved $28.22\%$ accuracy.

---

## III. DroneRF Dataset and Protocol Topologies

### A. Dataset Composition and Hardware Configuration
The DroneRF dataset [7] was collected using two synchronized National Instruments USRP-2943R software-defined radio (SDR) receivers. Each receiver recorded real-valued voltage amplitudes at a sampling rate of $F_s = 100\text{ MS/s}$ with 14-bit ADC resolution across a 40 MHz instantaneous bandwidth. Both receivers were configured to cover the 2.4 GHz WiFi spectrum (2.40--2.48 GHz): Receiver 1 captured the lower band ($L$, center frequency 2.422 GHz, spanning 2.40--2.44 GHz), while Receiver 2 recorded the upper band ($H$, center frequency 2.462 GHz, spanning 2.44--2.48 GHz), together covering an 80 MHz contiguous bandwidth spanning WiFi channels 1 through 13.

The dataset consists of 454 individual CSV files representing 227 synchronized $(L, H)$ recording pairs across 23 discrete recording sessions. Each raw file contains 10,000,000 real voltage samples. The recordings represent four major RF classes:
1. **Background RF Activities** (Label 0): Ambient RF noise, commercial WiFi traffic, and Bluetooth activity in the absence of UAVs (41 pairs, 82 files).
2. **Parrot Bebop Drone** (Label 1): Bebop quadcopter operating across four flight modes: connected/on, hovering, flying without video, and flying with video streaming (84 pairs, 168 files).
3. **Parrot AR Drone 2.0** (Label 2): AR Drone quadcopter operating across the same four flight modes (81 pairs, 162 files).
4. **DJI Phantom 3 Standard** (Label 3): Phantom 3 quadcopter operating in connected/on mode (21 pairs, 42 files).

### Table 2: DroneRF Dataset Organization and Class Statistics

| Class Name | Label | CSV Files | Synchronized Pairs | 2048-Sample Segments ($N=2048$) | Class Ratio (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| Background RF Activities | 0 | 82 | 41 | 4,100 | 18.06% |
| Parrot Bebop Drone | 1 | 168 | 84 | 8,400 | 37.00% |
| Parrot AR Drone 2.0 | 2 | 162 | 81 | 8,100 | 35.68% |
| DJI Phantom 3 Standard | 3 | 42 | 21 | 2,100 | 9.25% |
| **Total** | — | **454** | **227** | **22,700** | **100.00%** |

### B. Binary Unique Identifier (BUI) Labeling
Each recording is cataloged with a 5-bit Binary Unique Identifier $\text{BUI} = [b_4 b_3 b_2 b_1 b_0]$:
$$\text{BUI} = [\text{msBUI}, \text{lsBUI}], \tag{1}$$
where $\text{msBUI} \in \{000, 100, 101, 110\}$ denotes the experiment and drone type (Background, Bebop, AR, Phantom), and $\text{lsBUI} \in \{00, 01, 10, 11\}$ denotes the operational mode (on/connected, hovering, flying, video recording). Background activity is assigned $\text{BUI} = 00000$.

### C. Evaluation Protocol Topologies
1. **Standardized 10-Fold Segment-Level Protocol**: Each of the 227 synchronized recording pairs (containing 10,000,000 samples per channel) is divided into 100 non-overlapping contiguous blocks of 100,000 samples each, yielding 22,700 source blocks. For waveform-based models (VARDHAN-v3, Baseline1DCNN), a 2048-sample discrete-time segment ($N=2048$, corresponding to $20.48\,\mu\text{s}$ at $100\,\text{MS/s}$) is extracted from each 100,000-sample block. For the faithful FGCS reproduction, the full 100,000-sample block undergoes DC mean detrending prior to 2048-point DFT computation and $Q=10$ sub-band boundary concatenation. Evaluation is performed via 10-fold Stratified K-Fold cross-validation ($\text{shuffle}=\text{True}, \text{seed}=1$), with 20,430 training examples and 2,270 testing examples per fold.
2. **Strict Recording-Level Split (Zero Leakage)**: The 454 files (23 discrete sessions) are partitioned into:
   - **Training Set**: 308 files across 15 recording sessions (Bebop: 126, AR: 120, Background: 41, Phantom: 21).
   - **Validation Set**: 73 files across 4 recording sessions (Bebop: 21, AR: 21, Background: 21, Phantom: 10).
   - **Testing Set**: 73 files across 4 recording sessions (Bebop: 21, AR: 21, Background: 20, Phantom: 11).
   
   This partitioning enforces *zero recording overlap* and *zero file overlap* between training and evaluation splits. Notably, all 21 training files for the Phantom drone were recorded on the upper receiver $H$ (2.462 GHz), whereas validation and test Phantom files were recorded on lower receivers $L/L1/L2$ (2.422 GHz), introducing a cross-channel frequency shift domain gap.

---

## IV. Data Processing and Signal Formulations

### A. Generic RF Waveform Segmentation
Let a captured discrete-time real-valued RF waveform be represented as a finite-length vector $x \in \mathbb{R}^N$ with segment length $N = 2048$ samples:
$$x = [x[0], x[1], \dots, x[N-1]]^T. \tag{2}$$

### B. Per-Segment Mean Detrending
To eliminate zero-frequency DC bias and low-frequency receiver drift, sample mean detrending is applied:
$$\mu_x = \frac{1}{N} \sum_{n=0}^{N-1} x[n], \tag{3}$$
$$x_d[n] = x[n] - \mu_x, \quad \forall n \in \{0, 1, \dots, N-1\}. \tag{4}$$

### C. Per-Segment Waveform Normalization
For amplitude-invariant representation across disparate SDR gains and propagation distances, each segment is independently standardized:
$$\sigma_x = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} (x_d[n])^2 + \epsilon}, \tag{5}$$
$$x_{\text{norm}}[n] = \frac{x_d[n]}{\sigma_x}, \tag{6}$$
where $\epsilon = 10^{-8}$ prevents division by zero in near-silent segments.

### D. Complex Fourier Transform and DC Removal
The discrete Fourier transform of the real detrended sequence $x_{\text{norm}}$ is computed via the Real Fast Fourier Transform (rFFT):
$$X[k] = \sum_{n=0}^{N-1} x_{\text{norm}}[n] \exp\left(-j \frac{2\pi k n}{N}\right), \quad k \in \{0, 1, \dots, N/2\}. \tag{7}$$
For $N=2048$, the rFFT yields 1025 complex frequency bins. The DC component at $k=0$ is discarded, leaving $K = 1024$ positive-frequency bins ($k = 1, \dots, 1024$).

### E. VARDHAN Three-Channel Spectral Representation
Standard RF classifiers utilize only the magnitude-squared power spectrum $|X[k]|^2$, discarding complex phase information. In VARDHAN-v3, we construct a 3-channel spectral tensor $S \in \mathbb{R}^{3 \times 1024}$:
$$S = \begin{bmatrix} P[k] \\ R[k] \\ I[k] \end{bmatrix} \in \mathbb{R}^{3 \times 1024}, \tag{8}$$
defined bin-wise for $k \in \{1, \dots, 1024\}$ by:
$$P[k] = \ln\left(1 + \alpha \frac{|X[k]|^2}{N}\right), \quad (\alpha = 1000), \tag{9}$$
$$R[k] = \frac{\text{Re}(X[k])}{\sqrt{N}}, \tag{10}$$
$$I[k] = \frac{\text{Im}(X[k])}{\sqrt{N}}. \tag{11}$$
The logarithmic scaling in (9) compresses high dynamic range transmitter spikes, while the scaled real and imaginary components in (10)–(11) preserve frequency-domain phase information.

---

## V. Reproduction of Existing Baseline Models

Five baseline architectures were implemented in addition to the VARDHAN models to establish comparative performance benchmarks.

### A. FGCS2019DNN (Al-Sa'd et al., 2019)
- **Input**: 2048-point power spectrum stitched from $L$ and $H$ bands via [2]:
  $$y_i = [y_i^{(L)}, c \cdot y_i^{(H)}], \quad c = \frac{\sum_{q=0}^{Q-1} y_i^{(L)}(M-q)}{\sum_{q=0}^{Q-1} y_i^{(H)}(q)}, \tag{12}$$
  with $Q=10$ stitching boundary points.
- **Architecture Configurations**:
  - *Code-Mode Architecture* (author-released in `Classification.py`): Dense($2048 \rightarrow 128$) $\rightarrow$ ReLU $\rightarrow$ Dense($128 \rightarrow 128$) $\rightarrow$ ReLU $\rightarrow$ Dense($128 \rightarrow 128$) $\rightarrow$ ReLU $\rightarrow$ Dense($128 \rightarrow 4$) $\rightarrow$ Sigmoid (**295,812 parameters**). Used in the faithful standardized 10-fold reproduction.
  - *Paper-Mode Architecture* (described in FGCS 2019 paper text): Dense($2048 \rightarrow 256$) $\rightarrow$ ReLU $\rightarrow$ Dense($256 \rightarrow 128$) $\rightarrow$ ReLU $\rightarrow$ Dense($128 \rightarrow 64$) $\rightarrow$ ReLU $\rightarrow$ Dense($64 \rightarrow 4$) $\rightarrow$ Sigmoid (**565,956 parameters**). Evaluated under the historical strict benchmark.
- **Loss Function**: Mean Squared Error (MSE) on one-hot targets:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{C} \sum_{c=1}^C (y_c - \hat{y}_c)^2. \tag{13}$$
- **Optimizer**: Adam ($\text{lr} = 0.001$), batch size 10, 200 epochs.

### B. MC1DCNN (Allahham et al., 2020)
- **Input**: 8 sub-band channels $\times$ 256 frequency bins ($8 \times 256$).
- **Architecture**: Conv1D($8 \rightarrow 32, k=11, s=2, p=5$) $\rightarrow$ ReLU $\rightarrow$ MaxPool1D(2) $\rightarrow$ Conv1D($32 \rightarrow 64, k=5, s=1, p=2$) $\rightarrow$ ReLU $\rightarrow$ MaxPool1D(2) $\rightarrow$ Flatten(2048) $\rightarrow$ Dense($2048 \rightarrow 128$) $\rightarrow$ ReLU $\rightarrow$ Dense($128 \rightarrow 4$). Total parameters: **275,940**.
- **Loss \& Optimizer**: Cross-Entropy Loss, Adam ($\text{lr} = 0.001$), batch size 32, 100 epochs.

### C. DSCNN (Depthwise Separable CNN)
- **Formulation**: Depthwise convolution followed by $1 \times 1$ pointwise projection [9]:
  $$y_c = w_c \ast x_c, \quad y = W_{1\times 1} \ast y_{\text{dw}}. \tag{14}$$
- **Parameters**: 68,228 parameters. Input: raw 1D RF waveform ($1 \times 2048$).

### D. Compressive Sensing CNN (Mo et al., 2022)
- **Sampling**: Random Bernoulli projection matrix $\Phi \in \mathbb{R}^{M \times N}$ at $CR = M/N = 0.5$ ($M=1024$ measurements):
  $$y = \Phi x. \tag{15}$$
- **Parameters**: 1,059,908 parameters.

### E. MobileNet-Style Spectrogram Baseline
- **Input**: 2D Short-Time Fourier Transform (STFT) log-spectrogram tensor ($1 \times 128 \times 32$).
- **Parameters**: 11,588 parameters with depthwise separable 2D convolutions.

---

## VI. Proposed VARDHAN Framework

The VARDHAN-v3 architecture comprises two parallel domain-specific feature extractors, attention pooling mechanisms, and a bi-directional cross-domain gated fusion network.

### A. Multi-Scale Temporal Backbone (MS-TCN)
The temporal branch accepts the detrended, normalized waveform $x_{\text{norm}} \in \mathbb{R}^{B \times 1 \times 2048}$.
1. **Temporal Stem**: Conv1D($1 \rightarrow 32, k=15, s=2, p=7$) + BatchNorm1D(32) + GELU activation, downsampling to $(B, 32, 1024)$.
2. **Multi-Scale Temporal Blocks (MSTB)**: Four cascaded residual blocks split input channels $C_{\text{in}}$ across 4 parallel depthwise convolutional branches with kernel sizes $k \in \{3, 7, 15, 31\}$ and dilation factors $d \in \{1, 2, 4, 8\}$:
   $$\text{out}_{\text{dw}} = \left[ \text{DW}_3(x_1) \,\|\, \text{DW}_7(x_2) \,\|\, \text{DW}_{15}(x_3) \,\|\, \text{DW}_{31}(x_4) \right], \tag{16}$$
   where $x_i \in \mathbb{R}^{B \times (C_{\text{in}}/4) \times T}$. The concatenated output is projected via a $1 \times 1$ pointwise convolution, normalized by BatchNorm, activated via GELU, refined by Squeeze-and-Excitation (SE), and summed with a residual shortcut:
   $$y_{\text{MSTB}} = \text{SE}\left(\text{GELU}\left(\text{BN}\left(W_{\text{pw}} \ast \text{out}_{\text{dw}}\right)\right)\right) + \text{Res}(x). \tag{17}$$
   Channel progression: $32 \rightarrow 48 \rightarrow 64 \rightarrow 80 \rightarrow 80$ at sequence lengths $512 \rightarrow 256 \rightarrow 128 \rightarrow 128$.

### B. Multi-Scale Dual Spectral Backbone
The spectral branch operates on the 3-channel Fourier tensor $S \in \mathbb{R}^{B \times 3 \times 1024}$ from (8).
1. **Spectral Stem**: Conv1D($3 \rightarrow 32, k=11, s=2, p=5$) + BatchNorm1D(32) + GELU, downsampling to $(B, 32, 512)$.
2. **Multi-Scale Spectral Blocks (MSSB)**: Four cascaded blocks split channels across 2 parallel depthwise branches ($k \in \{5, 11\}$) with dilation schedule $d \in \{1, 2, 4, 8\}$ and SE refinement. Channel progression: $32 \rightarrow 48 \rightarrow 64 \rightarrow 80 \rightarrow 80$.

### C. Squeeze-and-Excitation (SE) Channel Attention
For an intermediate feature map $U \in \mathbb{R}^{B \times C \times L}$, SE [14] calculates channel descriptors via global average pooling:
$$z_c = \frac{1}{L} \sum_{i=1}^L U_c[i], \quad z \in \mathbb{R}^C. \tag{18}$$
Channel scaling vector $s \in \mathbb{R}^C$ is computed via a bottleneck projection ($r = 4$):
$$s = \sigma\left(W_2 \cdot \text{ReLU}\left(W_1 z + b_1\right) + b_2\right), \tag{19}$$
and features are scaled channel-wise: $\tilde{U}_c = s_c \cdot U_c$.

### D. Learnable Self-Attention Pooling
Rather than using static global average pooling, sequence features $H \in \mathbb{R}^{B \times C \times L}$ are aggregated using learnable attention weights $a \in \mathbb{R}^{B \times 1 \times L}$:
$$a = \text{Softmax}\left(W_{\text{att2}} \cdot \tanh\left(W_{\text{att1}} H + b_1\right) + b_2\right), \tag{20}$$
producing 80-dimensional branch summary vectors:
$$h_T = \sum_{l=1}^L a_T[l] \cdot H_T[l] \in \mathbb{R}^{B \times 80}, \quad h_F = \sum_{l=1}^L a_F[l] \cdot H_F[l] \in \mathbb{R}^{B \times 80}. \tag{21}$$

### E. Bi-Directional Cross-Domain Gated Fusion
To enable inter-domain feature modulation, the temporal feature $h_T$ is gated by spectral context $h_F$, and the spectral feature $h_F$ is gated by temporal context $h_T$:
$$g_T = \sigma\left(W_{gT2} \cdot \text{GELU}(W_{gT1} h_F + b_{gT1}) + b_{gT2}\right) \in \mathbb{R}^{B \times 80}, \tag{22}$$
$$g_F = \sigma\left(W_{gF2} \cdot \text{GELU}(W_{gF1} h_T + b_{gF1}) + b_{gF2}\right) \in \mathbb{R}^{B \times 80}. \tag{23}$$
Modulated representations are computed via Hadamard product:
$$h_T' = h_T \odot g_T, \quad h_F' = h_F \odot g_F. \tag{24}$$
The original and modulated vectors are concatenated into a 320-dimensional representation:
$$h_{\text{cat}} = [h_T \,\|\, h_F \,\|\, h_T' \,\|\, h_F'] \in \mathbb{R}^{B \times 320}. \tag{25}$$

### F. Classification Head
The classifier head maps $h_{\text{cat}}$ to class logits $\hat{y} \in \mathbb{R}^{B \times 4}$:
$$z_1 = \text{Dropout}_{0.2}\left(\text{GELU}\left(\text{BN}\left(W_{\text{fc1}} h_{\text{cat}} + b_{\text{fc1}}\right)\right)\right) \in \mathbb{R}^{B \times 64}, \tag{26}$$
$$\hat{y} = W_{\text{fc2}} z_1 + b_{\text{fc2}} \in \mathbb{R}^{B \times 4}. \tag{27}$$
During training, cross-entropy with label smoothing ($\alpha_{\text{ls}} = 0.05$) is minimized:
$$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^C \left[(1 - \alpha_{\text{ls}}) y_c + \frac{\alpha_{\text{ls}}}{C}\right] \ln\left(\frac{\exp(\hat{y}_c)}{\sum_{j=1}^C \exp(\hat{y}_j)}\right). \tag{28}$$

### G. Theoretical Receptive Field and Computational Complexity
The theoretical receptive field $R_l$ and feature jump $j_l$ through layer $l$ with kernel $k_l$, stride $s_l$, and dilation $d_l$ follow the recurrence:
$$R_l = R_{l-1} + (k_l - 1) d_l j_{l-1}, \quad j_l = j_{l-1} s_l. \tag{29}$$
For the temporal branch ($R_0=1, j_0=1$), the multi-scale branches yield a theoretical receptive field spanning **355 to 5,115 samples** ($3.55\,\mu\text{s}$ to $51.15\,\mu\text{s}$), fully covering the 2048-sample segment. The spectral branch achieves a receptive field spanning **435 to 1,071 frequency bins**.

VARDHAN-v3 contains exactly **119,806 trainable parameters**. The theoretical computational cost per 2048-sample segment is:
- **Multiply-Accumulate Operations (MACs)**: 11.8144 M MACs.
- **Floating-Point Operations (FLOPs)**: 23.6288 M FLOPs.

---

## VII. Experimental Setup

### A. Hardware and Execution Environment
Experiments were conducted using Python and PyTorch deep learning environments accelerated by NVIDIA Tesla T4 GPU hardware (16 GB VRAM).

### B. Model-Specific Training Configurations
To ensure complete reproducibility, Table 3 lists the verified model-specific training hyperparameter configurations across all standardized benchmark runs.

### Table 3: Model-Specific Experimental Configurations for Standardized 10-Fold Benchmark

| Parameter | FGCS2019DNN | MC1DCNN | VARDHAN-v3 |
|---|:---:|:---:|:---:|
| Architecture Mode | Code (128-128-128) | 8 Sub-bands | Dual-Domain TCN/FFT |
| Trainable Parameters | 295,812 | 275,940 | 119,806 |
| Optimizer | Adam | Adam | AdamW |
| Initial Learning Rate | $0.001$ | $0.001$ | $0.0003$ |
| Weight Decay | $0.0$ | $0.0$ | $10^{-4}$ |
| LR Scheduler | None (constant) | None (constant) | Cosine Annealing |
| Minimum Learning Rate | N/A | N/A | $10^{-6}$ |
| Batch Size | 10 | 32 | 32 |
| Epochs per Fold | 200 | 100 | 15 |
| Loss Function | MSELoss on one-hot | CrossEntropyLoss | CrossEntropy (Label Smooth 0.05) |
| Output Activation | Sigmoid | Linear (Logits) | Linear (Logits) |
| Normalization | Per-BUI Max Scaling | Per-BUI Max Scaling | Per-Segment Independent |
| Cross-Validation | 10-Fold Stratified | 10-Fold Stratified | 10-Fold Stratified |
| Random Seed | 1 | 1 | 1 |

*Note on Computational Constraint*: Due to computational-resource and execution-time constraints on cloud GPU instances, the full standardized 10-fold segment-level benchmark was completed only for FGCS2019DNN, MC1DCNN, and VARDHAN-v3.

### C. Strict Recording-Level Benchmark Configuration
Strict recording-level evaluation used:
- **Historical Baseline Suite**: $N_{\text{samp}} = 5$ segments per file (365 test segments), 5 epochs, Cross-Entropy Loss, seed 42.
- **Final VARDHAN-v3 Experiment**: $N_{\text{samp}} = 50$ segments per file (3,650 test segments: 1000 Background, 1050 Bebop, 1050 AR, 550 Phantom), 100 epochs, AdamW ($\text{lr}=0.0003$, $\text{min\_lr}=10^{-6}$), Cosine Annealing, per-segment normalization, seed 42. Checkpoint selection based on minimum validation loss.

---

## VIII. Results

### A. Standardized Segment-Level Benchmark Results
Table 4 reports aggregate 10-fold cross-validation results across the 22,700 segments of DroneRF.

### Table 4: Standardized 10-Fold Segment-Level Benchmark Performance ($N=22,700$)

| Model | Architecture Type | Trainable Params | Accuracy (%) | Macro-F1 Score | Balanced Accuracy (%) | Test Loss |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **FGCS2019DNN** [2] | 4-Layer MLP | 295,812 | $\mathbf{84.52 \pm 0.73}$ | $\mathbf{0.7899 \pm 0.0111}$ | N/A | N/A |
| **MC1DCNN** [8] | 1D CNN (8 Sub-bands) | 275,940 | $65.01 \pm 5.35$ | $0.6207 \pm 0.0533$ | N/A | N/A |
| **VARDHAN-v3 (Ours)** | Dual-Domain MS-TCN/FFT | **119,806** | $71.57 \pm 1.25$ | $0.6953 \pm 0.0168$ | $\mathbf{67.67 \pm 2.03}$ | $0.7737 \pm 0.0230$ |

FGCS2019DNN achieves the highest segment-level accuracy ($84.52\% \pm 0.73\%$), closely matching the published literature. VARDHAN-v3 achieves $71.57\% \pm 1.25\%$ accuracy and $0.6953 \pm 0.0168$ macro-F1 while requiring **59.5% fewer parameters** than FGCS2019DNN and **56.6% fewer parameters** than MC1DCNN ($65.01\% \pm 5.35\%$).

### B. Historical Strict Recording-Level Results ($N_{\text{samp}}=5$)
Table 5 presents the historical baseline benchmark on the strict recording-level split ($N_{\text{samp}}=5$, seed 42).

### Table 5: Historical Strict Recording-Level Benchmark ($N_{\text{samp}}=5$, Seed 42)

| Model | Input Domain | Parameters | Test Acc. (%) | Macro-F1 | Precision | Recall |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **FGCS2019DNN** [2] | Power Spectrum | 565,956 | 28.22% | 0.1100 | 0.0706 | 0.2500 |
| **Baseline1DCNN** [8] | 8 Sub-Bands | 275,940 | 28.77% | 0.1117 | 0.0719 | 0.2500 |
| **DSCNN** [9] | 1D Waveform | 68,228 | 26.58% | 0.1520 | 0.1800 | 0.2312 |
| **CS-CNN** [10] | CS Measurements | 1,059,908 | 28.77% | 0.1117 | 0.0719 | 0.2500 |
| **MobileNet-Style** | 2D Spectrogram | 11,588 | 28.77% | 0.1117 | 0.0719 | 0.2500 |
| **VARDHAN (v1)** | 1D Waveform | 6,692 | 26.30% | 0.1152 | 0.0740 | 0.2500 |

All models degrade to $26.30\%\text{--}28.77\%$ test accuracy, near the $25\%$ uniform-random four-class reference level. Macro-F1 scores ($0.1100\text{--}0.1520$) indicate that models collapse into predicting the majority class (Bebop).

### C. Final VARDHAN-v3 Strict Recording-Level Results ($N_{\text{samp}}=50$)
Table 6 details the final VARDHAN-v3 strict experiment ($N_{\text{samp}}=50$, 3,650 test segments, 100 epochs, per-segment normalization).

### Table 6: Final VARDHAN-v3 Strict Recording-Level Metrics ($N_{\text{samp}}=50$)

| Class Name | Precision (%) | Recall (%) | F1-Score (0–1) | Support |
|---|:---:|:---:|:---:|:---:|
| Background (No UAV) | 61.18% | 14.50% | 0.2344 | 1,000 |
| Parrot AR Drone | 25.28% | 19.52% | 0.2203 | 1,050 |
| Parrot Bebop Drone | 20.90% | 51.52% | 0.2974 | 1,050 |
| DJI Phantom 3 | 100.00% | 2.55% | 0.0496 | 550 |
| **Macro Average / Overall** | **51.84%** | **22.02%** | **0.2005** | **3,650** |
| **Summary Metrics** | \multicolumn{4}{c}{\textbf{Accuracy}: $24.79\%$ \quad \textbf{Balanced Accuracy}: $22.02\%$ \quad \textbf{Loss}: $1.4735$} |

### D. Waveform Normalization Diagnostic
We evaluated the effect of normalization strategies on VARDHAN-v3:
- **Global Train-Fold Z-Score Normalization**: Standardized 1-fold diagnostic yielded **$45.55\%$ accuracy** and **$0.3459$ macro-F1**, collapsing heavily toward Bebop (Background recall: 0.0%, Bebop recall: 99.29%).
- **Per-Segment Normalization**: Standardized 10-fold benchmark achieved **$71.57\% \pm 1.25\%$ accuracy** and **$0.6953 \pm 0.0168$ macro-F1** (Background recall: 83.17%, Bebop: 76.40%, AR: 68.09%, Phantom: 43.00%).

---

## IX. Generalization and Error Analysis

### A. The Cross-Recording Generalization Gap
The performance difference between segment-level ($71.57\%$) and recording-level ($24.79\%$) is consistent with segment-level evaluation benefiting from strong intra-recording correlation and recording-specific domain variation. In segment-level CV, the model can exploit stationary session-level characteristics. In strict recording-level evaluation, the test sessions represent independent physical recording sessions, producing substantial out-of-distribution degradation. Possible contributors include recording-specific interference conditions, receiver characteristics, propagation effects, transmitter-specific signatures, and frequency-dependent domain shifts. These factors were not independently isolated in the present study.

### B. Overfitting Dynamics on Strict Protocol
The 100-epoch training trajectory of VARDHAN-v3 on the strict split reveals empirical overfitting to the training sessions. By epoch 15, training accuracy exceeds $96\%$ ($\text{loss} < 0.33$), reaching $100.0\%$ by epoch 90. In contrast, validation accuracy remains bounded between $21\%$ and $38\%$ ($\text{loss} > 1.55$), with the minimum validation loss achieved at **Epoch 1** ($\text{val\_loss} = 1.5491$). This divergence provides clear empirical evidence of overfitting on the training sessions.

### C. Confusion Matrix and Normalization Scale Analysis
Analyzing the 10-fold confusion matrix for the faithful FGCS reproduction reveals that while Background ($99.85\%$ recall, 4094/4100) and Bebop ($98.45\%$ recall, 8270/8400) are classified reliably, $23.95\%$ of AR Drone segments (1940/8100) and $66.62\%$ of Phantom segments (1399/2100) are misclassified as Bebop.

Furthermore, the diagnostic results indicate that global train-fold standardization produced substantial scale disparities between classes in this dataset, coinciding with severe Background-class degradation. Diagnostic statistics reveal that the training set standard deviation was dominated by higher-amplitude recordings ($\sigma_{\text{Phantom}} \approx 1.8835, \sigma_{\text{AR}} \approx 0.6721$), while background recordings exhibited much lower variance ($\sigma_{\text{BG}} \approx 0.0152$). Per-segment normalization addressed this scaling disparity, restoring Background recall to $83.17\%$ in VARDHAN-v3.

---

## X. Edge-Oriented Deployment Considerations

VARDHAN-v3 was architected for resource-constrained edge computing. Table 7 compares model complexity.

### Table 7: Architectural Complexity and Parameter Efficiency

| Model | Input Type | Trainable Parameters | Memory (FP32) | Theoretical MACs | Theoretical FLOPs |
|---|---|:---:|:---:|:---:|:---:|
| **CS-CNN** [10] | CS ($CR=0.5$) | 1,059,908 | 4.24 MB | $\approx 54.2$ M | $\approx 108.4$ M |
| **FGCS2019DNN** [2] | Power Spectrum | 295,812 / 565,956 | 1.18 / 2.26 MB | 0.30 M | 0.59 M |
| **MC1DCNN** [8] | 8 Sub-Bands | 275,940 | 1.10 MB | 14.12 M | 28.24 M |
| **DSCNN** [9] | 1D Waveform | 68,228 | 0.27 MB | 3.49 M | 6.98 M |
| **MobileNet-Style** | 2D Spectrogram | 11,588 | 0.05 MB | 1.18 M | 2.36 M |
| **VARDHAN-v3** | **Dual-Domain** | **119,806** | **0.48 MB** | **11.81 M** | **23.63 M** |

While VARDHAN-v3 achieves compact parameter scaling ($119.8$ k parameters, $<0.48$ MB FP32 model size), *hardware-specific inference latency, runtime memory bandwidth, and power consumption on physical embedded hardware (e.g., NVIDIA Jetson, Raspberry Pi, ARM Cortex-M) remain unmeasured and represent future work*. Containerization and software pipelining demonstrate software deployment feasibility, but operational RF reliability requires dedicated over-the-air field validation.

---

## XI. Limitations

This study identifies several important limitations:
1. **Dataset Scope**: DroneRF represents a controlled experimental dataset recorded with specific USRP hardware and three drone models; it does not capture diverse real-world urban RF interference.
2. **Cross-Recording Generalization**: Performance under strict unseen recording evaluation remains low ($24.79\%$), highlighting a domain shift challenge.
3. **Standardized Benchmark Breadth**: Full 10-fold cross-validation was completed for FGCS, MC1DCNN, and VARDHAN-v3 due to computational resource constraints.
4. **Implementation Ambiguities**: Literature descriptions omit certain training and normalization details, necessitating faithful interpretations.
5. **Sample Count Differences**: Historical strict baselines used $N_{\text{samp}}=5$, whereas final VARDHAN-v3 strict used $N_{\text{samp}}=50$.
6. **Lack of External Validation**: Evaluation is confined to DroneRF without cross-dataset testing on Cardinal or private RF captures.
7. **Hardware Latency**: Inference latency was not benchmarked on physical edge microcontrollers.
8. **Closed-Set Constraint**: The framework assumes a closed 4-class classification space without open-set unknown drone rejection.

---

## XII. Future Work

Promising future research directions include:
- **Domain Adaptation \& Generalization**: Implementing adversarial domain adaptation, self-supervised pre-training (e.g., masked RF autoencoding), and cross-frequency contrastive learning to bridge recording-level domain shifts.
- **Data Augmentation for RF**: Applying synthetic multipath Rayleigh/Rician fading, phase noise injection, carrier frequency jitter, and SNR mixing.
- **Open-Set Recognition**: Designing energy-based out-of-distribution detectors and extreme value theory (EVT) classifiers to reject unknown drone models.
- **Hardware Optimization**: Quantizing VARDHAN-v3 to INT8/FP16 via TensorRT and ONNX Runtime for benchmarking on NVIDIA Jetson Orin and Raspberry Pi 5.
- **Multi-Modal Sensor Fusion**: Integrating RF fingerprinting with thermal imaging, micro-Doppler acoustic sensing, and optical camera tracking.

---

## XIII. Conclusion

This paper presented VARDHAN, a lightweight dual-domain deep learning framework for RF-based drone detection and classification, alongside a reproducibility and generalization study on the DroneRF dataset. VARDHAN-v3 integrates multi-scale temporal convolutions ($k \in \{3, 7, 15, 31\}$) and a 3-channel complex Fourier representation with Squeeze-and-Excitation attention, attention pooling, and bi-directional cross-domain gating with exactly 119,806 parameters. Under standardized segment-level 10-fold evaluation, VARDHAN-v3 achieves $71.57\% \pm 1.25\%$ accuracy and $0.6953$ macro-F1. However, under strict unseen recording-level evaluation, accuracy across evaluated models drops to $24\%\text{--}29\%$. The observed performance gap is consistent with segment-level evaluation benefiting from intra-recording correlation and recording-specific domain variation. Furthermore, we maintain clear distinction between the historical strict baseline benchmark ($N_{\text{samp}}=5$) and the final VARDHAN-v3 strict experiment ($N_{\text{samp}}=50$). These findings emphasize the necessity of strict recording-level evaluation protocols and establish a principled foundation for future domain-robust counter-UAS research.

---

## References

1. E. A. Debas, A. Albuali, and M. M. H. Rahman, "Forensic Examination of Drones: A Comprehensive Study of Frameworks, Challenges, and Machine Learning Applications," *IEEE Access*, vol. 12, pp. 111505–111522, Jul. 2024.
2. M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, "RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database," *Future Generation Computer Systems*, vol. 100, pp. 86–97, Nov. 2019.
3. B. Taha and A. Shoufan, "Machine Learning-Based Drone Detection and Classification: State-of-the-Art in Research," *IEEE Access*, vol. 7, pp. 138669–138682, 2019.
4. V. Sihag, G. Choudhary, P. Choudhary, and N. Dragoni, "Cyber4Drone: A Systematic Review of Cyber Security and Forensics in Next-Generation Drones," *Drones*, vol. 7, no. 7, p. 430, Jun. 2023.
5. M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, "Micro-UAV Detection and Classification from RF Fingerprints Using Machine Learning Techniques," in *Proc. IEEE Aerospace Conf.*, Big Sky, MT, USA, 2019, pp. 1–13.
6. M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, and I. Guvenc, "Detection and Classification of UAVs Using RF Fingerprints in the Presence of Wi-Fi and Bluetooth Interference," *IEEE Open Journal of the Communications Society*, vol. 1, pp. 60–76, 2020.
7. M. S. Allahham, M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, and A. Erbad, "DroneRF dataset: A dataset of drones for RF-based detection, classification, and identification," *Data in Brief*, vol. 26, p. 104313, Oct. 2019.
8. M. S. Allahham, T. Khattab, and A. Mohamed, "Deep Learning for RF-Based Drone Detection and Identification: A Multi-Channel 1-D Convolutional Neural Networks Approach," in *Proc. IEEE Int. Conf. Informatics, IoT, and Enabling Technologies (ICIoT)*, Doha, Qatar, 2020, pp. 112–117.
9. O. Medaiyese, A. Syed, and A. P. Lauf, "Machine Learning Framework for RF-Based Drone Detection and Identification System," in *Proc. 2nd Int. Conf. Smart Cities, Automation and Intelligent Computing Systems (ICON-SONICS)*, Tangerang, Indonesia, 2021, pp. 1–6.
10. Y. Mo, J. Huang, and G. Qian, "Deep Learning Approach to UAV Detection and Classification by Using Compressively Sensed RF Signal," *Sensors*, vol. 22, no. 8, p. 3072, Apr. 2022.
11. S. Al-Emadi and F. Al-Senaid, "Drone Detection Approach Based on Radio-Frequency Using Convolutional Neural Network," in *Proc. IEEE Int. Conf. Informatics, IoT, and Enabling Technologies (ICIoT)*, Doha, Qatar, 2020, pp. 29–34.
12. I. Nemer, T. Sheltami, I. Ahmad, A. U.-H. Yasar, and M. A. R. Abdeen, "RF-Based UAV Detection and Identification Using Hierarchical Learning Approach," *Sensors*, vol. 21, no. 6, p. 1947, Mar. 2021.
13. Y. Zhang, "RF-Based Drone Detection Using Machine Learning," in *Proc. 2nd Int. Conf. Computing and Data Science (CDS)*, Stanford, CA, USA, 2021, pp. 425–428.
14. J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141.
15. A. Howard et al., "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2019, pp. 1314–1324.
16. D. L. Donoho, "Compressed sensing," *IEEE Transactions on Information Theory*, vol. 52, no. 4, pp. 1289–1306, Apr. 2006.
17. E. J. Candès and M. B. Wakin, "An Introduction To Compressive Sampling," *IEEE Signal Processing Magazine*, vol. 25, no. 2, pp. 21–30, Mar. 2008.
18. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, CA, USA, 2015, pp. 1–15.
19. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. 7th Int. Conf. Learning Representations (ICLR)*, New Orleans, LA, USA, 2019, pp. 1–10.
20. D. Hendrycks and K. Gimpel, "Gaussian Error Linear Units (GELUs)," *arXiv preprint arXiv:1606.08415*, 2016.
