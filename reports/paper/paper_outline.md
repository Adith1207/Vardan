# Research Paper Draft: Vardan RF-Based Drone Detection

Drafting and assets compilation outline for publication (IEEE/Elsevier style). Final numbers must come from `results/` and `reports/VARDHAN_IEEE_Report.md`.

---

## 1. Title & Abstract (Proposed)
* **Title**: Lightweight Dual-Domain RF Classification of Drones and the Cost of Recording-Level Evaluation
* **Abstract**: (Draft placeholder)

## 2. Introduction
* Motivation: Security threats of low-altitude commercial drones; passive RF sensing.
* Problem: segment-level random splits leak recording-session information and inflate reported accuracy.
* Key Contributions of Vardan:
    1. Faithful reproduction of published DroneRF baselines.
    2. Strict recording-level benchmark with zero session overlap.
    3. VARDHAN-v3, a lightweight dual-domain (temporal + spectral) RF network.

## 3. Related Work
* RF-based drone detection on DroneRF and related datasets.
* Spectral, time-domain, and compressive-sensing deep-learning approaches.
* Lightweight network design (depthwise separable convolutions, squeeze-and-excitation).

## 4. Proposed Vardan Architecture
* **Signal Processing Pipeline**: Per-segment detrending and normalisation, 2048-point rFFT, three-channel spectral representation.
* **VARDHAN-v3 Model**: Multi-scale temporal and spectral backbones, attention pooling, cross-domain gated fusion of the two RF views.

## 5. Experimental Setup
* Dataset: DroneRF (RF).
* Protocols: segment-level 10-fold cross-validation and strict recording-level split.
* Training configs: Hyperparameters, hardware details.
* Software profiling: Parameter counts, MACs/FLOPs, CPU/GPU latency.

## 6. Results & Analysis
* **Segment-Level Benchmark**: FGCS2019DNN, MC1DCNN and VARDHAN-v3 10-fold results.
* **Strict Recording-Level Benchmark**: Generalisation gap and per-class analysis (incl. Phantom band shift).
* **Ablations**: Normalisation strategy; branch contributions.
* **Efficiency**: Accuracy vs. parameters and software-measured latency.

## 7. Conclusions & Future Work
* Summary of findings.
* Session-invariant training, multi-site data, open-set evaluation, and measured embedded deployment as future work.
