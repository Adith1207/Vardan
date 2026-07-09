# Research Paper Draft: Vardan Multi-Modal Counter-UAS

Drafting and assets compilation outline for publication (IEEE/Elsevier style).

---

## 1. Title & Abstract (Proposed)
* **Title**: Towards an Indigenous, Portable and Energy-Efficient Counter-UAS Detection System using Adaptive Multi-Modal Artificial Intelligence
* **Abstract**: (Draft placeholder)

## 2. Introduction
* Motivation: Security threats of low-altitude commercial drones.
* Limitations of single-modality RF or vision systems in high noise/obscurity.
* Key Contributions of Vardan:
    1. Multi-modal adaptivity.
    2. Embedded/TinyML efficiency suitable for edge microcontrollers.
    3. Performance improvements over state-of-the-art baselines.

## 3. Related Work
* RF-based drone detection (Review of DroneRF dataset papers).
* Acoustic drone signature classification.
* Vision-based detection on edge devices.
* Sensor Fusion techniques.

## 4. Proposed Vardan Architecture
* **Signal Processing Pipeline**: Windowing, STFT spectrogram extraction, wavelet decomposition.
* **VardhanRFNet Model**: Multi-scale feature branches, dilated convolutions.
* **Multi-Modal Fusion Module**: Entropy-based dynamic late fusion model.

## 5. Experimental Setup
* Datasets: DroneRF (RF), custom acoustic datasets, vision target profiles.
* Training configs: Hyperparameters, hardware details.
* Edge profiling suite: Parameter counts, flash/RAM utilization, latency.

## 6. Results & Analysis
* **RF Detection Accuracy**: Benchmark tables comparing baseline models (1D-CNN, DS-CNN, MobileNet) with VardhanRFNet.
* **Multimodal Fusion Performance**: Tables demonstrating single sensor vs. fused detection under interference.
* **Edge Optimization Benchmarks**: Pareto frontiers showing accuracy vs inference latency.

## 7. Conclusions & Future Work
* Summary of findings.
* Path forward to deployment on ARM Cortex microcontrollers (TinyML).
