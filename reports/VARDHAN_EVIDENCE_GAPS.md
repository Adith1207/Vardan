# VARDHAN Research Report Evidence Gaps and Verification Boundaries

This document transparently enumerates all known evidence gaps, unmeasured hardware properties, literature ambiguities, and methodological constraints identified during the forensic audit and benchmarking of the VARDHAN project.

---

## 1. Unmeasured Edge Hardware Properties

| Property | Current Project Status | Impact & Boundary Statement | Required Future Action |
|---|---|---|---|
| **Real-Time Edge Hardware Latency** | **Unmeasured on Target MCUs / SBCs** | Theoretical MACs (11.81 M) and FLOPs (23.63 M) are graph-derived. No empirical latency (ms/segment) has been benchmarked on physical NVIDIA Jetson Orin, Raspberry Pi 5, or ARM Cortex-M microcontrollers. | Export VARDHAN-v3 to ONNX / TensorRT / TFLite and profile inference on physical embedded hardware. |
| **Edge Memory Bandwidth & Flash Footprint** | **Calculated from FP32 Model Weights Only** | Static model weights require $\approx 0.48\,\text{MB}$ in FP32 ($119,806 \times 4$ bytes). Peak activation memory and scratchpad SRAM during streaming inference have not been measured on constrained devices. | Measure peak runtime memory using PyTorch Profiler / Valgrind on embedded Linux platforms. |
| **Hardware Power Consumption (Watts)** | **Unmeasured** | System power dissipation under continuous 100 MS/s RF stream ingestion is unquantified. | Profile dynamic power draw using a hardware power monitor (e.g., Monsoon Power Monitor). |
| **FP16 / INT8 Quantization Degradation** | **Unmeasured** | Model performance was evaluated in standard FP32 precision. Quantization-Aware Training (QAT) or Post-Training Quantization (PTQ) accuracy loss has not been tested. | Evaluate PTQ / QAT performance on quantized INT8 backbones. |

---

## 2. Literature Ambiguities & Discrepancies

| Topic / Paper | Documented Literature Ambiguity | Resolution in Project VARDHAN |
|---|---|---|
| **Al-Sa'd et al. (FGCS 2019) Architecture** | Paper text specifies hidden layer sizes Dense(256) $\rightarrow$ Dense(128) $\rightarrow$ Dense(64) (565,956 params). However, the author-released Python script `Classification.py` in the official DroneRF repository implemented Dense(128) $\rightarrow$ Dense(128) $\rightarrow$ Dense(128) (295,812 params). | The project faithfully reproduced the author-released code architecture (295,812 params), achieving $84.52\% \pm 0.73\%$ accuracy (matching the paper's reported $84.5\%$). Both architectures are explicitly documented in the report. |
| **Allahham et al. (MC1DCNN 2020) Architecture Details** | The paper specifies dividing 2048 DFT bins into 8 sub-bands ($8 \times 256$) and outlines 2 Conv1D stages, but does not explicitly provide exact kernel sizes or stride parameters in the text. | The repository constructed a representative 1D CNN with $k=11, s=2$ and $k=5, s=1$ totaling 275,940 parameters. This interpretation is explicitly documented as the project's implementation choice. |
| **Mo et al. (Sensors 2022) Compressive Sensing Details** | The paper describes Multi-Channel Random Demodulation (MCRD) using Bernoulli matrices and multi-stage classification, but the source code was not publicly released. | The project baseline implemented a deterministic Bernoulli random sensing projection matrix ($CR=0.5, M=1024$) and a 1D CNN classifier. |
| **Medaiyese et al. (2021) / DSCNN** | Medaiyese et al. combined wavelet decomposition and XGBoost / DSCNN for low-frequency spectrum classification. | The repository evaluated a depthwise separable 1D CNN baseline (68,228 parameters). |

---

## 3. Dataset & Protocol Limitations

| Limitation | Technical Context | Scientific Implication |
|---|---|---|
| **DroneRF Single-Site Capture** | All 454 files in DroneRF were recorded in a single physical environment using specific USRP-2943R receivers. | Cross-location, cross-city, and cross-hardware transferability cannot be established without external multi-site datasets. |
| **Phantom Receiver Frequency Shift** | In the strict recording-level split, all 21 training files for Phantom 3 were captured on the Upper band ($H$, 2.462 GHz), whereas validation and testing files were captured on the Lower bands ($L/L1/L2$, 2.422 GHz). | This cross-channel frequency shift represents a natural domain shift where the model was trained on upper-band emissions and tested on lower-band fundamentals, contributing to near-zero Phantom recall ($2.55\%$). |
| **Standardized Benchmark Scope** | Due to compute and execution time constraints on cloud GPU instances, the full 10-fold segment-level benchmark was executed for FGCS2019DNN, MC1DCNN, and VARDHAN-v3, but not for all six models. | The report explicitly states this constraint and separates the standardized 10-fold results from the historical strict baseline results. |
| **Sample Count Disparity** | Historical baseline strict benchmark used $N_{\text{samp}}=5$ segments per file (365 test segments), whereas the final VARDHAN-v3 strict experiment used $N_{\text{samp}}=50$ segments per file (3,650 test segments). | The report maintains strict separation between the $N_{\text{samp}}=5$ historical baseline table and the $N_{\text{samp}}=50$ VARDHAN-v3 strict table. |
| **Closed-Set Evaluation** | All experiments evaluate closed-set classification across 4 fixed classes (Background, Bebop, AR, Phantom). | Open-set unknown UAV rejection and confidence calibration remain unvalidated. |

---

## 4. Software vs. Operational Deployment Feasibility

| Deployment Facet | Status | Boundary Note |
|---|---|---|
| **Containerization & Docker Packaging** | Verified feasible | Validates software buildability, dependency isolation, and API reproducibility. |
| **Operational RF Detection Reliability** | **Not Verified in Real-World Flight** | Software container execution does not guarantee high real-world detection accuracy under complex multi-UAV, high-noise, or jamming environments. |
