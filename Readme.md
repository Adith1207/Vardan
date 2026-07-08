<div align="center">

# 🛰️ Portable Indigenous Multi-Modal Counter-UAS System

### AI-Powered Lightweight Multi-Sensor Drone Detection Framework for Edge Devices

![Status](https://img.shields.io/badge/Status-Research%20In%20Progress-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![TinyML](https://img.shields.io/badge/TinyML-Edge%20AI-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

### 🎓 Final Year B.Tech Research Project

**Department of Computer Science & Engineering**  
**Amrita Vishwa Vidyapeetham**

---

*"Towards an Indigenous, Portable and Energy-Efficient Counter-UAS Detection System using Multi-Modal Artificial Intelligence."*

</div>

---

# 📖 Overview

The rapid increase in the use of **Unmanned Aerial Vehicles (UAVs)** has transformed modern warfare, surveillance, logistics, and civilian applications. However, this advancement has also introduced significant security threats through unauthorized drone operations, making reliable drone detection an increasingly important research area.

This project aims to develop a **Portable Indigenous Multi-Modal Counter-UAS System** capable of detecting drones using multiple sensing modalities:

- 📡 Radio Frequency (RF)
- 🎙️ Acoustic Signals
- 📷 Computer Vision

Unlike conventional single-modality systems, our approach integrates information from multiple sensors to improve robustness under diverse environmental conditions while maintaining suitability for **resource-constrained embedded devices**.

---

# 🎯 Project Objectives

- Develop an intelligent RF-based drone detection module.
- Design an acoustic drone identification system.
- Build a vision-based drone detection pipeline.
- Develop an adaptive sensor fusion framework.
- Optimize models for TinyML deployment.
- Deploy the complete system on embedded hardware.

---

# 🏗️ Overall System Architecture

```text
                    Portable Counter-UAS System

                  ┌───────────────────────────────┐
                  │      Incoming Drone           │
                  └──────────────┬────────────────┘
                                 │
         ┌───────────────────────┼────────────────────────┐
         │                       │                        │
         ▼                       ▼                        ▼

   RF Detection            Acoustic Detection      Vision Detection

         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼

                    Adaptive Sensor Fusion Engine

                                 │
                                 ▼

                    Drone / No Drone Classification

                                 │
                                 ▼

                       Threat Level & Alert System
```

---

# 🚀 Development Roadmap

This project is being developed incrementally through multiple phases.

| Phase | Description | Status |
|--------|-------------|--------|
| Phase 1 | RF-Based Drone Detection | 🟡 In Progress |
| Phase 2 | Acoustic Drone Detection | ⏳ Planned |
| Phase 3 | Vision-Based Drone Detection | ⏳ Planned |
| Phase 4 | Multi-Modal Sensor Fusion | ⏳ Planned |
| Phase 5 | Embedded TinyML Deployment | ⏳ Planned |
| Phase 6 | Real-World Validation & Optimization | ⏳ Planned |

---

# 📍 Current Focus

## ✅ Phase 1 — RF-Based Drone Detection

Current research activities include:

- Literature Survey
- DroneRF Dataset Analysis
- RF Signal Visualization
- Signal Preprocessing
- Lightweight Deep Learning Models
- Model Benchmarking
- TinyML Optimization
- Embedded Deployment

---

# 🧠 Planned Lightweight Architectures

Since the final system is intended for **microcontroller deployment**, model selection prioritizes computational efficiency alongside detection performance.

The following architectures will be investigated:

| Model | Purpose |
|--------|---------|
| 1D CNN | Baseline RF Classification |
| DS-CNN | TinyML Optimized CNN |
| MobileNetV3 Small | Lightweight CNN |
| MCUNet | Microcontroller-Oriented Neural Network |

Each model will be evaluated using identical datasets and benchmarking protocols.

---

# 📊 Evaluation Metrics

Models will be compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC
- Inference Time
- Number of Parameters
- Model Size
- RAM Usage
- Flash Memory Usage
- TinyML Deployment Feasibility

---

# 💻 Technology Stack

## Programming

- Python

## Deep Learning

- PyTorch

## Signal Processing

- NumPy
- SciPy
- Librosa
- Matplotlib

## Machine Learning

- Scikit-Learn

## Embedded AI

- TensorFlow Lite *(Planned)*
- TensorFlow Lite for Microcontrollers *(Planned)*

## Version Control

- Git
- GitHub

---

# 📅 Project Milestones

- [x] Repository Initialization
- [ ] Literature Review
- [ ] DroneRF Dataset Analysis
- [ ] RF Signal Visualization
- [ ] Signal Preprocessing
- [ ] Baseline 1D CNN
- [ ] Lightweight Model Benchmarking
- [ ] TinyML Optimization
- [ ] Embedded Deployment
- [ ] Acoustic Module
- [ ] Vision Module
- [ ] Sensor Fusion Engine
- [ ] Final Portable Counter-UAS Prototype

---

# 🎯 Long-Term Vision

The ultimate objective is to develop a **lightweight, energy-efficient and portable drone detection framework** capable of operating in real-world environments where traditional infrastructure may not be available.

The completed system will integrate RF, Acoustic and Vision sensing to provide robust and reliable drone detection suitable for deployment on embedded platforms.

---

# 📂 Repository Structure

This repository will gradually evolve throughout the project.

```text
Portable-Indigenous-Multimodal-Counter-UAS-System/

│── README.md
│── LICENSE
│── .gitignore
│── requirements.txt

(Additional modules will be added as development progresses.)
```

---

# 👨‍💻 Team

**Team 53**

- Adith Narayan G
- S J Yuvan Dhurkesh
- Subash Santhanam K
- Sisthick S

**Guide**

Mr. Sumesh A K  
Assistant Professor  
Department of Computer Science & Engineering  
Amrita Vishwa Vidyapeetham

---

# 📜 License

This project is released under the **MIT License**.

---

<div align="center">

### 🚧 Research Under Active Development

*"Building the next generation of lightweight, intelligent and portable Counter-UAS systems."*

⭐ Star this repository to follow our progress.

</div>
