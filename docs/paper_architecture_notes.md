# Paper Architecture Notes

This document provides detailed, layer-by-layer architectural specifications and design classifications for each of the four baseline models implemented in the Vardan Counter-UAS framework.

Every parameter and configuration choice is tagged as one of:
- **`[PAPER-SPECIFIED]`**: Explicitly stated in the paper text, tables, or diagrams.
- **`[PAPER-INFERRED]`**: Unambiguously required or standard for the architecture/layer type.
- **`[OUR-DESIGN-CHOICE]`**: Custom choice in our framework (e.g., pipeline details or framework adapters).
- **`[NOT-SPECIFIED]`**: Omitted or left undefined in the publication.

---

## 1. FGCS2019DNN (Al-Sa'd et al., 2019)

### 1.1 Model Architecture Table

| Layer # | Layer Type | Input Shape | Output Shape | Layer Parameters / Weights | Activation |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **0** | Input | `(B, 2048)` | `(B, 2048)` | - | - |
| **1** | Linear (`fc1`) | `(B, 2048)` | `(B, 256)` | `[2048, 256]` `[PAPER-SPECIFIED]` | ReLU `[PAPER-SPECIFIED]` |
| **2** | Linear (`fc2`) | `(B, 256)` | `(B, 128)` | `[256, 128]` `[PAPER-SPECIFIED]` | ReLU `[PAPER-SPECIFIED]` |
| **3** | Linear (`fc3`) | `(B, 128)` | `(B, 64)` | `[128, 64]` `[PAPER-SPECIFIED]` | ReLU `[PAPER-SPECIFIED]` |
| **4** | Linear (`fc4`) | `(B, 64)` | `(B, 4)` | `[64, 4]` `[PAPER-SPECIFIED]` | Logits / CrossEntropyLoss `[OUR-DESIGN-CHOICE]` |

### 1.2 Configuration & Training Protocols

- **DC Removal**: `[PAPER-SPECIFIED]` (Mean of raw signal subtracted before FFT).
- **FFT Size**: `[PAPER-SPECIFIED]` (2048-point magnitude-squared power spectrum).
- **Normalization**: `[PAPER-INFERRED]` (Max magnitude scaling to `[0, 1]`).
- **Batch Size**: `[PAPER-SPECIFIED]` (10).
- **Optimizer**: `[PAPER-SPECIFIED]` (Adam).
- **Learning Rate**: `[PAPER-INFERRED]` (0.001).
- **Loss Function**: `[PAPER-INFERRED]` (Cross-Entropy Loss for multiclass).
- **Dropout**: `[PAPER-SPECIFIED]` (None).

---

## 2. Multi-Channel 1D CNN (Ezuma et al., 2020)

### 2.1 Model Architecture Table

| Layer # | Layer Type | Input Shape | Output Shape | Kernel | Stride / Padding | Activation |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Input | `(B, 2, 2048)` | `(B, 2, 2048)` | - | - | - |
| **1** | Conv1D (`conv1`) | `(B, 2, 2048)` | `(B, 32, 1024)` | 11 `[PAPER-SPECIFIED]` | Stride=2, Pad=5 `[PAPER-SPECIFIED]` | ReLU `[PAPER-SPECIFIED]` |
| **2** | MaxPool1D (`pool1`) | `(B, 32, 1024)` | `(B, 32, 512)` | 2 `[PAPER-SPECIFIED]` | Stride=2 `[PAPER-SPECIFIED]` | - |
| **3** | Conv1D (`conv2`) | `(B, 32, 512)` | `(B, 64, 512)` | 5 `[PAPER-SPECIFIED]` | Stride=1, Pad=2 `[PAPER-SPECIFIED]` | ReLU `[PAPER-SPECIFIED]` |
| **4** | MaxPool1D (`pool2`) | `(B, 64, 512)` | `(B, 64, 256)` | 2 `[PAPER-SPECIFIED]` | Stride=2 `[PAPER-SPECIFIED]` | - |
| **5** | Flatten | `(B, 64, 256)` | `(B, 16384)` | - | - | - |
| **6** | Linear (`fc1`) | `(B, 16384)` | `(B, 128)` | `[16384, 128]` `[PAPER-SPECIFIED]` | - | ReLU `[PAPER-SPECIFIED]` |
| **7** | Linear (`fc2`) | `(B, 128)` | `(B, 4)` | `[128, 4]` `[PAPER-SPECIFIED]` | - | Logits `[OUR-DESIGN-CHOICE]` |

### 2.2 Configuration & Training Protocols

- **Input Representation**: `[PAPER-SPECIFIED]` (Time-domain I/Q or multi-channel).
- **Normalization**: `[OUR-DESIGN-CHOICE]` (Z-score normalization fitted strictly on the training partition).
- **Batch Size**: `[OUR-DESIGN-CHOICE]` (32).
- **Optimizer**: `[PAPER-INFERRED]` (Adam).
- **Learning Rate**: `[PAPER-INFERRED]` (0.001).
- **Dropout**: `[NOT-SPECIFIED]` (None).

---

## 3. TinyML DSCNN (Medaiyese et al., 2022)

### 3.1 Model Architecture Table

| Layer # | Layer Type | Input Shape | Output Shape | Kernel / Groups | Stride / Padding | Activation |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Input | `(B, 2, 2048)` | `(B, 2, 2048)` | - | - | - |
| **1** | Conv1D (`conv1`) | `(B, 2, 2048)` | `(B, 32, 1024)` | Kernel=11 `[PAPER-SPECIFIED]` | Stride=2, Pad=5 `[PAPER-SPECIFIED]` | ReLU `[PAPER-SPECIFIED]` |
| **2** | Depthwise (`depth`) | `(B, 32, 1024)` | `(B, 32, 1024)` | Kernel=5, Groups=32 `[PAPER-SPECIFIED]` | Stride=1, Pad=2 `[PAPER-SPECIFIED]` | - |
| **3** | Pointwise (`point`) | `(B, 32, 1024)` | `(B, 64, 1024)` | Kernel=1, Groups=1 `[PAPER-SPECIFIED]` | Stride=1 `[PAPER-SPECIFIED]` | ReLU `[PAPER-SPECIFIED]` |
| **4** | MaxPool1D (`pool`) | `(B, 64, 1024)` | `(B, 64, 256)` | Kernel=4 `[PAPER-SPECIFIED]` | Stride=4 `[PAPER-SPECIFIED]` | - |
| **5** | Flatten | `(B, 64, 256)` | `(B, 16384)` | - | - | - |
| **6** | Linear (`fc`) | `(B, 16384)` | `(B, 4)` | `[16384, 4]` `[PAPER-SPECIFIED]` | - | Logits `[OUR-DESIGN-CHOICE]` |

### 3.2 Configuration & Training Protocols

- **Compression**: `[PAPER-INFERRED]` (Log dynamic range compression).
- **Normalization**: `[OUR-DESIGN-CHOICE]` (Z-score normalization).
- **Batch Size**: `[OUR-DESIGN-CHOICE]` (32).
- **Optimizer**: `[PAPER-INFERRED]` (Adam).

---

## 4. MobileNetV3Small (Howard et al., 2019 / Spectrogram Baseline)

### 4.1 Model Architecture Table

| Layer # | Layer Type | Input Shape | Output Shape | Parameters | Stride / Padding | Activation |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Input | `(B, 1, 65, 61)` | `(B, 1, 65, 61)` | - | - | - |
| **1** | Conv2D | `(B, 1, 65, 61)` | `(B, 16, 33, 31)` | Kernel=3x3, Filters=16 | Stride=2, Pad=1 | ReLU |
| **2** | BatchNorm2d | `(B, 16, 33, 31)` | `(B, 16, 33, 31)` | - | - | - |
| **3** | Depthwise Conv2D | `(B, 16, 33, 31)` | `(B, 32, 17, 16)` | Kernel=3x3, Groups=16 | Stride=2, Pad=1 | - |
| **4** | Pointwise Conv2D | `(B, 32, 17, 16)` | `(B, 64, 17, 16)` | Kernel=1x1, Filters=64 | Stride=1 | - |
| **5** | BatchNorm2d | `(B, 64, 17, 16)` | `(B, 64, 17, 16)` | - | - | ReLU |
| **6** | AdaptiveAvgPool2d | `(B, 64, 17, 16)` | `(B, 64, 1, 1)` | Output Size=(1, 1) | - | - |
| **7** | Flatten | `(B, 64, 1, 1)` | `(B, 64)` | - | - | - |
| **8** | Linear | `(B, 64)` | `(B, 128)` | `[64, 128]` | - | ReLU |
| **9** | Linear | `(B, 128)` | `(B, 4)` | `[128, 4]` | - | Logits |

### 4.2 Configuration & Training Protocols

- **STFT Parameters**: `[PAPER-INFERRED]` (Hann Window, $N_{\text{fft}} = 1024$, hop length = 256).
- **Log Scaling**: `[PAPER-INFERRED]` (Log dB magnitude spectrogram scaling: $20 \log_{10}(|STFT| + 10^{-12})$).
- **Normalization**: `[OUR-DESIGN-CHOICE]` (Z-score normalization).
