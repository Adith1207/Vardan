# VARDHAN-v3 Strict Recording-Level Benchmark (Per-Segment Normalization)

## Preserved Experiment Artifacts

> [!NOTE]
> This directory contains preserved evaluation artifacts reconstructed from the completed Kaggle benchmark execution on an NVIDIA Tesla T4 GPU. The Kaggle interactive session was terminated after completion, so the `.pt` binary checkpoints are no longer available on disk.

### Experiment Configuration
- **Model**: VARDHAN-v3 (Multi-Scale Dual-Domain RF-Net, exactly 119,806 trainable parameters)
- **Protocol**: Canonical Strict Recording-Level Split (Zero recording overlap, zero file overlap across Train / Val / Test)
- **Normalization**: `per_segment` (independent zero-mean unit-variance per 2048-sample waveform: $\mu = \text{mean}(x), \sigma = \text{std}(x) + 10^{-8}$)
- **Split Distribution**:
  - **Train**: 308 files / 15 physical recordings (15,400 samples)
  - **Validation**: 73 files / 4 physical recordings (3,650 samples)
  - **Test**: 73 files / 4 physical recordings (3,650 samples)
- **Hyperparameters**:
  - Seed: `42`
  - Epochs: `100`
  - Batch Size: `32`
  - Optimizer: `AdamW` ($\text{lr}=3\times 10^{-4}, \text{weight\_decay}=10^{-4}$)
  - Scheduler: `CosineAnnealingLR` ($\eta_{\min}=10^{-6}$)
  - Loss: `CrossEntropyLoss` ($\text{label\_smoothing}=0.05$, class weights disabled)
  - Model Selection: Epoch 1 (minimum validation loss)

### Final Test Results
- **Test Accuracy**: **24.79%**
- **Test Macro-F1**: **0.2005**
- **Test Balanced Accuracy**: **22.02%**
- **Test Loss**: **1.4735**

#### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `no_drone` (Background) | 61.18% | 14.50% | 0.2344 | 1,000 |
| `ar_drone` | 25.28% | 19.52% | 0.2203 | 1,050 |
| `bebop_drone` | 20.90% | 51.52% | 0.2974 | 1,050 |
| `phantom_drone` | 100.00% | 2.55% | 0.0496 | 550 |
| **Macro Average** | **51.84%** | **22.02%** | **0.2005** | **3,650** |

### Preserved Files
- `aggregate_metrics.json`: Full structured test metrics and per-class summary.
- `run_config.json`: Complete execution and model hyperparameter configuration.
- `test_metrics.csv`: Per-class precision, recall, F1, and support table.
- `training_log.csv`: Epoch-by-epoch training and validation loss, accuracy, F1, balanced accuracy, LR, and epoch time.
