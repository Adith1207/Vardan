# Vardan Data Directory Organization

This directory stores data used in the project, separated by stages of processing to ensure reproducibility and clean lineage.

## Directory Structure

*   `raw/`: Contains original, unmodified datasets.
    *   `DroneRF/`: Landing directory for the DroneRF dataset.
    *   *Rule*: **Never modify files in this folder.** Raw data is read-only.
*   `interim/`: Intermediate files generated during feature extraction or preprocessing.
    *   Examples include normalized signals, truncated samples, or temporary formats.
*   `processed/`: Fully processed datasets ready for training and evaluation.
    *   Examples include STFT spectrogram matrices, wavelet transformed sequences, and train/val/test splits.

## Raw Data Download Instructions

The DroneRF dataset is not included in this repository due to size and licensing constraints. 
To obtain the raw data:
1. Download the official **DroneRF** dataset from Mendeley Data (or official source).
2. Extract the dataset files.
3. Place them into `data/raw/DroneRF/` following this structure:
   ```text
   data/raw/DroneRF/
   ├── 10000/
   │   ├── 10000L.csv
   │   ├── 10000H.csv
   │   └── ...
   ```
