# Vardan Counter-UAS Framework Implementation Audit

This document summarizes the comprehensive system audit performed on the Vardan Drone RF fingerprinting codebase and baseline pipelines. It details the critical bugs discovered, their implications on research integrity, and the corresponding fixes implemented.

---

## 1. Data Loader Zero-Padding & Silent Data Leakage

### 1.1 Identified Bug
In `src/data/loader.py`, the `_read_segment` method was designed to read a raw signal file line-by-line using Python's `readline()` up to `offset + 3` lines:
```python
with open(file_path, "r") as f:
    lines = [f.readline() for _ in range(offset + 3)]
target_line = lines[-1]
```
However, the raw DroneRF CSV files are single-row files (containing 10,000,000 values on row 0). Calling `readline()` multiple times returns empty strings `""` for lines past index 0 (which corresponds to lines 2, 3, etc.).
Due to a try-except block that caught all exceptions and fell back to returning `np.zeros()`, the dataloader was silently fabricating arrays of all zeros for `offset > 0` samples.

### 1.2 Resolution
- Rewrote the loader to use character-bounded line loading: `line = f.readline(limit)`, which restricts the character read limit to only the needed segment.
- Split the resulting single-line raw signal via string partitioning and sliced it cleanly based on the `offset * segment_length` index.
- Removed the silent try-except zero fallback by default, raising `FileNotFoundError` immediately when dataset files are missing.
- Added a safe `mock: bool = False` flag to allow running tests/smoke tests on synthetic signals.

---

## 2. Inconsistent Class Label Mapping

### 2.1 Identified Bug
We identified conflicting label-to-index mappings across the preprocessing and dataset loading scripts:
- **`src/data/preprocess_dataset.py` (Old)**:
  - Background RF activities -> 0
  - Phantom drone -> 1
  - Bepop drone -> 2
  - AR Drone -> 3
- **`src/data/loader.py` & `src/constants.py` (Old)**:
  - Background RF activities -> 0
  - AR Drone -> 1
  - Bepop drone -> 2
  - Phantom drone -> 3

This discrepancy caused mismatching targets between preprocessed cached `.npz` records and lazy-loaded datasets.

### 2.2 Resolution
- Centralized `RAW_CLASS_TO_INDEX` and `LABEL_MAP` in `src/constants.py`.
- Replaced the local mappings in both `loader.py` and `preprocess_dataset.py` with imports from `src/constants.py`.

---

## 3. Path Mismatches in Metadata & Disk Layout

### 3.1 Identified Bug
The split files (e.g. `data/splits/train.csv`) reference paths under a nested `unzipped_data/` directory (e.g., `data/raw/DroneRF/unzipped_data/AR Drone/RF Data_10100_H/10100H_0.csv`).
However, physical files extracted from raw archives are stored under nested paths with slight variations:
`data/raw/DroneRF/DroneRF/AR drone/RF Data_10100_H/RF Data_10100_H/10100H_0.csv`
This naming mismatch (lower-case "drone" and nested directories) broke path resolution.

### 3.2 Resolution
- Created a robust `resolve_raw_path` function in the dataloader that checks fallback naming strategies (resolving directory casing and nested layouts) to guarantee real files are correctly resolved on disk.

---

## 4. F1 Macro Average Metric Discrepancy

### 4.1 Identified Bug
In `src/evaluation/metrics.py`, `f1_macro` was computed by taking the harmonic mean of the macro-precision and macro-recall values:
```python
f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)
```
This is mathematically incorrect. The true macro F1 score is the arithmetic mean of the per-class F1-scores.

### 4.2 Resolution
- Refactored `calculate_metrics` to calculate the individual F1-score for each class and average them directly, ensuring compliance with the standard macro F1 definition used by packages like `scikit-learn`.

---

## 5. Console Printing & Windows Encoding Crash

### 5.1 Identified Bug
The print statements in `trainer.py` and run scripts used the checkmark character `✓` (`\u2713`). On Windows consoles running CP1252 or ASCII, this triggered a fatal `UnicodeEncodeError`, causing training runs to crash.

### 5.2 Resolution
- Replaced the checkmark characters across `src/models/trainer.py`, `scripts/run_smoke_test.py`, `scripts/verify_preprocessing.py`, and `scripts/train_baselines.py` with standard ASCII bracket codes (e.g. `[OK]`).
