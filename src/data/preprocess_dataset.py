"""
preprocess_dataset.py
---------------------

Dataset preprocessing script for the DroneRF dataset.
Processes raw RF CSV signals using DroneRFPreprocessor and exports
train/val/test split artifacts to data/processed/DroneRF/.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import FFT_SIZE, SAMPLING_RATE
from constants import RAW_CLASS_TO_INDEX
from preprocessing.pipeline import DroneRFPreprocessor
from utils.paths import DATA_DIR, PROCESSED_DATA_DIR


# Label maps
CLASS_MAPPING_4 = RAW_CLASS_TO_INDEX


CLASS_MAPPING_5 = {
    "Backround RF activities": 0,
    "Phantom drone": 1,  # Can be 1 or 2 depending on receiver/mode
    "Bepop drone": 3,
    "AR Drone": 4,
}


def load_and_preprocess_dataset(
    max_segments_per_file: int = 50,
    segment_length: int = 2048,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
):
    """
    Load raw DroneRF CSV files, extract 2048-sample segments, preprocess them,
    and save stratified train/val/test datasets.
    """
    start_time = time.time()
    np.random.seed(random_seed)

    print("=========================================================")
    print("      Vardan Counter-UAS Dataset Preprocessing          ")
    print("=========================================================")

    metadata_path = DATA_DIR / "metadata" / "dronerf_metadata.csv"
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df_meta = pd.read_csv(metadata_path)
    print(f"Loaded metadata for {len(df_meta)} CSV files.")

    preprocessor = DroneRFPreprocessor(
        fft_size=FFT_SIZE,
        remove_dc=True,
        normalization="max",
        channel_count=8,
        channel_overlap=0.50,
        fs=SAMPLING_RATE,
    )

    all_raw_signals = []
    all_fgcs_spectra = []
    all_multichannel_spectra = []
    all_labels_4 = []
    all_labels_5 = []

    file_count = 0
    total_files = len(df_meta)

    for idx, row in df_meta.iterrows():
        rel_path = row["relative_path"]
        from data.loader import resolve_raw_path
        abs_path = resolve_raw_path(rel_path)

        if not abs_path.exists():
            continue

        drone_class = row["drone_class"]
        receiver = str(row["receiver"]).upper()

        # Determine 4-class label
        label_4 = CLASS_MAPPING_4.get(drone_class, 0)

        # Determine 5-class label
        if drone_class == "Phantom drone":
            if "2" in receiver or "VIDEO" in receiver:
                label_5 = 2  # phantom_4_video
            else:
                label_5 = 1  # phantom_4_active
        else:
            label_5 = CLASS_MAPPING_5.get(drone_class, 0)

        try:
            # Read first chunk of samples from CSV file
            chunk_df = pd.read_csv(abs_path, header=None, nrows=max_segments_per_file)
            flat_vals = chunk_df.to_numpy().reshape(-1)
            flat_vals = pd.to_numeric(flat_vals, errors="coerce")
            flat_vals = flat_vals[~np.isnan(flat_vals)].astype(np.float32)

            num_segments = len(flat_vals) // segment_length
            if num_segments == 0:
                continue

            for s_idx in range(num_segments):
                seg = flat_vals[s_idx * segment_length : (s_idx + 1) * segment_length]

                # Preprocess representations
                fgcs_spec = preprocessor.process_fgcs(seg)
                mc_spec = preprocessor.process_multichannel(seg)

                # Normalize 1D waveform
                seg_norm = seg - np.mean(seg)
                max_abs = np.max(np.abs(seg_norm))
                if max_abs > 1e-8:
                    seg_norm /= max_abs

                all_raw_signals.append(seg_norm)
                all_fgcs_spectra.append(fgcs_spec)
                all_multichannel_spectra.append(mc_spec)
                all_labels_4.append(label_4)
                all_labels_5.append(label_5)

            file_count += 1
            if file_count % 50 == 0 or file_count == total_files:
                print(f"Processed {file_count}/{total_files} files ({len(all_raw_signals)} total segments extracted)...")

        except Exception as e:
            print(f"Skipping {abs_path.name}: {e}")
            continue

    if not all_raw_signals:
        print("Error: No segments extracted.")
        return

    # Convert lists to NumPy arrays
    X_raw = np.array(all_raw_signals, dtype=np.float32)
    X_fgcs = np.array(all_fgcs_spectra, dtype=np.float32)
    X_mc = np.array(all_multichannel_spectra, dtype=np.float32)
    y_4 = np.array(all_labels_4, dtype=np.int64)
    y_5 = np.array(all_labels_5, dtype=np.int64)

    total_samples = len(y_4)
    print(f"\nSuccessfully extracted {total_samples} dataset segments!")
    print(f" - Raw waveforms shape:        {X_raw.shape}")
    print(f" - FGCS power spectra shape:   {X_fgcs.shape}")
    print(f" - Multi-channel spectra shape: {X_mc.shape}")

    # Stratified Dataset Splitting
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    out_dir = PROCESSED_DATA_DIR / "DroneRF"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    print("\nSaving dataset splits to:", out_dir)
    for split_name, split_indices in splits.items():
        save_path = out_dir / f"{split_name}.npz"
        np.savez_compressed(
            save_path,
            x_raw=X_raw[split_indices],
            x_fgcs=X_fgcs[split_indices],
            x_multichannel=X_mc[split_indices],
            y_4class=y_4[split_indices],
            y_5class=y_5[split_indices],
        )
        print(f" ✓ {split_name}.npz ({len(split_indices)} samples) saved to {save_path.name}")

    elapsed = time.time() - start_time
    print(f"\n=========================================================")
    print(f" Preprocessing completed successfully in {elapsed:.2f}s!")
    print(f"=========================================================")


if __name__ == "__main__":
    load_and_preprocess_dataset()
