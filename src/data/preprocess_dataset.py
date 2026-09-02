"""
preprocess_dataset.py
---------------------

Dataset preprocessing script for the DroneRF dataset.
Consumes pre-computed recording-level split manifests (train.csv, val.csv, test.csv)
and extracts preprocessed representations to data/processed/DroneRF/ without data leakage.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
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


def preprocess_split(
    split_csv_path: Path,
    preprocessor: DroneRFPreprocessor,
    max_segments_per_file: int = 50,
    segment_length: int = 2048,
    raw_data_dir: Optional[Path] = None,
) -> dict:
    """Extract and preprocess segments for a single split dataframe."""
    from data.loader import resolve_raw_path

    df_split = pd.read_csv(split_csv_path)
    all_raw = []
    all_fgcs = []
    all_mc = []
    all_labels = []

    total_files = len(df_split)

    for idx, row in df_split.iterrows():
        rel_path = row["relative_path"]
        abs_path = resolve_raw_path(rel_path, raw_data_dir=raw_data_dir)

        if not abs_path.exists():
            continue

        drone_class = row["drone_class"]
        label = RAW_CLASS_TO_INDEX.get(drone_class, 0)

        try:
            chunk_df = pd.read_csv(abs_path, header=None, nrows=max_segments_per_file)
            flat_vals = chunk_df.to_numpy().reshape(-1)
            flat_vals = pd.to_numeric(flat_vals, errors="coerce")
            flat_vals = flat_vals[~np.isnan(flat_vals)].astype(np.float32)

            num_segments = len(flat_vals) // segment_length
            if num_segments == 0:
                continue

            for s_idx in range(num_segments):
                seg = flat_vals[s_idx * segment_length : (s_idx + 1) * segment_length]

                fgcs_spec = preprocessor.process_fgcs(seg)
                mc_spec = preprocessor.process_multichannel(seg)

                seg_norm = seg - np.mean(seg)
                max_abs = np.max(np.abs(seg_norm))
                if max_abs > 1e-8:
                    seg_norm /= max_abs

                all_raw.append(seg_norm)
                all_fgcs.append(fgcs_spec)
                all_mc.append(mc_spec)
                all_labels.append(label)

        except Exception as e:
            print(f"Skipping {abs_path.name}: {e}")
            continue

    if not all_raw:
        return {
            "x_raw": np.empty((0, segment_length), dtype=np.float32),
            "x_fgcs": np.empty((0, segment_length), dtype=np.float32),
            "x_multichannel": np.empty((0, 8, 256), dtype=np.float32),
            "y_4class": np.empty((0,), dtype=np.int64),
        }

    return {
        "x_raw": np.array(all_raw, dtype=np.float32),
        "x_fgcs": np.array(all_fgcs, dtype=np.float32),
        "x_multichannel": np.array(all_mc, dtype=np.float32),
        "y_4class": np.array(all_labels, dtype=np.int64),
    }


def load_and_preprocess_dataset(
    splits_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_segments_per_file: int = 50,
    segment_length: int = 2048,
    raw_data_dir: Optional[Path] = None,
):
    """
    Preprocess DroneRF dataset split-by-split using pre-computed recording-level split CSVs.
    Guarantees strict isolation between Train, Val, and Test.
    """
    start_time = time.time()

    print("=========================================================")
    print("      Vardan Counter-UAS Dataset Preprocessing          ")
    print("=========================================================")

    splits_path = Path(splits_dir) if splits_dir else DATA_DIR / "splits"
    out_path = Path(output_dir) if output_dir else PROCESSED_DATA_DIR / "DroneRF"
    out_path.mkdir(parents=True, exist_ok=True)

    preprocessor = DroneRFPreprocessor(
        fft_size=FFT_SIZE,
        remove_dc=True,
        normalization="max",
        channel_count=8,
        channel_overlap=0.50,
        fs=SAMPLING_RATE,
    )

    for split_name in ["train", "val", "test"]:
        csv_file = splits_path / f"{split_name}.csv"
        if not csv_file.exists():
            print(f"Split CSV not found: {csv_file}. Please run create_splits.py first.")
            return

        print(f"\nProcessing {split_name} split from {csv_file.name}...")
        split_data = preprocess_split(
            split_csv_path=csv_file,
            preprocessor=preprocessor,
            max_segments_per_file=max_segments_per_file,
            segment_length=segment_length,
            raw_data_dir=raw_data_dir,
        )

        n_samples = len(split_data["y_4class"])
        if n_samples > 0:
            save_path = out_path / f"{split_name}.npz"
            np.savez_compressed(
                save_path,
                x_raw=split_data["x_raw"],
                x_fgcs=split_data["x_fgcs"],
                x_multichannel=split_data["x_multichannel"],
                y_4class=split_data["y_4class"],
            )
            print(f" [OK] {split_name}.npz ({n_samples} samples) saved to {save_path.name}")
        else:
            print(f" [Notice] 0 segments processed for {split_name} (raw data not present locally; run on Kaggle).")

    elapsed = time.time() - start_time
    print(f"\n=========================================================")
    print(f" Preprocessing completed in {elapsed:.2f}s!")
    print(f"=========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DroneRF dataset based on recording-level split manifests.")
    parser.add_argument("--splits_dir", type=str, default=None, help="Path to splits directory (containing train.csv, val.csv, test.csv)")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory for .npz files")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Path to raw DroneRF dataset")
    parser.add_argument("--max_segments_per_file", type=int, default=50, help="Max segments to extract per CSV file")
    parser.add_argument("--segment_length", type=int, default=2048, help="Sample count per segment")
    args = parser.parse_args()

    load_and_preprocess_dataset(
        splits_dir=Path(args.splits_dir) if args.splits_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_segments_per_file=args.max_segments_per_file,
        segment_length=args.segment_length,
        raw_data_dir=Path(args.raw_data_dir) if args.raw_data_dir else None,
    )
