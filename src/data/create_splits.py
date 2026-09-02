"""
create_splits.py
----------------
Deterministic, reproducible stratified recording-level train/validation/test splitting
for the DroneRF dataset metadata.

Guarantees:
- Recording-level isolation: All CSV segments from a single continuous recording session
  (identified by drone_class + experiment_id + receiver) remain strictly together in the same split.
- Zero data leakage: 0 recording overlap and 0 file overlap between Train, Val, and Test.
- Stratification: Every canonical class appears in all three splits.
- Reproducibility: Fully deterministic given random_seed (default 42).
- Saves split manifests to:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constants import RAW_CLASS_TO_INDEX
from utils.paths import DATA_DIR, ensure_directories


def generate_deterministic_splits(
    metadata_path: Path = None,
    output_dir: Path = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate stratified recording-level train/val/test split CSVs ensuring zero recording leakage."""
    ensure_directories()
    
    meta_path = Path(metadata_path) if metadata_path else DATA_DIR / "metadata" / "dronerf_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata index not found at {meta_path}")

    df_meta = pd.read_csv(meta_path)
    total_raw_files = len(df_meta)
    print(f"Loaded metadata index with {total_raw_files} file entries from {meta_path.name}.")

    # Form unique recording session identifier: drone_class + experiment_id + receiver
    df_meta["recording_id"] = (
        df_meta["drone_class"].astype(str)
        + "_"
        + df_meta["experiment_id"].astype(str)
        + "_"
        + df_meta["receiver"].astype(str)
    )

    splits_dir = Path(output_dir) if output_dir else DATA_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(random_seed)

    classes = sorted(df_meta["drone_class"].unique())
    print(f"\nDiscovered {len(classes)} classes across {df_meta['recording_id'].nunique()} unique recording sessions:")

    train_recordings = []
    val_recordings = []
    test_recordings = []

    recording_summary = []

    for drone_cls in classes:
        cls_df = df_meta[df_meta["drone_class"] == drone_cls]
        cls_recordings = sorted(cls_df["recording_id"].unique())
        n_total_recs = len(cls_recordings)

        # Shuffle recordings deterministically per class
        shuffled_recs = list(cls_recordings)
        rng.shuffle(shuffled_recs)

        n_test = max(1, int(round(n_total_recs * test_ratio)))
        n_val = max(1, int(round(n_total_recs * val_ratio)))
        n_train = n_total_recs - n_val - n_test

        if n_train < 1:
            raise ValueError(f"Not enough recordings for class '{drone_cls}' to populate Train, Val, and Test splits.")

        te_recs = shuffled_recs[:n_test]
        va_recs = shuffled_recs[n_test : n_test + n_val]
        tr_recs = shuffled_recs[n_test + n_val :]

        train_recordings.extend(tr_recs)
        val_recordings.extend(va_recs)
        test_recordings.extend(te_recs)

        recording_summary.append({
            "class": drone_cls,
            "total_recordings": n_total_recs,
            "train_recordings": len(tr_recs),
            "val_recordings": len(va_recs),
            "test_recordings": len(te_recs),
            "total_files": len(cls_df),
        })

    # Build split DataFrames from assigned recording IDs
    df_train = df_meta[df_meta["recording_id"].isin(train_recordings)].sort_values(["drone_class", "relative_path"]).reset_index(drop=True)
    df_val = df_meta[df_meta["recording_id"].isin(val_recordings)].sort_values(["drone_class", "relative_path"]).reset_index(drop=True)
    df_test = df_meta[df_meta["recording_id"].isin(test_recordings)].sort_values(["drone_class", "relative_path"]).reset_index(drop=True)

    # -------------------------------------------------------------------
    # VERIFICATION: INTEGRITY & ZERO LEAKAGE
    # -------------------------------------------------------------------
    total_split_files = len(df_train) + len(df_val) + len(df_test)
    assert total_split_files == total_raw_files, f"File count mismatch: expected {total_raw_files}, got {total_split_files}"

    # Zero recording overlap
    tr_recs_set = set(df_train["recording_id"])
    va_recs_set = set(df_val["recording_id"])
    te_recs_set = set(df_test["recording_id"])

    assert len(tr_recs_set & va_recs_set) == 0, "ERROR: Recording overlap between Train and Val!"
    assert len(tr_recs_set & te_recs_set) == 0, "ERROR: Recording overlap between Train and Test!"
    assert len(va_recs_set & te_recs_set) == 0, "ERROR: Recording overlap between Val and Test!"

    # Zero file overlap
    train_paths = set(df_train["relative_path"])
    val_paths = set(df_val["relative_path"])
    test_paths = set(df_test["relative_path"])

    assert len(train_paths & val_paths) == 0, "ERROR: File overlap between Train and Val!"
    assert len(train_paths & test_paths) == 0, "ERROR: File overlap between Train and Test!"
    assert len(val_paths & test_paths) == 0, "ERROR: File overlap between Val and Test!"

    # All classes present in all splits
    for split_name, df_s in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        split_classes = set(df_s["drone_class"])
        assert len(split_classes) == len(classes), f"Split {split_name} missing classes: expected {len(classes)}, got {len(split_classes)}"

    # Save split manifests
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print("\n[OK] Recording-level stratified splits successfully generated:")
    print(f"  - Train: {len(df_train)} files across {len(tr_recs_set)} recordings ({train_csv.name})")
    print(f"  - Val:   {len(df_val)} files across {len(va_recs_set)} recordings ({val_csv.name})")
    print(f"  - Test:  {len(df_test)} files across {len(te_recs_set)} recordings ({test_csv.name})")
    print(f"  - Total: {total_split_files} files across {df_meta['recording_id'].nunique()} recordings.")

    print("\nRecording Distribution by Class:")
    print("class                    | Train Recs | Val Recs | Test Recs | Total Recs")
    print("--------------------------------------------------------------------------")
    for r in recording_summary:
        print(f"{r['class']:24s} | {r['train_recordings']:10d} | {r['val_recordings']:8d} | {r['test_recordings']:9d} | {r['total_recordings']:10d}")
    print("--------------------------------------------------------------------------")

    print("\nFile Distribution by Class:")
    print("class                    | Train Files | Val Files | Test Files | Total Files")
    print("--------------------------------------------------------------------------")
    for drone_cls in classes:
        tr_c = len(df_train[df_train["drone_class"] == drone_cls])
        va_c = len(df_val[df_val["drone_class"] == drone_cls])
        te_c = len(df_test[df_test["drone_class"] == drone_cls])
        tot_c = tr_c + va_c + te_c
        print(f"{drone_cls:24s} | {tr_c:11d} | {va_c:9d} | {te_c:10d} | {tot_c:11d}")
    print("--------------------------------------------------------------------------")

    return df_train, df_val, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recording-level stratified splits for DroneRF metadata.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Path to dronerf_metadata.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save train.csv, val.csv, test.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_deterministic_splits(
        metadata_path=Path(args.metadata_path) if args.metadata_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        random_seed=args.seed,
    )
