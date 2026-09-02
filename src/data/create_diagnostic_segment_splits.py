"""
create_diagnostic_segment_splits.py
-----------------------------------

Diagnostic randomized segment-level train/val/test split generator for DroneRF.

Purpose:
Create a deliberately less strict, segment-level/randomized evaluation split using the
SAME current DroneRF raw files and SAME current 2048-sample window extraction.

Allows different 2048-sample windows from the same recording/file to appear across
Train, Validation, and Test splits while ensuring that the exact same window
(relative_path + offset) is never duplicated across splits.

This experiment is specifically designed to isolate and measure the performance
difference between:
1. Strict recording-level isolation (Primary Benchmark)
2. Randomized segment-level splitting (Published Al-Sa'd et al. 2019 / Allahham et al. 2020 protocol)
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constants import RAW_CLASS_TO_INDEX
from utils.paths import DATA_DIR, ensure_directories


def generate_diagnostic_segment_splits(
    metadata_path: Path = None,
    output_dir: Path = None,
    samples_per_file: int = 5,
    segment_length: int = 2048,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Generate randomized segment-level stratified train/val/test splits."""
    ensure_directories()

    meta_path = Path(metadata_path) if metadata_path else DATA_DIR / "metadata" / "dronerf_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata index not found at {meta_path}")

    df_meta = pd.read_csv(meta_path)
    total_raw_files = len(df_meta)

    # 1. Build all discrete segment records
    segment_records = []
    for idx, row in df_meta.iterrows():
        rec_id = f"{row['drone_class']}_{row['experiment_id']}_{row['receiver']}"
        rel_p = row["relative_path"]
        drone_cls = row["drone_class"]
        exp_id = row["experiment_id"]
        receiver = row["receiver"]
        file_seg_id = row["segment_id"]

        for offset in range(samples_per_file):
            seg_uid = f"{rel_p}#offset_{offset}"
            segment_records.append({
                "drone_class": drone_cls,
                "experiment_id": exp_id,
                "receiver": receiver,
                "file_segment_id": file_seg_id,
                "relative_path": rel_p,
                "recording_id": rec_id,
                "segment_offset": offset,
                "window_start_sample": offset * segment_length,
                "window_end_sample": (offset + 1) * segment_length,
                "segment_unique_id": seg_uid,
            })

    df_segments = pd.DataFrame(segment_records)
    total_segments = len(df_segments)

    # 2. Stratified randomized partitioning per class
    rng = np.random.RandomState(random_seed)
    classes = sorted(df_segments["drone_class"].unique())

    train_rows = []
    val_rows = []
    test_rows = []

    for drone_cls in classes:
        cls_df = df_segments[df_segments["drone_class"] == drone_cls].copy()
        n_cls = len(cls_df)

        # Deterministic shuffle
        shuffled_indices = rng.permutation(n_cls)
        cls_shuffled = cls_df.iloc[shuffled_indices].reset_index(drop=True)

        n_test = int(round(n_cls * test_ratio))
        n_val = int(round(n_cls * val_ratio))
        n_train = n_cls - n_val - n_test

        te_df = cls_shuffled.iloc[:n_test]
        va_df = cls_shuffled.iloc[n_test : n_test + n_val]
        tr_df = cls_shuffled.iloc[n_test + n_val :]

        train_rows.append(tr_df)
        val_rows.append(va_df)
        test_rows.append(te_df)

    df_train = pd.concat(train_rows, ignore_index=True)
    df_val = pd.concat(val_rows, ignore_index=True)
    df_test = pd.concat(test_rows, ignore_index=True)

    # 3. Verification of Zero Window Leakage
    train_uids = set(df_train["segment_unique_id"])
    val_uids = set(df_val["segment_unique_id"])
    test_uids = set(df_test["segment_unique_id"])

    assert len(train_uids.intersection(val_uids)) == 0, "Train and Val share exact segment windows!"
    assert len(train_uids.intersection(test_uids)) == 0, "Train and Test share exact segment windows!"
    assert len(val_uids.intersection(test_uids)) == 0, "Val and Test share exact segment windows!"
    assert len(df_train) + len(df_val) + len(df_test) == total_segments

    # 4. Compute overlap metrics (Intentional for segment-level diagnostics)
    train_files = set(df_train["relative_path"])
    val_files = set(df_val["relative_path"])
    test_files = set(df_test["relative_path"])

    train_recs = set(df_train["recording_id"])
    val_recs = set(df_val["recording_id"])
    test_recs = set(df_test["recording_id"])

    file_overlap_tr_va = len(train_files.intersection(val_files))
    file_overlap_tr_te = len(train_files.intersection(test_files))
    file_overlap_va_te = len(val_files.intersection(test_files))

    rec_overlap_tr_va = len(train_recs.intersection(val_recs))
    rec_overlap_tr_te = len(train_recs.intersection(test_recs))
    rec_overlap_va_te = len(val_recs.intersection(test_recs))

    # 5. Class Distribution Summary
    class_dist = {}
    for cls in classes:
        tr_c = len(df_train[df_train["drone_class"] == cls])
        va_c = len(df_val[df_val["drone_class"] == cls])
        te_c = len(df_test[df_test["drone_class"] == cls])
        tot_c = tr_c + va_c + te_c
        class_dist[cls] = {
            "train": tr_c,
            "val": va_c,
            "test": te_c,
            "total": tot_c,
            "train_pct": round(tr_c / tot_c * 100, 2),
            "val_pct": round(va_c / tot_c * 100, 2),
            "test_pct": round(te_c / tot_c * 100, 2),
        }

    # 6. Save Manifests
    out_dir = Path(output_dir) if output_dir else DATA_DIR / "splits" / "diagnostic_segment"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)

    metadata = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "Diagnostic randomized segment-level split to measure performance impact of recording-level isolation vs segment-level evaluation (Al-Sa'd et al. 2019 / Allahham et al. 2020 protocol).",
        "random_seed": random_seed,
        "segment_length": segment_length,
        "samples_per_file": samples_per_file,
        "number_of_source_files": total_raw_files,
        "number_of_generated_segments": total_segments,
        "split_segment_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
            "total": total_segments,
            "train_pct": round(len(df_train) / total_segments * 100, 2),
            "val_pct": round(len(df_val) / total_segments * 100, 2),
            "test_pct": round(len(df_test) / total_segments * 100, 2),
        },
        "exact_window_overlap": {
            "train_val_window_overlap": len(train_uids.intersection(val_uids)),
            "train_test_window_overlap": len(train_uids.intersection(test_uids)),
            "val_test_window_overlap": len(val_uids.intersection(test_uids)),
            "zero_exact_window_leakage": True,
        },
        "file_overlap_across_splits": {
            "train_unique_files": len(train_files),
            "val_unique_files": len(val_files),
            "test_unique_files": len(test_files),
            "train_val_shared_files": file_overlap_tr_va,
            "train_test_shared_files": file_overlap_tr_te,
            "val_test_shared_files": file_overlap_va_te,
        },
        "recording_overlap_across_splits": {
            "train_unique_recordings": len(train_recs),
            "val_unique_recordings": len(val_recs),
            "test_unique_recordings": len(test_recs),
            "train_val_shared_recordings": rec_overlap_tr_va,
            "train_test_shared_recordings": rec_overlap_tr_te,
            "val_test_shared_recordings": rec_overlap_va_te,
        },
        "class_distribution": class_dist,
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Diagnostic segment splits generated successfully in: {out_dir}")
    print(f"     Train Segments: {len(df_train)} ({metadata['split_segment_counts']['train_pct']}%)")
    print(f"     Val Segments:   {len(df_val)} ({metadata['split_segment_counts']['val_pct']}%)")
    print(f"     Test Segments:  {len(df_test)} ({metadata['split_segment_counts']['test_pct']}%)")
    print(f"     Total Segments: {total_segments}")
    print(f"     Exact Window Leakage: ZERO ({len(train_uids.intersection(val_uids))} shared)")
    print(f"     Source File Overlap (Intentional): {file_overlap_tr_va} Train-Val, {file_overlap_tr_te} Train-Test")
    print(f"     Recording Overlap (Intentional):   {rec_overlap_tr_va} Train-Val, {rec_overlap_tr_te} Train-Test")

    return df_train, df_val, df_test, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diagnostic segment-level splits for DroneRF.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Path to dronerf_metadata.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for split CSVs and metadata.json")
    parser.add_argument("--samples_per_file", type=int, default=5, help="Number of 2048-sample segments per CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    args = parser.parse_args()

    generate_diagnostic_segment_splits(
        metadata_path=Path(args.metadata_path) if args.metadata_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        samples_per_file=args.samples_per_file,
        random_seed=args.seed,
    )
