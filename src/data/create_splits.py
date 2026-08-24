"""
create_splits.py
----------------
Deterministic, reproducible stratified train/validation/test file-level splitting
for the DroneRF dataset metadata.

Requirements:
- seed = 42
- train / validation / test are mutually exclusive
- every one of the 4 classes MUST appear in all three splits
- preserve class proportions as closely as possible
- save split metadata to:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils.paths import DATA_DIR, ensure_directories


def generate_deterministic_splits(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
):
    """Generate stratified file-level train/val/test split CSVs ensuring all 4 classes appear in all splits."""
    ensure_directories()
    metadata_path = DATA_DIR / "metadata" / "dronerf_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata index not found at {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    print(f"Loaded metadata index with {len(df_meta)} file entries.")

    rng = np.random.RandomState(random_seed)

    splits_dir = DATA_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_indices = []
    val_indices = []
    test_indices = []

    classes = sorted(df_meta["drone_class"].unique())
    print(f"Drone classes found ({len(classes)}): {classes}")

    for drone_cls in classes:
        cls_mask = df_meta["drone_class"] == drone_cls
        file_indices = df_meta[cls_mask].index.to_numpy().copy()
        rng.shuffle(file_indices)

        n_total = len(file_indices)
        n_test = max(1, int(round(n_total * test_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_train = n_total - n_val - n_test

        te_idx = file_indices[:n_test]
        va_idx = file_indices[n_test : n_test + n_val]
        tr_idx = file_indices[n_test + n_val :]

        train_indices.extend(tr_idx)
        val_indices.extend(va_idx)
        test_indices.extend(te_idx)

    df_train = df_meta.loc[train_indices].sort_values(["drone_class", "relative_path"]).reset_index(drop=True)
    df_val = df_meta.loc[val_indices].sort_values(["drone_class", "relative_path"]).reset_index(drop=True)
    df_test = df_meta.loc[test_indices].sort_values(["drone_class", "relative_path"]).reset_index(drop=True)

    # Verification: Total files = 454
    total_split_files = len(df_train) + len(df_val) + len(df_test)
    assert total_split_files == 454, f"Expected 454 files, got {total_split_files}"

    # Verification: Zero file overlap
    train_paths = set(df_train["relative_path"])
    val_paths = set(df_val["relative_path"])
    test_paths = set(df_test["relative_path"])

    assert len(train_paths & val_paths) == 0, "ERROR: Overlap between Train and Val!"
    assert len(train_paths & test_paths) == 0, "ERROR: Overlap between Train and Test!"
    assert len(val_paths & test_paths) == 0, "ERROR: Overlap between Val and Test!"

    # Verification: All 4 classes in all 3 splits
    for split_name, df_s in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        split_classes = set(df_s["drone_class"])
        assert len(split_classes) == 4, f"Split {split_name} missing classes: expected 4, got {len(split_classes)}"

    # Save CSVs
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print("\n✓ Stratified splits successfully created and saved:")
    print(f"  - Train: {len(df_train)} files ({train_csv.name})")
    print(f"  - Val:   {len(df_val)} files ({val_csv.name})")
    print(f"  - Test:  {len(df_test)} files ({test_csv.name})")
    print(f"  - Total: {total_split_files} files.")

    print("\nclass | train | val | test | total")
    print("-----------------------------------")
    for drone_cls in classes:
        tr_c = len(df_train[df_train["drone_class"] == drone_cls])
        va_c = len(df_val[df_val["drone_class"] == drone_cls])
        te_c = len(df_test[df_test["drone_class"] == drone_cls])
        tot_c = tr_c + va_c + te_c
        print(f"{drone_cls:24s} | {tr_c:5d} | {va_c:3d} | {te_c:4d} | {tot_c:5d}")
    print("-----------------------------------")

    return df_train, df_val, df_test


if __name__ == "__main__":
    generate_deterministic_splits()
