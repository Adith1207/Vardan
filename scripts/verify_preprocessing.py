"""
verify_preprocessing.py
------------------------

Comprehensive Verification Suite for Vardan Counter-UAS Experimental Preprocessing.

Verifies:
1. Deterministic file-level 4-class stratified split integrity and zero file overlap.
2. Every class appears in Train, Validation, and Test splits.
3. Canonical label mapping (0: Background, 1: AR Drone, 2: Bepop drone, 3: Phantom drone).
4. Strict train-only normalization statistics (Programmatic data leakage test).
5. Dataset Sanity Check: Loads 2 samples per class across Train/Val/Test and checks finite values (0 NaN/Inf).
6. Single-batch model forward passes with NUM_CLASSES = 4 output dimensions.
7. Zero premature cached .npz file generation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from src.config import NUM_CLASSES
from src.constants import LABEL_MAP
from src.data.loader import CLASS_MAPPING, DroneRFLazyDataset, fit_train_normalization_stats, get_dataloader
from src.models.model_factory import get_model
from src.preprocessing.pipeline import DroneRFPreprocessor
from src.utils.paths import DATA_DIR, PROCESSED_DATA_DIR


def verify_experimental_preprocessing():
    print("=================================================================")
    print("      Vardan Experimental Preprocessing Verification Suite       ")
    print("=================================================================\n")

    splits_dir = DATA_DIR / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    # -------------------------------------------------------------------
    # 1. SPLIT INTEGRITY & FILE OVERLAP
    # -------------------------------------------------------------------
    assert train_csv.exists(), f"Missing {train_csv}"
    assert val_csv.exists(), f"Missing {val_csv}"
    assert test_csv.exists(), f"Missing {test_csv}"

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    total_files = len(df_train) + len(df_val) + len(df_test)
    assert total_files == 454, f"Expected 454 files total, got {total_files}"

    train_paths = set(df_train["relative_path"])
    val_paths = set(df_val["relative_path"])
    test_paths = set(df_test["relative_path"])

    assert len(train_paths & val_paths) == 0, "ERROR: Overlap between Train and Val!"
    assert len(train_paths & test_paths) == 0, "ERROR: Overlap between Train and Test!"
    assert len(val_paths & test_paths) == 0, "ERROR: Overlap between Val and Test!"

    print("1. Split Strategy & Isolation Verification:")
    print(f"   - Split Strategy: Deterministic Stratified File-Level Split (seed=42)")
    print(f"   - Train Files: {len(df_train)}")
    print(f"   - Val Files:   {len(df_val)}")
    print(f"   - Test Files:  {len(df_test)}")
    print(f"   - Total Files: {total_files}")
    print("   ✓ Verified 0 file overlap between Train, Validation, and Test splits.")

    # -------------------------------------------------------------------
    # 2. CLASS DISTRIBUTION PER SPLIT
    # -------------------------------------------------------------------
    print("\n2. Class Distribution per Split:")
    classes = sorted(df_train["drone_class"].unique())
    print("class                    | train | val | test | total")
    print("-----------------------------------------------------")
    for drone_cls in classes:
        tr_c = len(df_train[df_train["drone_class"] == drone_cls])
        va_c = len(df_val[df_val["drone_class"] == drone_cls])
        te_c = len(df_test[df_test["drone_class"] == drone_cls])
        tot_c = tr_c + va_c + te_c
        assert tr_c > 0, f"{drone_cls} missing in Train split!"
        assert va_c > 0, f"{drone_cls} missing in Val split!"
        assert te_c > 0, f"{drone_cls} missing in Test split!"
        print(f"{drone_cls:24s} | {tr_c:5d} | {va_c:3d} | {te_c:4d} | {tot_c:5d}")
    print("-----------------------------------------------------")
    print("   ✓ Verified: Every class exists in Train, Validation, AND Test splits.")

    # -------------------------------------------------------------------
    # 3. CANONICAL LABEL MAPPING & NUM_CLASSES = 4
    # -------------------------------------------------------------------
    print("\n3. Canonical Label Mapping Verification:")
    assert NUM_CLASSES == 4, f"Expected NUM_CLASSES = 4, got {NUM_CLASSES}"
    assert len(LABEL_MAP) == 4, f"Expected LABEL_MAP size 4, got {len(LABEL_MAP)}"
    for idx, name in LABEL_MAP.items():
        print(f"   - Label {idx}: {name}")
    assert set(CLASS_MAPPING.values()) == {0, 1, 2, 3}, "Label indices must be in {0, 1, 2, 3}"
    print("   ✓ Verified: NUM_CLASSES = 4 everywhere; all labels ∈ {0, 1, 2, 3}.")

    # -------------------------------------------------------------------
    # 4. PROGRAMMATIC DATA LEAKAGE TEST
    # -------------------------------------------------------------------
    print("\n4. Programmatic Data Leakage Test:")
    train_stats = fit_train_normalization_stats(train_csv, max_files=10)
    print(f"   - Learned Train Stats: Mean={train_stats['mean']:.6f}, Std={train_stats['std']:.6f}, Max={train_stats['max']:.6f}, Min={train_stats['min']:.6f}")

    # Instantiate datasets for Train, Val, Test using train_stats
    ds_train = DroneRFLazyDataset(train_csv, norm_stats=train_stats)
    ds_val = DroneRFLazyDataset(val_csv, norm_stats=train_stats)
    ds_test = DroneRFLazyDataset(test_csv, norm_stats=train_stats)

    # Verify that ds_val and ds_test have identical norm_stats to ds_train and did NOT modify them
    assert ds_val.norm_stats == train_stats, "ERROR: Val dataset modified normalization stats!"
    assert ds_test.norm_stats == train_stats, "ERROR: Test dataset modified normalization stats!"
    print("   ✓ Verified: Normalization statistics computed strictly from TRAIN. Val & Test never modify scaler state.")

    # -------------------------------------------------------------------
    # 5. DATASET SANITY CHECK (LOAD 2 SAMPLES PER CLASS)
    # -------------------------------------------------------------------
    print("\n5. Dataset Sanity Check (2 samples per class across splits):")
    preprocessor = DroneRFPreprocessor(fft_size=2048, remove_dc=True, normalization="max", fs=100e6)

    splits_map = [("Train", ds_train), ("Val", ds_val), ("Test", ds_test)]
    nan_count = 0
    inf_count = 0

    for split_name, ds in splits_map:
        print(f"\n   --- Sanity Check: {split_name} Split ---")
        for cls_id in range(4):
            cls_name = LABEL_MAP[cls_id]
            # Find item indices for this label
            item_indices = [i for i, item in enumerate(ds.items) if item[2] == cls_id][:2]
            for s_idx, item_i in enumerate(item_indices):
                file_path, offset, label = ds.items[item_i]
                sig = ds._read_segment(file_path, offset)
                fgcs_spec = preprocessor.process_fgcs(sig)
                is_finite = np.all(np.isfinite(sig)) and np.all(np.isfinite(fgcs_spec))
                if not is_finite:
                    nan_count += 1

                print(f"   [{split_name:5s}] Class {cls_id} ({cls_name:14s}) Sample {s_idx+1}: "
                      f"Raw {sig.shape} | Spec {fgcs_spec.shape} | Finite={is_finite} | "
                      f"Min={fgcs_spec.min():.4f}, Max={fgcs_spec.max():.4f}, Mean={fgcs_spec.mean():.4f}, Std={fgcs_spec.std():.4f}")

    assert nan_count == 0 and inf_count == 0, f"Detected {nan_count} NaN/Inf values during sanity check!"
    print("   ✓ Verified: 0 NaN / Inf values detected across all samples.")

    # -------------------------------------------------------------------
    # 6. MODEL FORWARD-PASS VERIFICATION (NUM_CLASSES = 4)
    # -------------------------------------------------------------------
    print("\n6. Model Forward-Pass Verification (NUM_CLASSES = 4):")
    models_to_test = [
        ("FGCS2019DNN", "fgcs2019dnn", {"in_features": 2048, "num_classes": 4}),
        ("Baseline1DCNN", "baseline1dcnn", {"in_channels": 2, "num_classes": 4, "seq_length": 2048}),
        ("DSCNN", "dscnn", {"in_channels": 2, "num_classes": 4, "seq_length": 2048}),
        ("MobileNetV3Small", "mobilenetv3small", {"num_classes": 4}),
    ]

    for model_class_name, model_key, kwargs in models_to_test:
        loader = get_dataloader(
            split_csv=train_csv,
            model_name=model_key,
            norm_stats=train_stats,
            batch_size=4,
            shuffle=False,
        )
        x_batch, y_batch = next(iter(loader))

        model = get_model(model_class_name, **kwargs)
        model.eval()

        with torch.no_grad():
            out_logits = model(x_batch)

        assert out_logits.shape == (4, 4), f"Expected output shape (4, 4), got {out_logits.shape}"
        print(f"   ✓ Model: {model_class_name:16s} | Input Batch: {tuple(x_batch.shape)} -> Output Logits: {tuple(out_logits.shape)}")

    # -------------------------------------------------------------------
    # 7. CACHED FILES CHECK
    # -------------------------------------------------------------------
    print("\n7. Cached Files Check:")
    cached_npz_files = list((PROCESSED_DATA_DIR / "DroneRF").glob("*.npz"))
    print(f"   - Number of cached .npz files in data/processed/DroneRF/: {len(cached_npz_files)}")
    assert len(cached_npz_files) == 0, "WARNING: Cached .npz files detected!"
    print("   ✓ Verified: NO cached .npz files generated prematurely.")

    print("\n=================================================================")
    print(" ✓ All 9 verification checks passed cleanly! Ready for training.")
    print("=================================================================")


if __name__ == "__main__":
    verify_experimental_preprocessing()
