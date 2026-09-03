"""
train_baseline1dcnn_faithful_reproduction.py
--------------------------------------------

Master Training & 10-Fold Cross-Validation Runner for Faithful MC1DCNN Reproduction.
Reproduction of the Multi-Channel 1D CNN methodology from Allahham et al. (IEEE ICIoT 2020)
applied to the complete available DroneRF dataset.

Experiment Specifications:
- Dataset: All 227 synchronized (L, H) recording pairs -> 22,700 segments of 100,000 samples.
- Preprocessing: Al-Sa'd faithful preprocessing (100k DC removal, 2048-point DFT, Q=10 scaling, global max normalization).
- Channelization: 2048-point full-spectrum power representation reshaped to (8, 256),
  representing 8 uniform 10 MHz sub-bands across the 80 MHz spectrum.
- Model: Baseline1DCNN (2-stage Conv1d + MaxPool1d + MLP, 275,940 parameters).
- Training: CrossEntropyLoss, Adam optimizer (lr=0.001), batch_size=32, epochs=100, seed=1.
  (Note: Hyperparameters lr/batch_size/epochs and exact layer dimensions are project-specific choices
  consistent with the paper's architecture, as specific numerical values were omitted from the paper text).
- Evaluation: StratifiedKFold (n_splits=10, shuffle=True, random_state=1) across segments.
  (Note: Segment-level CV reproduces the published-style protocol; segments from the same recording session
  can appear in both train and test folds, which is not recording-level isolated).
- Results Output: results/baseline1dcnn_faithful_reproduction/
  - fold_metrics.csv
  - aggregate_metrics.json
  - confusion_matrix.csv
  - run_config.json
- Checkpoints Output: models/checkpoints/baseline1dcnn_faithful_reproduction/
  - checkpoint_fold_1.pt ... checkpoint_fold_10.pt
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.fgcs_faithful import (
    FAITHFUL_FGCS_CLASS_TO_INDEX,
    FAITHFUL_FGCS_INDEX_TO_CLASS,
    BUI_TO_CLASS,
    normalize_global_max,
)
from data.fgcs_faithful_loader import (
    build_faithful_manifest,
    discover_and_pair_dronerf_files,
    read_raw_signal_10m,
    process_faithful_fgcs_pair_vectorized,
)
from models.baselines import Baseline1DCNN


def set_seed(seed: int = 1) -> None:
    """Set random seeds for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preflight_verification(
    df_manifest: pd.DataFrame,
    num_folds: int = 10,
    seed: int = 1,
) -> bool:
    """
    Verify complete integrity of dataset manifest before training:
    - Exactly 22,700 samples (227 pairs * 100 segments)
    - Class counts: 4100 / 8400 / 8100 / 2100
    - Model forward-pass produces (B, 4) with parameter count = 275,940
    - All 4 classes present in every fold
    """
    print("\n" + "=" * 75)
    print("      FAITHFUL MC1DCNN PREFLIGHT VERIFICATION CHECK                  ")
    print("=" * 75)

    total_samples = len(df_manifest)
    print(f"1. Total Dataset Samples: {total_samples}")
    assert total_samples == 22700, f"Expected 22,700 samples, found {total_samples}"

    # Class distribution check
    counts = df_manifest["faithful_label"].value_counts().to_dict()
    print("2. Class Segment Distribution:")
    expected_counts = {0: 4100, 1: 8400, 2: 8100, 3: 2100}
    for lbl in range(4):
        cls_name = FAITHFUL_FGCS_INDEX_TO_CLASS[lbl]
        cnt = counts.get(lbl, 0)
        exp = expected_counts[lbl]
        print(f"   - Label {lbl} ({cls_name:25s}): {cnt:5d} segments (Expected: {exp:5d})")
        assert cnt == exp, f"Class {lbl} count mismatch: got {cnt}, expected {exp}"

    # Model architecture verification
    dummy_model = Baseline1DCNN(in_channels=8, num_classes=4, seq_length=256)
    dummy_x = torch.zeros(2, 8, 256)
    dummy_out = dummy_model(dummy_x)
    param_count = sum(p.numel() for p in dummy_model.parameters())
    print(f"\n3. Model Architecture Verification:")
    print(f"   - Input Shape:        (B, 8, 256)")
    print(f"   - Output Logits Shape: {dummy_out.shape}")
    print(f"   - Total Parameters:   {param_count:,}")
    assert dummy_out.shape == (2, 4), f"Expected (2, 4), got {dummy_out.shape}"
    assert param_count == 275940, f"Expected 275,940 params, got {param_count}"

    # Stratified K-Fold balance verification
    print(f"\n4. Verifying Stratified {num_folds}-Fold Split Distribution:")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    labels = df_manifest["faithful_label"].to_numpy()
    dummy_samples = np.zeros(len(labels))

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(dummy_samples, labels), 1):
        tr_lbls = labels[tr_idx]
        te_lbls = labels[te_idx]
        assert len(np.unique(tr_lbls)) == 4, f"Fold {fold_idx} train is missing classes!"
        assert len(np.unique(te_lbls)) == 4, f"Fold {fold_idx} test is missing classes!"
        if fold_idx <= 2:
            print(f"   - Fold {fold_idx:2d}: Train={len(tr_idx):5d}, Test={len(te_idx):5d}, All 4 classes present")

    print(f"   - ... (all {num_folds} folds verified with all 4 classes present)")
    print("\n[PREFLIGHT PASS] Dataset manifest, model structure, and fold partition verified.")
    print("=" * 75 + "\n")
    return True


def materialize_mc1dcnn_features(
    df_manifest: pd.DataFrame,
    mock: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract, globally normalize, and channelize all 22,700 feature vectors into memory.
    Converts 2048-dim power spectrum into 8-channel x 256-bin sub-band representations: (N, 8, 256).
    Total memory footprint: 22,700 x 8 x 256 x 4 bytes ~= 185.95 MB.
    """
    n_samples = len(df_manifest)
    X = np.zeros((n_samples, 8, 256), dtype=np.float32)
    y = df_manifest["faithful_label"].to_numpy(dtype=np.int64)

    if mock:
        print(f"Generating deterministic mock (8, 256) features for {n_samples} samples...")
        for i in range(n_samples):
            rng = np.random.RandomState(i)
            feat_2048 = rng.rand(2048).astype(np.float32)
            X[i] = feat_2048.reshape(8, 256)
        return X, y

    print(f"\nMaterializing 8-channel (8, 256) features across {n_samples} segments (227 pairs)...")
    start_all = time.time()

    bui_groups = df_manifest.groupby("bui", sort=False)
    total_buis = len(bui_groups)

    for bui_idx, (bui_name, group_df) in enumerate(bui_groups, 1):
        t_bui_start = time.time()
        cls_name = group_df["drone_class"].iloc[0]
        lbl_idx = group_df["faithful_label"].iloc[0]

        pair_groups = group_df.groupby("pair_id", sort=False)
        n_pairs = len(pair_groups)
        n_segments = len(group_df)
        bui_features_2048 = np.zeros((n_segments, 2048), dtype=np.float32)

        pair_offset = 0
        for pair_id, pair_df in pair_groups:
            first_row = pair_df.iloc[0]
            l_path = first_row["l_path"]
            h_path = first_row["h_path"]
            l_rar = first_row.get("l_rar")
            h_rar = first_row.get("h_rar")
            l_inner = first_row.get("l_inner")
            h_inner = first_row.get("h_inner")

            # 1. Read raw signals (10M samples)
            raw_l = read_raw_signal_10m(l_path, rar_path=l_rar, inner_file=l_inner)
            raw_h = read_raw_signal_10m(h_path, rar_path=h_rar, inner_file=h_inner)

            # 2. Vectorized 100-segment processing -> (100, 2048)
            pair_features = process_faithful_fgcs_pair_vectorized(
                raw_l=raw_l,
                raw_h=raw_h,
                q=10,
                m=2048,
                segments_per_pair=100,
                segment_length=100000,
            )

            bui_features_2048[pair_offset : pair_offset + 100] = pair_features
            pair_offset += 100
            del raw_l, raw_h

        # 3. Per-BUI global max normalization matching Matlab/Main_2_Data_labeling.m
        bui_norm_2048, g_max = normalize_global_max(bui_features_2048)

        # 4. Channelize into 8 equal non-overlapping sub-bands of 256 bins: (N, 8, 256)
        bui_channels = bui_norm_2048.reshape(-1, 8, 256)

        indices = group_df.index.to_numpy()
        X[indices] = bui_channels

        t_bui_elapsed = time.time() - t_bui_start
        print(f"  [{bui_idx:2d}/{total_buis:2d}] BUI '{bui_name}' ({cls_name:22s}, Label {lbl_idx}) | "
              f"{n_pairs:2d} pairs ({n_segments:5d} segments) in {t_bui_elapsed:5.1f}s | Global Max: {g_max:.4e}")

    total_time = time.time() - start_all
    print(f"\n[DONE] All {n_samples} (8, 256) feature tensors materialized in {total_time:.1f}s ({total_time/60:.2f} min).")
    assert np.all(np.isfinite(X)), "Materialized features contain NaN or Inf"
    return X, y


def train_single_fold(
    fold_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    checkpoints_dir: Optional[Path] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Train and evaluate a single fold for Baseline1DCNN:
    - Model: Baseline1DCNN(in_channels=8, num_classes=4, seq_length=256)
    - Loss: CrossEntropyLoss (Eq. 3 in Allahham et al.)
    - Optimizer: Adam(lr=0.001)
    - Batch size: 32
    - Epochs: 100
    """
    model = Baseline1DCNN(in_channels=8, num_classes=4, seq_length=256)
    model.to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n--- Training Fold {fold_idx}/10 ({len(X_train)} train, {len(X_test)} test, {epochs} epochs) ---")
    start_t = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            logits = model(b_x)
            loss = loss_fn(logits, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_x)

        epoch_loss = total_loss / len(X_train)
        if epoch % 25 == 0 or epoch == epochs or epoch == 1:
            print(f"  [Fold {fold_idx:2d}] Epoch {epoch:3d}/{epochs:3d} | Train CE Loss: {epoch_loss:.6f}")

    train_time = time.time() - start_t

    # Evaluation on Test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_loss = loss_fn(test_logits, y_test_tensor).item()
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
    test_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)

    print(f"  --> Fold {fold_idx:2d} Result: Accuracy = {test_acc*100:.2f}%, Macro-F1 = {test_f1:.4f}, Time = {train_time:.1f}s")

    # Save fold checkpoint
    if checkpoints_dir is not None:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoints_dir / f"checkpoint_fold_{fold_idx}.pt"
        torch.save({
            "fold": fold_idx,
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_loss": test_loss,
        }, ckpt_path)

    return {
        "fold": fold_idx,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_loss": epoch_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "test_precision_macro": test_prec,
        "test_recall_macro": test_rec,
        "train_time_sec": train_time,
        "y_true": y_test,
        "y_pred": test_preds,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="EXP_BASELINE1DCNN_FAITHFUL 10-Fold Reproduction Experiment")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Base path to raw DroneRF dataset")
    parser.add_argument("--output_dir", type=str, default="results/baseline1dcnn_faithful_reproduction", help="Results export directory")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints/baseline1dcnn_faithful_reproduction", help="Checkpoints export directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs per fold (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam (default: 0.001)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--num_folds", type=int, default=10, help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--max_folds", type=int, default=None, help="Maximum number of folds to execute (default: all folds, use 1 for smoke testing)")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic mock data for preflight verification")
    parser.add_argument("--preflight_only", action="store_true", help="Perform preflight checks and exit without training")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    checkpoints_dir = PROJECT_ROOT / args.checkpoints_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 75)
    print("EXP_BASELINE1DCNN_FAITHFUL: 10-FOLD REPRODUCTION RUNNER")
    print("=" * 75)
    print(f"Device:          {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Seed:            {args.seed}")
    print(f"Folds:           {args.num_folds}" + (f" (executing up to fold {args.max_folds})" if args.max_folds else ""))
    print(f"Epochs per fold: {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Learning rate:   {args.lr}")
    print(f"Results Dir:     {output_dir}")
    print(f"Checkpoints Dir: {checkpoints_dir}")

    # Build or verify manifest
    df_manifest = build_faithful_manifest(raw_data_dir=args.raw_data_dir, segments_per_pair=100)
    preflight_ok = preflight_verification(df_manifest, num_folds=args.num_folds, seed=args.seed)
    assert preflight_ok, "Preflight verification failed"

    run_config = {
        "experiment_name": "EXP_BASELINE1DCNN_FAITHFUL",
        "description": "Faithful 10-fold reproduction of Allahham et al. MC1DCNN on complete DroneRF benchmark",
        "timestamp": datetime.datetime.now().isoformat(),
        "num_folds": args.num_folds,
        "max_folds": args.max_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "feature_dim": "8x256 (8 uniform sub-bands of 10 MHz each from 2048-pt DFT)",
        "num_classes": 4,
        "class_mapping": FAITHFUL_FGCS_CLASS_TO_INDEX,
        "architecture": "Conv1d(8->32, k=11, s=2, p=5) -> ReLU -> MaxPool1d(2) -> Conv1d(32->64, k=5, s=1, p=2) -> ReLU -> MaxPool1d(2) -> Flatten(2048) -> Linear(2048->128) -> ReLU -> Linear(128->4)",
        "parameter_count": 275940,
        "loss": "CrossEntropyLoss",
        "optimizer": "Adam",
        "normalization": "Per-BUI mode global max scaling (Main_2_Data_labeling.m)",
        "cv_protocol": "StratifiedKFold(n_splits=10, shuffle=True, random_state=1) across segments (unisolated segment-level CV)",
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    if args.preflight_only:
        print("\n[--preflight_only specified] Exiting cleanly without starting training.")
        return

    # Materialize features -> (22700, 8, 256)
    X, y = materialize_mc1dcnn_features(df_manifest, mock=args.mock)

    # Execute 10-fold cross validation
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_records: List[Dict] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    print("\n" + "=" * 75)
    print(f"STARTING {args.num_folds}-FOLD STRATIFIED CROSS VALIDATION")
    print("=" * 75)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        if args.max_folds is not None and fold_idx > args.max_folds:
            print(f"\n[--max_folds {args.max_folds} reached] Stopping cross-validation early.")
            break

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        res = train_single_fold(
            fold_idx=fold_idx,
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoints_dir=checkpoints_dir,
        )

        all_y_true.extend(res["y_true"])
        all_y_pred.extend(res["y_pred"])

        fold_records.append({
            "fold": res["fold"],
            "train_samples": res["train_samples"],
            "test_samples": res["test_samples"],
            "train_loss": res["train_loss"],
            "test_loss": res["test_loss"],
            "test_accuracy": res["test_accuracy"],
            "test_f1_macro": res["test_f1_macro"],
            "test_precision_macro": res["test_precision_macro"],
            "test_recall_macro": res["test_recall_macro"],
            "train_time_sec": res["train_time_sec"],
        })

    # Export fold metrics CSV
    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(output_dir / "fold_metrics.csv", index=False)

    # Compute aggregate metrics
    mean_acc = float(df_folds["test_accuracy"].mean())
    std_acc = float(df_folds["test_accuracy"].std())
    mean_f1 = float(df_folds["test_f1_macro"].mean())
    std_f1 = float(df_folds["test_f1_macro"].std())

    # Overall 4x4 confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{FAITHFUL_FGCS_INDEX_TO_CLASS[i]}" for i in range(4)],
        columns=[f"Pred_{FAITHFUL_FGCS_INDEX_TO_CLASS[i]}" for i in range(4)],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    aggregate_summary = {
        "experiment": "EXP_BASELINE1DCNN_FAITHFUL",
        "num_folds": len(fold_records),
        "mean_test_accuracy": mean_acc,
        "std_test_accuracy": std_acc if len(fold_records) > 1 else 0.0,
        "mean_test_f1_macro": mean_f1,
        "std_test_f1_macro": std_f1 if len(fold_records) > 1 else 0.0,
        "total_test_predictions": len(all_y_true),
        "confusion_matrix": cm.tolist(),
        "class_labels": FAITHFUL_FGCS_INDEX_TO_CLASS,
    }
    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    print("\n" + "=" * 75)
    print("FAITHFUL MC1DCNN 10-FOLD CROSS VALIDATION COMPLETE")
    print("=" * 75)
    print(f"Mean Test Accuracy: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
    print(f"Mean Macro-F1:      {mean_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Results exported to: {output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
