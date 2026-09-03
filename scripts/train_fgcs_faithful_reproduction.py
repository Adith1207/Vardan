"""
train_fgcs_faithful_reproduction.py
-----------------------------------

Master Training & 10-Fold Cross-Validation Runner for EXP_FGCS_FAITHFUL.
Faithful reproduction of Al-Sa'd et al. (FGCS 2019 / Python/Classification.py).

Experiment Specifications:
- Dataset: All 227 synchronized (L, H) recording pairs -> 22,700 segments of 100,000 samples.
- Feature Representation: 2048-dimensional stitched power spectrum (|FFT|^2 with Q=10 scaling).
- Normalization: Global maximum scaling per BUI mode matching Main_2_Data_labeling.m.
- Labels: 0: Background (4100), 1: Bebop (8400), 2: AR (8100), 3: Phantom (2100).
- Model: FGCSFaithfulDNN (Linear 2048 -> 128 -> 128 -> 128 -> 4, ReLU, Sigmoid output).
- Training: MSE Loss on one-hot targets, Adam optimizer (lr=0.001), batch_size=10, epochs=200, seed=1.
- Evaluation: StratifiedKFold (n_splits=10, shuffle=True, random_state=1) across segments.
- Results Output: results/fgcs_faithful_reproduction/
  - fold_metrics.csv
  - aggregate_metrics.json
  - confusion_matrix.csv
  - run_config.json
- Checkpoints Output: models/checkpoints/fgcs_faithful_reproduction/
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
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
    process_faithful_fgcs_segment,
    normalize_global_max,
)
from data.fgcs_faithful_loader import (
    discover_and_pair_dronerf_files,
    build_faithful_manifest,
    FGCSFaithfulLazyDataset,
)
from models.fgcs_faithful_dnn import FGCSFaithfulDNN


def set_seed(seed: int = 1) -> None:
    """Set random seeds for full deterministic reproducibility matching np.random.seed(1)."""
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
    - Exactly 22,700 samples
    - Class counts: 4100 / 8400 / 8100 / 2100
    - All 4 classes present in every fold
    """
    print("\n" + "=" * 75)
    print("      FAITHFUL FGCS PREFLIGHT VERIFICATION CHECK                     ")
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

    # Stratified K-Fold balance verification
    print(f"\n3. Verifying Stratified {num_folds}-Fold Split Distribution:")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    labels = df_manifest["faithful_label"].to_numpy()
    dummy_x = np.zeros(len(labels))

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(dummy_x, labels), 1):
        tr_lbls = labels[tr_idx]
        te_lbls = labels[te_idx]
        tr_unique = np.unique(tr_lbls)
        te_unique = np.unique(te_lbls)
        assert len(tr_unique) == 4, f"Fold {fold_idx} train is missing classes!"
        assert len(te_unique) == 4, f"Fold {fold_idx} test is missing classes!"
        if fold_idx <= 2:
            print(f"   - Fold {fold_idx:2d}: Train={len(tr_idx):5d}, Test={len(te_idx):5d}, All 4 classes present")

    print(f"   - ... (all {num_folds} folds verified with all 4 classes present)")
    print("\n[PREFLIGHT PASS] Dataset manifest and fold partition verified.")
    print("=" * 75 + "\n")
    return True


def materialize_features_for_training(
    df_manifest: pd.DataFrame,
    mock: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and globally normalize all 22,700 feature vectors into memory.
    Total memory footprint: 22,700 x 2048 x 4 bytes ~= 185 MB.
    """
    n_samples = len(df_manifest)
    X = np.zeros((n_samples, 2048), dtype=np.float32)
    y = df_manifest["faithful_label"].to_numpy(dtype=np.int64)

    if mock:
        print(f"Generating deterministic mock features for {n_samples} samples...")
        for i in range(n_samples):
            rng = np.random.RandomState(i)
            X[i] = rng.rand(2048).astype(np.float32)
        X, _ = normalize_global_max(X)
        return X, y

    # Process by BUI group to match Main_2_Data_labeling.m per-BUI max normalization
    print(f"Materializing 2048-dim features across {n_samples} segments...")
    bui_groups = df_manifest.groupby("bui")

    for bui_name, group_df in bui_groups:
        indices = group_df.index.to_numpy()
        bui_features = np.zeros((len(group_df), 2048), dtype=np.float32)

        for local_idx, (_, row) in enumerate(group_df.iterrows()):
            offset = int(row["segment_offset"])
            l_path = Path(row["l_path"])
            h_path = Path(row["h_path"])

            # Read 100,000 samples
            limit = (offset + 1) * 100000 * 15 + 10000
            with open(l_path, "r") as f:
                line_l = f.readline(limit)
            with open(h_path, "r") as f:
                line_h = f.readline(limit)

            vals_l = np.fromstring(line_l.rsplit(",", 1)[0], sep=",", dtype=np.float64)
            vals_h = np.fromstring(line_h.rsplit(",", 1)[0], sep=",", dtype=np.float64)

            st = offset * 100000
            fi = st + 100000
            x_seg = vals_l[st:fi]
            y_seg = vals_h[st:fi]

            feat = process_faithful_fgcs_segment(x_seg, y_seg, q=10, m=2048)
            bui_features[local_idx] = feat

        # Per-BUI global max normalization matching Matlab/Main_2_Data_labeling.m
        bui_norm, g_max = normalize_global_max(bui_features)
        X[indices] = bui_norm
        print(f"  Processed BUI '{bui_name}' ({len(group_df):4d} segments) | Global Max: {g_max:.4e}")

    assert np.all(np.isfinite(X)), "Materialized features contain NaN or Inf"
    return X, y


def train_single_fold(
    fold_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 10,
    lr: float = 0.001,
    checkpoints_dir: Optional[Path] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Train and evaluate a single fold matching Classification.py:
    - Model: 128 -> 128 -> 128 -> 4 (Sigmoid)
    - Loss: MSE on one-hot targets
    - Optimizer: Adam(lr=0.001)
    - Batch size: 10
    - Epochs: 200 (no early stopping)
    """
    # Build model
    model = FGCSFaithfulDNN(in_features=2048, num_classes=4, architecture_mode="code")
    model.to(device)

    # Convert to one-hot tensors for MSE loss matching Keras
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_train_onehot = torch.zeros(len(y_train), 4).scatter_(1, y_train_tensor.unsqueeze(1), 1.0).to(device)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    y_test_onehot = torch.zeros(len(y_test), 4).scatter_(1, y_test_tensor.unsqueeze(1), 1.0).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_onehot),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"\n--- Training Fold {fold_idx}/10 ({len(X_train)} train, {len(X_test)} test, {epochs} epochs) ---")
    start_t = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            out_probs = model(b_x, return_logits=False)
            loss = loss_fn(out_probs, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_x)

        epoch_loss = total_loss / len(X_train)
        if epoch % 50 == 0 or epoch == epochs:
            print(f"  [Fold {fold_idx:2d}] Epoch {epoch:3d}/{epochs:3d} | Train MSE Loss: {epoch_loss:.6f}")

    train_time = time.time() - start_t

    # Evaluation on Test set
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_tensor, return_logits=False)
        test_loss = loss_fn(test_probs, y_test_onehot).item()
        test_preds = torch.argmax(test_probs, dim=1).cpu().numpy()

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
    parser = argparse.ArgumentParser(description="EXP_FGCS_FAITHFUL 10-Fold Reproduction Experiment")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Base path to raw DroneRF dataset")
    parser.add_argument("--output_dir", type=str, default="results/fgcs_faithful_reproduction", help="Results export directory")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints/fgcs_faithful_reproduction", help="Checkpoints export directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs per fold (default: 200)")
    parser.add_argument("--batch_size", type=int, default=10, help="Training batch size (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam (default: 0.001)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--num_folds", type=int, default=10, help="Number of cross-validation folds (default: 10)")
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

    # Device resolution: single GPU if cuda requested
    if args.device is not None:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 75)
    print("EXP_FGCS_FAITHFUL: 10-FOLD REPRODUCTION RUNNER")
    print("=" * 75)
    print(f"Device:          {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Seed:            {args.seed}")
    print(f"Folds:           {args.num_folds}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Learning rate:   {args.lr}")
    print(f"Results Dir:     {output_dir}")
    print(f"Checkpoints Dir: {checkpoints_dir}")

    # Build or verify manifest
    df_manifest = build_faithful_manifest(raw_data_dir=args.raw_data_dir, segments_per_pair=100)
    preflight_ok = preflight_verification(df_manifest, num_folds=args.num_folds, seed=args.seed)
    assert preflight_ok, "Preflight verification failed"

    # Save run configuration
    run_config = {
        "experiment_name": "EXP_FGCS_FAITHFUL",
        "description": "Faithful 10-fold reproduction of Al-Sa'd DroneRF benchmark",
        "timestamp": datetime.datetime.now().isoformat(),
        "num_folds": args.num_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "feature_dim": 2048,
        "num_classes": 4,
        "class_mapping": FAITHFUL_FGCS_CLASS_TO_INDEX,
        "architecture": "Linear(2048->128) -> ReLU -> Linear(128->128) -> ReLU -> Linear(128->128) -> ReLU -> Linear(128->4) -> Sigmoid",
        "loss": "MSELoss on one-hot targets",
        "optimizer": "Adam",
        "normalization": "Per-BUI mode global max scaling (Main_2_Data_labeling.m)",
        "cv_protocol": "StratifiedKFold(n_splits=10, shuffle=True, random_state=1) across segments",
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    if args.preflight_only:
        print("\n[--preflight_only specified] Exiting cleanly without starting training.")
        return

    # Materialize features
    X, y = materialize_features_for_training(df_manifest, mock=args.mock)

    # Execute 10-fold cross validation
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_records: List[Dict] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    print("\n" + "=" * 75)
    print(f"STARTING {args.num_folds}-FOLD STRATIFIED CROSS VALIDATION")
    print("=" * 75)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
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
        "experiment": "EXP_FGCS_FAITHFUL",
        "num_folds": args.num_folds,
        "mean_test_accuracy": mean_acc,
        "std_test_accuracy": std_acc,
        "mean_test_f1_macro": mean_f1,
        "std_test_f1_macro": std_f1,
        "total_test_predictions": len(all_y_true),
        "confusion_matrix": cm.tolist(),
        "class_labels": FAITHFUL_FGCS_INDEX_TO_CLASS,
    }
    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    print("\n" + "=" * 75)
    print("FAITHFUL 10-FOLD CROSS VALIDATION COMPLETE")
    print("=" * 75)
    print(f"Mean Test Accuracy: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
    print(f"Mean Macro-F1:      {mean_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Results exported to: {output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
