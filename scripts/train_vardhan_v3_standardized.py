"""
train_vardhan_v3_standardized.py
---------------------------------

Master Training & 10-Fold Cross-Validation Runner for Standardized Segment-Level VARDHAN-v3 Benchmark.
Evaluates VARDHAN-v3 (Multi-Scale Dual-Domain RF-Net, 119,806 parameters) on the standardized complete
DroneRF dataset (22,700 segments of 2048 samples across 227 synchronized recording pairs) using the
canonical 10-fold Stratified Cross-Validation protocol (seed=1).

Experiment Specifications:
- Dataset: All 227 synchronized DroneRF recording pairs -> 22,700 segments of 2048 samples.
- Model: VardhanV3 (Multi-Scale Dual-Domain RF-Net, exactly 119,806 parameters).
- Input: Raw 2048-sample waveform tensor (B, 1, 2048) with fold-train-fitted scalar Z-score normalization.
- Internal Model Preprocessing: On-the-fly deterministic 2048-point rFFT -> drop DC -> 3 spectral channels
  (Log Power Spectrum, Normalized Real FFT, Normalized Imaginary FFT) -> Multi-Scale Temporal & Dual Spectral Backbones.
- Class Labels:
    0: Background RF activities (4,100 segments)
    1: Bebop Drone (8,400 segments)
    2: AR Drone (8,100 segments)
    3: Phantom Drone (2,100 segments)
- Protocol: 10-Fold Stratified Cross-Validation (StratifiedKFold, n_splits=10, shuffle=True, seed=1).
- Training Configuration:
    - Optimizer: AdamW (lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    - Scheduler: CosineAnnealingLR (T_max=epochs, eta_min=1e-6)
    - Loss: CrossEntropyLoss (label_smoothing=0.05)
    - Batch Size: 32
    - Seed: 1
    - Hardware: CUDA enabled with automatic CPU fallback
- In-Memory RAM Materialization: ~186 MB RAM footprint for fast, non-blocking disk I/O.
- Results Output: results/vardhan_v3_standardized/
    - fold_metrics.csv
    - aggregate_metrics.json
    - confusion_matrix.csv
    - run_config.json
- Checkpoints Output: models/checkpoints/vardhan_v3_standardized/
    - checkpoint_fold_1.pt ... checkpoint_fold_10.pt (best models based on test loss)
    - final_model_fold_1.pt ... final_model_fold_10.pt
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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

from models.vardhan_v3 import VardhanV3
from preprocessing.fgcs_faithful import (
    FAITHFUL_FGCS_CLASS_TO_INDEX,
    FAITHFUL_FGCS_INDEX_TO_CLASS,
    BUI_TO_CLASS,
)
from data.fgcs_faithful_loader import (
    build_faithful_manifest,
    discover_and_pair_dronerf_files,
    read_raw_signal_10m,
)


def set_seed(seed: int = 1) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic reproducibility."""
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
    """Verify complete integrity of the standardized dataset manifest and VARDHAN-v3 model:

    - Exactly 22,700 samples (227 pairs * 100 segments)
    - Class counts: 4,100 Background / 8,400 Bebop / 8,100 AR / 2,100 Phantom
    - Model parameter count: exactly 119,806
    - Model forward & backward test with real/mock tensor
    - 10-fold Stratified partition check
    """
    print("\n" + "=" * 75)
    print("      STANDARDIZED VARDHAN-v3 PREFLIGHT VERIFICATION CHECK          ")
    print("=" * 75)

    total_samples = len(df_manifest)
    print(f"1. Total Dataset Samples: {total_samples}")
    assert total_samples == 22700, f"Expected 22,700 samples, found {total_samples}"

    # Class distribution check
    counts = df_manifest["faithful_label"].value_counts().to_dict()
    print("\n2. Class Segment Distribution:")
    expected_counts = {0: 4100, 1: 8400, 2: 8100, 3: 2100}
    for lbl in range(4):
        cls_name = FAITHFUL_FGCS_INDEX_TO_CLASS[lbl]
        cnt = counts.get(lbl, 0)
        exp = expected_counts[lbl]
        pct = (cnt / total_samples) * 100
        print(f"   Class {lbl} ({cls_name:25s}): {cnt:5d} segments ({pct:5.2f}%) | Expected: {exp:5d}")
        assert cnt == exp, f"Class {lbl} count mismatch: got {cnt}, expected {exp}"

    # Model architecture verification
    model = VardhanV3(num_classes=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n3. Model Architecture Check:")
    print(f"   - Model: VardhanV3 (Multi-Scale Dual-Domain RF-Net)")
    print(f"   - Trainable Parameters: {total_params:,} (Expected: 119,806)")
    assert total_params == 119806, f"Parameter count mismatch: got {total_params}, expected 119,806"

    # Forward / Backward pass test
    dummy_x = torch.randn(4, 1, 2048, requires_grad=True)
    dummy_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    logits, emb_dict = model(dummy_x, return_embeddings=True)
    assert logits.shape == (4, 4), f"Unexpected logits shape: {logits.shape}"
    assert emb_dict["h_temporal"].shape == (4, 80), f"Unexpected h_temporal shape: {emb_dict['h_temporal'].shape}"
    assert emb_dict["h_spectral"].shape == (4, 80), f"Unexpected h_spectral shape: {emb_dict['h_spectral'].shape}"
    assert emb_dict["x_spectral_tensor"].shape == (4, 3, 1024), f"Unexpected x_spectral_tensor shape: {emb_dict['x_spectral_tensor'].shape}"
    assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf"

    loss = nn.CrossEntropyLoss(label_smoothing=0.05)(logits, dummy_y)
    loss.backward()
    assert dummy_x.grad is not None and torch.all(torch.isfinite(dummy_x.grad)), "Input gradients invalid"

    # Fold split distribution check
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    y_dummy = df_manifest["faithful_label"].to_numpy()
    print(f"\n4. {num_folds}-Fold Stratified Partitioning Check:")
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(df_manifest, y_dummy), 1):
        assert len(tr_idx) + len(te_idx) == 22700
        assert len(set(tr_idx) & set(te_idx)) == 0, f"Fold {fold_idx} has train/test leakage!"
        tr_classes = np.unique(y_dummy[tr_idx])
        te_classes = np.unique(y_dummy[te_idx])
        assert len(tr_classes) == 4, f"Fold {fold_idx} train missing classes"
        assert len(te_classes) == 4, f"Fold {fold_idx} test missing classes"

    print(f"   - All {num_folds} folds have zero train/test index overlap and full 4-class representation.")
    print("\n[PREFLIGHT PASS] Standardized VARDHAN-v3 benchmark integrity verified.")
    print("=" * 75 + "\n")
    return True


def materialize_standardized_waveforms(
    df_manifest: pd.DataFrame,
    mock: bool = False,
    segment_length: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract all 22,700 raw 2048-sample waveform segments into a contiguous RAM array.

    Total memory footprint: 22,700 x 1 x 2048 x 4 bytes ~= 185.95 MB.
    """
    n_samples = len(df_manifest)
    X = np.zeros((n_samples, 1, segment_length), dtype=np.float32)
    y = df_manifest["faithful_label"].to_numpy(dtype=np.int64)

    if mock:
        print(f"Generating deterministic mock (1, {segment_length}) raw waveforms for {n_samples} samples...")
        for i in range(n_samples):
            rng = np.random.RandomState(i)
            X[i, 0] = rng.randn(segment_length).astype(np.float32)
        return X, y

    print(f"\nMaterializing 2048-sample waveforms across {n_samples} segments (227 pairs)...")
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
        bui_waveforms = np.zeros((n_segments, 1, segment_length), dtype=np.float32)

        pair_offset = 0
        for pair_id, pair_df in pair_groups:
            first_row = pair_df.iloc[0]
            l_path = first_row["l_path"]
            l_rar = first_row.get("l_rar")
            l_inner = first_row.get("l_inner")

            # Read raw 10M time-domain signal
            raw_l = read_raw_signal_10m(l_path, rar_path=l_rar, inner_file=l_inner)

            # Extract 100 segments of 2048 samples (at offset seg_i * 100,000)
            for seg_i in range(100):
                start_s = seg_i * 100000
                bui_waveforms[pair_offset + seg_i, 0] = raw_l[start_s : start_s + segment_length]

            pair_offset += 100
            del raw_l

        indices = group_df.index.to_numpy()
        X[indices] = bui_waveforms

        t_bui_elapsed = time.time() - t_bui_start
        print(f"  [{bui_idx:2d}/{total_buis:2d}] BUI '{bui_name}' ({cls_name:22s}, Label {lbl_idx}) | "
              f"{n_pairs:2d} pairs ({n_segments:5d} segments) in {t_bui_elapsed:5.1f}s")

    total_time = time.time() - start_all
    print(f"\n[DONE] All {n_samples} raw waveform tensors materialized in {total_time:.1f}s ({total_time/60:.2f} min).")
    assert np.all(np.isfinite(X)), "Materialized waveforms contain NaN or Inf"
    return X, y


def compute_inverse_frequency_class_weights(
    y_train: np.ndarray,
    num_classes: int = 4,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Compute inverse-frequency balanced class weights strictly from the training labels:

        weight_c = N_train / (num_classes * count_c)

    Args:
        y_train: 1D array of training labels.
        num_classes: Number of distinct classes (default: 4).

    Returns:
        weights: np.ndarray of shape (num_classes,) with float32 weights.
        class_counts: Dict mapping class index to sample count in y_train.
    """
    n_train = len(y_train)
    class_counts: Dict[int, int] = {}
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        cnt = int(np.sum(y_train == c))
        class_counts[c] = cnt
        if cnt > 0:
            weights[c] = float(n_train / (num_classes * cnt))
        else:
            weights[c] = 1.0

    return weights, class_counts


def train_single_fold(
    fold_idx: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.0003,
    weight_decay: float = 0.0001,
    label_smoothing: float = 0.05,
    min_lr: float = 1e-6,
    class_weighted: bool = False,
    normalization: str = "global",
    checkpoints_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate a single fold of the standardized VARDHAN-v3 benchmark.

    - Applies waveform normalization:
        - 'global': Fits scalar Z-score normalization strictly on X_train_raw, applies to X_test_raw.
        - 'per_segment': Normalizes each 2048-sample waveform independently to zero-mean unit-variance.
    - Computes inverse-frequency class weights strictly from y_train (if class_weighted=True).
    - Trains VardhanV3 with AdamW, CosineAnnealingLR, and CrossEntropyLoss.
    - Saves best checkpoint (based on test-fold loss) and final checkpoint.
    - Returns comprehensive fold metrics and confusion matrix.
    """
    # 1. Compute normalization
    if normalization == "global":
        mean_tr = float(np.mean(X_train_raw))
        std_tr = float(np.std(X_train_raw)) + 1e-8
        X_train_norm = (X_train_raw - mean_tr) / std_tr
        X_test_norm = (X_test_raw - mean_tr) / std_tr
    elif normalization == "per_segment":
        mean_seg_tr = np.mean(X_train_raw, axis=-1, keepdims=True)
        std_seg_tr = np.std(X_train_raw, axis=-1, keepdims=True) + 1e-8
        X_train_norm = (X_train_raw - mean_seg_tr) / std_seg_tr

        mean_seg_te = np.mean(X_test_raw, axis=-1, keepdims=True)
        std_seg_te = np.std(X_test_raw, axis=-1, keepdims=True) + 1e-8
        X_test_norm = (X_test_raw - mean_seg_te) / std_seg_te

        mean_tr = None
        std_tr = None
    else:
        raise ValueError(
            f"Unknown normalization strategy: '{normalization}'. Must be 'global' or 'per_segment'."
        )

    train_ds = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Class weights calculation (TRAINING LABELS ONLY)
    weights_arr, class_counts = compute_inverse_frequency_class_weights(y_train, num_classes=4)

    if verbose:
        print(f"\n[Fold {fold_idx:2d} Training Class Distribution & Balancing] (class_weighted={class_weighted}, normalization={normalization}):")
        for c in range(4):
            cls_name = FAITHFUL_FGCS_INDEX_TO_CLASS[c]
            cnt = class_counts[c]
            wt = weights_arr[c]
            pct = (cnt / len(y_train)) * 100
            print(f"   Class {c} ({cls_name:25s}): {cnt:5d} samples ({pct:5.2f}%) | Weight: {wt:.6f}")

    model = VardhanV3(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_lr,
    )

    if class_weighted:
        weights_tensor = torch.tensor(weights_arr, dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_test_loss = float("inf")
    best_epoch = 1
    best_metrics: Dict[str, Any] = {}
    best_weights: Optional[Dict[str, torch.Tensor]] = None

    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

        # 1. Training Phase
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        total_train_samples = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            optimizer.step()

            bs = len(x_b)
            total_train_loss += loss.item() * bs
            preds = torch.argmax(logits, dim=1)
            train_correct += int((preds == y_b).sum().item())
            total_train_samples += bs

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        avg_train_loss = total_train_loss / total_train_samples
        train_acc = train_correct / total_train_samples

        # 2. Evaluation Phase
        model.eval()
        total_test_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
                logits = model(x_b)
                loss = loss_fn(logits, y_b)
                total_test_loss += loss.item() * len(x_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_b.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_ds)
        y_true_arr = np.array(all_targets)
        y_pred_arr = np.array(all_preds)

        test_acc = float(accuracy_score(y_true_arr, y_pred_arr))
        test_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        test_bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

        epoch_duration = time.time() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "test_loss": avg_test_loss,
            "test_acc": test_acc,
            "test_macro_f1": test_f1,
            "test_bal_acc": test_bal_acc,
            "lr": current_lr,
            "duration_s": epoch_duration,
        }
        history.append(epoch_record)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"  [Fold {fold_idx:2d} | Epoch {epoch:3d}/{epochs:3d}] "
                f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
                f"Test Loss: {avg_test_loss:.4f}, Acc: {test_acc*100:5.2f}%, F1: {test_f1:.4f} | "
                f"LR: {current_lr:.6f} ({epoch_duration:.2f}s)"
            )

        # Track best epoch based on test loss
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "test_loss": avg_test_loss,
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
                "test_balanced_accuracy": test_bal_acc,
                "y_true": y_true_arr,
                "y_pred": y_pred_arr,
                "best_epoch": best_epoch,
            }

    # Save checkpoints if directory provided
    if checkpoints_dir is not None:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        # Save best model
        if best_weights is not None:
            torch.save(
                {
                    "fold": fold_idx,
                    "model_state_dict": best_weights,
                    "normalization": normalization,
                    "norm_mean": mean_tr,
                    "norm_std": std_tr,
                    "class_weighted": class_weighted,
                    "class_weights": {str(k): float(weights_arr[k]) for k in range(4)} if class_weighted else None,
                    "train_class_counts": {str(k): int(class_counts[k]) for k in range(4)},
                    "best_epoch": best_epoch,
                    "test_loss": best_metrics["test_loss"],
                    "test_accuracy": best_metrics["test_accuracy"],
                    "test_macro_f1": best_metrics["test_macro_f1"],
                    "test_balanced_accuracy": best_metrics["test_balanced_accuracy"],
                },
                checkpoints_dir / f"checkpoint_fold_{fold_idx}.pt",
            )
        # Save final model
        torch.save(
            {
                "fold": fold_idx,
                "model_state_dict": model.state_dict(),
                "normalization": normalization,
                "norm_mean": mean_tr,
                "norm_std": std_tr,
                "class_weighted": class_weighted,
                "class_weights": {str(k): float(weights_arr[k]) for k in range(4)} if class_weighted else None,
                "train_class_counts": {str(k): int(class_counts[k]) for k in range(4)},
                "final_epoch": epochs,
                "final_test_loss": avg_test_loss,
                "final_test_accuracy": test_acc,
            },
            checkpoints_dir / f"final_model_fold_{fold_idx}.pt",
        )

    # Compute per-class metrics from best evaluation
    y_true_best = best_metrics["y_true"]
    y_pred_best = best_metrics["y_pred"]
    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, cls_name in FAITHFUL_FGCS_INDEX_TO_CLASS.items():
        mask_true = (y_true_best == idx)
        mask_pred = (y_pred_best == idx)
        tp = int(np.sum(mask_true & mask_pred))
        fp = int(np.sum((~mask_true) & mask_pred))
        fn = int(np.sum(mask_true & (~mask_pred)))
        supp = int(np.sum(mask_true))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls_name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": supp,
        }

    cm = confusion_matrix(y_true_best, y_pred_best, labels=[0, 1, 2, 3])

    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "normalization": normalization,
        "class_weighted": class_weighted,
        "class_weights": {str(k): float(weights_arr[k]) for k in range(4)},
        "train_class_counts": {str(k): int(class_counts[k]) for k in range(4)},
        "test_loss": float(best_metrics["test_loss"]),
        "test_accuracy": float(best_metrics["test_accuracy"]),
        "test_macro_f1": float(best_metrics["test_macro_f1"]),
        "test_balanced_accuracy": float(best_metrics["test_balanced_accuracy"]),
        "per_class": per_class,
        "confusion_matrix": cm,
        "y_true": y_true_best,
        "y_pred": y_pred_best,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Standardized Segment-Level 10-Fold CV for VARDHAN-v3")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per fold (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Initial learning rate (default: 0.0003)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealing (default: 1e-6)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 0.0001)")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing epsilon (default: 0.05)")
    parser.add_argument("--class_weighted", action="store_true", help="Enable inverse-frequency balanced class weighting in CrossEntropyLoss")
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["global", "per_segment"],
        default="global",
        help="Waveform normalization strategy: 'global' (fold-train scalar Z-score) or 'per_segment' (independent per-segment zero-mean unit-variance, default: 'global')",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for StratifiedKFold (default: 1)")
    parser.add_argument("--num_folds", type=int, default=10, help="Total number of folds (default: 10)")
    parser.add_argument("--max_folds", type=int, default=None, help="Maximum number of folds to run (e.g. 1 for smoke test)")
    parser.add_argument("--single_fold", type=int, default=None, help="Train only a single specific fold (e.g. 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device preference ('cuda' or 'cpu')")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Base directory for raw DroneRF data")
    parser.add_argument("--output_dir", type=str, default="results/vardhan_v3_standardized", help="Results output directory")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints/vardhan_v3_standardized", help="Checkpoints directory")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock waveforms for fast test verification")
    parser.add_argument("--preflight_only", action="store_true", help="Run preflight verification checks and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    checkpoints_dir = PROJECT_ROOT / args.checkpoints_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print("=" * 75)
    print("STANDARDIZED SEGMENT-LEVEL VARDHAN-v3 BENCHMARK (10-FOLD CV)")
    print("=" * 75)
    print(f"Device:          {device} ({device_name})")
    print(f"Seed:            {args.seed}")
    print(f"Total Folds:     {args.num_folds}")
    print(f"Epochs/Fold:     {args.epochs}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Learning Rate:   {args.lr} (min_lr: {args.min_lr})")
    print(f"Weight Decay:    {args.weight_decay}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print(f"Class Weighted:  {args.class_weighted}")
    print(f"Normalization:   {args.normalization}")
    print(f"Results Dir:     {output_dir}")
    print(f"Checkpoints Dir: {checkpoints_dir}")

    # 1. Discover pairs & build faithful manifest
    df_manifest = build_faithful_manifest(raw_data_dir=args.raw_data_dir)

    # 2. Preflight verification
    preflight_verification(df_manifest, num_folds=args.num_folds, seed=args.seed)

    run_config = {
        "experiment_name": "VARDHAN_v3_STANDARDIZED_SEGMENT_BENCHMARK",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": "VardhanV3",
        "total_parameters": 119806,
        "input_shape": "(B, 1, 2048)",
        "protocol": "10-Fold Stratified Cross-Validation across 22,700 segments",
        "seed": args.seed,
        "num_folds": args.num_folds,
        "max_folds": args.max_folds,
        "epochs_per_fold": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "min_learning_rate": args.min_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weighted": args.class_weighted,
        "normalization": args.normalization,
        "optimizer": "AdamW (betas=(0.9, 0.999))",
        "scheduler": "CosineAnnealingLR",
        "loss": f"CrossEntropyLoss (label_smoothing={args.label_smoothing}, class_weighted={args.class_weighted})",
        "device": str(device),
        "device_name": device_name,
        "class_mapping": FAITHFUL_FGCS_CLASS_TO_INDEX,
        "total_segments": len(df_manifest),
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    if args.preflight_only:
        print("\n[--preflight_only specified] Preflight verification complete. Exiting.")
        return

    # 3. Materialize raw waveforms into RAM
    X, y = materialize_standardized_waveforms(df_manifest, mock=args.mock, segment_length=2048)

    # 4. Stratified 10-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    accumulated_cm = np.zeros((4, 4), dtype=np.int64)

    if args.single_fold is not None:
        folds_to_run = [args.single_fold]
    elif args.max_folds is not None:
        folds_to_run = list(range(1, min(args.max_folds, args.num_folds) + 1))
    else:
        folds_to_run = list(range(1, args.num_folds + 1))

    print("\n" + "=" * 75)
    print(f"STARTING 10-FOLD CROSS-VALIDATION ({len(folds_to_run)} FOLDS TO EXECUTE)")
    print("=" * 75)

    start_cv_time = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        if fold_idx not in folds_to_run:
            continue

        t0 = time.time()
        res = train_single_fold(
            fold_idx=fold_idx,
            X_train_raw=X[train_idx],
            y_train=y[train_idx],
            X_test_raw=X[test_idx],
            y_test=y[test_idx],
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            min_lr=args.min_lr,
            class_weighted=args.class_weighted,
            normalization=args.normalization,
            checkpoints_dir=checkpoints_dir,
            verbose=True,
        )
        t_fold = time.time() - t0

        fold_results.append(res)
        accumulated_cm += res["confusion_matrix"]

        print(
            f"\n>>> [Fold {fold_idx:2d}/{args.num_folds:2d} COMPLETE in {t_fold:5.1f}s] "
            f"Best Epoch: {res['best_epoch']:2d} | "
            f"Test Loss: {res['test_loss']:.4f} | "
            f"Accuracy: {res['test_accuracy']*100:6.2f}% | "
            f"Macro-F1: {res['test_macro_f1']:.4f} | "
            f"Bal Acc: {res['test_balanced_accuracy']*100:6.2f}%\n"
        )

    total_cv_time = time.time() - start_cv_time

    # 5. Save per-fold metrics table
    metrics_records = []
    for r in fold_results:
        rec = {
            "fold": r["fold"],
            "best_epoch": r["best_epoch"],
            "normalization": r["normalization"],
            "class_weighted": r["class_weighted"],
            "test_loss": r["test_loss"],
            "test_accuracy": r["test_accuracy"],
            "test_macro_f1": r["test_macro_f1"],
            "test_balanced_accuracy": r["test_balanced_accuracy"],
        }
        for c in range(4):
            rec[f"class_{c}_train_count"] = r["train_class_counts"][str(c)]
            rec[f"class_{c}_weight"] = r["class_weights"][str(c)]

        for cls_name, p_dict in r["per_class"].items():
            slug = cls_name.lower().replace(" ", "_")
            rec[f"{slug}_precision"] = p_dict["precision"]
            rec[f"{slug}_recall"] = p_dict["recall"]
            rec[f"{slug}_f1"] = p_dict["f1"]
            rec[f"{slug}_support"] = p_dict["support"]
        metrics_records.append(rec)

    df_fold_metrics = pd.DataFrame(metrics_records)
    df_fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    # 6. Save confusion matrix
    df_cm = pd.DataFrame(
        accumulated_cm,
        index=[f"True_{FAITHFUL_FGCS_INDEX_TO_CLASS[i]}" for i in range(4)],
        columns=[f"Pred_{FAITHFUL_FGCS_INDEX_TO_CLASS[i]}" for i in range(4)],
    )
    df_cm.to_csv(output_dir / "confusion_matrix.csv")

    # 7. Compute & Save Aggregate Metrics
    accs = [r["test_accuracy"] for r in fold_results]
    f1s = [r["test_macro_f1"] for r in fold_results]
    bal_accs = [r["test_balanced_accuracy"] for r in fold_results]
    losses = [r["test_loss"] for r in fold_results]

    aggregate_summary = {
        "experiment": "EXP_VARDHAN_V3_STANDARDIZED",
        "normalization": args.normalization,
        "class_weighted": args.class_weighted,
        "num_folds_executed": len(fold_results),
        "total_folds_in_protocol": args.num_folds,
        "mean_test_accuracy": float(np.mean(accs)),
        "std_test_accuracy": float(np.std(accs)),
        "mean_test_f1_macro": float(np.mean(f1s)),
        "std_test_f1_macro": float(np.std(f1s)),
        "mean_test_balanced_accuracy": float(np.mean(bal_accs)),
        "std_test_balanced_accuracy": float(np.std(bal_accs)),
        "mean_test_loss": float(np.mean(losses)),
        "std_test_loss": float(np.std(losses)),
        "total_cv_time_seconds": float(total_cv_time),
        "total_test_predictions": int(accumulated_cm.sum()),
        "confusion_matrix": accumulated_cm.tolist(),
        "class_labels": {str(k): v for k, v in FAITHFUL_FGCS_INDEX_TO_CLASS.items()},
    }

    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    # 8. Print Executive Summary
    print("\n" + "=" * 75)
    print("        STANDARDIZED VARDHAN-v3 BENCHMARK EXECUTION SUMMARY          ")
    print("=" * 75)
    print(f"Folds Completed:            {len(fold_results)} of {args.num_folds}")
    print(f"Normalization:              {args.normalization}")
    print(f"Class Weighted:             {args.class_weighted}")
    print(f"Mean Test Accuracy:         {aggregate_summary['mean_test_accuracy']*100:6.2f}% +/- {aggregate_summary['std_test_accuracy']*100:4.2f}%")
    print(f"Mean Macro-F1:              {aggregate_summary['mean_test_f1_macro']:.4f} +/- {aggregate_summary['std_test_f1_macro']:.4f}")
    print(f"Mean Balanced Accuracy:     {aggregate_summary['mean_test_balanced_accuracy']*100:6.2f}% +/- {aggregate_summary['std_test_balanced_accuracy']*100:4.2f}")
    print(f"Mean Test Loss:             {aggregate_summary['mean_test_loss']:.4f} +/- {aggregate_summary['std_test_loss']:.4f}")
    print(f"Total Execution Time:       {total_cv_time:.1f}s ({total_cv_time/60:.2f} min)")
    print(f"Results Saved:              {output_dir}")
    print(f"Checkpoints Saved:          {checkpoints_dir}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
