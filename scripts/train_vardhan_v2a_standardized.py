"""
train_vardhan_v2a_standardized.py
---------------------------------

Master Training & 10-Fold Cross-Validation Runner for Standardized Segment-Level VARDHAN-v2A Benchmark.
Evaluates VARDHAN-v2A (Tri-Branch Multi-Representation RF-Net) on the standardized complete DroneRF dataset
using the same segment-level 10-fold cross-validation protocol established in FGCS (Al-Sa'd et al., 2019)
and MC1DCNN (Allahham et al., 2020).

Experiment Specifications:
- Dataset: All 227 synchronized DroneRF recording pairs -> 22,700 segments of 2048 samples.
- Model: VardhanV2A (Tri-Branch RF-Net, exactly 69,559 parameters, unmodified).
- Input: Raw 2048-sample waveform tensor (B, 1, 2048) with fold-train-fitted scalar Z-score normalization.
- Internal Model Preprocessing: On-the-fly 2048-point rFFT -> drop DC -> 1024 positive bins -> 8x128 sub-bands.
- Class Labels:
    0: Background RF activities (4,100 segments)
    1: Bebop Drone (8,400 segments)
    2: AR Drone (8,100 segments)
    3: Phantom Drone (2,100 segments)
- Protocol: 10-Fold Stratified Cross-Validation (StratifiedKFold, n_splits=10, shuffle=True, seed=1).
- Training: CrossEntropyLoss, AdamW (lr=0.0003, weight_decay=0.0001), batch_size=32, epochs=100.
- Optimization: In-memory RAM materialization (~186 MB) for high-speed non-blocking execution.
- Results Output: results/vardhan_v2a_standardized/
    - fold_metrics.csv
    - aggregate_metrics.json
    - confusion_matrix.csv
    - run_config.json
- Checkpoints Output: models/checkpoints/vardhan_v2a_standardized/
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

from models.vardhan_v2a import VardhanV2A
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
    """
    Verify complete integrity of the standardized dataset manifest before training:
    - Exactly 22,700 samples (227 pairs * 100 segments)
    - Class counts: 4,100 Background / 8,400 Bebop / 8,100 AR / 2,100 Phantom
    - Model parameter count: exactly 69,559
    - Model forward & backward test with real/mock tensor
    """
    print("\n" + "=" * 75)
    print("      STANDARDIZED VARDHAN-v2A PREFLIGHT VERIFICATION CHECK         ")
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
    model = VardhanV2A(num_classes=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n3. Model Architecture Check:")
    print(f"   - Model: VardhanV2A")
    print(f"   - Trainable Parameters: {total_params:,} (Expected: 69,559)")
    assert total_params == 69559, f"Parameter count mismatch: got {total_params}, expected 69,559"

    # Forward / Backward pass test
    dummy_x = torch.randn(4, 1, 2048, requires_grad=True)
    dummy_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    logits, emb_dict = model(dummy_x, return_embeddings=True)
    assert logits.shape == (4, 4), f"Unexpected logits shape: {logits.shape}"
    assert emb_dict["h_fused"].shape == (4, 192), f"Unexpected fusion shape: {emb_dict['h_fused'].shape}"
    assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf"

    loss = nn.CrossEntropyLoss()(logits, dummy_y)
    loss.backward()
    assert dummy_x.grad is not None and torch.all(torch.isfinite(dummy_x.grad)), "Input gradients invalid"

    # Fold split distribution check
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    y_dummy = df_manifest["faithful_label"].to_numpy()
    print(f"\n4. 10-Fold Stratified Partitioning Check:")
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(df_manifest, y_dummy), 1):
        assert len(tr_idx) + len(te_idx) == 22700
        assert len(set(tr_idx) & set(te_idx)) == 0, f"Fold {fold_idx} has train/test leakage!"
        tr_classes = np.unique(y_dummy[tr_idx])
        te_classes = np.unique(y_dummy[te_idx])
        assert len(tr_classes) == 4, f"Fold {fold_idx} train missing classes"
        assert len(te_classes) == 4, f"Fold {fold_idx} test missing classes"

    print(f"   - All {num_folds} folds have zero train/test index overlap and full 4-class representation.")
    print("\n[PREFLIGHT PASS] Standardized VARDHAN-v2A benchmark integrity verified.")
    print("=" * 75 + "\n")
    return True


def materialize_standardized_waveforms(
    df_manifest: pd.DataFrame,
    mock: bool = False,
    segment_length: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all 22,700 raw 2048-sample waveform segments into a contiguous RAM array.
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

            # Extract 100 segments of 2048 samples (at offset j * 100,000)
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
    checkpoints_dir: Optional[Path] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Train and evaluate a single fold of the standardized VARDHAN-v2A benchmark.
    - Fits scalar Z-score normalization strictly on X_train_raw.
    - Applies the same normalization to X_test_raw.
    - Trains VardhanV2A with AdamW and CrossEntropyLoss.
    - Returns fold accuracy, macro-F1, balanced accuracy, per-class metrics, and predictions.
    """
    # 1. Compute fold-train-only normalization statistics
    mean_tr = float(np.mean(X_train_raw))
    std_tr = float(np.std(X_train_raw)) + 1e-8

    X_train_norm = (X_train_raw - mean_tr) / std_tr
    X_test_norm = (X_test_raw - mean_tr) / std_tr

    train_ds = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = VardhanV2A(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_b)

        avg_train_loss = total_loss / len(train_ds)
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model weights for test fold evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    total_test_loss = 0.0

    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            logits = model(x_b)
            loss = loss_fn(logits, y_b)
            total_test_loss += loss.item() * len(x_b)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_b.cpu().numpy())

    y_true_arr = np.array(all_targets)
    y_pred_arr = np.array(all_preds)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    test_loss = total_test_loss / len(test_ds)

    # Per-class metrics
    per_class = {}
    for idx, cls_name in FAITHFUL_FGCS_INDEX_TO_CLASS.items():
        mask_true = (y_true_arr == idx)
        mask_pred = (y_pred_arr == idx)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum((~mask_true) & mask_pred)
        fn = np.sum(mask_true & (~mask_pred))
        supp = int(np.sum(mask_true))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls_name] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": supp}

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3])

    if checkpoints_dir is not None:
        torch.save({
            "fold": fold_idx,
            "model_state_dict": model.state_dict(),
            "norm_mean": mean_tr,
            "norm_std": std_tr,
            "test_accuracy": acc,
            "test_macro_f1": f1,
            "test_balanced_accuracy": bal_acc,
        }, checkpoints_dir / f"checkpoint_fold_{fold_idx}.pt")

    return {
        "fold": fold_idx,
        "test_loss": float(test_loss),
        "test_accuracy": float(acc),
        "test_macro_f1": float(f1),
        "test_balanced_accuracy": float(bal_acc),
        "per_class": per_class,
        "confusion_matrix": cm,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Standardized Segment-Level 10-Fold CV for VARDHAN-v2A")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per fold (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate (default: 0.0003)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 0.0001)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for StratifiedKFold (default: 1)")
    parser.add_argument("--num_folds", type=int, default=10, help="Number of folds (default: 10)")
    parser.add_argument("--device", type=str, default="cuda", help="Device preference ('cuda' or 'cpu')")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Base directory for raw DroneRF data")
    parser.add_argument("--output_dir", type=str, default="results/vardhan_v2a_standardized", help="Results output directory")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints/vardhan_v2a_standardized", help="Checkpoints directory")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock waveforms for verification")
    parser.add_argument("--preflight_only", action="store_true", help="Run preflight verification checks and exit")
    parser.add_argument("--single_fold", type=int, default=None, help="Train only a single fold (e.g. 1) for smoke testing")
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
    print("STANDARDIZED SEGMENT-LEVEL VARDHAN-v2A BENCHMARK (10-FOLD CV)")
    print("=" * 75)
    print(f"Device:          {device} ({device_name})")
    print(f"Seed:            {args.seed}")
    print(f"Folds:           {args.num_folds}")
    print(f"Epochs/Fold:     {args.epochs}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Weight Decay:    {args.weight_decay}")
    print(f"Results Dir:     {output_dir}")
    print(f"Checkpoints Dir: {checkpoints_dir}")

    # 1. Discover pairs & build faithful manifest
    df_manifest = build_faithful_manifest(raw_data_dir=args.raw_data_dir)

    # 2. Preflight verification
    preflight_verification(df_manifest, num_folds=args.num_folds, seed=args.seed)

    run_config = {
        "experiment_name": "VARDHAN_v2A_STANDARDIZED_SEGMENT_BENCHMARK",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": "VardhanV2A",
        "total_parameters": 69559,
        "input_shape": "(B, 1, 2048)",
        "protocol": "10-Fold Stratified Cross-Validation across 22,700 segments",
        "seed": args.seed,
        "num_folds": args.num_folds,
        "epochs_per_fold": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
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

    folds_to_run = [args.single_fold] if args.single_fold is not None else list(range(1, args.num_folds + 1))

    print("\n" + "=" * 75)
    print(f"STARTING 10-FOLD CROSS-VALIDATION ({len(folds_to_run)} FOLDS)")
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
            checkpoints_dir=checkpoints_dir,
        )
        fold_time = time.time() - t0

        fold_results.append(res)
        accumulated_cm += res["confusion_matrix"]

        print(f"  Fold {fold_idx:2d}/{args.num_folds:2d} | "
              f"Test Acc: {res['test_accuracy']*100:5.2f}% | "
              f"Macro-F1: {res['test_macro_f1']:.4f} | "
              f"BalAcc: {res['test_balanced_accuracy']*100:5.2f}% | "
              f"Loss: {res['test_loss']:.4f} | "
              f"Time: {fold_time:5.1f}s")

    total_cv_time = time.time() - start_cv_time
    print("-" * 75)
    print(f"Cross-Validation Complete in {total_cv_time:.1f}s ({total_cv_time/60:.2f} min).")

    # 5. Export fold metrics CSV
    fold_rows = []
    for r in fold_results:
        row = {
            "fold": r["fold"],
            "test_accuracy": r["test_accuracy"],
            "test_macro_f1": r["test_macro_f1"],
            "test_balanced_accuracy": r["test_balanced_accuracy"],
            "test_loss": r["test_loss"],
        }
        for cls_name, m in r["per_class"].items():
            row[f"{cls_name}_precision"] = m["precision"]
            row[f"{cls_name}_recall"] = m["recall"]
            row[f"{cls_name}_f1"] = m["f1"]
        fold_rows.append(row)

    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)

    # 6. Aggregate metrics across folds
    acc_list = [r["test_accuracy"] for r in fold_results]
    f1_list = [r["test_macro_f1"] for r in fold_results]
    bal_list = [r["test_balanced_accuracy"] for r in fold_results]
    loss_list = [r["test_loss"] for r in fold_results]

    agg_per_class = {}
    for cls_name in FAITHFUL_FGCS_INDEX_TO_CLASS.values():
        agg_per_class[cls_name] = {
            "mean_precision": float(np.mean([r["per_class"][cls_name]["precision"] for r in fold_results])),
            "std_precision": float(np.std([r["per_class"][cls_name]["precision"] for r in fold_results])),
            "mean_recall": float(np.mean([r["per_class"][cls_name]["recall"] for r in fold_results])),
            "std_recall": float(np.std([r["per_class"][cls_name]["recall"] for r in fold_results])),
            "mean_f1": float(np.mean([r["per_class"][cls_name]["f1"] for r in fold_results])),
            "std_f1": float(np.std([r["per_class"][cls_name]["f1"] for r in fold_results])),
        }

    agg_summary = {
        "model_name": "VardhanV2A",
        "total_parameters": 69559,
        "num_folds_evaluated": len(fold_results),
        "mean_accuracy": float(np.mean(acc_list)),
        "std_accuracy": float(np.std(acc_list)),
        "mean_macro_f1": float(np.mean(f1_list)),
        "std_macro_f1": float(np.std(f1_list)),
        "mean_balanced_accuracy": float(np.mean(bal_list)),
        "std_balanced_accuracy": float(np.std(bal_list)),
        "mean_loss": float(np.mean(loss_list)),
        "std_loss": float(np.std(loss_list)),
        "per_class_metrics": agg_per_class,
        "total_cv_time_seconds": total_cv_time,
        "confusion_matrix": accumulated_cm.tolist(),
    }
    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(agg_summary, f, indent=2)

    # Save accumulated confusion matrix CSV
    cm_df = pd.DataFrame(
        accumulated_cm,
        index=[f"True_{FAITHFUL_FGCS_INDEX_TO_CLASS[i]}" for i in range(4)],
        columns=[f"Pred_{FAITHFUL_FGCS_INDEX_TO_CLASS[i]}" for i in range(4)],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    print("=" * 75)
    print("STANDARDIZED BENCHMARK AGGREGATE SUMMARY")
    print("=" * 75)
    print(f"Mean Accuracy:          {agg_summary['mean_accuracy']*100:.2f}% +/- {agg_summary['std_accuracy']*100:.2f}%")
    print(f"Mean Macro-F1:          {agg_summary['mean_macro_f1']:.4f} +/- {agg_summary['std_macro_f1']:.4f}")
    print(f"Mean Balanced Accuracy: {agg_summary['mean_balanced_accuracy']*100:.2f}% +/- {agg_summary['std_balanced_accuracy']*100:.2f}%")
    print(f"Artifacts saved to:     {output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
