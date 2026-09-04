"""
train_vardhan_v3.py
-------------------

Master Training, Benchmark & Evaluation Runner for VARDHAN-v3 (Multi-Scale Dual-Domain RF-Net, 119,806 parameters).
Evaluated on the canonical strict recording-level benchmark (Zero Recording Overlap).

Features:
- Single-channel (B, 1, 2048) input protocol with configurable normalization:
    1. 'global': Train-fitted scalar Z-score normalization (mean_tr, std_tr)
    2. 'per_segment': Independent per-segment zero-mean unit-variance (mean(x), std(x) + 1e-8)
- Internal deterministic spectral extraction (rFFT 2048 -> drop DC -> 1024 positive bins -> 3 spectral channels).
- Parameter count verification: exactly 119,806 trainable parameters.
- Comprehensive pre-flight split verification (zero recording leakage, zero file overlap).
- Real-data forward/backward pipeline verification before training.
- High-performance in-memory dataset caching (eliminates repeated quadratic ASCII parsing).
- Class-weighted CrossEntropyLoss with inverse frequency weighting and label smoothing.
- Cosine Annealing learning rate scheduling.
- AdamW optimizer, single GPU (CUDA when available) with CPU fallback.
- Exports metrics.json, history.csv, confusion_matrix.csv, val_confusion_matrix.csv, run_config.json, best.pt, and last.pt.
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

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from models.vardhan_v3 import VardhanV3
from data.loader import DroneRFLazyDataset, fit_train_normalization_stats, get_dataloader
from constants import LABEL_MAP, RAW_CLASS_TO_INDEX


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducible execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def verify_strict_split_integrity(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
) -> bool:
    """Verify zero file overlap, zero recording overlap, and full 4-class representation."""
    print("=" * 75)
    print("      STRICT RECORDING-LEVEL SPLIT INTEGRITY VERIFICATION           ")
    print("=" * 75)

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_te = pd.read_csv(test_csv)

    tr_files = set(df_tr["relative_path"])
    va_files = set(df_va["relative_path"])
    te_files = set(df_te["relative_path"])

    assert len(tr_files & va_files) == 0, "Train and Validation share files! Leakage detected."
    assert len(tr_files & te_files) == 0, "Train and Test share files! Leakage detected."
    assert len(va_files & te_files) == 0, "Validation and Test share files! Leakage detected."
    print("[OK] Zero file overlap verified across Train, Validation, and Test.")

    tr_recs = set(df_tr["relative_path"].apply(lambda p: Path(p).parent.name))
    va_recs = set(df_va["relative_path"].apply(lambda p: Path(p).parent.name))
    te_recs = set(df_te["relative_path"].apply(lambda p: Path(p).parent.name))

    assert len(tr_recs & va_recs) == 0, "Train and Validation share physical recordings! Leakage detected."
    assert len(tr_recs & te_recs) == 0, "Train and Test share physical recordings! Leakage detected."
    assert len(va_recs & te_recs) == 0, "Validation and Test share physical recordings! Leakage detected."
    print("[OK] Zero recording overlap verified across Train, Validation, and Test.")

    splits_info = [
        ("Train", df_tr, tr_recs),
        ("Validation", df_va, va_recs),
        ("Test", df_te, te_recs),
    ]

    for name, df, recs in splits_info:
        classes = df["drone_class"].value_counts().to_dict()
        assert len(classes) == 4, f"{name} split is missing classes! Found: {list(classes.keys())}"
        print(f"\n--- {name} Split ({len(df)} files, {len(recs)} recordings) ---")
        for cls_name, count in sorted(classes.items()):
            print(f"  - {cls_name:25s}: {count:3d} files")

    print("\n[PREFLIGHT PASS] Strict recording-level benchmark partition confirmed.")
    print("=" * 75)
    return True


def compute_train_class_weights(
    train_csv: Path,
    samples_per_file: int = 50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute inverse class frequency weights strictly from training split sample counts:

    weight_c = N_total / (num_classes * N_c)
    """
    df_tr = pd.read_csv(train_csv)
    raw_counts = df_tr["drone_class"].value_counts().to_dict()

    counts = [0] * 4
    for raw_cls, idx in RAW_CLASS_TO_INDEX.items():
        counts[idx] = raw_counts.get(raw_cls, 0) * samples_per_file

    n_total = sum(counts)
    num_classes = len(counts)
    weights = [n_total / (num_classes * c) if c > 0 else 1.0 for c in counts]
    w_tensor = torch.tensor(weights, dtype=torch.float32)

    weights_dict = {LABEL_MAP[i]: float(w_tensor[i].item()) for i in range(4)}
    return w_tensor, weights_dict


def normalize_waveform_segment(
    x: Union[np.ndarray, torch.Tensor],
    normalization: str = "global",
    norm_stats: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Normalize a single 2048-sample waveform tensor according to strategy:

    - 'global': (x - mean_tr) / (std_tr + 1e-8) using train-fold-only statistics
    - 'per_segment': (x - mean(x)) / (std(x) + 1e-8) computed independently per waveform
    """
    if isinstance(x, np.ndarray):
        t = torch.tensor(x, dtype=torch.float32)
    else:
        t = x.clone().detach().to(dtype=torch.float32)

    if t.ndim == 1:
        t = t.unsqueeze(0)  # (1, 2048)

    if normalization == "global":
        if norm_stats is None:
            raise ValueError("norm_stats required for global normalization")
        mean_val = norm_stats["mean"]
        std_val = norm_stats["std"] + 1e-8
        return (t - mean_val) / std_val
    elif normalization == "per_segment":
        mean_seg = t.mean(dim=-1, keepdim=True)
        std_seg = t.std(dim=-1, keepdim=True) + 1e-8
        return (t - mean_seg) / std_seg
    else:
        raise ValueError(f"Unknown normalization: '{normalization}'. Must be 'global' or 'per_segment'.")


def preload_split_dataset(
    split_csv: Path,
    norm_stats: Optional[Dict[str, float]] = None,
    samples_per_file: int = 50,
    segment_length: int = 2048,
    raw_data_dir: Optional[Union[str, Path]] = None,
    mock: bool = False,
    normalization: str = "global",
) -> TensorDataset:
    """Pre-extract all segments from the split into contiguous RAM memory with chosen normalization."""
    lazy_ds = DroneRFLazyDataset(
        split_csv=split_csv,
        model_name="vardhan",
        norm_stats={"mean": 0.0, "std": 1.0, "max": 1.0, "min": -1.0},
        segment_length=segment_length,
        samples_per_file=samples_per_file,
        raw_data_dir=raw_data_dir,
        mock=mock,
    )

    n_samples = len(lazy_ds)
    X = torch.zeros(n_samples, 1, segment_length, dtype=torch.float32)
    y = torch.zeros(n_samples, dtype=torch.long)

    t0 = time.time()
    for i in range(n_samples):
        item_x, item_y = lazy_ds[i]
        raw_tensor = item_x if isinstance(item_x, torch.Tensor) else torch.tensor(item_x, dtype=torch.float32)
        X[i] = normalize_waveform_segment(raw_tensor, normalization=normalization, norm_stats=norm_stats)
        y[i] = item_y

    elapsed = time.time() - t0
    print(f"  --> Preloaded {n_samples:5d} segments ({normalization}) from {Path(split_csv).name} into RAM ({X.element_size() * X.nelement() / (1024**2):.1f} MB) in {elapsed:5.1f}s.")
    return TensorDataset(X, y)


def verify_model_and_real_pipeline(
    model: VardhanV3,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> bool:
    """Verify model parameter count (119,806), tensor shapes through DataLoader, and backward gradients."""
    print("\n" + "=" * 75)
    print("      MODEL ARCHITECTURE & PIPELINE FORWARD/BACKWARD CHECK          ")
    print("=" * 75)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"1. Model Parameters: {total_params:,} (Expected: 119,806)")
    if total_params != 119806:
        raise ValueError(f"VardhanV3 parameter mismatch! Got {total_params}, expected exactly 119,806.")
    print("[OK] Exact parameter count (119,806) verified.")

    # Real data batch test
    model.eval()
    batch_x, batch_y = next(iter(train_loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    print(f"\n2. Real Data Batch Shapes (Batch Size: {batch_x.shape[0]}):")
    print(f"   - Input Raw Waveform: {batch_x.shape}")
    assert batch_x.ndim == 3 and batch_x.shape[1] == 1 and batch_x.shape[2] == 2048, f"Unexpected shape {batch_x.shape}"

    x_spectral = model.extract_spectral_representation(batch_x)
    print(f"   - Extracted 3-Channel Spectral Tensor: {x_spectral.shape}")
    assert x_spectral.shape == (batch_x.shape[0], 3, 1024)

    logits, emb_dict = model(batch_x, return_embeddings=True)
    print(f"   - Temporal Feature (h_T): {emb_dict['h_temporal'].shape}")
    print(f"   - Spectral Feature (h_F): {emb_dict['h_spectral'].shape}")
    print(f"   - Output Logits:          {logits.shape}")

    assert emb_dict["h_temporal"].shape == (batch_x.shape[0], 80)
    assert emb_dict["h_spectral"].shape == (batch_x.shape[0], 80)
    assert logits.shape == (batch_x.shape[0], 4)
    assert torch.all(torch.isfinite(logits)), "Forward logits contain NaN or Inf"

    # Backward gradient verification
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, batch_y)
    assert torch.isfinite(loss), "Initial loss is not finite"
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None and torch.all(torch.isfinite(p.grad)), f"Parameter {name} has invalid gradient!"

    model.zero_grad()
    print("\n[PASS] Model architecture and real-data pipeline verification passed.")
    print("=" * 75 + "\n")
    return True


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """Train for one epoch and return average loss, accuracy, macro-F1, and balanced accuracy."""
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(all_targets) if len(all_targets) > 0 else 0.0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0

    return avg_loss, acc, f1, bal_acc


def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Evaluate on validation or test split without gradient computation."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * len(x)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(all_targets) if len(all_targets) > 0 else 0.0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0

    return avg_loss, acc, f1, bal_acc, y_true, y_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Strict Recording-Level VARDHAN-v3 Training Benchmark")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Initial learning rate (default: 0.0003)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for AdamW (default: 0.0001)")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="CrossEntropy label smoothing (default: 0.05)")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "none"], help="LR scheduler (default: cosine)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for Cosine scheduler (default: 1e-6)")
    parser.add_argument("--use_class_weights", action="store_true", help="Enable inverse frequency class weighting")
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["global", "per_segment"],
        default="global",
        help="Waveform normalization strategy: 'global' (train-fold scalar Z-score) or 'per_segment' (independent per-segment zero-mean unit-variance, default: 'global')",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--samples_per_file", type=int, default=50, help="Non-overlapping segments per raw file (default: 50)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (default: 0)")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory for GPU transfer")
    parser.add_argument("--no_cache", action="store_true", help="Disable in-memory RAM caching")
    parser.add_argument("--splits_dir", type=str, default=None, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Base directory for raw DroneRF data")
    parser.add_argument("--output_dir", type=str, default="results/vardhan_v3_strict", help="Results output directory")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints/vardhan_v3_strict", help="Checkpoints directory")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock waveforms for fast testing")
    parser.add_argument("--preflight_only", action="store_true", help="Run preflight verification checks and exit")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Cap training batches per epoch (for smoke testing)")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Cap validation batches per epoch (for smoke testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    splits_dir = Path(args.splits_dir) if args.splits_dir else PROJECT_ROOT / "data" / "splits"
    output_dir = PROJECT_ROOT / args.output_dir
    checkpoints_dir = PROJECT_ROOT / args.checkpoints_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    use_cache = not args.no_cache

    print("=" * 75)
    print("VARDHAN-v3 BENCHMARK RUNNER (STRICT RECORDING-LEVEL SPLIT)")
    print("=" * 75)
    print(f"Device:          {device} ({device_name})")
    print(f"Seed:            {args.seed}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Initial LR:      {args.lr}")
    print(f"Weight Decay:    {args.weight_decay}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print(f"LR Scheduler:    {args.lr_scheduler} (min_lr: {args.min_lr})")
    print(f"Class Weights:   {'ENABLED' if args.use_class_weights else 'DISABLED'}")
    print(f"Normalization:   {args.normalization}")
    print(f"Samples/File:    {args.samples_per_file}")
    print(f"Data Caching:    {'ENABLED (In-Memory RAM Preloading)' if use_cache else 'DISABLED (On-Demand File Reading)'}")
    print(f"Results Dir:     {output_dir}")
    print(f"Checkpoints Dir: {checkpoints_dir}")

    # 1. Verify strict recording-level split
    verify_strict_split_integrity(train_csv, val_csv, test_csv)

    # 2. Compute train-only scalar normalization statistics (if global normalization)
    if args.normalization == "global":
        print("\nComputing train-only global normalization statistics...")
        norm_stats = fit_train_normalization_stats(
            train_split_csv=train_csv,
            segment_length=2048,
            raw_data_dir=args.raw_data_dir,
        )
        print(f"Train Normalization Stats: {norm_stats}")
    else:
        print("\nUsing per-segment independent waveform normalization (mean(x), std(x) + 1e-8)...")
        norm_stats = None

    # 3. Compute Train-Only Class Weights
    class_weights_tensor, class_weights_dict = compute_train_class_weights(train_csv, args.samples_per_file)
    print(f"\nComputed Train-Only Class Weights (N_total / (C * N_c)):")
    for cls_name, w in class_weights_dict.items():
        print(f"  - {cls_name:15s}: {w:.4f}")

    # 4. Instantiate Datasets & DataLoaders
    print("\n[Dataset Preparation] Loading dataset splits into memory...")
    train_ds = preload_split_dataset(
        split_csv=train_csv,
        norm_stats=norm_stats,
        samples_per_file=args.samples_per_file,
        segment_length=2048,
        raw_data_dir=args.raw_data_dir,
        mock=args.mock,
        normalization=args.normalization,
    )
    val_ds = preload_split_dataset(
        split_csv=val_csv,
        norm_stats=norm_stats,
        samples_per_file=args.samples_per_file,
        segment_length=2048,
        raw_data_dir=args.raw_data_dir,
        mock=args.mock,
        normalization=args.normalization,
    )
    test_ds = preload_split_dataset(
        split_csv=test_csv,
        norm_stats=norm_stats,
        samples_per_file=args.samples_per_file,
        segment_length=2048,
        raw_data_dir=args.raw_data_dir,
        mock=args.mock,
        normalization=args.normalization,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" and args.pin_memory),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" and args.pin_memory),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" and args.pin_memory),
    )

    # 5. Instantiate VARDHAN-v3 model
    model = VardhanV3(num_classes=4).to(device)

    # 6. Model & real-data pipeline verification
    verify_model_and_real_pipeline(model, train_loader, device)

    run_config = {
        "experiment_name": "VARDHAN_v3_STRICT_RECORDING_SPLIT",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": "VardhanV3",
        "total_parameters": 119806,
        "seed": args.seed,
        "device": str(device),
        "device_name": device_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "use_class_weights": args.use_class_weights,
        "class_weights": class_weights_dict if args.use_class_weights else None,
        "normalization": args.normalization,
        "lr_scheduler": args.lr_scheduler,
        "min_lr": args.min_lr,
        "optimizer": "AdamW (betas=(0.9, 0.999))",
        "loss": "CrossEntropyLoss(weighted, smoothed)" if args.use_class_weights else "CrossEntropyLoss(smoothed)",
        "samples_per_file": args.samples_per_file,
        "data_caching": use_cache,
        "raw_data_dir": str(args.raw_data_dir),
        "splits_dir": str(splits_dir),
        "class_mapping": RAW_CLASS_TO_INDEX,
        "norm_stats": norm_stats,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    if args.preflight_only:
        print("\n[--preflight_only specified] Preflight verification complete. Exiting.")
        return

    # 7. Optimizer, Loss & Scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    weights_arg = class_weights_tensor.to(device) if args.use_class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weights_arg, label_smoothing=args.label_smoothing)

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    else:
        scheduler = None

    # 8. Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics = {}
    best_model_state = None
    history_records = []

    print("\n" + "=" * 75)
    print(f"STARTING VARDHAN-v3 TRAINING ({args.epochs} EPOCHS, normalization={args.normalization})")
    print("=" * 75)

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc, tr_f1, tr_bal = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            max_batches=args.max_train_batches,
        )

        va_loss, va_acc, va_f1, va_bal, y_val_true, y_val_pred = evaluate_split(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            max_batches=args.max_val_batches,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_duration = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs:3d} | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:5.2f}%, F1: {tr_f1:.4f} | "
            f"Val Loss: {va_loss:.4f}, Acc: {va_acc*100:5.2f}%, F1: {va_f1:.4f}, Bal: {va_bal*100:5.2f}% | "
            f"LR: {current_lr:.6f} ({epoch_duration:4.1f}s)"
        )

        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_macro_f1": tr_f1,
            "train_bal_acc": tr_bal,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_macro_f1": va_f1,
            "val_bal_acc": va_bal,
            "lr": current_lr,
            "duration_s": epoch_duration,
        }
        history_records.append(rec)

        # Track best model based on validation loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            best_val_metrics = {
                "val_loss": va_loss,
                "val_accuracy": va_acc,
                "val_macro_f1": va_f1,
                "val_balanced_accuracy": va_bal,
            }
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_metrics": best_val_metrics,
                    "normalization": args.normalization,
                    "norm_stats": norm_stats,
                    "run_config": run_config,
                },
                checkpoints_dir / "best.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "normalization": args.normalization,
                    "norm_stats": norm_stats,
                },
                output_dir / "best.pt",
            )

            # Save best validation confusion matrix
            val_cm = confusion_matrix(y_val_true, y_val_pred, labels=[0, 1, 2, 3])
            val_cm_df = pd.DataFrame(
                val_cm,
                index=[f"True_{LABEL_MAP[i]}" for i in range(4)],
                columns=[f"Pred_{LABEL_MAP[i]}" for i in range(4)],
            )
            val_cm_df.to_csv(output_dir / "val_confusion_matrix.csv")

    total_training_sec = time.time() - start_train_time

    # Save final last.pt checkpoint
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalization": args.normalization,
            "norm_stats": norm_stats,
            "run_config": run_config,
        },
        checkpoints_dir / "last.pt",
    )
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "normalization": args.normalization,
            "norm_stats": norm_stats,
        },
        output_dir / "last.pt",
    )

    # Save history dataframe
    df_history = pd.DataFrame(history_records)
    df_history.to_csv(output_dir / "history.csv", index=False)

    # 9. Final Test Set Evaluation using best model checkpoint
    print("\n" + "=" * 75)
    print(f"LOADING BEST MODEL (Epoch {best_epoch}) FOR TEST SET EVALUATION")
    print("=" * 75)

    best_ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    te_loss, te_acc, te_f1, te_bal, y_true, y_pred = evaluate_split(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    # Per-class metrics
    per_class = {}
    for idx, cls_name in LABEL_MAP.items():
        mask_true = (y_true == idx)
        mask_pred = (y_pred == idx)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum((~mask_true) & mask_pred)
        fn = np.sum(mask_true & (~mask_pred))
        support = int(np.sum(mask_true))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls_name] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_c),
            "support": support,
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{LABEL_MAP[i]}" for i in range(4)],
        columns=[f"Pred_{LABEL_MAP[i]}" for i in range(4)],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")
    cm_df.to_csv(output_dir / "test_confusion_matrix.csv")

    metrics_summary = {
        "model_name": "VardhanV3",
        "total_parameters": 119806,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": best_val_metrics,
        "test_loss": float(te_loss),
        "test_accuracy": float(te_acc),
        "test_macro_f1": float(te_f1),
        "test_balanced_accuracy": float(te_bal),
        "normalization": args.normalization,
        "use_class_weights": args.use_class_weights,
        "class_weights": class_weights_dict if args.use_class_weights else None,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "initial_lr": args.lr,
        "min_lr": args.min_lr,
        "per_class_metrics": per_class,
        "total_training_time_seconds": total_training_sec,
        "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("=" * 75)
    print("FINAL TEST EVALUATION SUMMARY (STRICT RECORDING-LEVEL SPLIT)")
    print("=" * 75)
    print(f"Normalization:          {args.normalization}")
    print(f"Test Accuracy:          {te_acc*100:.2f}%")
    print(f"Test Macro-F1:          {te_f1:.4f}")
    print(f"Test Balanced Accuracy: {te_bal*100:.2f}%")
    print(f"Test Loss:              {te_loss:.4f}")
    print(f"\nPer-Class Breakdown:")
    for cls_name, m in per_class.items():
        print(f"  - {cls_name:15s}: Prec={m['precision']*100:5.2f}%, Rec={m['recall']*100:5.2f}%, F1={m['f1']:.4f} (Support: {m['support']})")
    print("=" * 75)


if __name__ == "__main__":
    main()
