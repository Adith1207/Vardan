"""
train_vardhan_v2a.py
--------------------

Master Training, Benchmark & Profiling Runner for VARDHAN-v2A (Tri-Branch Multi-Representation RF-Net).
Evaluated on the canonical strict recording-level benchmark (Zero Recording Overlap).

Features:
- Single-file (B, 1, 2048) input protocol with train-fitted scalar Z-score normalization.
- Internal deterministic spectral extraction (rfft 2048 -> drop DC -> 1024 positive bins -> 8x128 sub-bands).
- Parameter count verification: exactly 69,559 trainable parameters.
- Comprehensive pre-flight split verification (zero recording leakage, zero file overlap).
- Real-data forward/backward pipeline verification before training.
- High-performance in-memory dataset caching (eliminates repeated quadratic ASCII parsing).
- Class-weighted CrossEntropyLoss with inverse frequency weighting and label smoothing.
- Cosine Annealing learning rate scheduling.
- Profiling mode (--profile) with microsecond breakdown across I/O, transfer, forward, backward, and optimizer steps.
- AdamW optimizer, single GPU (CUDA when available).
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

from models.vardhan_v2a import VardhanV2A
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
    """Compute inverse class frequency weights strictly from training split sample counts.

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


def preload_split_dataset(
    split_csv: Path,
    norm_stats: Dict[str, float],
    samples_per_file: int = 50,
    segment_length: int = 2048,
    raw_data_dir: Optional[Union[str, Path]] = None,
    mock: bool = False,
) -> TensorDataset:
    """Pre-extract all segments from the split into contiguous RAM memory."""
    lazy_ds = DroneRFLazyDataset(
        split_csv=split_csv,
        model_name="vardhan",
        norm_stats=norm_stats,
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
        X[i] = item_x if isinstance(item_x, torch.Tensor) else torch.tensor(item_x)
        y[i] = item_y

    elapsed = time.time() - t0
    print(f"  --> Preloaded {n_samples:5d} segments from {Path(split_csv).name} into RAM ({X.element_size() * X.nelement() / (1024**2):.1f} MB) in {elapsed:5.1f}s.")
    return TensorDataset(X, y)


def verify_model_and_real_pipeline(
    model: VardhanV2A,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> bool:
    """Verify model parameter count, tensor shapes through the real DataLoader, and backward gradients."""
    print("\n" + "=" * 75)
    print("      MODEL ARCHITECTURE & PIPELINE FORWARD/BACKWARD CHECK          ")
    print("=" * 75)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"1. Model Parameters: {total_params:,} (Expected: 69,559)")
    if total_params != 69559:
        raise ValueError(f"VardhanV2A parameter mismatch! Got {total_params}, expected exactly 69,559.")
    print("[OK] Exact parameter count (69,559) verified.")

    # Real data batch test
    model.eval()
    batch_x, batch_y = next(iter(train_loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    print(f"\n2. Real Data Batch Shapes (Batch Size: {batch_x.shape[0]}):")
    print(f"   - Input Raw Waveform: {batch_x.shape}")
    assert batch_x.ndim == 3 and batch_x.shape[1] == 1 and batch_x.shape[2] == 2048, f"Unexpected shape {batch_x.shape}"

    freq_input, mb_input = model.extract_spectral_representations(batch_x)
    print(f"   - Extracted Fine FFT: {freq_input.shape}")
    print(f"   - Extracted Sub-Bands:{mb_input.shape}")
    assert freq_input.shape == (batch_x.shape[0], 1, 1024)
    assert mb_input.shape == (batch_x.shape[0], 8, 128)

    logits, emb_dict = model(batch_x, return_embeddings=True)
    print(f"   - Temporal Embedding: {emb_dict['h_temporal'].shape}")
    print(f"   - Spectral Embedding: {emb_dict['h_spectral'].shape}")
    print(f"   - Multi-Band Embedding: {emb_dict['h_multiband'].shape}")
    print(f"   - Fused Embedding:    {emb_dict['h_fused'].shape}")
    print(f"   - Output Logits:      {logits.shape}")

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


def run_profiling_benchmark(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    profile_batches: int = 50,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Execute high-resolution microsecond profiling across the training loop."""
    print("=" * 75)
    print(f"STARTING HIGH-RESOLUTION PROFILING BENCHMARK ({profile_batches} BATCHES)")
    print("=" * 75)

    is_cuda = (device.type == "cuda")
    if is_cuda:
        torch.cuda.synchronize()

    times_loader_wait = []
    times_transfer = []
    times_forward = []
    times_loss = []
    times_backward = []
    times_optimizer = []
    times_batch_total = []

    model.train()
    loader_iter = iter(loader)
    t_prev = time.perf_counter()

    for b_idx in range(profile_batches):
        t_start_wait = time.perf_counter()
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        t_after_wait = time.perf_counter()
        times_loader_wait.append((t_after_wait - t_start_wait) * 1000.0)

        # CPU -> GPU transfer
        if is_cuda:
            torch.cuda.synchronize()
        t_transfer_start = time.perf_counter()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if is_cuda:
            torch.cuda.synchronize()
        t_transfer_end = time.perf_counter()
        times_transfer.append((t_transfer_end - t_transfer_start) * 1000.0)

        # Forward pass
        optimizer.zero_grad()
        if is_cuda:
            torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        logits = model(x)
        if is_cuda:
            torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        times_forward.append((t_fwd_end - t_fwd_start) * 1000.0)

        # Loss
        if is_cuda:
            torch.cuda.synchronize()
        t_loss_start = time.perf_counter()
        loss = loss_fn(logits, y)
        if is_cuda:
            torch.cuda.synchronize()
        t_loss_end = time.perf_counter()
        times_loss.append((t_loss_end - t_loss_start) * 1000.0)

        # Backward pass
        if is_cuda:
            torch.cuda.synchronize()
        t_bwd_start = time.perf_counter()
        loss.backward()
        if is_cuda:
            torch.cuda.synchronize()
        t_bwd_end = time.perf_counter()
        times_backward.append((t_bwd_end - t_bwd_start) * 1000.0)

        # Optimizer step
        if is_cuda:
            torch.cuda.synchronize()
        t_opt_start = time.perf_counter()
        optimizer.step()
        if is_cuda:
            torch.cuda.synchronize()
        t_opt_end = time.perf_counter()
        times_optimizer.append((t_opt_end - t_opt_start) * 1000.0)

        t_total_batch = (t_opt_end - t_start_wait) * 1000.0
        times_batch_total.append(t_total_batch)

    # Compute statistics
    mean_wait = float(np.mean(times_loader_wait))
    mean_xfer = float(np.mean(times_transfer))
    mean_fwd = float(np.mean(times_forward))
    mean_loss = float(np.mean(times_loss))
    mean_bwd = float(np.mean(times_backward))
    mean_opt = float(np.mean(times_optimizer))
    mean_batch = float(np.mean(times_batch_total))

    est_epoch_sec = (mean_batch / 1000.0) * len(loader)

    print("\nPROFILING SUMMARY TABLE (Mean per Batch over %d batches):" % profile_batches)
    print("-" * 75)
    print(f"  1. DataLoader / Segment I/O Wait (CPU):   {mean_wait:8.3f} ms  ({mean_wait/mean_batch*100:5.1f}%)")
    print(f"  2. CPU -> GPU Data Transfer:              {mean_xfer:8.3f} ms  ({mean_xfer/mean_batch*100:5.1f}%)")
    print(f"  3. Model Forward Pass (GPU/Model):        {mean_fwd:8.3f} ms  ({mean_fwd/mean_batch*100:5.1f}%)")
    print(f"  4. CrossEntropy Loss Computation:         {mean_loss:8.3f} ms  ({mean_loss/mean_batch*100:5.1f}%)")
    print(f"  5. Backward Pass / Gradient Calc:         {mean_bwd:8.3f} ms  ({mean_bwd/mean_batch*100:5.1f}%)")
    print(f"  6. Optimizer Step (AdamW):                {mean_opt:8.3f} ms  ({mean_opt/mean_batch*100:5.1f}%)")
    print("-" * 75)
    print(f"  --> Mean Total Batch Time:                {mean_batch:8.3f} ms")
    print(f"  --> Estimated Full Epoch Time (482 btch): {est_epoch_sec:8.2f} seconds ({est_epoch_sec/60:.2f} min)")
    print("=" * 75)

    profile_report = {
        "device": str(device),
        "num_batches_profiled": profile_batches,
        "dataloader_num_workers": getattr(loader, "num_workers", 0),
        "dataloader_pin_memory": getattr(loader, "pin_memory", False),
        "mean_dataloader_wait_ms": mean_wait,
        "mean_cpu_gpu_transfer_ms": mean_xfer,
        "mean_forward_ms": mean_fwd,
        "mean_loss_ms": mean_loss,
        "mean_backward_ms": mean_bwd,
        "mean_optimizer_ms": mean_opt,
        "mean_total_batch_ms": mean_batch,
        "estimated_epoch_time_seconds": est_epoch_sec,
    }

    if output_dir is not None:
        with open(output_dir / "profiling_report.json", "w") as f:
            json.dump(profile_report, f, indent=2)

    return profile_report


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

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    return avg_loss, acc, f1, bal_acc


def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Evaluate on a validation or test loader."""
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
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    return avg_loss, acc, f1, bal_acc, np.array(all_targets), np.array(all_preds)


def parse_args():
    parser = argparse.ArgumentParser(description="VARDHAN-v2A Strict Recording-Level Benchmark Training Runner")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for AdamW (default: 0.0003)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW (default: 0.01)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for CrossEntropyLoss (default: 0.1)")
    parser.add_argument("--use_class_weights", action="store_true", default=True, help="Use inverse frequency class weights on train split (default: True)")
    parser.add_argument("--no_class_weights", action="store_false", dest="use_class_weights", help="Disable class weighting")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"], help="Learning rate scheduler (default: cosine)")
    parser.add_argument("--min_lr", type=float, default=0.00001, help="Minimum learning rate for Cosine Annealing (default: 0.00001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda", help="Device preference ('cuda' or 'cpu')")
    parser.add_argument("--samples_per_file", type=int, default=50, help="Samples extracted per raw CSV file (default: 50)")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Base path to raw DroneRF dataset")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output_dir", type=str, default="results/vardhan_v2a_controlled_exp_c", help="Output directory for metrics and results")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints/vardhan_v2a_controlled_exp_c", help="Output directory for checkpoints")
    parser.add_argument("--no_cache", action="store_true", help="Disable RAM preloading/caching and use on-demand lazy file reading")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (default: 0)")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Enable DataLoader pin_memory (default: True)")
    parser.add_argument("--profile", action="store_true", help="Run high-resolution timing profiling on training loop")
    parser.add_argument("--profile_batches", type=int, default=50, help="Number of batches to profile (default: 50)")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Max train batches per epoch (for smoke testing)")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Max val batches per epoch (for smoke testing)")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic mock signals for preflight testing")
    parser.add_argument("--preflight_only", action="store_true", help="Run preflight integrity checks and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    checkpoints_dir = PROJECT_ROOT / args.checkpoints_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = PROJECT_ROOT / args.splits_dir
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    # Select single GPU if CUDA requested & available
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    use_cache = not args.no_cache

    print("=" * 75)
    print("VARDHAN-v2A BENCHMARK RUNNER (STRICT RECORDING-LEVEL SPLIT)")
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
    print(f"Samples/File:    {args.samples_per_file}")
    print(f"Data Caching:    {'ENABLED (In-Memory RAM Preloading)' if use_cache else 'DISABLED (On-Demand File Reading)'}")
    print(f"Results Dir:     {output_dir}")
    print(f"Checkpoints Dir: {checkpoints_dir}")

    # 1. Verify strict recording-level split
    verify_strict_split_integrity(train_csv, val_csv, test_csv)

    # 2. Compute train-only scalar normalization statistics
    print("\nComputing train-only normalization statistics...")
    norm_stats = fit_train_normalization_stats(
        train_split_csv=train_csv,
        segment_length=2048,
        raw_data_dir=args.raw_data_dir,
    )
    print(f"Train Normalization Stats: {norm_stats}")

    # 3. Compute Train-Only Class Weights
    class_weights_tensor, class_weights_dict = compute_train_class_weights(train_csv, args.samples_per_file)
    print(f"\nComputed Train-Only Class Weights (N_total / (C * N_c)):")
    for cls_name, w in class_weights_dict.items():
        print(f"  - {cls_name:15s}: {w:.4f}")

    # 4. Instantiate Datasets & DataLoaders
    if use_cache:
        print("\n[Data Caching] Preloading dataset splits into RAM memory...")
        train_ds = preload_split_dataset(train_csv, norm_stats, args.samples_per_file, 2048, args.raw_data_dir, args.mock)
        val_ds = preload_split_dataset(val_csv, norm_stats, args.samples_per_file, 2048, args.raw_data_dir, args.mock)
        test_ds = preload_split_dataset(test_csv, norm_stats, args.samples_per_file, 2048, args.raw_data_dir, args.mock)

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
    else:
        print("\n[On-Demand Loading] Using lazy file reader DataLoaders...")
        train_loader = get_dataloader(
            split_csv=train_csv,
            model_name="vardhan",
            norm_stats=norm_stats,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda" and args.pin_memory),
            samples_per_file=args.samples_per_file,
            raw_data_dir=args.raw_data_dir,
            mock=args.mock,
        )
        val_loader = get_dataloader(
            split_csv=val_csv,
            model_name="vardhan",
            norm_stats=norm_stats,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda" and args.pin_memory),
            samples_per_file=args.samples_per_file,
            raw_data_dir=args.raw_data_dir,
            mock=args.mock,
        )
        test_loader = get_dataloader(
            split_csv=test_csv,
            model_name="vardhan",
            norm_stats=norm_stats,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda" and args.pin_memory),
            samples_per_file=args.samples_per_file,
            raw_data_dir=args.raw_data_dir,
            mock=args.mock,
        )

    # 5. Instantiate VARDHAN-v2A model
    model = VardhanV2A(num_classes=4).to(device)

    # 6. Model & real-data pipeline verification
    verify_model_and_real_pipeline(model, train_loader, device)

    run_config = {
        "experiment_name": "VARDHAN_v2A_CONTROLLED_EXP_C",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": "VardhanV2A",
        "total_parameters": 69559,
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
        "lr_scheduler": args.lr_scheduler,
        "min_lr": args.min_lr,
        "optimizer": "AdamW",
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    weights_arg = class_weights_tensor.to(device) if args.use_class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weights_arg, label_smoothing=args.label_smoothing)

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    else:
        scheduler = None

    if args.profile:
        run_profiling_benchmark(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            profile_batches=args.profile_batches,
            output_dir=output_dir,
        )
        print("\n[--profile complete] Profiling benchmark finished. Exiting without full training.")
        return

    # 8. Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics = {}
    history_records = []

    print("\n" + "=" * 75)
    print(f"STARTING CONTROLLED VARDHAN-v2A TRAINING ({args.epochs} EPOCHS)")
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

        epoch_time = time.time() - t0
        is_best = va_loss < best_val_loss

        if is_best:
            best_val_loss = va_loss
            best_epoch = epoch
            best_val_metrics = {
                "val_loss": va_loss,
                "val_accuracy": va_acc,
                "val_macro_f1": va_f1,
                "val_balanced_accuracy": va_bal,
            }

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": va_loss,
                "val_accuracy": va_acc,
                "val_f1_macro": va_f1,
                "val_balanced_accuracy": va_bal,
                "norm_stats": norm_stats,
                "run_config": run_config,
            }, checkpoints_dir / "best.pt")

            # Save validation confusion matrix for best epoch
            val_cm = confusion_matrix(y_val_true, y_val_pred, labels=[0, 1, 2, 3])
            val_cm_df = pd.DataFrame(
                val_cm,
                index=[f"True_{LABEL_MAP[i]}" for i in range(4)],
                columns=[f"Pred_{LABEL_MAP[i]}" for i in range(4)],
            )
            val_cm_df.to_csv(output_dir / "val_confusion_matrix.csv")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": va_loss,
            "val_accuracy": va_acc,
            "val_f1_macro": va_f1,
            "val_balanced_accuracy": va_bal,
            "norm_stats": norm_stats,
            "run_config": run_config,
        }, checkpoints_dir / "last.pt")

        history_records.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1_macro": tr_f1,
            "train_balanced_accuracy": tr_bal,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_f1_macro": va_f1,
            "val_balanced_accuracy": va_bal,
            "epoch_time_sec": epoch_time,
            "is_best": is_best,
        })

        best_marker = " *" if is_best else ""
        print(f"  Epoch {epoch:2d}/{args.epochs:2d} (lr: {current_lr:.6f}) | "
              f"Tr Loss: {tr_loss:.4f}, Acc: {tr_acc*100:5.2f}%, F1: {tr_f1:.4f} | "
              f"Val Loss: {va_loss:.4f}, Acc: {va_acc*100:5.2f}%, F1: {va_f1:.4f}, BalAcc: {va_bal*100:5.2f}% | "
              f"Time: {epoch_time:4.1f}s{best_marker}")

    total_training_sec = time.time() - start_train_time
    print("-" * 75)
    print(f"Training Complete in {total_training_sec:.1f}s. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")

    # Export history CSV
    pd.DataFrame(history_records).to_csv(output_dir / "history.csv", index=False)

    # 9. Final Test Evaluation using best.pt
    print("\nLoading best checkpoint for Test Split Evaluation...")
    best_ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device)
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
        "model_name": "VardhanV2A",
        "total_parameters": 69559,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": best_val_metrics,
        "test_loss": float(te_loss),
        "test_accuracy": float(te_acc),
        "test_macro_f1": float(te_f1),
        "test_balanced_accuracy": float(te_bal),
        "class_weights": class_weights_dict,
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
    print("FINAL TEST EVALUATION SUMMARY (EXPERIMENT C: REGULARIZED & BALANCED)")
    print("=" * 75)
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
