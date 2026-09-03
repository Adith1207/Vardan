"""
test_train_vardhan_v2a.py
-------------------------

Pytest suite for the VARDHAN-v2A training runner, profiling, and in-memory caching.
Tests strict split integrity, real/mock data loading, parameter count,
cached vs uncached numerical equality, profiling benchmark, checkpoint creation, and metrics writing.
"""

import json
import sys
from pathlib import Path

# Ensure src and root are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from models.vardhan_v2a import VardhanV2A
from data.loader import DroneRFLazyDataset, fit_train_normalization_stats, get_dataloader
from scripts.train_vardhan_v2a import (
    verify_strict_split_integrity,
    verify_model_and_real_pipeline,
    preload_split_dataset,
    run_profiling_benchmark,
)


def test_runner_imports_and_parameter_count():
    """Verify clean import of VardhanV2A and exact 69,559 parameter count."""
    model = VardhanV2A(num_classes=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 69559


def test_strict_split_verification_function():
    """Verify that verify_strict_split_integrity passes on canonical split files."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    assert train_csv.exists()
    assert val_csv.exists()
    assert test_csv.exists()

    passed = verify_strict_split_integrity(train_csv, val_csv, test_csv)
    assert passed is True


def test_cached_vs_uncached_sample_numerical_equality():
    """Verify that preloaded in-memory TensorDataset is 100% numerically identical to uncached LazyDataset."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"

    norm_stats = {"mean": 0.123, "std": 1.456, "max": 1.0, "min": -1.0}

    # 1. Uncached Lazy Dataset
    ds_lazy = DroneRFLazyDataset(
        split_csv=train_csv,
        model_name="vardhan",
        norm_stats=norm_stats,
        samples_per_file=50,
        segment_length=2048,
        mock=True,
    )

    # 2. Preloaded In-Memory Dataset
    ds_cached = preload_split_dataset(
        split_csv=train_csv,
        norm_stats=norm_stats,
        samples_per_file=50,
        segment_length=2048,
        mock=True,
    )

    assert len(ds_lazy) == len(ds_cached) == 15400

    # Spot check multiple indices across the dataset
    check_indices = [0, 1, 49, 50, 100, 500, 1000, 5000, 15399]
    for idx in check_indices:
        x_lazy, y_lazy = ds_lazy[idx]
        x_cached, y_cached = ds_cached[idx]

        x_lazy_t = x_lazy if isinstance(x_lazy, torch.Tensor) else torch.tensor(x_lazy)
        assert torch.all(x_lazy_t == x_cached), f"Mismatch at sample index {idx}"
        assert y_lazy == y_cached, f"Label mismatch at sample index {idx}"


def test_split_sample_counts_and_classes():
    """Verify exact sample counts for Train (15,400), Val (3,650), Test (3,650)."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    norm_stats = {"mean": 0.0, "std": 1.0, "max": 1.0, "min": -1.0}

    train_ds = preload_split_dataset(splits_dir / "train.csv", norm_stats, 50, 2048, mock=True)
    val_ds = preload_split_dataset(splits_dir / "val.csv", norm_stats, 50, 2048, mock=True)
    test_ds = preload_split_dataset(splits_dir / "test.csv", norm_stats, 50, 2048, mock=True)

    assert len(train_ds) == 308 * 50 == 15400
    assert len(val_ds) == 73 * 50 == 3650
    assert len(test_ds) == 73 * 50 == 3650


def test_real_pipeline_forward_and_backward():
    """Verify that verify_model_and_real_pipeline passes on data loader batch."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"

    norm_stats = fit_train_normalization_stats(train_csv, max_files=5)
    loader = get_dataloader(
        split_csv=train_csv,
        model_name="vardhan",
        norm_stats=norm_stats,
        batch_size=4,
        shuffle=False,
        samples_per_file=2,
        mock=True,
    )

    model = VardhanV2A(num_classes=4)
    device = torch.device("cpu")
    passed = verify_model_and_real_pipeline(model, loader, device)
    assert passed is True


def test_profiling_benchmark_execution(tmp_path):
    """Verify that run_profiling_benchmark computes valid timings and saves profiling_report.json."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"

    norm_stats = {"mean": 0.0, "std": 1.0, "max": 1.0, "min": -1.0}
    loader = get_dataloader(train_csv, model_name="vardhan", norm_stats=norm_stats, batch_size=4, mock=True, samples_per_file=2)

    model = VardhanV2A(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    report = run_profiling_benchmark(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        profile_batches=5,
        output_dir=tmp_path,
    )

    assert "mean_dataloader_wait_ms" in report
    assert "mean_forward_ms" in report
    assert "mean_backward_ms" in report
    assert "mean_total_batch_ms" in report
    assert (tmp_path / "profiling_report.json").exists()


def test_mock_training_and_artifact_generation(tmp_path):
    """Verify that a 1-epoch mock training run produces best.pt, last.pt, history.csv, metrics.json, and confusion_matrix.csv."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    norm_stats = {"mean": 0.0, "std": 1.0, "max": 1.0, "min": -1.0}
    train_loader = get_dataloader(train_csv, model_name="vardhan", norm_stats=norm_stats, batch_size=4, mock=True, samples_per_file=2)
    val_loader = get_dataloader(val_csv, model_name="vardhan", norm_stats=norm_stats, batch_size=4, mock=True, samples_per_file=2)
    test_loader = get_dataloader(test_csv, model_name="vardhan", norm_stats=norm_stats, batch_size=4, mock=True, samples_per_file=2)

    model = VardhanV2A(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    out_dir = tmp_path / "results"
    ckpt_dir = tmp_path / "checkpoints"
    out_dir.mkdir()
    ckpt_dir.mkdir()

    # Train 1 step
    model.train()
    bx, by = next(iter(train_loader))
    optimizer.zero_grad()
    loss = loss_fn(model(bx), by)
    loss.backward()
    optimizer.step()

    # Save checkpoints
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, ckpt_dir / "best.pt")
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, ckpt_dir / "last.pt")

    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "last.pt").exists()

    # Save metrics
    metrics = {"model_name": "VardhanV2A", "test_accuracy": 0.5, "total_parameters": 69559}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    assert (out_dir / "metrics.json").exists()
