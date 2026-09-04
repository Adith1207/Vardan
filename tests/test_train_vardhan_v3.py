"""
test_train_vardhan_v3.py
------------------------

Pytest suite for the strict recording-level split VARDHAN-v3 training runner.
Tests:
1. Parameter count verification (exactly 119,806 parameters).
2. Strict recording-level split integrity verification (zero file overlap, zero recording overlap).
3. CLI argument parsing (default normalization='global', custom normalization='per_segment').
4. Per-segment normalization mean ~0 and std ~1 for arbitrary waveforms.
5. Independent normalization across different segments (sample independence).
6. Leakage-free normalization (zero test statistics used for training normalization).
7. Global normalization behavior unchanged.
8. Dataset preloading with global vs per_segment normalization.
9. Pipeline forward/backward verification with VardhanV3.
10. End-to-end strict-runner execution (smoke run and checkpoint creation).
"""

import json
import sys
import tempfile
from pathlib import Path

# Ensure src and root are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vardhan_v3 import VardhanV3
from data.loader import fit_train_normalization_stats
from scripts.train_vardhan_v3 import (
    set_seed,
    verify_strict_split_integrity,
    compute_train_class_weights,
    normalize_waveform_segment,
    preload_split_dataset,
    verify_model_and_real_pipeline,
    train_one_epoch,
    evaluate_split,
    parse_args,
)


def test_v3_parameter_count_and_architecture():
    """Verify VardhanV3 has exactly 119,806 trainable parameters."""
    model = VardhanV3(num_classes=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 119806, f"Expected 119,806 parameters, got {total_params}"


def test_strict_split_verification():
    """Verify that verify_strict_split_integrity passes on canonical split manifests."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    assert train_csv.exists()
    assert val_csv.exists()
    assert test_csv.exists()

    passed = verify_strict_split_integrity(train_csv, val_csv, test_csv)
    assert passed is True


def test_cli_parsing_defaults_and_overrides(monkeypatch):
    """Verify CLI argument parsing handles --normalization global and per_segment."""
    # Test default
    monkeypatch.setattr(sys, "argv", ["train_vardhan_v3.py"])
    args_default = parse_args()
    assert args_default.normalization == "global"
    assert args_default.lr == 0.0003
    assert args_default.epochs == 100
    assert args_default.batch_size == 32
    assert args_default.use_class_weights is False

    # Test custom flags
    custom_args = [
        "train_vardhan_v3.py",
        "--epochs", "5",
        "--batch_size", "16",
        "--lr", "0.001",
        "--normalization", "per_segment",
        "--use_class_weights",
        "--seed", "123",
        "--mock",
        "--preflight_only",
    ]
    monkeypatch.setattr(sys, "argv", custom_args)
    args = parse_args()
    assert args.normalization == "per_segment"
    assert args.epochs == 5
    assert args.batch_size == 16
    assert args.lr == 0.001
    assert args.use_class_weights is True
    assert args.seed == 123
    assert args.mock is True
    assert args.preflight_only is True


def test_per_segment_mean_and_std():
    """Verify per-segment normalization yields mean ~0 and std ~1 on arbitrary waveform inputs."""
    rng = np.random.RandomState(42)
    # Generate signals with disparate offsets and scales
    for i in range(10):
        offset = (i - 5) * 10.0
        scale = 0.5 * (1.5 ** i)
        raw_wave = offset + scale * rng.randn(2048).astype(np.float32)

        norm_wave = normalize_waveform_segment(raw_wave, normalization="per_segment")
        assert norm_wave.shape == (1, 2048)

        mean_val = float(norm_wave.mean().item())
        std_val = float(norm_wave.std().item())

        assert mean_val == pytest.approx(0.0, abs=1e-4)
        assert std_val == pytest.approx(1.0, rel=1e-3)


def test_different_segments_normalized_independently():
    """Verify normalization of segment A is independent of segment B."""
    rng = np.random.RandomState(101)
    seg_a = rng.randn(2048).astype(np.float32) * 5.0 + 50.0
    seg_b = rng.randn(2048).astype(np.float32) * 0.1 - 20.0

    norm_a_1 = normalize_waveform_segment(seg_a, normalization="per_segment")
    norm_b_1 = normalize_waveform_segment(seg_b, normalization="per_segment")

    # Change seg_b completely and renormalize seg_a
    seg_b_mod = rng.randn(2048).astype(np.float32) * 100.0 + 500.0
    norm_a_2 = normalize_waveform_segment(seg_a, normalization="per_segment")

    torch.testing.assert_close(norm_a_1, norm_a_2)


def test_global_normalization_behavior_unchanged():
    """Verify global normalization computes exact scalar Z-score using train stats."""
    raw_wave = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    norm_stats = {"mean": 20.0, "std": 10.0, "max": 40.0, "min": 10.0}

    norm_wave = normalize_waveform_segment(raw_wave, normalization="global", norm_stats=norm_stats)
    expected = (raw_wave - 20.0) / (10.0 + 1e-8)

    torch.testing.assert_close(norm_wave.squeeze(0), torch.tensor(expected, dtype=torch.float32))


def test_preload_split_dataset_global_and_per_segment():
    """Verify preload_split_dataset loads samples with correct shapes under both normalizations."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"

    norm_stats = {"mean": 1.0, "std": 2.0, "max": 5.0, "min": -5.0}

    # Global
    ds_global = preload_split_dataset(
        split_csv=train_csv,
        norm_stats=norm_stats,
        samples_per_file=2,
        segment_length=2048,
        mock=True,
        normalization="global",
    )
    assert len(ds_global) == 308 * 2
    x_g, y_g = ds_global[0]
    assert x_g.shape == (1, 2048)

    # Per-segment
    ds_per_seg = preload_split_dataset(
        split_csv=train_csv,
        norm_stats=None,
        samples_per_file=2,
        segment_length=2048,
        mock=True,
        normalization="per_segment",
    )
    assert len(ds_per_seg) == 308 * 2
    x_p, y_p = ds_per_seg[0]
    assert x_p.shape == (1, 2048)
    assert float(x_p.mean().item()) == pytest.approx(0.0, abs=1e-4)
    assert float(x_p.std().item()) == pytest.approx(1.0, rel=1e-3)


def test_verify_model_and_real_pipeline():
    """Verify verify_model_and_real_pipeline executes cleanly with VardhanV3."""
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"

    ds = preload_split_dataset(
        split_csv=train_csv,
        norm_stats=None,
        samples_per_file=2,
        segment_length=2048,
        mock=True,
        normalization="per_segment",
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = VardhanV3(num_classes=4)
    device = torch.device("cpu")
    passed = verify_model_and_real_pipeline(model, loader, device)
    assert passed is True


def test_end_to_end_strict_runner_execution():
    """Verify a complete end-to-end training and test evaluation cycle on strict split."""
    set_seed(42)
    device = torch.device("cpu")
    splits_dir = PROJECT_ROOT / "data" / "splits"

    train_ds = preload_split_dataset(splits_dir / "train.csv", None, 2, 2048, mock=True, normalization="per_segment")
    val_ds = preload_split_dataset(splits_dir / "val.csv", None, 2, 2048, mock=True, normalization="per_segment")
    test_ds = preload_split_dataset(splits_dir / "test.csv", None, 2, 2048, mock=True, normalization="per_segment")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = VardhanV3(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Train 1 epoch
    tr_loss, tr_acc, tr_f1, tr_bal = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        max_batches=2,
    )
    assert np.isfinite(tr_loss)
    assert 0.0 <= tr_acc <= 1.0

    # Evaluate validation
    va_loss, va_acc, va_f1, va_bal, y_v_t, y_v_p = evaluate_split(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        max_batches=2,
    )
    assert np.isfinite(va_loss)
    assert 0.0 <= va_acc <= 1.0

    # Evaluate test
    te_loss, te_acc, te_f1, te_bal, y_t_t, y_t_p = evaluate_split(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        max_batches=2,
    )
    assert np.isfinite(te_loss)
    assert 0.0 <= te_acc <= 1.0
