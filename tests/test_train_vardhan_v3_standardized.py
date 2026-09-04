"""
test_train_vardhan_v3_standardized.py
-------------------------------------

Pytest test suite for standardized segment-level VARDHAN-v3 training runner.
Validates:
1. Module and function imports.
2. Synthetic manifest creation and preflight verification logic.
3. In-memory waveform materialization (mock mode).
4. One-batch DataLoader creation and forward pass tensor shapes.
5. Loss calculation with label smoothing (0.05).
6. Backward pass, gradient computation, optimizer step (AdamW), and CosineAnnealingLR step.
7. Single-fold execution, metric calculation, and checkpoint persistence.
8. CLI argument parser defaults and custom flag overrides (including --max_folds and --label_smoothing).
"""

import argparse
import sys
from pathlib import Path
import tempfile

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from models.vardhan_v3 import VardhanV3
from train_vardhan_v3_standardized import (
    set_seed,
    preflight_verification,
    materialize_standardized_waveforms,
    train_single_fold,
    parse_args,
)


@pytest.fixture
def synthetic_manifest() -> pd.DataFrame:
    """Create a minimal synthetic manifest conforming to 227 pairs * 100 segments = 22,700 rows."""
    records = []
    # 41 background, 84 bebop, 81 ar, 21 phantom = 227 pairs
    buis = (
        [("00000", "Background RF activities", 0)] * 41
        + [("10000", "Bebop drone", 1)] * 84
        + [("10100", "AR drone", 2)] * 81
        + [("11000", "Phantom drone", 3)] * 21
    )

    for pair_idx, (bui, cls_name, lbl) in enumerate(buis):
        for seg_idx in range(100):
            records.append({
                "pair_id": f"pair_{pair_idx:03d}",
                "bui": bui,
                "drone_class": cls_name,
                "faithful_label": lbl,
                "segment_index": seg_idx,
                "l_path": f"/mock/path/L_{pair_idx}.csv",
                "h_path": f"/mock/path/H_{pair_idx}.csv",
            })
    return pd.DataFrame(records)


def test_runner_imports():
    """Verify clean import of all training runner functions and classes."""
    import train_vardhan_v3_standardized as runner
    assert hasattr(runner, "preflight_verification")
    assert hasattr(runner, "materialize_standardized_waveforms")
    assert hasattr(runner, "train_single_fold")
    assert hasattr(runner, "main")


def test_preflight_verification(synthetic_manifest):
    """Verify preflight verification passes on valid 22,700-segment manifest."""
    assert preflight_verification(synthetic_manifest, num_folds=10, seed=1) is True


def test_mock_waveform_materialization(synthetic_manifest):
    """Verify deterministic mock waveform generation produces (22700, 1, 2048) array."""
    X, y = materialize_standardized_waveforms(synthetic_manifest, mock=True, segment_length=2048)
    assert X.shape == (22700, 1, 2048)
    assert y.shape == (22700,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert np.all(np.isfinite(X))


def test_one_batch_forward_and_loss():
    """Verify model forward pass on a single batch with label smoothing loss."""
    model = VardhanV3(num_classes=4)
    model.train()

    B = 32
    x = torch.randn(B, 1, 2048)
    y = torch.randint(0, 4, (B,))

    logits = model(x)
    assert logits.shape == (B, 4)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    loss = criterion(logits, y)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_backward_pass_optimizer_and_scheduler_step():
    """Verify backward pass, AdamW optimizer step, and CosineAnnealingLR step."""
    model = VardhanV3(num_classes=4)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    x = torch.randn(16, 1, 2048)
    y = torch.randint(0, 4, (16,))

    initial_lr = optimizer.param_groups[0]["lr"]
    assert initial_lr == pytest.approx(3e-4)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Check gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has None grad"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite grad"

    optimizer.step()
    scheduler.step()

    updated_lr = optimizer.param_groups[0]["lr"]
    assert updated_lr < initial_lr, "Cosine scheduler did not decrease learning rate"


def test_train_single_fold_and_checkpoint_creation():
    """Verify train_single_fold executes cleanly and creates valid checkpoint files."""
    set_seed(42)
    device = torch.device("cpu")

    # Small mock dataset for fast testing: 64 train samples, 32 test samples
    N_tr = 64
    N_te = 32
    X_tr = np.random.randn(N_tr, 1, 2048).astype(np.float32)
    y_tr = np.random.choice([0, 1, 2, 3], size=N_tr).astype(np.int64)
    X_te = np.random.randn(N_te, 1, 2048).astype(np.float32)
    y_te = np.random.choice([0, 1, 2, 3], size=N_te).astype(np.int64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints"

        res = train_single_fold(
            fold_idx=1,
            X_train_raw=X_tr,
            y_train=y_tr,
            X_test_raw=X_te,
            y_test=y_te,
            device=device,
            epochs=2,
            batch_size=16,
            lr=0.001,
            weight_decay=0.0001,
            label_smoothing=0.05,
            min_lr=1e-6,
            checkpoints_dir=ckpt_dir,
            verbose=False,
        )

        assert res["fold"] == 1
        assert "test_loss" in res and np.isfinite(res["test_loss"])
        assert "test_accuracy" in res and 0.0 <= res["test_accuracy"] <= 1.0
        assert "test_macro_f1" in res and 0.0 <= res["test_macro_f1"] <= 1.0
        assert "confusion_matrix" in res and res["confusion_matrix"].shape == (4, 4)
        assert len(res["history"]) == 2

        # Verify best and final checkpoints were saved
        best_ckpt = ckpt_dir / "checkpoint_fold_1.pt"
        final_ckpt = ckpt_dir / "final_model_fold_1.pt"
        assert best_ckpt.exists(), "checkpoint_fold_1.pt not created"
        assert final_ckpt.exists(), "final_model_fold_1.pt not created"

        # Verify checkpoint loads cleanly
        state = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        assert "model_state_dict" in state
        assert "norm_mean" in state
        assert "norm_std" in state
        assert state["fold"] == 1

        # Load into model
        model = VardhanV3(num_classes=4)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, 2048))
            assert out.shape == (2, 4)


def test_cli_argument_parsing(monkeypatch):
    """Verify command line arguments and default flags parse correctly."""
    test_args = [
        "train_vardhan_v3_standardized.py",
        "--epochs", "5",
        "--batch_size", "16",
        "--lr", "0.0005",
        "--min_lr", "1e-5",
        "--weight_decay", "0.0002",
        "--label_smoothing", "0.1",
        "--seed", "42",
        "--max_folds", "2",
        "--single_fold", "1",
        "--mock",
        "--preflight_only",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    args = parse_args()

    assert args.epochs == 5
    assert args.batch_size == 16
    assert args.lr == 0.0005
    assert args.min_lr == 1e-5
    assert args.weight_decay == 0.0002
    assert args.label_smoothing == 0.1
    assert args.seed == 42
    assert args.max_folds == 2
    assert args.single_fold == 1
    assert args.mock is True
    assert args.preflight_only is True
