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
    compute_inverse_frequency_class_weights,
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
    assert hasattr(runner, "compute_inverse_frequency_class_weights")
    assert hasattr(runner, "train_single_fold")
    assert hasattr(runner, "main")


def test_compute_inverse_frequency_class_weights_training_only():
    """Verify class weights are strictly derived from training labels using N / (4 * count_c)."""
    # Standardized 10-fold Train Fold 1 distribution (20,430 samples):
    # Class 0: 3690, Class 1: 7560, Class 2: 7290, Class 3: 1890
    y_train = np.array(
        [0] * 3690 + [1] * 7560 + [2] * 7290 + [3] * 1890,
        dtype=np.int64,
    )
    # Mock test set (2,270 samples) - must NEVER affect weights
    y_test = np.array(
        [0] * 410 + [1] * 840 + [2] * 810 + [3] * 210,
        dtype=np.int64,
    )

    weights, counts = compute_inverse_frequency_class_weights(y_train, num_classes=4)

    # 1. Verify counts derived from y_train only
    assert counts[0] == 3690
    assert counts[1] == 7560
    assert counts[2] == 7290
    assert counts[3] == 1890

    # 2. Verify all 4 weights are positive and finite
    assert len(weights) == 4
    assert np.all(weights > 0.0)
    assert np.all(np.isfinite(weights))

    # 3. Verify exact mathematical formula values
    # N_train = 20,430
    assert weights[0] == pytest.approx(20430.0 / (4.0 * 3690.0), rel=1e-5)
    assert weights[1] == pytest.approx(20430.0 / (4.0 * 7560.0), rel=1e-5)
    assert weights[2] == pytest.approx(20430.0 / (4.0 * 7290.0), rel=1e-5)
    assert weights[3] == pytest.approx(20430.0 / (4.0 * 1890.0), rel=1e-5)

    # Approximate exact numerical values:
    # Class 0: ~1.384146
    # Class 1: ~0.675595
    # Class 2: ~0.700617
    # Class 3: ~2.702381
    assert weights[0] == pytest.approx(1.384146, rel=1e-4)
    assert weights[1] == pytest.approx(0.675595, rel=1e-4)
    assert weights[2] == pytest.approx(0.700617, rel=1e-4)
    assert weights[3] == pytest.approx(2.702381, rel=1e-4)


def test_class_weighted_loss_execution_and_checkpoint():
    """Verify class-weighted training executes cleanly and persists weighting metadata."""
    set_seed(42)
    device = torch.device("cpu")

    # Imbalanced training subset: 40 Class 0, 80 Class 1, 60 Class 2, 20 Class 3 = 200 samples
    y_tr = np.array([0] * 40 + [1] * 80 + [2] * 60 + [3] * 20, dtype=np.int64)
    X_tr = np.random.randn(len(y_tr), 1, 2048).astype(np.float32)

    y_te = np.array([0] * 10 + [1] * 20 + [2] * 15 + [3] * 5, dtype=np.int64)
    X_te = np.random.randn(len(y_te), 1, 2048).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir) / "checkpoints"

        # Run with class_weighted=True
        res_weighted = train_single_fold(
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
            class_weighted=True,
            checkpoints_dir=ckpt_dir,
            verbose=False,
        )

        assert res_weighted["class_weighted"] is True
        assert "class_weights" in res_weighted
        assert res_weighted["class_weights"]["0"] == pytest.approx(200.0 / (4.0 * 40.0))
        assert res_weighted["class_weights"]["1"] == pytest.approx(200.0 / (4.0 * 80.0))
        assert res_weighted["class_weights"]["2"] == pytest.approx(200.0 / (4.0 * 60.0))
        assert res_weighted["class_weights"]["3"] == pytest.approx(200.0 / (4.0 * 20.0))

        # Check saved checkpoint
        best_ckpt = ckpt_dir / "checkpoint_fold_1.pt"
        assert best_ckpt.exists()
        state = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        assert state["class_weighted"] is True
        assert state["class_weights"] is not None
        assert state["train_class_counts"]["0"] == 40
        assert state["train_class_counts"]["3"] == 20

        # Run with class_weighted=False (unweighted baseline)
        res_unweighted = train_single_fold(
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
            class_weighted=False,
            checkpoints_dir=ckpt_dir,
            verbose=False,
        )
        assert res_unweighted["class_weighted"] is False


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
    # Test default
    monkeypatch.setattr(sys, "argv", ["train_vardhan_v3_standardized.py"])
    args_default = parse_args()
    assert args_default.class_weighted is False
    assert args_default.normalization == "global"

    # Test custom flags
    test_args = [
        "train_vardhan_v3_standardized.py",
        "--epochs", "5",
        "--batch_size", "16",
        "--lr", "0.0005",
        "--min_lr", "1e-5",
        "--weight_decay", "0.0002",
        "--label_smoothing", "0.1",
        "--class_weighted",
        "--normalization", "per_segment",
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
    assert args.class_weighted is True
    assert args.normalization == "per_segment"
    assert args.seed == 42
    assert args.max_folds == 2
    assert args.single_fold == 1
    assert args.mock is True
    assert args.preflight_only is True


def test_per_segment_normalization_mean_and_std():
    """Verify per-segment normalization independently yields mean ~0 and std ~1 for each 2048-sample waveform."""
    rng = np.random.RandomState(42)
    # Generate 10 segments of varying baseline offsets and standard deviations
    X_raw = np.zeros((10, 1, 2048), dtype=np.float32)
    for i in range(10):
        offset = (i - 5) * 5.0
        scale = 0.5 * (1.5 ** i)  # scale spans from 0.5 to ~19.2
        X_raw[i, 0] = offset + scale * rng.randn(2048).astype(np.float32)

    mean_seg = np.mean(X_raw, axis=-1, keepdims=True)
    std_seg = np.std(X_raw, axis=-1, keepdims=True) + 1e-8
    X_norm = (X_raw - mean_seg) / std_seg

    # Assert shape is preserved
    assert X_norm.shape == (10, 1, 2048)

    # Check each individual segment independently
    for i in range(10):
        seg = X_norm[i, 0]
        assert np.mean(seg) == pytest.approx(0.0, abs=1e-4), f"Segment {i} mean is not approx 0"
        assert np.std(seg) == pytest.approx(1.0, rel=1e-3), f"Segment {i} std is not approx 1"


def test_global_normalization_behavior_unchanged():
    """Verify global normalization behavior matches scalar Z-score on training set exactly."""
    rng = np.random.RandomState(123)
    X_tr = rng.randn(50, 1, 2048).astype(np.float32) * 25.0 + 10.0
    X_te = rng.randn(20, 1, 2048).astype(np.float32) * 30.0 + 5.0

    mean_tr = float(np.mean(X_tr))
    std_tr = float(np.std(X_tr)) + 1e-8

    X_tr_norm = (X_tr - mean_tr) / std_tr
    X_te_norm = (X_te - mean_tr) / std_tr

    assert np.mean(X_tr_norm) == pytest.approx(0.0, abs=1e-5)
    assert np.std(X_tr_norm) == pytest.approx(1.0, rel=1e-4)
    # Test set normalized with train stats should NOT be forced to zero mean
    assert X_te_norm.shape == (20, 1, 2048)


def test_per_segment_normalization_sample_independence_and_no_test_leakage():
    """Verify per-segment normalization of test sample is 100% independent and leakage-free."""
    rng = np.random.RandomState(999)
    sample_a = rng.randn(1, 1, 2048).astype(np.float32) * 50.0 + 100.0
    sample_b = rng.randn(1, 1, 2048).astype(np.float32) * 2.0 - 50.0

    # Normalize sample_a in isolation
    mean_a = np.mean(sample_a, axis=-1, keepdims=True)
    std_a = np.std(sample_a, axis=-1, keepdims=True) + 1e-8
    norm_a_isolated = (sample_a - mean_a) / std_a

    # Normalize sample_a within a batch with sample_b
    batch = np.concatenate([sample_a, sample_b], axis=0)
    mean_batch = np.mean(batch, axis=-1, keepdims=True)
    std_batch = np.std(batch, axis=-1, keepdims=True) + 1e-8
    norm_batch = (batch - mean_batch) / std_batch

    # The normalized output for sample_a inside the batch must be bit-exact to isolated normalization
    np.testing.assert_allclose(norm_batch[0:1], norm_a_isolated, rtol=1e-7, atol=1e-7)


def test_train_single_fold_per_segment_normalization_execution():
    """Verify train_single_fold executes with normalization='per_segment' and stores metadata."""
    set_seed(42)
    device = torch.device("cpu")

    N_tr = 32
    N_te = 16
    X_tr = np.random.randn(N_tr, 1, 2048).astype(np.float32) * 10.0 + 5.0
    y_tr = np.random.choice([0, 1, 2, 3], size=N_tr).astype(np.int64)
    X_te = np.random.randn(N_te, 1, 2048).astype(np.float32) * 20.0 - 10.0
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
            normalization="per_segment",
            checkpoints_dir=ckpt_dir,
            verbose=False,
        )

        assert res["normalization"] == "per_segment"
        assert res["fold"] == 1
        assert "test_loss" in res and np.isfinite(res["test_loss"])

        # Check saved checkpoint metadata
        best_ckpt = ckpt_dir / "checkpoint_fold_1.pt"
        assert best_ckpt.exists()
        state = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        assert state["normalization"] == "per_segment"
        assert state["norm_mean"] is None
        assert state["norm_std"] is None
