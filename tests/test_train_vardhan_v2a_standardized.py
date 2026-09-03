"""
test_train_vardhan_v2a_standardized.py
--------------------------------------

Pytest suite for the standardized segment-level VARDHAN-v2A benchmark runner.
Tests dataset construction, exact sample counts (22,700), class distribution (4100/8400/8100/2100),
fold split non-leakage, train-only normalization, model forward/backward, and single-fold mock execution.
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
from sklearn.model_selection import StratifiedKFold

from models.vardhan_v2a import VardhanV2A
from preprocessing.fgcs_faithful import (
    FAITHFUL_FGCS_CLASS_TO_INDEX,
    FAITHFUL_FGCS_INDEX_TO_CLASS,
)
from data.fgcs_faithful_loader import (
    discover_and_pair_dronerf_files,
    build_faithful_manifest,
)
from scripts.train_vardhan_v2a_standardized import (
    preflight_verification,
    materialize_standardized_waveforms,
    train_single_fold,
)


def test_standardized_runner_parameter_count():
    """Verify clean import of VardhanV2A and exact 69,559 parameter count."""
    model = VardhanV2A(num_classes=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 69559


def test_faithful_manifest_construction_and_counts():
    """Verify that build_faithful_manifest creates exactly 22,700 rows with exact class distributions."""
    df_pairs = discover_and_pair_dronerf_files()
    assert len(df_pairs) == 227

    df_manifest = build_faithful_manifest()
    assert len(df_manifest) == 22700

    counts = df_manifest["faithful_label"].value_counts().to_dict()
    assert counts[0] == 4100  # Background
    assert counts[1] == 8400  # Bebop
    assert counts[2] == 8100  # AR
    assert counts[3] == 2100  # Phantom


def test_preflight_verification_function():
    """Verify that preflight_verification passes completely on the faithful manifest."""
    df_manifest = build_faithful_manifest()
    passed = preflight_verification(df_manifest, num_folds=10, seed=1)
    assert passed is True


def test_materialize_mock_waveforms():
    """Verify that materialize_standardized_waveforms produces (22700, 1, 2048) finite float32 arrays."""
    df_manifest = build_faithful_manifest()

    X, y = materialize_standardized_waveforms(df_manifest, mock=True, segment_length=2048)
    assert X.shape == (22700, 1, 2048)
    assert y.shape == (22700,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert np.all(np.isfinite(X))


def test_stratified_kfold_leakage_and_representation():
    """Verify zero overlap between train and test indices across all 10 folds and full 4-class representation."""
    df_manifest = build_faithful_manifest()
    y = df_manifest["faithful_label"].to_numpy()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(df_manifest, y), 1):
        assert len(tr_idx) == 20430
        assert len(te_idx) == 2270
        assert len(set(tr_idx) & set(te_idx)) == 0
        assert len(np.unique(y[tr_idx])) == 4
        assert len(np.unique(y[te_idx])) == 4


def test_train_single_fold_mock_execution(tmp_path):
    """Verify training a single fold on a small mock dataset produces valid fold metrics and checkpoint."""
    n_samples = 100
    X_train = np.random.randn(80, 1, 2048).astype(np.float32)
    y_train = np.random.randint(0, 4, size=80, dtype=np.int64)
    X_test = np.random.randn(20, 1, 2048).astype(np.float32)
    y_test = np.random.randint(0, 4, size=20, dtype=np.int64)

    device = torch.device("cpu")
    res = train_single_fold(
        fold_idx=1,
        X_train_raw=X_train,
        y_train=y_train,
        X_test_raw=X_test,
        y_test=y_test,
        device=device,
        epochs=1,
        batch_size=8,
        lr=0.001,
        checkpoints_dir=tmp_path,
    )

    assert "test_accuracy" in res
    assert "test_macro_f1" in res
    assert "test_balanced_accuracy" in res
    assert "per_class" in res
    assert (tmp_path / "checkpoint_fold_1.pt").exists()
