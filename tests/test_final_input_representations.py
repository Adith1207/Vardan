"""
test_final_input_representations.py
-----------------------------------

Rigorous test suite verifying finalized input representations:
- Exact tensor shapes for all 6 models from DataLoader and Preprocessor:
    * FGCS2019DNN: (2048,) -> (B, 2048)
    * Baseline1DCNN (MC1DCNN): (8, 256) -> (B, 8, 256)
    * DSCNN (TinyML): (1, 2048) -> (B, 1, 2048)
    * Compressive Sensing: (1, 1024) -> (B, 1, 1024)
    * VardhanRFNet: (1, 2048) -> (B, 1, 2048)
    * MobileNetV3Small: (1, 65, 61) -> (B, 1, 65, 61)
- Model forward pass outputting shape (B, 4) with finite logits
- CompressiveSensingMatrix determinism and constancy across multiple instances/inferences
- Zero synthetic I/Q (np.roll) presence
- Zero split leakage verification
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.loader import DroneRFLazyDataset, get_dataloader
from src.models import (
    FGCS2019DNN,
    Baseline1DCNN,
    CompressiveSensingCNN,
    DSCNN,
    MobileNetV3Small,
    VardhanRFNet,
    get_model,
)
from src.preprocessing.compression import CompressiveSensingMatrix
from src.preprocessing.pipeline import DroneRFPreprocessor
from src.utils.paths import DATA_DIR


def test_preprocessing_shapes():
    """Verify exact tensor shapes produced by DroneRFLazyDataset for all 6 models."""
    split_csv = DATA_DIR / "splits" / "val.csv"
    dummy_stats = {"mean": 0.0, "std": 1.0}

    expected_shapes = {
        "fgcs2019dnn": (2048,),
        "baseline1dcnn": (8, 256),
        "dscnn": (1, 2048),
        "compressed_sensing": (1, 1024),
        "vardhan": (1, 2048),
        "mobilenetv3small": (1, 65, 61),
    }

    for model_name, expected_shape in expected_shapes.items():
        dataset = DroneRFLazyDataset(
            split_csv=split_csv,
            model_name=model_name,
            norm_stats=dummy_stats,
            segment_length=2048,
            samples_per_file=2,
            mock=True,
        )
        sample, label = dataset[0]
        assert sample.shape == expected_shape, (
            f"Model {model_name} produced shape {sample.shape}, expected {expected_shape}"
        )
        assert torch.isfinite(sample).all(), f"Model {model_name} produced non-finite values"


def test_batch_dataloader_shapes():
    """Verify DataLoader yields batched tensors with correct shapes and finite values."""
    split_csv = DATA_DIR / "splits" / "val.csv"
    batch_size = 4

    expected_batch_shapes = {
        "fgcs2019dnn": (batch_size, 2048),
        "baseline1dcnn": (batch_size, 8, 256),
        "dscnn": (batch_size, 1, 2048),
        "compressed_sensing": (batch_size, 1, 1024),
        "vardhan": (batch_size, 1, 2048),
        "mobilenetv3small": (batch_size, 1, 65, 61),
    }

    for model_name, expected_shape in expected_batch_shapes.items():
        loader = get_dataloader(
            split_csv=split_csv,
            model_name=model_name,
            norm_stats={"mean": 0.0, "std": 1.0},
            batch_size=batch_size,
            samples_per_file=2,
            segment_length=2048,
            mock=True,
        )
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == expected_shape, (
            f"DataLoader for {model_name} produced {batch_x.shape}, expected {expected_shape}"
        )
        assert batch_y.shape == (batch_size,)
        assert torch.isfinite(batch_x).all()


def test_model_forward_passes():
    """Verify forward pass on every model produces finite (B, 4) class logits."""
    batch_size = 3
    test_cases = [
        ("fgcs2019dnn", torch.randn(batch_size, 2048)),
        ("baseline1dcnn", torch.randn(batch_size, 8, 256)),
        ("dscnn", torch.randn(batch_size, 1, 2048)),
        ("compressed_sensing", torch.randn(batch_size, 1, 1024)),
        ("vardhan", torch.randn(batch_size, 1, 2048)),
        ("mobilenetv3small", torch.randn(batch_size, 1, 65, 61)),
    ]

    for model_key, in_tensor in test_cases:
        model = get_model(model_key, num_classes=4)
        model.eval()
        with torch.no_grad():
            out = model(in_tensor)
        assert out.shape == (batch_size, 4), f"{model_key} output shape {out.shape} != ({batch_size}, 4)"
        assert torch.isfinite(out).all(), f"{model_key} produced non-finite logits"


def test_cs_matrix_determinism_and_constancy():
    """Verify CompressiveSensingMatrix is deterministic and invariant across instances."""
    cs1 = CompressiveSensingMatrix(n_input=2048, n_compressed=1024, seed=42)
    cs2 = CompressiveSensingMatrix(n_input=2048, n_compressed=1024, seed=42)

    # 1. Exact matrix equality
    assert np.array_equal(cs1.phi, cs2.phi), "Two CS instances with seed=42 produced different Phi matrices"
    assert cs1.phi.shape == (1024, 2048)

    # 2. Projection transform consistency
    dummy_signal = np.ones(2048, dtype=np.float32)
    y1 = cs1.transform(dummy_signal)
    y2 = cs2.transform(dummy_signal)
    assert np.array_equal(y1, y2)
    assert y1.shape == (1024,)
    assert np.isfinite(y1).all()

    # 3. Repeated transform on same instance
    y1_rep = cs1.transform(dummy_signal)
    assert np.array_equal(y1, y1_rep)


def test_no_pseudo_iq_in_dataset():
    """Verify that no np.roll or 2-channel pseudo-I/Q is produced for 1D models."""
    split_csv = DATA_DIR / "splits" / "val.csv"
    for m_name in ["dscnn", "vardhan", "compressed_sensing"]:
        dataset = DroneRFLazyDataset(
            split_csv=split_csv,
            model_name=m_name,
            norm_stats={"mean": 0.0, "std": 1.0},
            segment_length=2048,
            samples_per_file=1,
            mock=True,
        )
        sample, _ = dataset[0]
        # Must be single channel (1, L), NOT (2, L)
        assert sample.shape[0] == 1, f"Model {m_name} has channel count {sample.shape[0]}, expected 1 (no pseudo-IQ)"


def test_zero_leakage_across_splits():
    """Verify that dataset splits remain strictly disjoint by recording and relative path."""
    import pandas as pd
    train_df = pd.read_csv(DATA_DIR / "splits" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "splits" / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "splits" / "test.csv")

    train_files = set(train_df["relative_path"].unique())
    val_files = set(val_df["relative_path"].unique())
    test_files = set(test_df["relative_path"].unique())

    assert len(train_files.intersection(val_files)) == 0, "Train and Val share files"
    assert len(train_files.intersection(test_files)) == 0, "Train and Test share files"
    assert len(val_files.intersection(test_files)) == 0, "Val and Test share files"

    train_recs = set(train_df["recording_id"].unique())
    val_recs = set(val_df["recording_id"].unique())
    test_recs = set(test_df["recording_id"].unique())

    assert len(train_recs.intersection(val_recs)) == 0, "Train and Val share recording sessions"
    assert len(train_recs.intersection(test_recs)) == 0, "Train and Test share recording sessions"
    assert len(val_recs.intersection(test_recs)) == 0, "Val and Test share recording sessions"

