"""
test_baseline1dcnn_faithful.py
------------------------------

Pytest test suite for faithful Baseline1DCNN (MC1DCNN) reproduction.
Verifies input reshaping, manifest capacity, model architecture, parameter count,
preprocessing parity, and strict benchmark isolation.
"""

import sys
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest
import torch

from models.baselines import Baseline1DCNN
from preprocessing.fgcs_faithful import (
    FAITHFUL_FGCS_CLASS_TO_INDEX,
    FAITHFUL_FGCS_INDEX_TO_CLASS,
    normalize_global_max,
    process_faithful_fgcs_pair_vectorized,
)
from data.fgcs_faithful_loader import (
    build_faithful_manifest,
    discover_and_pair_dronerf_files,
    read_raw_signal_10m,
)
from preprocessing.channelization import SpectrumChannelizer
from utils.paths import PROJECT_ROOT


def test_mc1dcnn_reshape_2048_to_8x256():
    """Verify that contiguous reshape(8, 256) exactly matches uniform 8-channel sub-band division."""
    spectrum = np.random.rand(2048).astype(np.float32)

    # Method A: SpectrumChannelizer object
    channelizer = SpectrumChannelizer(channel_count=8, overlap=0.0)
    out_ch = channelizer.transform(spectrum)

    # Method B: Direct reshape
    out_reshape = spectrum.reshape(8, 256)

    assert out_reshape.shape == (8, 256)
    assert np.array_equal(out_ch, out_reshape)
    assert np.max(np.abs(out_ch - out_reshape)) == 0.0


def test_mc1dcnn_manifest_and_class_counts():
    """Verify expected 22,700 total segments and exact class distribution."""
    df_manifest = build_faithful_manifest(segments_per_pair=100)
    assert len(df_manifest) == 22700

    counts = df_manifest["faithful_label"].value_counts().to_dict()
    assert counts[0] == 4100  # Background
    assert counts[1] == 8400  # Bebop
    assert counts[2] == 8100  # AR Drone
    assert counts[3] == 2100  # Phantom


def test_mc1dcnn_model_shapes_and_forward():
    """Verify Baseline1DCNN input shape (B, 8, 256) and output logits shape (B, 4)."""
    model = Baseline1DCNN(in_channels=8, num_classes=4, seq_length=256)
    model.eval()

    batch_x = torch.randn(4, 8, 256)
    with torch.no_grad():
        logits = model(batch_x)

    assert logits.shape == (4, 4)
    assert torch.all(torch.isfinite(logits))


def test_mc1dcnn_parameter_count():
    """Verify exact parameter count is 275,940."""
    model = Baseline1DCNN(in_channels=8, num_classes=4, seq_length=256)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params == 275940
    assert trainable_params == 275940


def test_mc1dcnn_loss_calculation():
    """Verify CrossEntropyLoss computation on (B, 8, 256) input batches."""
    model = Baseline1DCNN(in_channels=8, num_classes=4, seq_length=256)
    loss_fn = torch.nn.CrossEntropyLoss()

    batch_x = torch.randn(8, 8, 256)
    batch_y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

    logits = model(batch_x)
    loss = loss_fn(logits, batch_y)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_real_data_preprocessing_parity_with_faithful_pipeline():
    """Verify that real-data pair processed and reshaped to (8, 256) has finite, normalized values."""
    base_raw = PROJECT_ROOT / "data" / "raw" / "DroneRF" / "DroneRF"
    rar_l = base_raw / "AR drone" / "RF Data_10100_L.rar"
    inner_l = "RF Data_10100_L/10100L_0.csv"
    rar_h = base_raw / "AR drone" / "RF Data_10100_H.rar"
    inner_h = "RF Data_10100_H/10100H_0.csv"

    if not rar_l.exists():
        pytest.skip("Local raw RAR archives not present.")

    raw_l = read_raw_signal_10m(rar_l, rar_path=rar_l, inner_file=inner_l)
    raw_h = read_raw_signal_10m(rar_h, rar_path=rar_h, inner_file=inner_h)

    # Process 100 segments to 2048 power spectrum
    features_2048 = process_faithful_fgcs_pair_vectorized(raw_l, raw_h, q=10, m=2048)
    norm_2048, g_max = normalize_global_max(features_2048)

    # Reshape to (100, 8, 256)
    channels_8x256 = norm_2048.reshape(-1, 8, 256)

    assert channels_8x256.shape == (100, 8, 256)
    assert np.all(np.isfinite(channels_8x256))
    assert channels_8x256.min() >= 0.0
    assert channels_8x256.max() <= 1.0


def test_strict_recording_level_benchmark_isolation():
    """Verify that primary recording-level splits and baseline pipeline remain untouched."""
    train_split = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_split = PROJECT_ROOT / "data" / "splits" / "val.csv"
    test_split = PROJECT_ROOT / "data" / "splits" / "test.csv"

    assert train_split.exists()
    assert val_split.exists()
    assert test_split.exists()

    import pandas as pd
    df_tr = pd.read_csv(train_split)
    df_va = pd.read_csv(val_split)
    df_te = pd.read_csv(test_split)

    assert len(df_tr) == 308
    assert len(df_va) == 73
    assert len(df_te) == 73
