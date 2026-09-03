"""
test_fgcs_faithful.py
---------------------

Pytest test suite for faithful FGCS DroneRF preprocessing, pairing, and model definitions.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.preprocessing.fgcs_faithful import (
    FAITHFUL_FGCS_CLASS_TO_INDEX,
    FAITHFUL_FGCS_INDEX_TO_CLASS,
    BUI_TO_CLASS,
    parse_dronerf_filename,
    process_faithful_fgcs_segment,
    normalize_global_max,
)
from src.data.fgcs_faithful_loader import (
    discover_and_pair_dronerf_files,
    build_faithful_manifest,
    FGCSFaithfulLazyDataset,
)
from src.models.fgcs_faithful_dnn import FGCSFaithfulDNN


def test_faithful_label_mapping():
    """Verify original 4-class label ordering: 0: Background, 1: Bebop, 2: AR, 3: Phantom."""
    assert FAITHFUL_FGCS_CLASS_TO_INDEX["Background RF activities"] == 0
    assert FAITHFUL_FGCS_CLASS_TO_INDEX["Bebop drone"] == 1
    assert FAITHFUL_FGCS_CLASS_TO_INDEX["AR drone"] == 2
    assert FAITHFUL_FGCS_CLASS_TO_INDEX["Phantom drone"] == 3

    assert FAITHFUL_FGCS_INDEX_TO_CLASS[0] == "Background RF activities"
    assert FAITHFUL_FGCS_INDEX_TO_CLASS[1] == "Bebop drone"
    assert FAITHFUL_FGCS_INDEX_TO_CLASS[2] == "AR drone"
    assert FAITHFUL_FGCS_INDEX_TO_CLASS[3] == "Phantom drone"


def test_parse_dronerf_filename():
    """Verify filename parsing for representative DroneRF files."""
    p_bg = parse_dronerf_filename("00000L_12.csv")
    assert p_bg is not None
    assert p_bg["bui"] == "00000"
    assert p_bg["receiver"] == "L"
    assert p_bg["file_segment_num"] == 12
    assert p_bg["faithful_label"] == 0

    p_bebop = parse_dronerf_filename("10001H_3.csv")
    assert p_bebop is not None
    assert p_bebop["bui"] == "10001"
    assert p_bebop["receiver"] == "H"
    assert p_bebop["faithful_label"] == 1

    p_ar = parse_dronerf_filename("10110L_0.csv")
    assert p_ar is not None
    assert p_ar["bui"] == "10110"
    assert p_ar["receiver"] == "L"
    assert p_ar["faithful_label"] == 2

    p_phantom = parse_dronerf_filename("11000H_20.csv")
    assert p_phantom is not None
    assert p_phantom["bui"] == "11000"
    assert p_phantom["receiver"] == "H"
    assert p_phantom["faithful_label"] == 3

    assert parse_dronerf_filename("invalid_file.csv") is None


def test_process_faithful_fgcs_segment_math():
    """Verify mathematical transformations of faithful FGCS signal processing."""
    rng = np.random.RandomState(123)
    x = rng.randn(100000)
    y = rng.randn(100000) * 2.0

    power_vec = process_faithful_fgcs_segment(x, y, q=10, m=2048)

    assert power_vec.shape == (2048,)
    assert power_vec.dtype == np.float32
    assert np.all(np.isfinite(power_vec))
    assert np.all(power_vec >= 0.0)

    # Step-by-step verification
    x_detrend = x - np.mean(x)
    y_detrend = y - np.mean(y)
    fft_x = np.fft.fft(x_detrend[:2048], n=2048)
    fft_y = np.fft.fft(y_detrend[:2048], n=2048)
    xf = np.abs(np.fft.fftshift(fft_x))[1024:]
    yf = np.abs(np.fft.fftshift(fft_y))[1024:]
    c = np.mean(xf[-10:]) / (np.mean(yf[:10]) + 1e-12)
    expected = (np.concatenate([xf, c * yf]) ** 2).astype(np.float32)

    assert np.allclose(power_vec, expected, atol=1e-5)


def test_normalize_global_max():
    """Verify global maximum normalization function."""
    data = np.array([[10.0, 20.0], [5.0, 40.0]], dtype=np.float32)
    norm, g_max = normalize_global_max(data)
    assert g_max == 40.0
    assert norm.max() == 1.0
    assert norm[0, 0] == 0.25


def test_pairing_discovery():
    """Verify that discover_and_pair_dronerf_files finds 227 pairs."""
    df_pairs = discover_and_pair_dronerf_files()
    assert len(df_pairs) == 227
    assert (df_pairs["drone_class"] == "Background RF activities").sum() == 41
    assert (df_pairs["drone_class"] == "Bebop drone").sum() == 84
    assert (df_pairs["drone_class"] == "AR drone").sum() == 81
    assert (df_pairs["drone_class"] == "Phantom drone").sum() == 21


def test_manifest_building():
    """Verify 22,700-segment capacity manifest construction."""
    df_manifest = build_faithful_manifest(segments_per_pair=100)
    assert len(df_manifest) == 22700
    assert "faithful_label" in df_manifest.columns
    assert "segment_offset" in df_manifest.columns


def test_lazy_dataset_mock():
    """Verify FGCSFaithfulLazyDataset item retrieval in mock mode."""
    df_manifest = build_faithful_manifest(segments_per_pair=2)
    dataset = FGCSFaithfulLazyDataset(df_manifest, mock=True, global_max=100.0)
    assert len(dataset) == 227 * 2

    x_tensor, y_label = dataset[0]
    assert isinstance(x_tensor, torch.Tensor)
    assert x_tensor.shape == (2048,)
    assert 0 <= y_label <= 3


def test_faithful_dnn_forward():
    """Verify FGCSFaithfulDNN forward pass and output activations."""
    dummy = torch.randn(8, 2048)

    model_code = FGCSFaithfulDNN(in_features=2048, num_classes=4, architecture_mode="code")
    logits_code = model_code(dummy, return_logits=True)
    probs_code = model_code(dummy, return_logits=False)
    assert logits_code.shape == (8, 4)
    assert probs_code.shape == (8, 4)
    assert torch.all((probs_code >= 0.0) & (probs_code <= 1.0))

    model_paper = FGCSFaithfulDNN(in_features=2048, num_classes=4, architecture_mode="paper")
    logits_paper = model_paper(dummy, return_logits=True)
    assert logits_paper.shape == (8, 4)
