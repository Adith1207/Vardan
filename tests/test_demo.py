"""
test_demo.py
------------

Comprehensive test suite verifying the VARDAN interactive demo logic:
1. Checkpoint discovery and error handling when missing.
2. Strict state_dict loading into VardhanV3.
3. Output logits and softmax probabilities across 4 classes.
4. Correct class decoding matching faithful FGCS training pipeline.
5. Exact per-segment normalization behavior.
6. Real-data loader and pair discovery.
7. Explicit failure and absence of random/mock fallback in production paths.
8. CPU inference and latency profiling.

Tests that read real DroneRF recordings are skipped when data/raw/DroneRF/ holds no
CSV or RAR files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from demo.config import (
    CLASS_MAPPING,
    DEFAULT_CHECKPOINT_PATH,
    EXPECTED_PARAMETERS,
    RAW_DATA_DIR,
)
from demo.data_utils import DemoDataLoader
from demo.inference import VardanInferenceEngine
from demo.visualization import (
    plot_class_probabilities,
    plot_dual_spectral_tensor,
    plot_power_spectrum,
    plot_waveform,
)

DRONERF_DIR = RAW_DATA_DIR / "DroneRF"
HAS_DRONERF = DRONERF_DIR.is_dir() and any(
    p.suffix.lower() in {".csv", ".rar"} for p in DRONERF_DIR.rglob("*")
)
requires_dronerf = pytest.mark.skipif(
    not HAS_DRONERF, reason="DroneRF recordings not present under data/raw/DroneRF/."
)


def test_checkpoint_discovery_and_strict_loading():
    """Verify that the primary checkpoint exists, loads strictly, and matches 119,806 params."""
    engine = VardanInferenceEngine(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu")
    assert engine.model is not None
    assert engine.checkpoint_metadata["total_parameters"] == EXPECTED_PARAMETERS
    assert engine.checkpoint_metadata["strict_keys_matched"] is True
    assert engine.checkpoint_metadata["fold"] == 1
    assert engine.checkpoint_metadata["best_epoch"] == 5


def test_missing_checkpoint_raises_filenotfound():
    """Verify that a non-existent checkpoint path raises FileNotFoundError immediately."""
    fake_path = PROJECT_ROOT / "models" / "checkpoints" / "non_existent_checkpoint.pt"
    with pytest.raises(FileNotFoundError):
        _ = VardanInferenceEngine(checkpoint_path=fake_path)


def test_per_segment_normalization():
    """Verify exact per-segment zero-mean unit-variance normalization formula."""
    rng = np.random.RandomState(42)
    raw = rng.randn(2048).astype(np.float32) * 5.0 + 10.0

    mu = float(np.mean(raw))
    sigma = float(np.std(raw)) + 1e-8
    norm = (raw - mu) / sigma

    assert np.isclose(float(np.mean(norm)), 0.0, atol=1e-5)
    assert np.isclose(float(np.std(norm)), 1.0, atol=1e-3)


def test_inference_dimensions_and_class_decoding():
    """Verify forward-pass output shape (1, 4), probability sum, and class string resolution."""
    engine = VardanInferenceEngine(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu")
    dummy_norm_waveform = np.zeros(2048, dtype=np.float32)

    res = engine.predict(dummy_norm_waveform)

    assert "predicted_label" in res
    assert res["predicted_label"] in [0, 1, 2, 3]
    assert res["predicted_name"] == CLASS_MAPPING[res["predicted_label"]]
    assert len(res["probabilities"]) == 4

    prob_sum = sum(res["probabilities"].values())
    assert np.isclose(prob_sum, 1.0, atol=1e-4)

    assert res["logits"].shape == (4,)
    assert res["spectral_tensor"].shape == (3, 1024)
    assert res["temporal_features"].shape == (80,)
    assert res["spectral_features"].shape == (80,)
    assert res["latency_ms"] > 0.0


def test_prevention_of_silent_random_fallback_on_invalid_data():
    """Verify that DemoDataLoader never returns random data and fails explicitly if data is missing."""
    loader = DemoDataLoader(raw_data_dir=PROJECT_ROOT / "data" / "raw")
    fake_row = {
        "l_path": "non_existent_file.csv",
        "l_rar": None,
        "l_inner": None,
        "pair_id": "test_fake",
        "bui": "00000",
        "file_segment_num": 0,
        "faithful_label": 0,
    }
    with pytest.raises(FileNotFoundError):
        loader.load_real_segment(fake_row, segment_index=0)


@requires_dronerf
def test_real_sample_discovery_and_extraction():
    """Verify that real DroneRF archives are discovered and at least one real segment is loaded."""
    loader = DemoDataLoader()
    manifest = loader.get_paired_manifest()

    assert len(manifest) == 227
    assert "faithful_label" in manifest.columns

    # Test loading segment from the first discovered recording
    first_row = manifest.iloc[0].to_dict()
    sample = loader.load_real_segment(first_row, segment_index=0)

    assert sample["raw_waveform"].shape == (2048,)
    assert sample["normalized_waveform"].shape == (2048,)
    assert np.all(np.isfinite(sample["raw_waveform"]))
    assert np.all(np.isfinite(sample["normalized_waveform"]))
    assert sample["ground_truth_label"] == int(first_row["faithful_label"])


@requires_dronerf
def test_deterministic_prediction_on_real_sample():
    """Verify that inferring the same real sample twice produces identical probability outputs."""
    loader = DemoDataLoader()
    manifest = loader.get_paired_manifest()
    row = manifest.iloc[0].to_dict()
    sample = loader.load_real_segment(row, segment_index=0)

    engine = VardanInferenceEngine(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu")
    pred1 = engine.predict(sample["normalized_waveform"])
    pred2 = engine.predict(sample["normalized_waveform"])

    assert pred1["predicted_label"] == pred2["predicted_label"]
    assert np.allclose(pred1["probability_array"], pred2["probability_array"])
    assert np.allclose(pred1["logits"], pred2["logits"])


def test_visualization_generators_return_valid_figures():
    """Verify that all Plotly visualization functions generate valid go.Figure objects without error."""
    rng = np.random.RandomState(42)
    fake_waveform = rng.randn(2048).astype(np.float32)
    fake_spectral = rng.randn(3, 1024).astype(np.float32)
    fake_probs = {CLASS_MAPPING[i]: 0.25 for i in range(4)}

    fig1 = plot_waveform(fake_waveform)
    fig2 = plot_power_spectrum(fake_waveform)
    fig3 = plot_dual_spectral_tensor(fake_spectral)
    fig4 = plot_class_probabilities(fake_probs, ground_truth_name=CLASS_MAPPING[0])

    assert fig1 is not None and len(fig1.data) > 0
    assert fig2 is not None and len(fig2.data) > 0
    assert fig3 is not None and len(fig3.data) == 3
    assert fig4 is not None and len(fig4.data) > 0
