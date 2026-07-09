"""Verification test suite for the Vardan repository structure and module resolution."""

import sys
from pathlib import Path

# Ensure src/ is in python path

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))


def test_paths():
    """Verify that the paths utility resolves files relative to the project root."""
    from src.utils.paths import (
        PROJECT_ROOT,
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        CONFIGS_DIR,
    )
    
    assert PROJECT_ROOT.exists(), f"Project root does not exist at {PROJECT_ROOT}"
    assert PROJECT_ROOT.is_dir()
    
    # Check key directories exist
    assert DATA_DIR.exists(), f"Data dir does not exist at {DATA_DIR}"
    assert RAW_DATA_DIR.exists()
    assert INTERIM_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert CONFIGS_DIR.exists()


def test_configs_load():
    """Verify that YAML configuration files load correctly."""
    from src.utils.helpers import load_yaml
    from src.utils.paths import CONFIGS_DIR
    
    dataset_cfg = load_yaml(CONFIGS_DIR / "dataset.yaml")
    assert "dataset" in dataset_cfg
    assert dataset_cfg["dataset"]["name"] == "DroneRF"
    assert "classes" in dataset_cfg
    assert len(dataset_cfg["classes"]) == 5
    
    preprocessing_cfg = load_yaml(CONFIGS_DIR / "preprocessing.yaml")
    assert "stft" in preprocessing_cfg
    assert "wavelet" in preprocessing_cfg
    
    training_cfg = load_yaml(CONFIGS_DIR / "training.yaml")
    assert "optimizer" in training_cfg
    assert "scheduler" in training_cfg


def test_constants():
    """Verify global constants and maps."""
    from src.constants import SUPPORTED_MODALITIES, LABEL_MAP
    
    assert "rf" in SUPPORTED_MODALITIES
    assert "acoustic" in SUPPORTED_MODALITIES
    assert "vision" in SUPPORTED_MODALITIES
    
    assert LABEL_MAP[0] == "no_drone"
    assert len(LABEL_MAP) == 5


def test_models_instantiation():
    """Test model architectures compile and can perform a forward pass."""
    import torch
    from src.models.baselines import Baseline1DCNN, DSCNN, MobileNetV3Small
    from src.models.vardhan import VardhanRFNet
    
    # Check 1D waveform baselines
    x_1d = torch.randn(2, 2, 2048)  # batch size 2, 2 channels, 2048 sequence length
    
    model_1d = Baseline1DCNN(in_channels=2, num_classes=5, seq_length=2048)
    out_1d = model_1d(x_1d)
    assert out_1d.shape == (2, 5)
    
    model_dscnn = DSCNN(in_channels=2, num_classes=5, seq_length=2048)
    out_dscnn = model_dscnn(x_1d)
    assert out_dscnn.shape == (2, 5)
    
    # Check custom Vardhan model
    model_vardhan = VardhanRFNet(in_channels=2, num_classes=5, seq_length=2048)
    out_vardhan = model_vardhan(x_1d)
    assert out_vardhan.shape == (2, 5)
    
    # Check 2D spectrogram baseline
    x_2d = torch.randn(2, 1, 513, 100)  # batch size 2, 1 channel, 513 frequency bins, 100 time frames
    model_2d = MobileNetV3Small(num_classes=5)
    out_2d = model_2d(x_2d)
    assert out_2d.shape == (2, 5)
