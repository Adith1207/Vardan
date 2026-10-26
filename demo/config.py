"""Configuration parameters and constants for the VARDAN research demonstration."""

from pathlib import Path
from typing import Dict

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "vardhan_v3_per_segment_15ep"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "checkpoint_fold_1.pt"
FINAL_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "final_model_fold_1.pt"
SMOKE_TEST_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "smoke_tests" / "vardhan_v3_standardized" / "checkpoint_fold_1.pt"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Verified 4-class mapping from the faithful FGCS training pipeline
CLASS_MAPPING: Dict[int, str] = {
    0: "Background RF activities",
    1: "Bebop drone",
    2: "AR drone",
    3: "Phantom drone",
}

CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "Interference, environmental radio activity, and noise without drone transmission",
    1: "Parrot Bebop quadcopter command/telemetry link in the 2.4 GHz ISM band",
    2: "Parrot AR.Drone 2.0 Wi-Fi communication and video transmission link",
    3: "DJI Phantom 3 Lightbridge/Wi-Fi remote control and digital video link",
}

CLASS_COLORS: Dict[int, str] = {
    0: "#6c757d",  # Neutral gray for background
    1: "#0d6efd",  # Blue for Bebop
    2: "#198754",  # Green for AR Drone
    3: "#dc3545",  # Red for Phantom
}

# DroneRF RF Signal Specifications
SAMPLING_RATE_HZ = 40_000_000.0  # 40 MS/s
SEGMENT_LENGTH = 2048            # samples per classification slice
SEGMENT_DURATION_US = (SEGMENT_LENGTH / SAMPLING_RATE_HZ) * 1e6  # 51.2 microseconds

# Architecture specifications
MODEL_NAME = "VARDHAN-v3 (Multi-Scale Dual-Domain RF-Net)"
EXPECTED_PARAMETERS = 119_806
INPUT_SHAPE = (1, 1, SEGMENT_LENGTH)
INPUT_DTYPE = "torch.float32"
NORMALIZATION_MODE = "per_segment"
