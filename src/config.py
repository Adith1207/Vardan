"""Configuration settings for Notebook 03: FGCS 2019 DNN Baseline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "DroneRF" / "unzipped_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_dnn"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "baseline_dnn"
FIGURES_DIR = PROJECT_ROOT / "figures" / "notebook03"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
EPOCHS = 200
BATCH_SIZE = 10
LEARNING_RATE = 1e-3  # Adam default
NUM_FOLDS = 10

# Model Settings
INPUT_FEATURES = 2048
NUM_CLASSES = 4 # AR Drone, Bebop Drone, Phantom Drone, Background
