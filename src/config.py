"""Configuration settings for Vardan Counter-UAS framework and preprocessing pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "DroneRF"
DATA_DIR = RAW_DATA_DIR / "unzipped_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_dnn"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "baseline_dnn"
FIGURES_DIR = PROJECT_ROOT / "figures" / "notebook03"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Signal Processing & Preprocessing Defaults
SAMPLING_RATE = 100_000_000  # 100 MHz for DroneRF dataset
SEGMENT_LENGTH = 2048
RAW_SIGNAL_LENGTH = 2048
OVERLAP = 0
REMOVE_DC_COMPONENT = True

# FFT Parameters
FFT_SIZE = 1024
HOP_LENGTH = 256
WINDOW_TYPE = "hann"

# Channelization Settings
NUM_CHANNELS = 4
CHANNEL_FILTER_TYPE = "uniform"

# Normalization & Compression
NORMALIZATION_METHOD = "standard"  # options: standard, minmax, robust, none
NORMALIZATION_EPSILON = 1e-8
COMPRESSION_METHOD = "log"  # options: log, mu_law, power_law, none
COMPRESSION_FACTOR = 1.0
MU_LAW_MU = 255

# Training Hyperparameters
EPOCHS = 200
BATCH_SIZE = 10
LEARNING_RATE = 1e-3  # Adam default
NUM_FOLDS = 10

# Model Settings
INPUT_FEATURES = 2048
from constants import NUM_CLASSES
