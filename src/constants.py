"""Immutable global constants for the Vardan project.

Ensures key-value lookups, sensor modality identifiers, and labels remain 
consistent across data ingestion, modeling, and evaluation modules.
"""

# Modalities supported by the system
MODALITY_RF = "rf"
MODALITY_ACOUSTIC = "acoustic"
MODALITY_VISION = "vision"

SUPPORTED_MODALITIES = [MODALITY_RF, MODALITY_ACOUSTIC, MODALITY_VISION]

# Label definitions mapping class IDs to human-readable strings (Canonical 4-class)
LABEL_MAP = {
    0: "no_drone",
    1: "ar_drone",
    2: "bebop_drone",
    3: "phantom_drone",
}

# Inverse label map
CLASS_TO_INDEX = {v: k for k, v in LABEL_MAP.items()}

# Canonical raw class mapping from DroneRF metadata (handles spelling in raw metadata)
RAW_CLASS_TO_INDEX = {
    "Backround RF activities": 0,
    "AR Drone": 1,
    "Bepop drone": 2,
    "Phantom drone": 3,
}
# Canonical 4-class count and class names list
NUM_CLASSES = len(LABEL_MAP)
CLASS_NAMES = [LABEL_MAP[i] for i in range(NUM_CLASSES)]
