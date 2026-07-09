"""Immutable global constants for the Vardan project.

Ensures key-value lookups, sensor modality identifiers, and labels remain 
consistent across data ingestion, modeling, and evaluation modules.
"""

# Modalities supported by the system
MODALITY_RF = "rf"
MODALITY_ACOUSTIC = "acoustic"
MODALITY_VISION = "vision"

SUPPORTED_MODALITIES = [MODALITY_RF, MODALITY_ACOUSTIC, MODALITY_VISION]

# Label definitions mapping class IDs to human-readable strings
LABEL_MAP = {
    0: "no_drone",
    1: "phantom_4_active",
    2: "phantom_4_video",
    3: "bebop_2",
    4: "ar_drone"
}

# Inverse label map
CLASS_TO_INDEX = {v: k for k, v in LABEL_MAP.items()}
