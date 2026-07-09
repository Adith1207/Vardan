"""Feature extraction pipeline for raw signals."""

import numpy as np


def extract_rf_features(signal: np.ndarray) -> np.ndarray:
    """Extract hand-crafted features from raw I/Q RF signals.

    Calculates statistical characteristics (mean, std, skewness, kurtosis) and
    energy metrics in time and frequency domains.

    Args:
        signal: Input 2D I/Q RF signal of shape (sequence_length, 2).

    Returns:
        1D feature vector.
    """
    # Placeholder: Extract basic properties for demonstration
    mean_i = np.mean(signal[:, 0])
    mean_q = np.mean(signal[:, 1])
    std_i = np.std(signal[:, 0])
    std_q = np.std(signal[:, 1])
    energy = np.sum(signal ** 2) / len(signal)
    
    return np.array([mean_i, mean_q, std_i, std_q, energy], dtype=np.float32)


def extract_acoustic_features(audio_waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract standard acoustic features (MFCCs, spectral centroid, etc.).

    Args:
        audio_waveform: 1D audio time-series array.
        sr: Sampling rate.

    Returns:
        1D feature vector.
    """
    # Placeholder: Return dummy features vector
    spectral_centroid = np.mean(audio_waveform)
    rms_energy = np.sqrt(np.mean(audio_waveform ** 2))
    
    return np.array([spectral_centroid, rms_energy], dtype=np.float32)
