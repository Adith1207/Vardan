"""
spectrogram.py
--------------

STFT / spectrogram representation for RF signals.
"""

import numpy as np
from scipy import signal


class SpectrogramProcessor:
    """
    Compute RF spectrogram representations.
    """

    def __init__(
        self,
        fs: float,
        nperseg: int = 128,
        noverlap: int = 96,
    ):

        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap

    def transform(
        self,
        rf_signal: np.ndarray,
    ):

        rf_signal = np.asarray(
            rf_signal,
            dtype=np.float32
        )

        frequencies, times, power = signal.spectrogram(
            rf_signal,
            fs=self.fs,
            nperseg=min(
                self.nperseg,
                len(rf_signal)
            ),
            noverlap=min(
                self.noverlap,
                max(
                    0,
                    len(rf_signal) - 1
                ),
            ),
            scaling="density",
            mode="psd",
        )

        power_db = 10.0 * np.log10(
            power + 1e-12
        )

        return (
            frequencies.astype(np.float32),
            times.astype(np.float32),
            power_db.astype(np.float32),
        )


SpectrogramGenerator = SpectrogramProcessor


def compute_spectrogram(signal: np.ndarray, fs: float = 100e6, nperseg: int = 128, noverlap: int = 96, **kwargs) -> np.ndarray:
    """Compute spectrogram power_db."""
    proc = SpectrogramProcessor(fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, p_db = proc.transform(signal)
    return p_db


def compute_stft(signal: np.ndarray, fs: float = 100e6, nperseg: int = 128, noverlap: int = 96, **kwargs) -> np.ndarray:
    """Compute STFT spectrogram matrix."""
    return compute_spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)