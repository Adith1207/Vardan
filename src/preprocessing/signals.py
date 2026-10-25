"""Signal processing operations for RF waveforms."""

from typing import Tuple
import numpy as np
from scipy.signal import get_window


def normalize_signal(signal: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalize signal using the specified method.

    Args:
        signal: Input NumPy array (1D or 2D).
        method: Normalization type ('standard', 'minmax', 'robust', 'none').

    Returns:
        Normalized NumPy array.
    """
    if method == "standard":
        mean = np.mean(signal)
        std = np.std(signal) + 1e-8
        return (signal - mean) / std
    elif method == "minmax":
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / (max_val - min_val + 1e-8)
    return signal


def apply_window(signal: np.ndarray, window_type: str = "hann") -> np.ndarray:
    """Apply a windowing function to the signal.

    Args:
        signal: Input 1D signal.
        window_type: Type of window (e.g., 'hann', 'hamming', 'blackman').

    Returns:
        Windowed signal array.
    """
    length = len(signal)
    if window_type == "hann":
        window = np.hanning(length)
    elif window_type == "hamming":
        window = np.hamming(length)
    elif window_type == "blackman":
        window = np.blackman(length)
    else:
        window = np.ones(length)
    return signal * window


def compute_fft(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Fast Fourier Transform (FFT) of a signal.

    Args:
        signal: Input 1D signal.

    Returns:
        A tuple of (frequencies, magnitude_spectrum).
    """
    n = len(signal)
    fft_val = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n)
    magnitudes = np.abs(fft_val)
    return freqs, magnitudes


def compute_stft(
    signal: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    window: str = "hann",
) -> np.ndarray:
    """Compute the Short-Time Fourier Transform (STFT) of a signal.

    Args:
        signal: Input 1D signal.
        n_fft: Size of FFT window.
        hop_length: Number of samples between successive frames.
        window: Window type accepted by ``scipy.signal.get_window``.

    Returns:
        Complex spectrogram of shape (n_fft // 2 + 1, num_frames).
    """
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop_length]
    win = get_window(window, n_fft)
    return np.fft.rfft(frames * win, axis=-1).T
