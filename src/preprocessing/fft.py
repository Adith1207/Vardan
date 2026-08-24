"""Fast Fourier Transform (FFT) and frequency domain feature processing algorithms."""

from typing import Tuple, Optional
import numpy as np


def compute_fft(
    signal: np.ndarray,
    n_fft: Optional[int] = None,
    sampling_rate: float = 1.0,
    norm: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute standard 1D FFT and return positive-frequency spectrum.

    Args:
        signal: Input 1D signal array.
        n_fft: FFT length. If None, defaults to signal length.
        sampling_rate: Sampling frequency (Hz) for frequency axis resolution.
        norm: Normalization mode for numpy.fft ('ortho' or None).

    Returns:
        Tuple of (frequencies_hz, magnitude_spectrum).
    """
    if n_fft is None:
        n_fft = len(signal)

    fft_vals = np.fft.fft(signal, n=n_fft, norm=norm)
    freqs = np.fft.fftfreq(n_fft, d=1.0 / sampling_rate)

    # Keep non-negative frequencies
    positive_mask = freqs >= 0
    freqs_pos = freqs[positive_mask]
    magnitudes = np.abs(fft_vals[positive_mask])

    return freqs_pos, magnitudes


def compute_rfft(
    signal: np.ndarray,
    n_fft: Optional[int] = None,
    sampling_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Real FFT (rfft) optimized for real-valued signals.

    Args:
        signal: Input 1D real-valued signal array.
        n_fft: FFT length. Defaults to length of signal.
        sampling_rate: Sampling frequency in Hz.

    Returns:
        Tuple of (frequencies_hz, rfft_magnitudes).
    """
    if n_fft is None:
        n_fft = len(signal)

    rfft_vals = np.fft.rfft(signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
    magnitudes = np.abs(rfft_vals)

    return freqs, magnitudes


def compute_power_spectral_density(
    signal: np.ndarray,
    sampling_rate: float = 1.0,
    n_fft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density (PSD) using Periodogram method.

    Args:
        signal: Input 1D real signal array.
        sampling_rate: Sampling frequency in Hz.
        n_fft: Number of FFT points.

    Returns:
        Tuple of (frequencies_hz, psd_values).
    """
    if n_fft is None:
        n_fft = len(signal)

    freqs, magnitudes = compute_rfft(signal, n_fft=n_fft, sampling_rate=sampling_rate)
    psd = (magnitudes ** 2) / (sampling_rate * n_fft)
    return freqs, psd


class FFTProcessor:
    """Configurable FFT processor for batched signal processing."""

    def __init__(
        self,
        n_fft: int = 1024,
        fft_size: Optional[int] = None,
        remove_dc: bool = False,
        sampling_rate: float = 1.0,
        real_only: bool = True,
        return_db: bool = False,
    ):
        self.n_fft = fft_size if fft_size is not None else n_fft
        self.fft_size = self.n_fft
        self.remove_dc = remove_dc
        self.sampling_rate = sampling_rate
        self.real_only = real_only
        self.return_db = return_db

    def process(
        self,
        signal: np.ndarray,
        representation: str = "magnitude",
        keep_full_spectrum: bool = False,
    ) -> np.ndarray:
        """Process 1D or 2D (batch, time) array into magnitude or power spectrum.

        Args:
            signal: Input signal of shape (N,) or (B, N).
            representation: 'magnitude', 'power', or 'db'.
            keep_full_spectrum: If True, keep negative frequencies / full spectrum length.

        Returns:
            Spectrum array.
        """
        if self.remove_dc:
            mean = np.mean(signal, axis=-1, keepdims=True)
            signal = signal - mean

        if signal.ndim == 1:
            if keep_full_spectrum or not self.real_only:
                fft_vals = np.fft.fft(signal, n=self.n_fft)
                if representation == "power":
                    mag = np.abs(fft_vals) ** 2
                else:
                    mag = np.abs(fft_vals)
            else:
                if representation == "power":
                    _, rfft_mag = compute_rfft(signal, n_fft=self.n_fft, sampling_rate=self.sampling_rate)
                    mag = rfft_mag ** 2
                else:
                    _, mag = compute_rfft(signal, n_fft=self.n_fft, sampling_rate=self.sampling_rate)
        elif signal.ndim == 2:
            if keep_full_spectrum or not self.real_only:
                fft_vals = np.fft.fft(signal, n=self.n_fft, axis=-1)
                if representation == "power":
                    mag = np.abs(fft_vals) ** 2
                else:
                    mag = np.abs(fft_vals)
            else:
                rfft_vals = np.fft.rfft(signal, n=self.n_fft, axis=-1)
                if representation == "power":
                    mag = np.abs(rfft_vals) ** 2
                else:
                    mag = np.abs(rfft_vals)
        else:
            raise ValueError(f"Expected 1D or 2D signal, got ndim={signal.ndim}")

        if self.return_db or representation == "db":
            mag = 20.0 * np.log10(np.maximum(mag, 1e-10))

        return mag


_processor = FFTProcessor()


def fft(signal: np.ndarray) -> np.ndarray:
    """Convenience wrapper for FFTProcessor."""
    return _processor.process(signal)
