"""Dynamic range compression and non-linear logarithmic transformation functions."""

from typing import Optional
import numpy as np


def log_compression(
    spectrum: np.ndarray,
    epsilon: float = 1e-6,
    scale: float = 1.0,
    use_log10: bool = True,
) -> np.ndarray:
    """Logarithmic dynamic range compression for spectrum/spectrogram arrays.

    Args:
        spectrum: Magnitude or power spectrum NumPy array.
        epsilon: Small additive float constant to avoid log(0).
        scale: Scaling factor inside or outside log transform.
        use_log10: If True, uses base-10 log; otherwise uses natural log.

    Returns:
        Log-compressed array.
    """
    scaled = np.maximum(spectrum, 0.0) * scale + epsilon
    if use_log10:
        return np.log10(scaled)
    return np.log(scaled)


def mu_law_compression(
    signal: np.ndarray,
    mu: int = 255,
) -> np.ndarray:
    r"""$\mu$-law dynamic range compression (ITU-T G.711 standard).

    Compresses high peak-to-average-power ratio (PAPR) RF waveforms.

    Args:
        signal: Normalized input signal array with values in [-1, 1].
        mu: Compression parameter ($\mu$). Default 255.

    Returns:
        $\mu$-law compressed signal array with values bounded in [-1, 1].
    """
    clipped = np.clip(signal, -1.0, 1.0)
    magnitude = np.abs(clipped)
    compressed_mag = np.log(1.0 + mu * magnitude) / np.log(1.0 + mu)
    return np.sign(clipped) * compressed_mag


def power_law_compression(
    spectrum: np.ndarray,
    gamma: float = 0.3,
) -> np.ndarray:
    """Power-law (gamma) compression $y = x^{\\gamma}$.

    Args:
        spectrum: Positive magnitude spectrum array.
        gamma: Exponent factor (typically between 0.1 and 0.5).

    Returns:
        Power-law compressed spectrum.
    """
    pos = np.maximum(spectrum, 0.0)
    return np.power(pos, gamma)


def compress_dynamic_range(
    data: np.ndarray,
    method: str = "log",
    scale: float = 1.0,
    epsilon: float = 1e-6,
    mu: int = 255,
    gamma: float = 0.3,
) -> np.ndarray:
    """Dispatch function for dynamic range compression.

    Args:
        data: Input signal or spectrum NumPy array.
        method: Method type ('log', 'mu_law', 'power_law', 'none').
        scale: Log scale multiplier.
        epsilon: Log stability constant.
        mu: Mu-law parameter.
        gamma: Power-law exponent.

    Returns:
        Compressed array.
    """
    m = method.lower()
    if m == "log":
        return log_compression(data, epsilon=epsilon, scale=scale)
    elif m == "mu_law":
        return mu_law_compression(data, mu=mu)
    elif m == "power_law":
        return power_law_compression(data, gamma=gamma)
    elif m in ["none", "passthrough"]:
        return data
    else:
        raise ValueError(f"Unknown compression method '{method}'. Options: log, mu_law, power_law, none.")


class DynamicRangeCompressor:
    """Stateful dynamic range compression processor."""

    def __init__(
        self,
        method: str = "log",
        compression_ratio: Optional[float] = None,
        scale: float = 1.0,
        epsilon: float = 1e-6,
        mu: int = 255,
        gamma: float = 0.3,
    ):
        self.method = method
        self.compression_ratio = compression_ratio
        self.scale = scale
        self.epsilon = epsilon
        self.mu = mu
        self.gamma = gamma

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression to input array."""
        return compress_dynamic_range(
            data,
            method=self.method,
            scale=self.scale,
            epsilon=self.epsilon,
            mu=self.mu,
            gamma=self.gamma,
        )


CompressedSensingProcessor = DynamicRangeCompressor
