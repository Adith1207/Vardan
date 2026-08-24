"""
channelization.py
-----------------

Frequency-spectrum channelization for the
multi-channel 1D CNN baseline.
"""

import numpy as np


class SpectrumChannelizer:
    """
    Split a frequency spectrum into multiple channels.
    """

    def __init__(
        self,
        channel_count: int = 8,
        overlap: float = 0.50,
    ):

        if channel_count <= 0:
            raise ValueError(
                "channel_count must be positive."
            )

        if not 0.0 <= overlap < 1.0:
            raise ValueError(
                "overlap must be in [0, 1)."
            )

        self.channel_count = channel_count
        self.overlap = overlap

    def transform(
        self,
        spectrum: np.ndarray,
    ) -> np.ndarray:

        spectrum = np.asarray(
            spectrum,
            dtype=np.float32
        )

        if spectrum.ndim != 1:
            raise ValueError(
                "Spectrum must be one-dimensional."
            )

        length = spectrum.size

        if self.channel_count > length:
            raise ValueError(
                "Number of channels cannot exceed "
                "spectrum length."
            )

        # --------------------------------------------------------------
        # Equal-width non-overlapping division as the deterministic
        # baseline. Overlap can be incorporated later if the exact
        # paper implementation requires it.
        # --------------------------------------------------------------

        boundaries = np.linspace(
            0,
            length,
            self.channel_count + 1,
            dtype=int,
        )

        channels = []

        for i in range(
            self.channel_count
        ):

            start = boundaries[i]
            end = boundaries[i + 1]

            channels.append(
                spectrum[start:end]
            )

        # All channels must have equal dimensions.
        min_length = min(
            len(channel)
            for channel in channels
        )

        channels = [
            channel[:min_length]
            for channel in channels
        ]

        return np.stack(
            channels,
            axis=0
        ).astype(np.float32)


Channelizer = SpectrumChannelizer


def channelize_signal(
    signal: np.ndarray,
    num_channels: int = 4,
    filter_type: str = "uniform",
) -> np.ndarray:
    """Channelize signal into frequency channels."""
    channelizer = SpectrumChannelizer(channel_count=num_channels)
    if signal.ndim == 1:
        return channelizer.transform(signal)
    elif signal.ndim == 2:
        return np.array([channelizer.transform(s) for s in signal])
    return signal


def extract_subbands(
    signal: np.ndarray,
    sampling_rate: float = 100_000_000,
    band_edges: list = None,
) -> np.ndarray:
    """Extract subbands from signal."""
    return channelize_signal(signal)