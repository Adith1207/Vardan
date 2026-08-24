"""
raw_loader.py
-------------

Raw DroneRF signal loading and validation utilities.

Responsibilities
----------------
- Load RF samples from CSV files.
- Convert samples to float32.
- Remove invalid/non-numeric values.
- Validate signal length.
- Return a deterministic 1-D NumPy array.

No FFT, normalization, feature extraction or augmentation
should occur in this module.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from config import RAW_SIGNAL_LENGTH


PathLike = Union[str, Path]


class DroneRFLoader:
    """
    Loader for individual DroneRF CSV segments.
    """

    def __init__(
        self,
        expected_length: int = RAW_SIGNAL_LENGTH,
    ):
        self.expected_length = expected_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: PathLike) -> np.ndarray:
        """
        Load one DroneRF CSV segment.

        Parameters
        ----------
        file_path : str or Path
            Path to DroneRF CSV.

        Returns
        -------
        np.ndarray
            One-dimensional float32 RF signal.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"DroneRF file not found: {file_path}"
            )

        dataframe = pd.read_csv(
            file_path,
            header=None
        )

        values = dataframe.to_numpy().reshape(-1)

        # Convert to numeric safely
        values = pd.to_numeric(
            values,
            errors="coerce"
        )

        # Remove invalid values
        values = values[~np.isnan(values)]

        signal = values.astype(
            np.float32,
            copy=False
        )

        self._validate(signal, file_path)

        return signal

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        signal: np.ndarray,
        file_path: Path,
    ) -> None:

        if signal.size == 0:
            raise ValueError(
                f"Empty RF signal: {file_path}"
            )

        if not np.all(np.isfinite(signal)):
            raise ValueError(
                f"Non-finite values detected: {file_path}"
            )

        if signal.size != self.expected_length:
            raise ValueError(
                f"Unexpected signal length for {file_path}. "
                f"Expected {self.expected_length}, "
                f"got {signal.size}."
            )


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------

_loader = DroneRFLoader()


def load_signal(file_path: PathLike) -> np.ndarray:
    """
    Convenience wrapper.
    """
    return _loader.load(file_path)


load_raw_rf_data = load_signal
RawDataLoader = DroneRFLoader


def segment_signal(signal: np.ndarray, segment_length: int = 2048, overlap: int = 0) -> np.ndarray:
    """Split continuous 1D signal into uniform length segments."""
    if signal.ndim != 1:
        return signal
    step = max(1, segment_length - overlap)
    n_samples = len(signal)
    indices = list(range(0, n_samples - segment_length + 1, step))
    if not indices:
        return np.empty((0, segment_length), dtype=signal.dtype)
    return np.array([signal[i : i + segment_length] for i in indices])