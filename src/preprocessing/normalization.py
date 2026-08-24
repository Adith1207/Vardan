"""
normalization.py
----------------

Normalization utilities for DroneRF experiments.

Supported methods
-----------------
- none
- minmax
- max
- zscore
"""

import numpy as np


class SignalNormalizer:
    """
    Explicit normalization utility.
    """

    def __init__(
        self,
        method: str = "max",
        epsilon: float = 1e-12,
    ):

        valid_methods = {
            "none",
            "minmax",
            "max",
            "zscore",
        }

        if method not in valid_methods:
            raise ValueError(
                f"Unknown normalization method: {method}. "
                f"Expected one of {valid_methods}."
            )

        self.method = method
        self.epsilon = epsilon

    # ==================================================================
    # Public API
    # ==================================================================

    def transform(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        signal = np.asarray(
            signal,
            dtype=np.float32
        )

        if self.method == "none":

            return signal.copy()

        if self.method == "max":

            return self._max_normalize(signal)

        if self.method == "minmax":

            return self._minmax_normalize(signal)

        if self.method == "zscore":

            return self._zscore_normalize(signal)

        raise RuntimeError(
            "Unsupported normalization method."
        )

    # ==================================================================
    # Methods
    # ==================================================================

    def _max_normalize(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        maximum = np.max(
            np.abs(signal)
        )

        if maximum < self.epsilon:

            return np.zeros_like(
                signal,
                dtype=np.float32
            )

        return (
            signal / maximum
        ).astype(np.float32)

    def _minmax_normalize(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        minimum = np.min(signal)
        maximum = np.max(signal)

        denominator = (
            maximum - minimum
        )

        if denominator < self.epsilon:

            return np.zeros_like(
                signal,
                dtype=np.float32
            )

        return (
            (signal - minimum)
            / denominator
        ).astype(np.float32)

    def _zscore_normalize(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        mean = np.mean(signal)
        std = np.std(signal)

        if std < self.epsilon:

            return np.zeros_like(
                signal,
                dtype=np.float32
            )

        return (
            (signal - mean) / std
        ).astype(np.float32)


def normalize_signal(signal: np.ndarray, method: str = "max", epsilon: float = 1e-12) -> np.ndarray:
    """Normalize signal using SignalNormalizer."""
    return SignalNormalizer(method=method, epsilon=epsilon).transform(signal)


def zscore_normalize(signal: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Z-score normalize signal."""
    return SignalNormalizer(method="zscore", epsilon=epsilon).transform(signal)


def minmax_normalize(signal: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Min-max normalize signal."""
    return SignalNormalizer(method="minmax", epsilon=epsilon).transform(signal)


def robust_normalize(signal: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Robust normalize signal."""
    return SignalNormalizer(method="zscore", epsilon=epsilon).transform(signal)