"""
pipeline.py
-----------

High-level DroneRF preprocessing pipeline.

This module combines:
    raw loading
    zero-centering
    FFT / PSD
    normalization
    channelization
    compressed sensing
    spectrogram generation

The pipeline is model-aware but keeps each operation modular.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    FFT_SIZE,
    REMOVE_DC_COMPONENT,
)

from .raw_loader import DroneRFLoader
from .fft import FFTProcessor
from .normalization import SignalNormalizer
from .channelization import SpectrumChannelizer
from .compression import CompressiveSensingMatrix, CompressedSensingProcessor
from .spectrogram import SpectrogramProcessor


class DroneRFPreprocessor:
    """
    Unified preprocessing interface for DroneRF experiments.
    """

    def __init__(
        self,
        fft_size: int = FFT_SIZE,
        remove_dc: bool = REMOVE_DC_COMPONENT,
        normalization: str = "max",
        channel_count: int = 8,
        channel_overlap: float = 0.0,
        compression_ratio: float = 0.50,
        fs: Optional[float] = None,
    ):

        self.loader = DroneRFLoader()

        self.fft_processor = FFTProcessor(
            fft_size=fft_size,
            remove_dc=remove_dc,
        )

        self.normalizer = SignalNormalizer(
            method=normalization
        )

        self.channelizer = SpectrumChannelizer(
            channel_count=channel_count,
            overlap=channel_overlap,
        )

        self.cs_matrix = CompressiveSensingMatrix(
            n_input=2048,
            n_compressed=1024,
            seed=42,
        )

        self.compressor = self.cs_matrix


        self.fs = fs

        if fs is not None:

            self.spectrogram_processor = (
                SpectrogramProcessor(
                    fs=fs
                )
            )

        else:

            self.spectrogram_processor = None

    # ==================================================================
    # Raw
    # ==================================================================

    def load(
        self,
        file_path: str | Path,
    ) -> np.ndarray:

        return self.loader.load(
            file_path
        )

    # ==================================================================
    # FGCS DNN representation
    # ==================================================================

    def process_fgcs(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        spectrum = self.fft_processor.process(
            signal,
            representation="power",
            keep_full_spectrum=True,
        )

        spectrum = self.normalizer.transform(
            spectrum
        )

        return spectrum.astype(
            np.float32
        )

    # ==================================================================
    # Multi-channel 1D CNN representation
    # ==================================================================

    def process_multichannel(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        spectrum = self.fft_processor.process(
            signal,
            representation="power",
            keep_full_spectrum=True,
        )

        spectrum = self.normalizer.transform(
            spectrum
        )

        channels = self.channelizer.transform(
            spectrum
        )

        return channels.astype(
            np.float32
        )

    # ==================================================================
    # Compressed-sensing baseline
    # ==================================================================

    def process_compressed(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:

        signal = np.asarray(
            signal,
            dtype=np.float32
        )

        if REMOVE_DC_COMPONENT:

            signal = (
                signal
                - np.mean(signal)
            )

        compressed = (
            self.cs_matrix.transform(
                signal
            )
        )

        compressed_norm = (compressed - np.mean(compressed)) / (np.std(compressed) + 1e-8)

        return compressed_norm.astype(
            np.float32
        )

    # ==================================================================
    # Spectrogram
    # ==================================================================

    def process_spectrogram(
        self,
        signal: np.ndarray,
    ):

        if self.spectrogram_processor is None:

            raise RuntimeError(
                "Sampling frequency 'fs' must be "
                "provided for spectrogram processing."
            )

        return self.spectrogram_processor.transform(
            signal
        )

    # ==================================================================
    # Model dispatcher
    # ==================================================================

    def process(
        self,
        signal: np.ndarray,
        model_name: str,
    ) -> np.ndarray:

        if model_name == "baseline_dnn":

            return self.process_fgcs(
                signal
            )

        if model_name == "baseline_mc1dcnn":

            return self.process_multichannel(
                signal
            )

        if model_name == "baseline_cnn":

            return self.process_compressed(
                signal
            )

        raise ValueError(
            f"No preprocessing pipeline defined "
            f"for model '{model_name}'."
        )


PreprocessingPipeline = DroneRFPreprocessor
create_default_pipeline = DroneRFPreprocessor