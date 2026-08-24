"""Signal preprocessing package for Vardan Counter-UAS framework."""

from .raw_loader import (
    DroneRFLoader,
    load_signal,
    load_raw_rf_data,
    segment_signal,
)
from .fft import (
    FFTProcessor,
    fft,
    compute_fft,
    compute_rfft,
    compute_power_spectral_density,
)
from .normalization import (
    SignalNormalizer,
    normalize_signal,
    zscore_normalize,
    minmax_normalize,
    robust_normalize,
)
from .channelization import (
    Channelizer,
    SpectrumChannelizer,
    channelize_signal,
    extract_subbands,
)
from .compression import (
    DynamicRangeCompressor,
    CompressedSensingProcessor,
    compress_dynamic_range,
    log_compression,
    mu_law_compression,
    power_law_compression,
)
from .spectrogram import (
    SpectrogramGenerator,
    SpectrogramProcessor,
    compute_spectrogram,
    compute_stft,
)
from .pipeline import (
    DroneRFPreprocessor,
    PreprocessingPipeline,
    create_default_pipeline,
)

__all__ = [
    "DroneRFLoader",
    "load_signal",
    "load_raw_rf_data",
    "segment_signal",
    "FFTProcessor",
    "fft",
    "compute_fft",
    "compute_rfft",
    "compute_power_spectral_density",
    "SignalNormalizer",
    "normalize_signal",
    "zscore_normalize",
    "minmax_normalize",
    "robust_normalize",
    "Channelizer",
    "SpectrumChannelizer",
    "channelize_signal",
    "extract_subbands",
    "DynamicRangeCompressor",
    "CompressedSensingProcessor",
    "compress_dynamic_range",
    "log_compression",
    "mu_law_compression",
    "power_law_compression",
    "SpectrogramGenerator",
    "SpectrogramProcessor",
    "compute_spectrogram",
    "compute_stft",
    "DroneRFPreprocessor",
    "PreprocessingPipeline",
    "create_default_pipeline",
]