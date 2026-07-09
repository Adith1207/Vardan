"""Model profiling and benchmarking utilities for Edge AI and TinyML."""

from .bench import (
    benchmark_inference_latency,
    profile_model_footprint,
)

__all__ = [
    "benchmark_inference_latency",
    "profile_model_footprint",
]
