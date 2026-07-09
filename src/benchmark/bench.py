"""Edge AI benchmarking tools for tracking computation cost and parameters."""

import time
from typing import Any, Dict
# pyrefly: ignore [missing-import]
import numpy as np

try:
    import torch  # type: ignore
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def benchmark_inference_latency(
    model: Any, input_shape: tuple, num_runs: int = 100, device: str = "cpu"
) -> Dict[str, float]:
    """Measure model inference latency on a specific hardware device.

    Args:
        model: PyTorch model object.
        input_shape: Shape of the input tensor (excluding batch dimension).
        num_runs: Number of forward runs for statistics.
        device: 'cpu' or 'cuda' target device.

    Returns:
        Dictionary with 'mean_latency_ms' and 'std_latency_ms'.
    """
    if not HAS_TORCH:
        return {"mean_latency_ms": 0.0, "std_latency_ms": 0.0}
        
    model.eval()
    model.to(device)
    
    # Generate dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup runs
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Measure latency
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # to ms
        
    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
    }


def profile_model_footprint(model: Any) -> Dict[str, Any]:
    """Count model parameters and estimate memory footprint.

    Args:
        model: PyTorch model object.

    Returns:
        Dictionary containing parameter count and size estimates.
    """
    if not HAS_TORCH:
        return {"total_parameters": 0, "trainable_parameters": 0, "size_mb": 0.0}
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size assuming float32 (4 bytes per parameter)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "estimated_size_mb": float(size_mb),
    }
