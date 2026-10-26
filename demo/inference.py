"""Inference engine for loading VARDHAN-v3 and executing reproducible classification."""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.vardhan_v3 import VardhanV3
from demo.config import (
    CLASS_MAPPING,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_PATH,
    EXPECTED_PARAMETERS,
    FINAL_CHECKPOINT_PATH,
    INPUT_SHAPE,
    SEGMENT_LENGTH,
    SMOKE_TEST_CHECKPOINT,
)


class VardanInferenceEngine:
    """Manages VARDHAN-v3 checkpoint discovery, model loading, and forward-pass inference."""

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        self.device = self._resolve_device(preferred_device=device)
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.model: Optional[VardhanV3] = None
        self.checkpoint_metadata: Dict[str, Any] = {}
        self.load_model()

    @staticmethod
    def _resolve_device(preferred_device: Optional[str] = None) -> torch.device:
        if preferred_device:
            return torch.device(preferred_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def discover_available_checkpoints() -> List[Path]:
        """Scan standard locations for valid VARDHAN-v3 checkpoints."""
        candidates = [
            DEFAULT_CHECKPOINT_PATH,
            FINAL_CHECKPOINT_PATH,
            SMOKE_TEST_CHECKPOINT,
        ]
        if DEFAULT_CHECKPOINT_DIR.exists():
            for p in DEFAULT_CHECKPOINT_DIR.glob("*.pt"):
                if p not in candidates:
                    candidates.append(p)
        return [p for p in candidates if p.exists() and p.is_file()]

    def _resolve_checkpoint(self, checkpoint_path: Optional[Union[str, Path]]) -> Path:
        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            if p.exists() and p.is_file():
                return p
            raise FileNotFoundError(f"Specified checkpoint does not exist: {p}")

        available = self.discover_available_checkpoints()
        if available:
            return available[0]

        raise FileNotFoundError(
            f"No trained VARDHAN-v3 checkpoint found! Expected at {DEFAULT_CHECKPOINT_PATH}. "
            "Please ensure the trained checkpoint file is present."
        )

    def load_model(self) -> None:
        """Instantiate VardhanV3 and load weights strictly from the checkpoint."""
        model = VardhanV3(num_classes=4)
        total_params = sum(p.numel() for p in model.parameters())
        if total_params != EXPECTED_PARAMETERS:
            raise ValueError(
                f"Model parameter mismatch: expected {EXPECTED_PARAMETERS}, got {total_params}."
            )

        # Load checkpoint weights (tensors and plain metadata only)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint format invalid (expected dict, got {type(checkpoint)}).")

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Strict loading ensures zero silent missing/unexpected keys
        load_result = model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()

        # Warm-up pass so the first reported latency reflects steady-state inference
        with torch.inference_mode():
            model(torch.zeros(INPUT_SHAPE, dtype=torch.float32, device=self.device))

        self.model = model
        self.checkpoint_metadata = {
            "path": str(self.checkpoint_path),
            "fold": checkpoint.get("fold", "Unknown"),
            "best_epoch": checkpoint.get("best_epoch", checkpoint.get("final_epoch", "Unknown")),
            "normalization": checkpoint.get("normalization", "per_segment"),
            "test_accuracy": checkpoint.get("test_accuracy", None),
            "test_macro_f1": checkpoint.get("test_macro_f1", None),
            "test_balanced_accuracy": checkpoint.get("test_balanced_accuracy", None),
            "test_loss": checkpoint.get("test_loss", None),
            "strict_keys_matched": len(load_result.missing_keys) == 0 and len(load_result.unexpected_keys) == 0,
            "device": str(self.device),
            "total_parameters": total_params,
        }

    def predict(
        self,
        normalized_waveform: np.ndarray,
    ) -> Dict[str, Any]:
        """Run single-segment inference on normalized 2048-sample waveform.

        Args:
            normalized_waveform: 1D array of shape (2048,) normalized to zero mean and unit variance.

        Returns:
            Dictionary with prediction results, class probabilities, latency, and branch embeddings.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        if normalized_waveform.shape != (SEGMENT_LENGTH,):
            raise ValueError(
                f"Input waveform must have shape ({SEGMENT_LENGTH},), got {normalized_waveform.shape}."
            )

        # Construct input tensor of shape (1, 1, 2048)
        input_tensor = (
            torch.tensor(normalized_waveform, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        t_start = time.perf_counter()
        with torch.inference_mode():
            logits, embeddings = self.model(input_tensor, return_embeddings=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            logits_np = logits[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_MAPPING.get(pred_idx, f"Class {pred_idx}")

        # Intermediate representations for visualization
        h_t = embeddings["h_temporal"][0].cpu().numpy()
        h_f = embeddings["h_spectral"][0].cpu().numpy()
        x_spec = embeddings["x_spectral_tensor"][0].cpu().numpy()  # (3, 1024)

        return {
            "predicted_label": pred_idx,
            "predicted_name": pred_class,
            "confidence": float(probs[pred_idx]),
            "probabilities": {CLASS_MAPPING[i]: float(probs[i]) for i in range(4)},
            "probability_array": probs,
            "logits": logits_np,
            "latency_ms": t_elapsed_ms,
            "device": str(self.device),
            "temporal_features": h_t,
            "spectral_features": h_f,
            "spectral_tensor": x_spec,
        }
