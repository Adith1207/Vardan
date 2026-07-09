"""Adaptive Multi-Sensor Fusion Engine."""

from typing import Dict, List, Optional
import numpy as np


class AdaptiveFusion:
    """Fuses detection probabilities from RF, Acoustic, and Vision streams.

    Supports late fusion using static weights, entropy-based dynamic weighting,
    or confidence threshold gating.
    """

    def __init__(self, static_weights: Optional[Dict[str, float]] = None):
        """Initialize the Fusion engine.

        Args:
            static_weights: Weights for each modality summing to 1.0.
                            Defaults to equal weighting.
        """
        if static_weights:
            self.weights = static_weights
        else:
            self.weights = {"rf": 0.4, "acoustic": 0.3, "vision": 0.3}
            
        # Ensure weights sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def late_fusion_static(
        self, rf_probs: np.ndarray, acoustic_probs: np.ndarray, vision_probs: np.ndarray
    ) -> np.ndarray:
        """Combine predictions using weighted linear averaging.

        Args:
            rf_probs: Probability vector from RF model of shape (num_classes,).
            acoustic_probs: Probability vector from Acoustic model.
            vision_probs: Probability vector from Vision model.

        Returns:
            Fused probability vector.
        """
        fused = (
            self.weights["rf"] * rf_probs
            + self.weights["acoustic"] * acoustic_probs
            + self.weights["vision"] * vision_probs
        )
        # Re-normalize to ensure valid probability distribution
        return fused / (np.sum(fused) + 1e-8)

    def late_fusion_entropy(
        self, modalities_probs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Fuse predictions using dynamic weights derived from prediction entropy.

        Lower entropy (higher confidence) receives higher dynamic weight.

        Args:
            modalities_probs: Dictionary mapping modality names to probability vectors.

        Returns:
            Fused probability vector.
        """
        entropies = {}
        for mod, probs in modalities_probs.items():
            # Shannon entropy: H(X) = -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            entropies[mod] = entropy
            
        # Calculate dynamic weights inversely proportional to entropy
        # Add epsilon to prevent division by zero
        inverse_entropies = {k: 1.0 / (v + 1e-8) for k, v in entropies.items()}
        sum_inv = sum(inverse_entropies.values())
        
        dynamic_weights = {k: v / sum_inv for k, v in inverse_entropies.items()}
        
        # Calculate final fused output
        fused = np.zeros_like(next(iter(modalities_probs.values())))
        for mod, probs in modalities_probs.items():
            fused += dynamic_weights[mod] * probs
            
        return fused / (np.sum(fused) + 1e-8)
