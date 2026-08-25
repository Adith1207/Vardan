"""Metrics computation module for model evaluation."""

from typing import Dict, Union
import numpy as np


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate classification metrics (accuracy, precision, recall, f1).

    Args:
        y_true: Ground truth 1D class labels.
        y_pred: Predicted 1D class labels.

    Returns:
        Dictionary containing metric values.
    """
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    accuracy = np.sum(y_true == y_pred) / total
    
    classes = np.unique(y_true)
    precisions = []
    recalls = []
    f1s = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    precision_macro = float(np.mean(precisions)) if precisions else 0.0
    recall_macro = float(np.mean(recalls)) if recalls else 0.0
    f1_macro = float(np.mean(f1s)) if f1s else 0.0
        
    return {
        "accuracy": float(accuracy),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_macro,  # Approximation when balanced
    }


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Generate confusion matrix of dimension (num_classes, num_classes).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Total number of classes. Default 4.

    Returns:
        NumPy confusion matrix array.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm
