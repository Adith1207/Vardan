"""Signal and results plotting utilities using Matplotlib."""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_rf_iq(
    signal: np.ndarray,
    title: str = "RF Signal I/Q Representation",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot the in-phase (I) and quadrature (Q) components of an RF signal.

    Args:
        signal: Input 2D I/Q RF signal of shape (sequence_length, 2).
        title: Title of the plot.
        save_path: Destination to save the plot figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(signal[:, 0], color="tab:blue")
    axes[0].set_ylabel("In-Phase (I)")
    axes[0].grid(True)
    
    axes[1].plot(signal[:, 1], color="tab:orange")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].set_xlabel("Time (Samples)")
    axes[1].grid(True)
    
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        
    return fig


def plot_spectrogram(
    spectrogram: np.ndarray,
    title: str = "Signal Spectrogram",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot complex or magnitude spectrogram.

    Args:
        spectrogram: Spectrogram matrix.
        title: Plot title.
        save_path: Destination to save the plot figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    magnitude = np.abs(spectrogram)
    db_spectrogram = 20 * np.log10(magnitude + 1e-8)
    
    img = ax.imshow(db_spectrogram, aspect="auto", origin="lower", cmap="viridis")
    fig.colorbar(img, ax=ax, label="Power (dB)")
    
    ax.set_title(title)
    ax.set_ylabel("Frequency Bin")
    ax.set_xlabel("Time Frame")
    
    fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot the confusion matrix for model evaluation.

    Args:
        cm: Confusion matrix array.
        classes: List of class name labels.
        title: Plot title.
        save_path: Destination to save the plot.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
            
    fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        
    return fig
