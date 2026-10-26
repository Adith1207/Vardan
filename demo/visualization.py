"""Interactive visualization utilities using Plotly for the VARDAN research demo."""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from demo.config import SAMPLING_RATE_HZ


def plot_waveform(
    waveform: np.ndarray,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    title: str = "Raw Time-Domain RF Voltage Waveform (2048 Samples)",
) -> go.Figure:
    """Plot time-domain RF voltage samples against time in microseconds."""
    n_samples = len(waveform)
    time_us = (np.arange(n_samples) / sampling_rate_hz) * 1e6

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_us,
            y=waveform,
            mode="lines",
            line=dict(color="#00ADB5", width=1.5),
            name="RF Voltage",
            hovertemplate="Time: %{x:.2f} µs<br>Amplitude: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#EEEEEE")),
        xaxis=dict(
            title="Time (µs)",
            gridcolor="#2D3748",
            showline=True,
            linecolor="#4A5568",
            ticksuffix=" µs",
        ),
        yaxis=dict(
            title="Normalized Voltage",
            gridcolor="#2D3748",
            showline=True,
            linecolor="#4A5568",
            zerolinecolor="#4A5568",
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#CBD5E1"),
        height=320,
    )
    return fig


def plot_power_spectrum(
    waveform: np.ndarray,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    title: str = "RF Frequency-Domain Power Spectral Density (PSD)",
) -> go.Figure:
    """Compute and plot the positive-frequency power spectrum in dB vs Frequency (MHz)."""
    # 2048-point Real FFT
    rfft_vals = np.fft.rfft(waveform, n=len(waveform))
    freqs_mhz = (np.fft.rfftfreq(len(waveform), d=1.0 / sampling_rate_hz)) / 1e6
    psd = (np.abs(rfft_vals) ** 2) / float(len(waveform))
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs_mhz,
            y=psd_db,
            mode="lines",
            line=dict(color="#F39C12", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(243, 156, 18, 0.15)",
            name="PSD (dB)",
            hovertemplate="Frequency: %{x:.2f} MHz<br>Power: %{y:.2f} dB<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#EEEEEE")),
        xaxis=dict(
            title="Baseband Frequency (MHz)",
            gridcolor="#2D3748",
            showline=True,
            linecolor="#4A5568",
            ticksuffix=" MHz",
        ),
        yaxis=dict(
            title="Power (dB)",
            gridcolor="#2D3748",
            showline=True,
            linecolor="#4A5568",
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#CBD5E1"),
        height=320,
    )
    return fig


def plot_dual_spectral_tensor(spectral_tensor: np.ndarray) -> go.Figure:
    """Visualize the 3 internal spectral channels extracted on-the-fly inside VARDHAN-v3:
    Channel 0: Log Power Spectrum (1024 positive bins)
    Channel 1: Normalized Real Fourier Component (1024 positive bins)
    Channel 2: Normalized Imaginary Fourier Component (1024 positive bins)
    """
    bins = np.arange(1, 1025)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Channel 0: Log Power Spectrum [log(1 + 1000 · |X|² / 2048)]",
            "Channel 1: Normalized Real Fourier Component [Re(X) / √2048]",
            "Channel 2: Normalized Imaginary Fourier Component [Im(X) / √2048]",
        ),
        vertical_spacing=0.18,
    )

    fig.add_trace(
        go.Scatter(x=bins, y=spectral_tensor[0], line=dict(color="#38BDF8", width=1.2), name="Log Power"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=bins, y=spectral_tensor[1], line=dict(color="#34D399", width=1.2), name="Real FFT"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=bins, y=spectral_tensor[2], line=dict(color="#F472B6", width=1.2), name="Imag FFT"),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=640,
        margin=dict(l=40, r=40, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#CBD5E1", size=11),
        showlegend=False,
    )
    # Set gridcolor for all axes, but place the x-axis label exclusively on the bottom subplot
    fig.update_xaxes(gridcolor="#2D3748")
    fig.update_xaxes(title_text="Positive Frequency Bin (1..1024)", row=3, col=1)
    fig.update_yaxes(gridcolor="#2D3748")

    # Format subplot titles to ensure clean spacing and no collisions
    fig.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#E2E8F0")))
    return fig


def plot_class_probabilities(
    probabilities: Dict[str, float],
    ground_truth_name: Optional[str] = None,
) -> go.Figure:
    """Horizontal bar chart showing ranked class probability distribution."""
    # Sort classes descending by probability
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [k for k, _ in sorted_items]
    probs = [v * 100.0 for _, v in sorted_items]

    # Bar colors: highlight prediction and match status
    top_class = classes[0]
    bar_colors = []
    for cls in classes:
        if cls == top_class:
            if ground_truth_name and cls == ground_truth_name:
                bar_colors.append("#10B981")  # Green for correct top prediction
            elif ground_truth_name and cls != ground_truth_name:
                bar_colors.append("#F59E0B")  # Amber for mismatch
            else:
                bar_colors.append("#3B82F6")  # Blue default
        elif cls == ground_truth_name:
            bar_colors.append("#10B981")      # Green for ground truth if not top
        else:
            bar_colors.append("#64748B")      # Muted slate

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#1E293B", width=1.5)),
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto",
            hovertemplate="Class: %{y}<br>Probability: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="VARDHAN-v3 Posterior Probability Distribution", font=dict(size=15, color="#EEEEEE")),
        xaxis=dict(
            title="Softmax Probability (%)",
            range=[0, 105],
            gridcolor="#2D3748",
            ticksuffix="%",
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#CBD5E1"),
        height=260,
    )
    return fig


def plot_confusion_matrix_heatmap(
    matrix: np.ndarray,
    class_names: List[str],
    title: str = "Standardized 10-Fold Confusion Matrix (22,700 Segments)",
) -> go.Figure:
    """Plot an interactive confusion matrix heatmap."""
    text_annotations = [[str(int(val)) for val in row] for row in matrix]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=class_names,
            y=class_names,
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#FFFFFF"),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Samples", tickcolor="#CBD5E1"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#EEEEEE")),
        xaxis=dict(title="Predicted Label", color="#CBD5E1"),
        yaxis=dict(title="True Label", autorange="reversed", color="#CBD5E1"),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#CBD5E1"),
        height=380,
    )
    return fig
