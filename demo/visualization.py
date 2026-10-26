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
