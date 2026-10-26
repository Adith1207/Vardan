"""VARDAN — Interactive Research Demonstration Application.

A professional, presentation-ready Streamlit interface demonstrating the trained
VARDHAN-v3 dual-domain RF drone signal classification model on real DroneRF recordings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from demo.config import (
    CLASS_DESCRIPTIONS,
    CLASS_MAPPING,
    EXPECTED_PARAMETERS,
    INPUT_DTYPE,
    INPUT_SHAPE,
    MODEL_NAME,
    RESULTS_DIR,
    SAMPLING_RATE_HZ,
    SEGMENT_DURATION_US,
)
from demo.data_utils import DemoDataLoader
from demo.inference import VardanInferenceEngine
from demo.visualization import (
    plot_class_probabilities,
    plot_confusion_matrix_heatmap,
    plot_dual_spectral_tensor,
    plot_power_spectrum,
    plot_waveform,
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="VARDAN — RF Drone Classification Demo",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for academic research aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .match-tag-success {
        color: #10B981;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .match-tag-fail {
        color: #F59E0B;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained VARDHAN-v3 checkpoint...")
def get_inference_engine(checkpoint_path_str: str) -> VardanInferenceEngine:
    """Load and cache the inference engine strictly in memory."""
    return VardanInferenceEngine(checkpoint_path=checkpoint_path_str)


@st.cache_resource(show_spinner="Discovering DroneRF recording pairs...")
def get_data_loader(raw_dir_str: str) -> DemoDataLoader:
    """Instantiate and cache the dataset loader."""
    return DemoDataLoader(raw_data_dir=raw_dir_str)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/PyTorch-Dark.svg",
        width=48,
    )
    st.markdown("### **VARDAN Control Panel**")
    st.markdown("---")

    # Checkpoint Selection
    st.markdown("#### 1. Model Checkpoint")
    available_ckpts = VardanInferenceEngine.discover_available_checkpoints()
    if not available_ckpts:
        st.error("No valid checkpoint found! Expected at `models/checkpoints/`.")
        st.stop()

    ckpt_options = [str(p) for p in available_ckpts]
    selected_ckpt_str = st.selectbox(
        "Select Checkpoint File:",
        options=ckpt_options,
        index=0,
        format_func=lambda p: Path(p).name,
        help="Loads pre-trained weights without retraining.",
    )

    # Initialize Engine
    try:
        engine = get_inference_engine(selected_ckpt_str)
        meta = engine.checkpoint_metadata
        st.success(f"Loaded: `{Path(selected_ckpt_str).name}`")
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    # Model Status Badges
    with st.expander("Model Checkpoint Details", expanded=False):
        st.write(f"**Architecture:** `{MODEL_NAME}`")
        st.write(f"**Trainable Parameters:** `{meta.get('total_parameters', 0):,}`")
        st.write(f"**Target Device:** `{engine.device}`")
        st.write(f"**Fold:** `{meta.get('fold', 'N/A')}`")
        st.write(f"**Best Epoch:** `{meta.get('best_epoch', 'N/A')}`")
        st.write(f"**Normalization:** `{meta.get('normalization', 'per_segment')}`")
        if meta.get("test_accuracy"):
            st.write(f"**Fold 1 Test Accuracy:** `{meta['test_accuracy']*100:.2f}%`")
        if meta.get("test_macro_f1"):
            st.write(f"**Fold 1 Macro-F1:** `{meta['test_macro_f1']:.4f}`")

    st.markdown("---")

    # Dataset & Sample Selection
    st.markdown("#### 2. DroneRF Sample Selection")
    data_loader = get_data_loader(str(PROJECT_ROOT / "data" / "raw"))

    try:
        choices = data_loader.get_sample_choices()
    except Exception as e:
        st.error(f"Dataset discovery error: {e}")
        st.stop()

    # Filter choices by class
    class_filter = st.selectbox(
        "Filter by Target Drone Class:",
        options=["All Classes"] + list(CLASS_MAPPING.values()),
        index=0,
    )

    filtered_choices = (
        choices
        if class_filter == "All Classes"
        else [c for c in choices if c["class_name"] == class_filter]
    )
    if not filtered_choices:
        st.warning(f"No recordings of class '{class_filter}' found in the dataset.")
        st.stop()

    selected_choice = st.selectbox(
        "Select Real Recording:",
        options=filtered_choices,
        format_func=lambda c: c["display"],
        index=0,
    )

    segment_idx = st.slider(
        "Segment Offset Window (0 to 99):",
        min_value=0,
        max_value=99,
        value=0,
        help="Extracts a non-overlapping 2048-sample slice at offset * 100,000 samples.",
    )

    st.markdown("---")
    st.caption(
        "ℹ️ Selecting a new sample runs live inference with the loaded checkpoint, without retraining. "
        "The latency card reports the time measured on this machine."
    )


# ==============================================================================
# MAIN DASHBOARD HEADER
# ==============================================================================

st.markdown('<div class="main-title">🛰️ VARDAN — RF Drone Classification Demo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Lightweight Dual-Domain Neural Network (VARDHAN-v3) for RF UAV Signal Identification</div>',
    unsafe_allow_html=True,
)

# Load Selected Real Sample
with st.spinner("Extracting and normalizing raw RF sample..."):
    try:
        sample_data = data_loader.load_real_segment(
            pair_row=selected_choice["row"],
            segment_index=segment_idx,
        )
    except Exception as e:
        st.error(f"Error reading real DroneRF sample: {e}")
        st.stop()

# Run Live Inference
try:
    pred_res = engine.predict(sample_data["normalized_waveform"])
except Exception as e:
    st.error(f"Inference execution failed: {e}")
    st.stop()
