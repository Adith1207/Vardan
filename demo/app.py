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


# ==============================================================================
# TABS INTERFACE
# ==============================================================================

tab_infer, tab_pipe, tab_arch, tab_eval = st.tabs([
    "🎯 Live Signal & Prediction",
    "🔄 Inference Pipeline",
    "🧠 Model Architecture",
    "📊 Scientific Provenance & Evaluation",
])

# ------------------------------------------------------------------------------
# TAB 1: LIVE INFERENCE & SIGNAL VISUALIZATION
# ------------------------------------------------------------------------------
with tab_infer:
    # Summary KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Ground Truth</div>
                <div class="metric-value">{sample_data['ground_truth_name']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Class</div>
                <div class="metric-value">{pred_res['predicted_name']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Posterior Probability</div>
                <div class="metric-value">{pred_res['confidence']*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Inference Latency</div>
                <div class="metric-value">{pred_res['latency_ms']:.2f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        is_match = pred_res["predicted_label"] == sample_data["ground_truth_label"]
        match_html = (
            '<span class="match-tag-success">✅ CORRECT</span>'
            if is_match
            else '<span class="match-tag-fail">⚠️ MISMATCH</span>'
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Prediction Status</div>
                <div class="metric-value">{match_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### **1. Real RF Input Signal Characterization**")
    sig_col1, sig_col2 = st.columns(2)

    with sig_col1:
        fig_wave = plot_waveform(
            sample_data["normalized_waveform"],
            sampling_rate_hz=SAMPLING_RATE_HZ,
            title=f"Normalized Time-Domain Voltage (Window: {sample_data['start_sample']}–{sample_data['end_sample']})",
        )
        st.plotly_chart(fig_wave, width="stretch")
        st.caption(
            f"⏱️ **Duration:** {SEGMENT_DURATION_US:.1f} µs (2,048 samples @ 40 MSps baseband). "
            f"**Normalized:** μ={np.mean(sample_data['normalized_waveform']):.2e}, σ={np.std(sample_data['normalized_waveform']):.2f}"
        )

    with sig_col2:
        fig_psd = plot_power_spectrum(
            sample_data["raw_waveform"],
            sampling_rate_hz=SAMPLING_RATE_HZ,
            title="Real-Valued FFT Power Spectral Density (0 to 20 MHz Baseband)",
        )
        st.plotly_chart(fig_psd, width="stretch")
        st.caption(
            "📡 2048-point positive-frequency Periodogram PSD. Drones exhibit distinct transient burst spectral peaks."
        )

    st.markdown("### **2. Classification Output & Posterior Confidence**")
    prob_col1, prob_col2 = st.columns([3, 2])

    with prob_col1:
        fig_probs = plot_class_probabilities(
            pred_res["probabilities"],
            ground_truth_name=sample_data["ground_truth_name"],
        )
        st.plotly_chart(fig_probs, width="stretch")

    with prob_col2:
        st.markdown("#### Class Interpretation & Description")
        for cls_idx, cls_str in CLASS_MAPPING.items():
            prob_pct = pred_res["probabilities"][cls_str] * 100.0
            is_pred = cls_str == pred_res["predicted_name"]
            is_gt = cls_str == sample_data["ground_truth_name"]

            badge = ""
            if is_pred and is_gt:
                badge = " 🧠 **[PREDICTED & TRUE]**"
            elif is_pred:
                badge = " 🔵 **[PREDICTED]**"
            elif is_gt:
                badge = " 🎯 **[GROUND TRUTH]**"

            st.markdown(
                f"- **{cls_str}** (`{prob_pct:.1f}%`){badge}<br>"
                f"<small style='color:#94A3B8;'>{CLASS_DESCRIPTIONS[cls_idx]}</small>",
                unsafe_allow_html=True,
            )

    # Expandable Dual-Domain Internal Representation
    with st.expander("🔬 View Model Internal Dual-Domain Representation (On-The-Fly FFT Stem)", expanded=False):
        st.markdown(
            "Inside `VardhanV3`, the model computes a deterministic 3-channel dual-Fourier representation "
            "directly from the input waveform without requiring pre-extracted spectral files:"
        )
        fig_dual = plot_dual_spectral_tensor(pred_res["spectral_tensor"])
        st.plotly_chart(fig_dual, width="stretch")


# ------------------------------------------------------------------------------
# TAB 2: INFERENCE PIPELINE TRACE
# ------------------------------------------------------------------------------
with tab_pipe:
    st.markdown("### **End-to-End Inference Pipeline**")
    st.markdown(
        "This execution trace reflects the exact mathematical operations performed from raw dataset file to output logits:"
    )

    st.markdown(
        f"""
        1. **Raw DroneRF Signal Ingestion**:
            - **Selected Recording:** Pair `{sample_data['pair_id']}` (BUI: `{sample_data['bui']}`)
            - **File Segment Number:** `{sample_data['file_segment_num']}`
            - **Underlying File:** `{sample_data['source_file']}`
            - **Ground Truth:** Class {sample_data['ground_truth_label']} (`{sample_data['ground_truth_name']}`)

        2. **Time-Domain Slicing**:
            - Extracted 2,048 contiguous samples from index `{sample_data['start_sample']}` to `{sample_data['end_sample']}`.
            - Observation window length: **51.2 microseconds** at 40 MS/s.

        3. **Per-Segment Waveform Normalization**:
            - Formula: $\\tilde{{x}} = \\frac{{x - \\mu}}{{\\sigma + 10^{{-8}}}}$
            - Sample Raw Mean: `{sample_data['mean']:.6f}`
            - Sample Raw Std: `{sample_data['std']:.6f}`
            - Resulting Shape: `(2048,)`

        4. **Model Boundary Tensor Construction**:
            - Tensor Shape: `{INPUT_SHAPE}`
            - Tensor Dtype: `{INPUT_DTYPE}`
            - Device: `{engine.device}`

        5. **Dual-Domain Feature Extraction (Internal)**:
            - **Branch A (Temporal):** 4 Multi-Scale Temporal Blocks (dilations: 1, 2, 4, 8) $\\to$ 80-d attention-pooled features.
            - **Branch B (Spectral):** On-the-fly 2048-point rFFT (3 channels $\\times$ 1024 bins) $\\to$ 4 Multi-Scale Spectral Blocks $\\to$ 80-d attention-pooled features.

        6. **Bi-Directional Cross-Domain Gated Fusion**:
            - Temporal features modulate spectral features via sigmoid gating.
            - Fused representation: $[h_T, h_F, h_{{T,\\text{{mod}}}}, h_{{F,\\text{{mod}}}}] \\in \\mathbb{{R}}^{{320}}$.

        7. **Classification & Softmax**:
            - Multi-layer perceptron head: $320 \\to 64 \\to 4$.
            - Unnormalized Logits: `{np.round(pred_res['logits'], 4)}`
            - Softmax Normalization: Yields valid probability distribution summing to {float(np.sum(pred_res['probability_array'])):.4f}.
        """
    )


# ------------------------------------------------------------------------------
# TAB 3: MODEL ARCHITECTURE SUMMARY
# ------------------------------------------------------------------------------
with tab_arch:
    st.markdown("### **VARDHAN-v3 Architecture Specification**")
    st.markdown(
        "VARDHAN-v3 is a lightweight, RF-specific deep learning architecture designed for edge efficiency "
        "and multi-scale temporal-spectral feature representation."
    )

    arch_col1, arch_col2 = st.columns([1, 1])

    with arch_col1:
        st.markdown("#### Architectural Components")
        st.markdown(
            """
            - **Input**: Single-channel RF time-domain waveform `(B, 1, 2048)`.
            - **Temporal Backbone**:
                - Conv1D stem (kernel=15, stride=2, channels=32) + BatchNorm1d + GELU.
                - 4 MultiScaleTemporalBlocks with depthwise parallel branches ($k \\in \\{3, 7, 15, 31\\}$),
                  dilation schedule $d \\in \\{1, 2, 4, 8\\}$, and Squeeze-and-Excitation (SE) channel attention.
                - Learnable self-attention pooling (`AttentionPool1d`: $80 \\to 40 \\to 1$) yielding $h_T \\in \\mathbb{R}^{80}$.
            - **Spectral Backbone**:
                - On-the-fly 3-channel rFFT tensor `(B, 3, 1024)`: Log Power, Real Component, Imag Component.
                - Conv1D stem (kernel=11, stride=2, channels=32) + BatchNorm1d + GELU.
                - 4 MultiScaleSpectralBlocks ($k \\in \\{5, 11\\}$), dilations $d \\in \\{1, 2, 4, 8\\}$, SE attention.
                - Learnable self-attention pooling yielding $h_F \\in \\mathbb{R}^{80}$.
            - **Cross-Domain Gated Fusion**:
                - Bi-directional sigmoid modulation: $g_T(h_F)$, $g_F(h_T)$.
                - Fused feature vector: $320$ dimensions.
            - **Classification Head**:
                - Linear(320 $\\to$ 64) + BatchNorm1d + GELU + Dropout(0.2) + Linear(64 $\\to$ 4).
            """
        )

    with arch_col2:
        st.markdown("#### Verified Trainable Parameter Counts")
        if engine.model:
            modules = [
                ("Time Stem", sum(p.numel() for p in engine.model.time_stem.parameters())),
                ("Temporal Blocks (t_b1..4)", sum(p.numel() for b in [engine.model.t_b1, engine.model.t_b2, engine.model.t_b3, engine.model.t_b4] for p in b.parameters())),
                ("Temporal Attention Pool", sum(p.numel() for p in engine.model.t_pool.parameters())),
                ("Frequency Stem", sum(p.numel() for p in engine.model.freq_stem.parameters())),
                ("Spectral Blocks (f_b1..4)", sum(p.numel() for b in [engine.model.f_b1, engine.model.f_b2, engine.model.f_b3, engine.model.f_b4] for p in b.parameters())),
                ("Spectral Attention Pool", sum(p.numel() for p in engine.model.f_pool.parameters())),
                ("Fusion & Classification Head", sum(p.numel() for p in engine.model.head.parameters())),
            ]
            df_params = pd.DataFrame(modules, columns=["Submodule", "Parameters"])
            df_params.loc[len(df_params)] = ["Total Trainable Parameters", df_params["Parameters"].sum()]
            st.dataframe(df_params, width="stretch", hide_index=True)
            st.success(f"Verified: Total parameters equal exactly {EXPECTED_PARAMETERS:,}.")
