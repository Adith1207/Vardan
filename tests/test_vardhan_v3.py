"""
test_vardhan_v3.py
------------------

Comprehensive test suite for VARDHAN-v3 (Multi-Scale Dual-Domain RF-Net).
Validates:
1. Exact trainable parameter count (strictly 119,806).
2. Exact tensor shapes at all intermediate layers.
3. Deterministic on-the-fly 3-channel spectral extraction.
4. Finite forward pass outputs (no NaN / Inf).
5. Finite loss computation.
6. Complete gradient backpropagation and finite gradients for all trainable parameters.
7. Train vs. Eval mode behavior (Dropout and BatchNorm).
8. Embedding dictionary extraction mode.
9. Support for variable batch sizes (B=1 in eval, B=2, 4, 16, 32).
10. Model factory registration and initialization.
"""

import math
import sys
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
import torch
import torch.nn as nn

from models.vardhan_v3 import (
    VardhanV3,
    MultiScaleTemporalBlock,
    MultiScaleSpectralBlock,
    AttentionPool1d,
    CrossDomainGatedFusion,
    SqueezeExcitation1d,
)
from models.model_factory import get_model


EXPECTED_V3_PARAM_COUNT = 119806


def test_vardhan_v3_exact_parameter_count():
    """Verify that VARDHAN-v3 has exactly 119,806 trainable parameters."""
    model = VardhanV3(num_classes=4)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params == EXPECTED_V3_PARAM_COUNT, (
        f"Expected {EXPECTED_V3_PARAM_COUNT} total parameters, got {total_params}"
    )
    assert trainable_params == EXPECTED_V3_PARAM_COUNT, (
        f"Expected {EXPECTED_V3_PARAM_COUNT} trainable parameters, got {trainable_params}"
    )


def test_vardhan_v3_module_parameter_breakdown():
    """Verify individual module parameter counts match design table."""
    model = VardhanV3(num_classes=4)

    # 1. Temporal Backbone
    p_time_stem = sum(p.numel() for p in model.time_stem.parameters())
    p_t_b1 = sum(p.numel() for p in model.t_b1.parameters())
    p_t_b2 = sum(p.numel() for p in model.t_b2.parameters())
    p_t_b3 = sum(p.numel() for p in model.t_b3.parameters())
    p_t_b4 = sum(p.numel() for p in model.t_b4.parameters())
    p_t_pool = sum(p.numel() for p in model.t_pool.parameters())
    p_temporal_total = p_time_stem + p_t_b1 + p_t_b2 + p_t_b3 + p_t_b4 + p_t_pool

    assert p_time_stem == 544, f"time_stem expected 544, got {p_time_stem}"
    assert p_t_b1 == 4828, f"t_b1 expected 4,828, got {p_t_b1}"
    assert p_t_b2 == 9072, f"t_b2 expected 9,072, got {p_t_b2}"
    assert p_t_b3 == 14596, f"t_b3 expected 14,596, got {p_t_b3}"
    assert p_t_b4 == 10980, f"t_b4 expected 10,980, got {p_t_b4}"
    assert p_t_pool == 3281, f"t_pool expected 3,281, got {p_t_pool}"
    assert p_temporal_total == 43301, f"temporal backbone expected 43,301, got {p_temporal_total}"

    # 2. Spectral Backbone
    p_freq_stem = sum(p.numel() for p in model.freq_stem.parameters())
    p_f_b1 = sum(p.numel() for p in model.f_b1.parameters())
    p_f_b2 = sum(p.numel() for p in model.f_b2.parameters())
    p_f_b3 = sum(p.numel() for p in model.f_b3.parameters())
    p_f_b4 = sum(p.numel() for p in model.f_b4.parameters())
    p_f_pool = sum(p.numel() for p in model.f_pool.parameters())
    p_spectral_total = p_freq_stem + p_f_b1 + p_f_b2 + p_f_b3 + p_f_b4 + p_f_pool

    assert p_freq_stem == 1120, f"freq_stem expected 1,120, got {p_freq_stem}"
    assert p_f_b1 == 4636, f"f_b1 expected 4,636, got {p_f_b1}"
    assert p_f_b2 == 8784, f"f_b2 expected 8,784, got {p_f_b2}"
    assert p_f_b3 == 14212, f"f_b3 expected 14,212, got {p_f_b3}"
    assert p_f_b4 == 10500, f"f_b4 expected 10,500, got {p_f_b4}"
    assert p_f_pool == 3281, f"f_pool expected 3,281, got {p_f_pool}"
    assert p_spectral_total == 42533, f"spectral backbone expected 42,533, got {p_spectral_total}"

    # 3. Fusion Head
    p_gate_t = sum(p.numel() for p in model.head.gate_t.parameters())
    p_gate_f = sum(p.numel() for p in model.head.gate_f.parameters())
    p_fc1 = sum(p.numel() for p in model.head.fc1.parameters())
    p_bn = sum(p.numel() for p in model.head.bn.parameters())
    p_fc2 = sum(p.numel() for p in model.head.fc2.parameters())
    p_head_total = sum(p.numel() for p in model.head.parameters())

    assert p_gate_t == 6520, f"gate_t expected 6,520, got {p_gate_t}"
    assert p_gate_f == 6520, f"gate_f expected 6,520, got {p_gate_f}"
    assert p_fc1 == 20544, f"fc1 expected 20,544, got {p_fc1}"
    assert p_bn == 128, f"bn expected 128, got {p_bn}"
    assert p_fc2 == 260, f"fc2 expected 260, got {p_fc2}"
    assert p_head_total == 33972, f"fusion head expected 33,972, got {p_head_total}"

    # Grand Total
    assert p_temporal_total + p_spectral_total + p_head_total == EXPECTED_V3_PARAM_COUNT


def test_vardhan_v3_intermediate_tensor_shapes():
    """Verify exact tensor propagation shapes at every intermediate stage."""
    model = VardhanV3(num_classes=4)
    model.eval()

    B = 4
    x = torch.randn(B, 1, 2048)

    # 1. Deterministic Spectral Extraction
    x_freq = model.extract_spectral_representation(x)
    assert x_freq.shape == (B, 3, 1024), f"Expected (B, 3, 1024), got {x_freq.shape}"

    # 2. Temporal Path
    t0 = model.time_stem(x)
    assert t0.shape == (B, 32, 1024), f"time_stem shape: {t0.shape}"
    t1 = model.t_b1(t0)
    assert t1.shape == (B, 48, 512), f"t_b1 shape: {t1.shape}"
    t2 = model.t_b2(t1)
    assert t2.shape == (B, 64, 256), f"t_b2 shape: {t2.shape}"
    t3 = model.t_b3(t2)
    assert t3.shape == (B, 80, 128), f"t_b3 shape: {t3.shape}"
    t4 = model.t_b4(t3)
    assert t4.shape == (B, 80, 128), f"t_b4 shape: {t4.shape}"
    h_t = model.t_pool(t4)
    assert h_t.shape == (B, 80), f"h_t shape: {h_t.shape}"

    # 3. Spectral Path
    f0 = model.freq_stem(x_freq)
    assert f0.shape == (B, 32, 512), f"freq_stem shape: {f0.shape}"
    f1 = model.f_b1(f0)
    assert f1.shape == (B, 48, 256), f"f_b1 shape: {f1.shape}"
    f2 = model.f_b2(f1)
    assert f2.shape == (B, 64, 128), f"f_b2 shape: {f2.shape}"
    f3 = model.f_b3(f2)
    assert f3.shape == (B, 80, 128), f"f_b3 shape: {f3.shape}"
    f4 = model.f_b4(f3)
    assert f4.shape == (B, 80, 128), f"f_b4 shape: {f4.shape}"
    h_f = model.f_pool(f4)
    assert h_f.shape == (B, 80), f"h_f shape: {h_f.shape}"

    # 4. Fusion and Head
    g_t = model.head.gate_t(h_f)
    g_f = model.head.gate_f(h_t)
    assert g_t.shape == (B, 80), f"g_t shape: {g_t.shape}"
    assert g_f.shape == (B, 80), f"g_f shape: {g_f.shape}"

    logits = model.head(h_t, h_f)
    assert logits.shape == (B, 4), f"logits shape: {logits.shape}"


def test_deterministic_spectral_extraction():
    """Verify that rFFT spectral extraction is 100% deterministic and reproducible."""
    model = VardhanV3(num_classes=4)
    x = torch.randn(8, 1, 2048)

    spec1 = model.extract_spectral_representation(x)
    spec2 = model.extract_spectral_representation(x)

    torch.testing.assert_close(spec1, spec2, rtol=1e-6, atol=1e-6)

    # Check that channel 0 is strictly non-negative (log1p of non-negative power)
    assert (spec1[:, 0] >= 0.0).all(), "Log power channel must be non-negative"

    # Check that Real and Imag components are bounded
    assert torch.isfinite(spec1[:, 1]).all(), "Real FFT component must be finite"
    assert torch.isfinite(spec1[:, 2]).all(), "Imag FFT component must be finite"


def test_forward_pass_finite_output():
    """Verify forward pass outputs finite logits for standard, zero, and noisy inputs."""
    model = VardhanV3(num_classes=4)
    model.eval()

    # Standard random Gaussian
    x_rand = torch.randn(4, 1, 2048)
    out_rand = model(x_rand)
    assert torch.isfinite(out_rand).all(), "Random input produced non-finite logits"

    # All-zero signal
    x_zero = torch.zeros(4, 1, 2048)
    out_zero = model(x_zero)
    assert torch.isfinite(out_zero).all(), "Zero input produced non-finite logits"

    # Large amplitude signal
    x_large = torch.randn(4, 1, 2048) * 100.0
    out_large = model(x_large)
    assert torch.isfinite(out_large).all(), "Large input produced non-finite logits"


def test_backward_pass_and_finite_gradients():
    """Verify complete end-to-end backpropagation and check that every parameter gets finite gradients."""
    model = VardhanV3(num_classes=4)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, 1, 2048)
    targets = torch.randint(0, 4, (8,))

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, targets)

    assert torch.isfinite(loss), f"Loss was non-finite: {loss.item()}"

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient (None)"
        assert torch.isfinite(param.grad).all(), f"Parameter '{name}' received non-finite gradients"

    optimizer.step()


def test_train_and_eval_mode_consistency():
    """Verify behavior across train and eval modes (eval mode should be fully deterministic)."""
    model = VardhanV3(num_classes=4)
    x = torch.randn(4, 1, 2048)

    # Eval mode: identical forward passes must produce identical outputs
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2, rtol=1e-6, atol=1e-6)

    # Train mode: forward pass should execute without errors
    model.train()
    out_train = model(x)
    assert out_train.shape == (4, 4)
    assert torch.isfinite(out_train).all()


def test_return_embeddings():
    """Verify return_embeddings=True returns dictionary with expected keys and shapes."""
    model = VardhanV3(num_classes=4)
    model.eval()

    x = torch.randn(3, 1, 2048)
    with torch.no_grad():
        logits, emb_dict = model(x, return_embeddings=True)

    assert logits.shape == (3, 4)
    assert isinstance(emb_dict, dict)
    assert "h_temporal" in emb_dict
    assert "h_spectral" in emb_dict
    assert "x_spectral_tensor" in emb_dict

    assert emb_dict["h_temporal"].shape == (3, 80)
    assert emb_dict["h_spectral"].shape == (3, 80)
    assert emb_dict["x_spectral_tensor"].shape == (3, 3, 1024)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 32])
def test_variable_batch_sizes(batch_size):
    """Verify model operates correctly on various batch sizes."""
    model = VardhanV3(num_classes=4)
    model.eval()

    x = torch.randn(batch_size, 1, 2048)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 4)
    assert torch.isfinite(out).all()


def test_model_factory_registration():
    """Verify VardhanV3 is registered and instantiable via get_model."""
    m1 = get_model("vardhan_v3")
    assert isinstance(m1, VardhanV3)
    assert sum(p.numel() for p in m1.parameters()) == EXPECTED_V3_PARAM_COUNT

    m2 = get_model("v3")
    assert isinstance(m2, VardhanV3)

    m3 = get_model("vardhanv3")
    assert isinstance(m3, VardhanV3)
