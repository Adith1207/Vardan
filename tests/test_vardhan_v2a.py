"""
test_vardhan_v2a.py
-------------------

Pytest test suite for VARDHAN-v2A (Tri-Branch Multi-Representation RF-Net).
Covers model construction, tensor shapes across all branches, FFT DC-removal,
attention pooling properties, exact parameter counts, receptive field,
finite gradients, and multi-batch size execution.
"""

import sys
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
import torch
import torch.nn as nn
from models.vardhan_v2a import VardhanV2A, TemporalBlock, AttentionPool1d


def test_model_construction():
    """Verify clean instantiation of VardhanV2A."""
    model = VardhanV2A(num_classes=4)
    assert isinstance(model, nn.Module)
    assert model.num_classes == 4


def test_input_output_shapes_batch_1_and_32():
    """Verify input (B, 1, 2048) -> output (B, 4) across batch sizes 1, 4, 32."""
    model = VardhanV2A(num_classes=4)
    model.eval()

    for bs in [1, 4, 32]:
        x = torch.randn(bs, 1, 2048)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (bs, 4)
        assert torch.all(torch.isfinite(out))


def test_fft_extraction_and_dc_removal():
    """Verify that FFT produces 1025 bins, drops DC bin (k=0), and yields 1024 positive bins."""
    model = VardhanV2A(num_classes=4)

    # Signal with known DC offset
    x = torch.randn(2, 1, 2048) + 10.0

    # 1. Per-sample detrending
    x_detrend = x - x.mean(dim=-1, keepdim=True)
    assert torch.allclose(x_detrend.mean(dim=-1), torch.zeros(2, 1), atol=1e-5)

    # 2. Raw rfft
    fft_raw = torch.fft.rfft(x_detrend.squeeze(1), n=2048)
    assert fft_raw.shape == (2, 1025)

    # 3. Model method extraction
    freq_input, mb_input = model.extract_spectral_representations(x)

    # Fine-grained view: (B, 1, 1024)
    assert freq_input.shape == (2, 1, 1024)
    # Coarse sub-band view: (B, 8, 128)
    assert mb_input.shape == (2, 8, 128)

    # Verify both views originate from exact same 1024-bin tensor
    assert torch.allclose(freq_input.squeeze(1), mb_input.view(2, 1024))
    assert torch.all(torch.isfinite(freq_input))
    assert torch.all(torch.isfinite(mb_input))


def test_temporal_branch_shapes():
    """Verify intermediate layer shapes through the temporal branch."""
    model = VardhanV2A(num_classes=4)
    x = torch.randn(2, 1, 2048)

    s = model.time_stem(x)
    assert s.shape == (2, 32, 1024)

    b1 = model.time_block1(s)
    assert b1.shape == (2, 48, 512)

    b2 = model.time_block2(b1)
    assert b2.shape == (2, 64, 256)

    b3 = model.time_block3(b2)
    assert b3.shape == (2, 64, 128)

    b4 = model.time_block4(b3)
    assert b4.shape == (2, 64, 128)

    h_t = model.time_pool(b4)
    assert h_t.shape == (2, 64)


def test_fine_grained_spectral_branch_shapes():
    """Verify intermediate layer shapes through the fine-grained spectral branch."""
    model = VardhanV2A(num_classes=4)
    freq_input = torch.randn(2, 1, 1024)

    s = model.freq_stem(freq_input)
    assert s.shape == (2, 32, 512)

    b1 = model.freq_block1(s)
    assert b1.shape == (2, 48, 256)

    b2 = model.freq_block2(b1)
    assert b2.shape == (2, 64, 128)

    b3 = model.freq_block3(b2)
    assert b3.shape == (2, 64, 128)

    h_f = model.freq_pool(b3)
    assert h_f.shape == (2, 64)


def test_coarse_spectral_band_branch_shapes():
    """Verify intermediate layer shapes through the coarse spectral-band branch."""
    model = VardhanV2A(num_classes=4)
    mb_input = torch.randn(2, 8, 128)

    s = model.mb_stem(mb_input)
    assert s.shape == (2, 32, 128)

    b1 = model.mb_block1(s)
    assert b1.shape == (2, 48, 64)

    b2 = model.mb_block2(b1)
    assert b2.shape == (2, 64, 64)

    h_m = model.mb_pool(b2)
    assert h_m.shape == (2, 64)


def test_attention_pool1d_weights_sum_to_one():
    """Verify that AttentionPool1d softmax attention weights sum exactly to 1.0 along sequence."""
    pool = AttentionPool1d(in_dim=64, hidden_dim=32)
    x = torch.randn(4, 64, 128)

    # Compute internal attention weights
    weights = pool.attn(x)  # (4, 1, 128)
    weight_sums = weights.sum(dim=-1)  # (4, 1)

    assert weights.shape == (4, 1, 128)
    assert torch.allclose(weight_sums, torch.ones(4, 1), atol=1e-6)
    assert torch.all(weights >= 0.0)

    out = pool(x)
    assert out.shape == (4, 64)


def test_exact_parameter_counts():
    """Verify exact parameter count matches design specification: 69,559."""
    model = VardhanV2A(num_classes=4)

    time_p = (
        sum(p.numel() for p in model.time_stem.parameters())
        + sum(p.numel() for p in model.time_block1.parameters())
        + sum(p.numel() for p in model.time_block2.parameters())
        + sum(p.numel() for p in model.time_block3.parameters())
        + sum(p.numel() for p in model.time_block4.parameters())
        + sum(p.numel() for p in model.time_pool.parameters())
    )
    freq_p = (
        sum(p.numel() for p in model.freq_stem.parameters())
        + sum(p.numel() for p in model.freq_block1.parameters())
        + sum(p.numel() for p in model.freq_block2.parameters())
        + sum(p.numel() for p in model.freq_block3.parameters())
        + sum(p.numel() for p in model.freq_pool.parameters())
    )
    mb_p = (
        sum(p.numel() for p in model.mb_stem.parameters())
        + sum(p.numel() for p in model.mb_block1.parameters())
        + sum(p.numel() for p in model.mb_block2.parameters())
        + sum(p.numel() for p in model.mb_pool.parameters())
    )
    head_p = sum(p.numel() for p in model.fusion_head.parameters())

    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert time_p == 26097
    assert freq_p == 16913
    assert mb_p == 13809
    assert head_p == 12740
    assert total_p == 69559


def test_finite_forward_and_backward_pass():
    """Verify forward and backward gradients are finite with no NaNs/Infs."""
    model = VardhanV2A(num_classes=4)
    model.train()

    x = torch.randn(4, 1, 2048, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)

    assert torch.isfinite(loss)
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.all(torch.isfinite(p.grad)), f"Parameter {name} gradient is not finite"


def test_receptive_field_calculation():
    """Verify mathematical receptive field along deepest temporal path matches 1,035 samples."""
    layers = [
        ("stem", 15, 2, 1, 15),
        ("block1", 7, 2, 1, 27),
        ("block2", 7, 2, 2, 75),
        ("block3", 7, 2, 4, 267),
        ("block4", 7, 1, 8, 1035),
    ]

    rf = 1
    j = 1
    for name, k, s, d, expected_rf in layers:
        k_eff = 1 + (k - 1) * d
        rf = rf + (k_eff - 1) * j
        j = j * s
        assert rf == expected_rf
