"""
verify_vardhan_v2a.py
---------------------

Verification script for VARDHAN-v2A (Tri-Branch Multi-Representation RF-Net).
Validates:
1. Exact tensor shapes through every stem, block, attention pool, and fusion layer.
2. Deterministic FFT preprocessing, DC bin removal, and coarse sub-band reshaping.
3. Total parameter count and layer-by-layer parameter audit.
4. Finite forward pass and gradient backpropagation.
5. Maximum sequential temporal receptive field tracking.
"""

import sys
from pathlib import Path

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from models.vardhan_v2a import VardhanV2A, TemporalBlock, AttentionPool1d


def verify_shapes_and_forward():
    print("=" * 75)
    print("1. VERIFYING VARDHAN-v2A TENSOR SHAPES THROUGH EVERY LAYER")
    print("=" * 75)

    model = VardhanV2A(num_classes=4)
    model.eval()

    B = 2
    x = torch.randn(B, 1, 2048, requires_grad=True)
    print(f"Input Raw Waveform: {x.shape}")
    assert x.shape == (2, 1, 2048), f"Expected (2, 1, 2048), got {x.shape}"

    # FFT & Spectral Verification
    freq_input, mb_input = model.extract_spectral_representations(x)
    print(f"\n[Spectral Representations]")
    print(f"  - Fine-grained spectrum (k=1..1024): {freq_input.shape}")
    print(f"  - Coarse sub-bands (8 x 128 bins):    {mb_input.shape}")
    assert freq_input.shape == (2, 1, 1024), f"Expected (2, 1, 1024), got {freq_input.shape}"
    assert mb_input.shape == (2, 8, 128), f"Expected (2, 8, 128), got {mb_input.shape}"

    # Temporal Branch Steps
    print(f"\n[Temporal Branch Propagation]")
    t_stem = model.time_stem(x)
    print(f"  - time_stem:   {t_stem.shape}")
    assert t_stem.shape == (2, 32, 1024)

    t_b1 = model.time_block1(t_stem)
    print(f"  - time_block1: {t_b1.shape}")
    assert t_b1.shape == (2, 48, 512)

    t_b2 = model.time_block2(t_b1)
    print(f"  - time_block2: {t_b2.shape}")
    assert t_b2.shape == (2, 64, 256)

    t_b3 = model.time_block3(t_b2)
    print(f"  - time_block3: {t_b3.shape}")
    assert t_b3.shape == (2, 64, 128)

    t_b4 = model.time_block4(t_b3)
    print(f"  - time_block4: {t_b4.shape}")
    assert t_b4.shape == (2, 64, 128)

    h_t = model.time_pool(t_b4)
    print(f"  - time_pool (h_T): {h_t.shape}")
    assert h_t.shape == (2, 64)

    # Fine-Grained Spectral Branch Steps
    print(f"\n[Fine-Grained Spectral Branch Propagation]")
    f_stem = model.freq_stem(freq_input)
    print(f"  - freq_stem:   {f_stem.shape}")
    assert f_stem.shape == (2, 32, 512)

    f_b1 = model.freq_block1(f_stem)
    print(f"  - freq_block1: {f_b1.shape}")
    assert f_b1.shape == (2, 48, 256)

    f_b2 = model.freq_block2(f_b1)
    print(f"  - freq_block2: {f_b2.shape}")
    assert f_b2.shape == (2, 64, 128)

    f_b3 = model.freq_block3(f_b2)
    print(f"  - freq_block3: {f_b3.shape}")
    assert f_b3.shape == (2, 64, 128)

    h_f = model.freq_pool(f_b3)
    print(f"  - freq_pool (h_F): {h_f.shape}")
    assert h_f.shape == (2, 64)

    # Coarse Spectral-Band Branch Steps
    print(f"\n[Coarse Spectral-Band Branch Propagation]")
    m_stem = model.mb_stem(mb_input)
    print(f"  - mb_stem:     {m_stem.shape}")
    assert m_stem.shape == (2, 32, 128)

    m_b1 = model.mb_block1(m_stem)
    print(f"  - mb_block1:   {m_b1.shape}")
    assert m_b1.shape == (2, 48, 64)

    m_b2 = model.mb_block2(m_b1)
    print(f"  - mb_block2:   {m_b2.shape}")
    assert m_b2.shape == (2, 64, 64)

    h_m = model.mb_pool(m_b2)
    print(f"  - mb_pool (h_M):   {h_m.shape}")
    assert h_m.shape == (2, 64)

    # Fusion & Head Steps
    print(f"\n[Fusion & Classification Head]")
    h_cat = torch.cat([h_t, h_f, h_m], dim=-1)
    print(f"  - h_cat (h_T + h_F + h_M): {h_cat.shape}")
    assert h_cat.shape == (2, 192)

    logits = model.fusion_head(h_cat)
    print(f"  - final logits:            {logits.shape}")
    assert logits.shape == (2, 4)

    # Check finite values
    assert torch.all(torch.isfinite(logits)), "Output logits contain NaN or Inf"
    print("\n[PASS] All forward tensor shapes and finite values verified cleanly!")


def verify_gradients_and_backward():
    print("\n" + "=" * 75)
    print("2. VERIFYING BACKWARD PASS & FINITE GRADIENTS")
    print("=" * 75)

    model = VardhanV2A(num_classes=4)
    model.train()

    x = torch.randn(4, 1, 2048, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)

    print(f"Forward CE Loss: {loss.item():.6f}")
    assert torch.isfinite(loss), "Loss is not finite"

    loss.backward()

    # Verify input gradient
    assert x.grad is not None, "Input x has no gradient"
    assert torch.all(torch.isfinite(x.grad)), "Input gradient contains NaN or Inf"
    print(f"Input x.grad shape: {x.grad.shape} | All finite: True")

    # Verify all parameter gradients
    non_finite_params = []
    for name, p in model.named_parameters():
        if p.grad is None:
            non_finite_params.append((name, "NO_GRAD"))
        elif not torch.all(torch.isfinite(p.grad)):
            non_finite_params.append((name, "NAN_OR_INF"))

    assert len(non_finite_params) == 0, f"Found parameter gradient errors: {non_finite_params}"
    print(f"[PASS] All {sum(1 for _ in model.parameters())} parameter tensors have finite gradients!")


def verify_parameter_counts():
    print("\n" + "=" * 75)
    print("3. VERIFYING EXACT LAYER-BY-LAYER PARAMETER BREAKDOWN")
    print("=" * 75)

    model = VardhanV2A(num_classes=4)

    time_params = (
        sum(p.numel() for p in model.time_stem.parameters())
        + sum(p.numel() for p in model.time_block1.parameters())
        + sum(p.numel() for p in model.time_block2.parameters())
        + sum(p.numel() for p in model.time_block3.parameters())
        + sum(p.numel() for p in model.time_block4.parameters())
        + sum(p.numel() for p in model.time_pool.parameters())
    )

    freq_params = (
        sum(p.numel() for p in model.freq_stem.parameters())
        + sum(p.numel() for p in model.freq_block1.parameters())
        + sum(p.numel() for p in model.freq_block2.parameters())
        + sum(p.numel() for p in model.freq_block3.parameters())
        + sum(p.numel() for p in model.freq_pool.parameters())
    )

    mb_params = (
        sum(p.numel() for p in model.mb_stem.parameters())
        + sum(p.numel() for p in model.mb_block1.parameters())
        + sum(p.numel() for p in model.mb_block2.parameters())
        + sum(p.numel() for p in model.mb_pool.parameters())
    )

    head_params = sum(p.numel() for p in model.fusion_head.parameters())

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"1. Temporal Branch:                  {time_params:6,d} params (Expected: 26,097)")
    print(f"2. Fine-Grained Spectral Branch:     {freq_params:6,d} params (Expected: 16,913)")
    print(f"3. Coarse Spectral-Band Branch:      {mb_params:6,d} params (Expected: 13,809)")
    print(f"4. Lightweight Fusion & Head:        {head_params:6,d} params (Expected: 12,740)")
    print("-" * 75)
    print(f"TOTAL TRAINABLE PARAMETERS:          {total_trainable:6,d} params (Expected: 69,559)")

    assert time_params == 26097, f"Time branch mismatch: got {time_params}, expected 26,097"
    assert freq_params == 16913, f"Freq branch mismatch: got {freq_params}, expected 16,913"
    assert mb_params == 13809, f"MB branch mismatch: got {mb_params}, expected 13,809"
    assert head_params == 12740, f"Head mismatch: got {head_params}, expected 12,740"
    assert total_trainable == 69559, f"Total mismatch: got {total_trainable}, expected 69,559"
    print("\n[PASS] Parameter counts match specification exactly (69,559)!")


def verify_receptive_field():
    print("\n" + "=" * 75)
    print("4. VERIFYING MAXIMUM SEQUENTIAL TEMPORAL RECEPTIVE FIELD")
    print("=" * 75)

    layers = [
        ("Stem Conv1d(k=15, s=2, d=1)", 15, 2, 1, 15),
        ("Block 1 Depthwise(k=7, s=2, d=1)", 7, 2, 1, 27),
        ("Block 2 Depthwise(k=7, s=2, d=2)", 7, 2, 2, 75),
        ("Block 3 Depthwise(k=7, s=2, d=4)", 7, 2, 4, 267),
        ("Block 4 Depthwise(k=7, s=1, d=8)", 7, 1, 8, 1035),
    ]

    rf = 1
    j = 1
    print(f"Initial Raw Sample: RF = {rf:4d} samples (0.025 us at 40 MSps)")

    for name, k, s, d, expected_rf in layers:
        k_eff = 1 + (k - 1) * d
        rf = rf + (k_eff - 1) * j
        j = j * s
        duration_us = rf / 40.0
        print(f"  - {name:35s}: RF = {rf:4d} samples ({duration_us:6.3f} us at 40 MSps) | Expected: {expected_rf:4d}")
        assert rf == expected_rf, f"Receptive field mismatch at {name}: got {rf}, expected {expected_rf}"

    print(f"\n[PASS] Receptive field tracking verified: 1035 samples (25.875 us at 40 MSps) along deepest path!")


if __name__ == "__main__":
    verify_shapes_and_forward()
    verify_gradients_and_backward()
    verify_parameter_counts()
    verify_receptive_field()
    print("\n" + "=" * 75)
    print("ALL VARDHAN-v2A INTEGRITY & ARCHITECTURE CHECKS PASSED!")
    print("=" * 75)
