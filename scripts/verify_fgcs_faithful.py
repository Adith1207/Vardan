"""
verify_fgcs_faithful.py
-----------------------

Standalone verification script for EXP_FGCS_FAITHFUL preprocessing and pairing pipeline.

Checks:
1. Complete L/H pairing across all 4 classes (227 pairs expected).
2. 100 segments per pair -> 22,700 total segment capacity.
3. Shape verification: output is (2048,).
4. Finite values, non-negative power, no NaN/Inf.
5. Q=10 boundary matching behavior.
6. Global maximum normalization behavior.
7. Faithful 4-class label ordering (0: Background, 1: Bebop, 2: AR, 3: Phantom).
8. Model forward pass verification.
"""

import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from preprocessing.fgcs_faithful import (
    FAITHFUL_FGCS_CLASS_TO_INDEX,
    FAITHFUL_FGCS_INDEX_TO_CLASS,
    BUI_TO_CLASS,
    parse_dronerf_filename,
    process_faithful_fgcs_segment,
    normalize_global_max,
)
from data.fgcs_faithful_loader import (
    discover_and_pair_dronerf_files,
    build_faithful_manifest,
    FGCSFaithfulLazyDataset,
)
from models.fgcs_faithful_dnn import FGCSFaithfulDNN


def run_verification() -> bool:
    print("=" * 70)
    print("STARTING FAITHFUL FGCS PREPROCESSING VERIFICATION (EXP_FGCS_FAITHFUL)")
    print("=" * 70)

    # 1. Verify Filename Parser & Label Ordering
    print("\n--- 1. Verifying Filename Parser and Faithful Label Mapping ---")
    test_cases = [
        ("00000L_0.csv", "00000", "L", 0, "Background RF activities", 0),
        ("00000H_40.csv", "00000", "H", 40, "Background RF activities", 0),
        ("10000L_5.csv", "10000", "L", 5, "Bebop drone", 1),
        ("10011H_20.csv", "10011", "H", 20, "Bebop drone", 1),
        ("10100L_10.csv", "10100", "L", 10, "AR drone", 2),
        ("10111H_17.csv", "10111", "H", 17, "AR drone", 2),
        ("11000L_0.csv", "11000", "L", 0, "Phantom drone", 3),
        ("11000H_20.csv", "11000", "H", 20, "Phantom drone", 3),
    ]
    for fname, exp_bui, exp_rec, exp_seg, exp_cls, exp_lbl in test_cases:
        p = parse_dronerf_filename(fname)
        assert p is not None, f"Failed to parse {fname}"
        assert p["bui"] == exp_bui, f"BUI mismatch for {fname}: got {p['bui']}, expected {exp_bui}"
        assert p["receiver"] == exp_rec, f"Receiver mismatch for {fname}"
        assert p["file_segment_num"] == exp_seg, f"Seg num mismatch for {fname}"
        assert p["drone_class"] == exp_cls, f"Class mismatch for {fname}"
        assert p["faithful_label"] == exp_lbl, f"Label mismatch for {fname}: got {p['faithful_label']}, expected {exp_lbl}"
        print(f"  [OK] {fname:15s} -> BUI={p['bui']}, Band={p['receiver']}, Seg={p['file_segment_num']:2d}, Class='{p['drone_class']}', Label={p['faithful_label']}")

    # Verify canonical mapping constants
    assert FAITHFUL_FGCS_CLASS_TO_INDEX["Background RF activities"] == 0
    assert FAITHFUL_FGCS_CLASS_TO_INDEX["Bebop drone"] == 1
    assert FAITHFUL_FGCS_CLASS_TO_INDEX["AR drone"] == 2
    assert FAITHFUL_FGCS_CLASS_TO_INDEX["Phantom drone"] == 3
    print("  [OK] Faithful 4-class label ordering verified: 0=Background, 1=Bebop, 2=AR, 3=Phantom")

    # 2. Verify L/H Pairing Discovery
    print("\n--- 2. Verifying L/H File Pairing Discovery ---")
    df_pairs = discover_and_pair_dronerf_files()
    print(f"  Total recording pairs discovered: {len(df_pairs)}")
    class_counts = df_pairs["drone_class"].value_counts().to_dict()
    print("  Discovered pairs per class:")
    for cls_name, cnt in class_counts.items():
        print(f"    - {cls_name:25s}: {cnt:3d} pairs")

    assert len(df_pairs) == 227, f"Expected 227 pairs, found {len(df_pairs)}"
    assert class_counts.get("Background RF activities", 0) == 41, "Expected 41 Background pairs"
    assert class_counts.get("Bebop drone", 0) == 84, "Expected 84 Bebop pairs"
    assert class_counts.get("AR drone", 0) == 81, "Expected 81 AR pairs"
    assert class_counts.get("Phantom drone", 0) == 21, "Expected 21 Phantom pairs"
    print("  [OK] Exactly 227 synchronized L/H recording pairs verified across all 4 classes.")

    # 3. Verify 22,700-Segment Capacity Manifest
    print("\n--- 3. Verifying 22,700-Segment Manifest Capacity ---")
    df_manifest = build_faithful_manifest(segments_per_pair=100)
    print(f"  Total manifest segment rows: {len(df_manifest)} (227 pairs * 100 segments)")
    assert len(df_manifest) == 22700, f"Expected 22,700 segment entries, got {len(df_manifest)}"
    manifest_counts = df_manifest["drone_class"].value_counts().to_dict()
    for cls_name, cnt in manifest_counts.items():
        print(f"    - {cls_name:25s}: {cnt:5d} segments ({cnt/len(df_manifest)*100:.2f}%)")
    print("  [OK] 22,700-segment capacity verified.")

    # 4. Mathematical Preprocessing Verification on Synthetic Signals
    print("\n--- 4. Verifying Signal Processing Steps (FFT, Shift, Q=10 Scaling, Power) ---")
    rng = np.random.RandomState(42)
    x_test = rng.randn(100000).astype(np.float32)
    y_test = rng.randn(100000).astype(np.float32) * 0.5  # Different amplitude for H

    power_vec = process_faithful_fgcs_segment(x_test, y_test, q=10, m=2048)
    
    assert power_vec.shape == (2048,), f"Shape mismatch: expected (2048,), got {power_vec.shape}"
    assert np.all(np.isfinite(power_vec)), "Non-finite values found in power spectrum"
    assert np.all(power_vec >= 0), "Power values must be non-negative"
    print(f"  [OK] Output vector shape: {power_vec.shape}")
    print(f"  [OK] All 2048 elements are finite and non-negative (min: {np.min(power_vec):.4e}, max: {np.max(power_vec):.4e})")

    # Step-by-step validation of positive half and Q=10 scaling
    x_detrend = x_test - np.mean(x_test)
    y_detrend = y_test - np.mean(y_test)
    fft_x = np.fft.fft(x_detrend[:2048], n=2048)
    fft_y = np.fft.fft(y_detrend[:2048], n=2048)
    xf_step = np.abs(np.fft.fftshift(fft_x))[1024:]
    yf_step = np.abs(np.fft.fftshift(fft_y))[1024:]
    c_step = np.mean(xf_step[-10:]) / (np.mean(yf_step[:10]) + 1e-12)
    expected_stitched = (np.concatenate([xf_step, c_step * yf_step]) ** 2).astype(np.float32)

    assert np.allclose(power_vec, expected_stitched, rtol=1e-4, atol=1e-4), "Step-by-step computation mismatch"
    print(f"  [OK] L-band contributes exactly 1024 bins (2.40 - 2.44 GHz)")
    print(f"  [OK] H-band contributes exactly 1024 bins (2.44 - 2.48 GHz)")
    print(f"  [OK] Boundary matching scale factor c = {c_step:.4f} verified using Q=10 points")

    # 5. Global Normalization Verification
    print("\n--- 5. Verifying Global Normalization ---")
    mock_matrix = np.stack([power_vec, power_vec * 0.5, power_vec * 2.0])
    norm_matrix, g_max = normalize_global_max(mock_matrix)
    assert np.isclose(g_max, np.max(mock_matrix)), "Global max mismatch"
    assert np.max(norm_matrix) == 1.0, "Max of globally normalized matrix must be 1.0"
    assert np.min(norm_matrix) >= 0.0, "Min must be non-negative"
    print(f"  [OK] Global maximum scaling verified: matrix scaled by global scalar {g_max:.4e}")

    # 6. Lazy Dataset Verification (Mock Mode)
    print("\n--- 6. Verifying Lazy Dataset Pipeline ---")
    mock_manifest = df_manifest.head(20).copy()
    dataset = FGCSFaithfulLazyDataset(mock_manifest, mock=True, global_max=g_max)
    assert len(dataset) == 20
    sample_tensor, sample_label = dataset[0]
    assert sample_tensor.shape == (2048,)
    assert 0 <= sample_label <= 3
    print(f"  [OK] Dataset item 0: Tensor shape={sample_tensor.shape}, Label={sample_label} ({FAITHFUL_FGCS_INDEX_TO_CLASS[sample_label]})")

    # 7. Model Architecture Forward Pass
    print("\n--- 7. Verifying Model Architecture Forward Pass ---")
    model_code = FGCSFaithfulDNN(in_features=2048, num_classes=4, architecture_mode="code")
    model_paper = FGCSFaithfulDNN(in_features=2048, num_classes=4, architecture_mode="paper")
    
    import torch
    dummy_in = torch.randn(10, 2048)
    out_code_logits = model_code(dummy_in, return_logits=True)
    out_code_sig = model_code(dummy_in, return_logits=False)
    out_paper_logits = model_paper(dummy_in, return_logits=True)
    
    assert out_code_logits.shape == (10, 4), f"Code model shape mismatch: {out_code_logits.shape}"
    assert out_code_sig.shape == (10, 4)
    assert out_paper_logits.shape == (10, 4)
    assert torch.all((out_code_sig >= 0.0) & (out_code_sig <= 1.0)), "Sigmoid output range [0, 1] violated"
    print("  [OK] FGCSFaithfulDNN ('code' mode: 128->128->128->4) forward pass verified.")
    print("  [OK] FGCSFaithfulDNN ('paper' mode: 256->128->64->4) forward pass verified.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (ZERO MODEL TRAINING PERFORMED)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
