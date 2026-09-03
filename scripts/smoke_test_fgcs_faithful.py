"""
smoke_test_fgcs_faithful.py
---------------------------

Lightweight real-data smoke test for EXP_FGCS_FAITHFUL preprocessing,
dataset batching, model forward pass, and loss computation.

Does NOT train any model.
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn as nn

from preprocessing.fgcs_faithful import (
    FAITHFUL_FGCS_CLASS_TO_INDEX,
    FAITHFUL_FGCS_INDEX_TO_CLASS,
    process_faithful_fgcs_segment,
    normalize_global_max,
)
from models.fgcs_faithful_dnn import FGCSFaithfulDNN


def extract_real_segment(rar_path: str, inner_csv: str, n_samples: int = 100000) -> np.ndarray:
    """Extract first n_samples from raw DroneRF CSV inside RAR archive using tar."""
    cmd = ["tar", "-xOf", rar_path, inner_csv]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    line = p.stdout.readline(n_samples * 15 + 1000).decode("utf-8", errors="ignore")
    vals = np.fromstring(line.rsplit(",", 1)[0], sep=",", dtype=np.float64)[:n_samples]
    p.kill()
    if len(vals) < n_samples:
        vals = np.pad(vals, (0, n_samples - len(vals)))
    return vals


def run_smoke_test():
    print("=" * 75)
    print("FAITHFUL FGCS REAL-DATA SMOKE TEST (EXP_FGCS_FAITHFUL)")
    print("=" * 75)

    base_raw = Path(r"c:\Users\subsa\Desktop\DRONE\Vardan\data\raw\DroneRF\DroneRF")

    test_cases = [
        {
            "bui": "00000",
            "class": "Background RF activities",
            "label": 0,
            "l_rar": base_raw / "Background RF activites" / "RF Data_00000_L1.rar",
            "l_file": "RF Data_00000_L1/00000L_0.csv",
            "h_rar": base_raw / "Background RF activites" / "RF Data_00000_H1.rar",
            "h_file": "00000H_0.csv",
        },
        {
            "bui": "10000",
            "class": "Bebop drone",
            "label": 1,
            "l_rar": base_raw / "Bepop drone" / "RF Data_10000_L.rar",
            "l_file": "RF Data_10000_L/10000L_0.csv",
            "h_rar": base_raw / "Bepop drone" / "RF Data_10000_H.rar",
            "h_file": "RF Data_10000_H/10000H_0.csv",
        },
        {
            "bui": "10100",
            "class": "AR drone",
            "label": 2,
            "l_rar": base_raw / "AR drone" / "RF Data_10100_L.rar",
            "l_file": "RF Data_10100_L/10100L_0.csv",
            "h_rar": base_raw / "AR drone" / "RF Data_10100_H.rar",
            "h_file": "RF Data_10100_H/10100H_0.csv",
        },
        {
            "bui": "11000",
            "class": "Phantom drone",
            "label": 3,
            "l_rar": base_raw / "Phantom drone" / "RF Data_11000_L1.rar",
            "l_file": "11000L_0.csv",
            "h_rar": base_raw / "Phantom drone" / "RF Data_11000_H.rar",
            "h_file": "RF Data_11000_H/11000H_0.csv",
        },
    ]

    features_list = []
    labels_list = []

    print("\n--- 1. Real-Data Segment Extraction & Transformation ---")
    for tc in test_cases:
        bui = tc["bui"]
        cls_name = tc["class"]
        lbl = tc["label"]

        # Read 100,000 samples from real data
        x_raw = extract_real_segment(str(tc["l_rar"]), tc["l_file"], n_samples=100000)
        y_raw = extract_real_segment(str(tc["h_rar"]), tc["h_file"], n_samples=100000)

        # Generate faithful 2048-dim feature
        feat = process_faithful_fgcs_segment(x_raw, y_raw, q=10, m=2048)

        # Verifications
        assert feat.shape == (2048,), f"Shape mismatch: {feat.shape}"
        assert np.all(np.isfinite(feat)), "Feature contains NaN or Inf"
        assert np.all(feat >= 0), "Feature power contains negative values"

        # Q scaling calculation for reporting
        x_det = x_raw - np.mean(x_raw)
        y_det = y_raw - np.mean(y_raw)
        fft_x = np.fft.fft(x_det[:2048], n=2048)
        fft_y = np.fft.fft(y_det[:2048], n=2048)
        xf = np.abs(np.fft.fftshift(fft_x))[1024:]
        yf = np.abs(np.fft.fftshift(fft_y))[1024:]
        q_scale = float(np.mean(xf[-10:]) / (np.mean(yf[:10]) + 1e-12))

        features_list.append(feat)
        labels_list.append(lbl)

        print(f"\n  BUI:        {bui} ({cls_name}, Label {lbl})")
        print(f"  L File:     {Path(tc['l_file']).name} (100,000 samples)")
        print(f"  H File:     {Path(tc['h_file']).name} (100,000 samples)")
        print(f"  Q Factor:   {q_scale:.6f}")
        print(f"  Shape:      {feat.shape} (L: bins 0..1023, H: bins 1024..2047)")
        print(f"  Min Value:  {np.min(feat):.4e}")
        print(f"  Max Value:  {np.max(feat):.4e}")
        print(f"  Mean Value: {np.mean(feat):.4e}")

    # 2. Normalization Check
    print("\n--- 2. Per-BUI / Mode Global Normalization Check ---")
    batch_raw = np.stack(features_list, axis=0)
    batch_norm, g_max = normalize_global_max(batch_raw)
    assert np.isclose(np.max(batch_norm), 1.0), "Normalized max must equal 1.0"
    assert np.min(batch_norm) >= 0.0, "Normalized min must be non-negative"
    print(f"  Global Max Scalar: {g_max:.4e}")
    print(f"  Batch Min after Norm: {np.min(batch_norm):.4e}")
    print(f"  Batch Max after Norm: {np.max(batch_norm):.4e}")
    print("  [OK] Normalization scales entire feature matrix cleanly to [0.0, 1.0].")

    # 3. Batch Tensor Shapes
    print("\n--- 3. PyTorch Tensor Batch Shapes ---")
    X_tensor = torch.tensor(batch_norm, dtype=torch.float32)
    y_tensor = torch.tensor(labels_list, dtype=torch.long)
    print(f"  X shape: {X_tensor.shape} (batch_size=4, in_features=2048)")
    print(f"  y shape: {y_tensor.shape} (batch_size=4, labels={labels_list})")
    assert X_tensor.shape == (4, 2048)
    assert y_tensor.shape == (4,)
    print("  [OK] Tensor shapes verified.")

    # 4. Model Forward Pass
    print("\n--- 4. Model Forward Pass ---")
    model = FGCSFaithfulDNN(in_features=2048, num_classes=4, architecture_mode="code")
    model.eval()

    logits = model(X_tensor, return_logits=True)
    probs = model(X_tensor, return_logits=False)

    print(f"  Logits shape:      {logits.shape}")
    print(f"  Sigmoid out shape: {probs.shape}")
    print(f"  Sample Probs (row 0):\n    {probs[0].detach().numpy()}")
    assert logits.shape == (4, 4)
    assert probs.shape == (4, 4)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    print("  [OK] Model forward pass verified.")

    # 5. Loss Calculation
    print("\n--- 5. Loss Calculation in Both Modes ---")
    
    # Mode A: Sigmoid + MSE (Exact match to released Classification.py)
    # y one-hot encoding
    y_one_hot = torch.zeros(4, 4)
    y_one_hot.scatter_(1, y_tensor.unsqueeze(1), 1.0)
    mse_loss_fn = nn.MSELoss()
    loss_mse = mse_loss_fn(probs, y_one_hot)
    print(f"  Mode A (Sigmoid + MSE):       Loss = {loss_mse.item():.6f}")
    assert torch.isfinite(loss_mse)

    # Mode B: Logits + CrossEntropy (Standard PyTorch mode)
    ce_loss_fn = nn.CrossEntropyLoss()
    loss_ce = ce_loss_fn(logits, y_tensor)
    print(f"  Mode B (Logits + CrossEntropy): Loss = {loss_ce.item():.6f}")
    assert torch.isfinite(loss_ce)
    print("  [OK] Loss computation verified in both modes.")

    print("\n" + "=" * 75)
    print("SMOKE TEST COMPLETED SUCCESSFULLY (ZERO MODEL TRAINING PERFORMED)")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
