"""
fgcs_faithful.py
----------------

Faithful reproduction of the original Al-Sa'd DroneRF preprocessing pipeline
from Matlab/Main_1_Data_aggregation.m and Matlab/Main_2_Data_labeling.m.

Pipeline Steps:
1. Synchronized L/H File Pairing:
   - Pairs low-band (L: 2.40 - 2.44 GHz) and high-band (H: 2.44 - 2.48 GHz) recordings.
2. 100,000-Sample Segmentation:
   - Divides 10,000,000-sample files into non-overlapping 100,000-sample segments (100 segments per pair).
3. DC / Mean Removal:
   - x_detrend = x_seg - mean(x_seg)
   - y_detrend = y_seg - mean(y_seg)
4. 2048-Point FFT and Shift:
   - Computes 2048-point FFT on detrended signal.
   - Applies fftshift and retains positive-frequency half (1024 bins each).
5. Q=10 Boundary Matching & Concatenation:
   - c = mean(xf[end-9:end]) / mean(yf[1:10])
   - stitched = [xf ; c * yf]  -> 2048 frequency bins spanning 2.40 - 2.48 GHz.
6. Power Conversion:
   - feature = stitched ** 2
7. Global Class Normalization:
   - feature = feature / max(max(Data))
8. Faithful 4-Class Ordering:
   - 0: Background RF activities (BUI: 00000)
   - 1: Bebop drone (BUI: 10000, 10001, 10010, 10011)
   - 2: AR drone (BUI: 10100, 10101, 10110, 10111)
   - 3: Phantom drone (BUI: 11000)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# Original FGCS 4-class label mapping (Al-Sa'd 2019 / Main_2_Data_labeling.m)
FAITHFUL_FGCS_CLASS_TO_INDEX: Dict[str, int] = {
    "Background": 0,
    "Background RF activites": 0,
    "Background RF activities": 0,
    "00000": 0,
    "Bebop": 1,
    "Bepop": 1,
    "Bebop drone": 1,
    "Bepop drone": 1,
    "Parrot Bebop": 1,
    "AR": 2,
    "AR Drone": 2,
    "AR drone": 2,
    "Parrot AR": 2,
    "Phantom": 3,
    "Phantom drone": 3,
    "DJI Phantom": 3,
    "DJI Phantom 3": 3,
}

FAITHFUL_FGCS_INDEX_TO_CLASS: Dict[int, str] = {
    0: "Background RF activities",
    1: "Bebop drone",
    2: "AR drone",
    3: "Phantom drone",
}

# Mapping from BUI prefix to canonical class name and faithful label index
BUI_TO_CLASS: Dict[str, Tuple[str, int]] = {
    "00000": ("Background RF activities", 0),
    "10000": ("Bebop drone", 1),
    "10001": ("Bebop drone", 1),
    "10010": ("Bebop drone", 1),
    "10011": ("Bebop drone", 1),
    "10100": ("AR drone", 2),
    "10101": ("AR drone", 2),
    "10110": ("AR drone", 2),
    "10111": ("AR drone", 2),
    "11000": ("Phantom drone", 3),
}


def parse_dronerf_filename(filename: str) -> Optional[Dict[str, Union[str, int]]]:
    """
    Parse a DroneRF CSV filename to extract BUI, receiver band (L or H), and file segment index.
    
    Examples:
        '00000L_0.csv' -> bui='00000', receiver='L', segment_num=0
        '10100H_15.csv' -> bui='10100', receiver='H', segment_num=15
        '11000L_9.csv' -> bui='11000', receiver='L', segment_num=9
    """
    base = Path(filename).name
    match = re.match(r"^([01]{5})([LH])_(\d+)\.csv$", base, re.IGNORECASE)
    if not match:
        return None
    bui = match.group(1)
    receiver = match.group(2).upper()
    seg_num = int(match.group(3))
    
    class_name, label_idx = BUI_TO_CLASS.get(bui, ("Unknown", -1))
    return {
        "filename": base,
        "bui": bui,
        "receiver": receiver,
        "file_segment_num": seg_num,
        "drone_class": class_name,
        "faithful_label": label_idx,
    }


def process_faithful_fgcs_segment(
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    q: int = 10,
    m: int = 2048,
) -> np.ndarray:
    """
    Faithfully process one pair of 100,000-sample segments into a 2048-dimensional power spectrum.
    
    Matches Matlab/Main_1_Data_aggregation.m:
        xf = abs(fftshift(fft(x(st:fi)-mean(x(st:fi)), M))); xf = xf(end/2+1:end);
        yf = abs(fftshift(fft(y(st:fi)-mean(y(st:fi)), M))); yf = yf(end/2+1:end);
        data(:,cnt) = [xf ; (yf*mean(xf((end-Q+1):end))./mean(yf(1:Q)))];
        Data = data.^2;
        
    Args:
        x_seg: 1D real-valued array of L-band samples (e.g. 100,000 samples).
        y_seg: 1D real-valued array of H-band samples (e.g. 100,000 samples).
        q: Number of boundary points for spectral continuity scaling (default: 10).
        m: Total number of FFT frequency bins (default: 2048).
        
    Returns:
        2048-dimensional float32 power spectrum vector.
    """
    x_seg = np.asarray(x_seg, dtype=np.float64)
    y_seg = np.asarray(y_seg, dtype=np.float64)

    # 1. DC / Mean removal over the segment
    x_detrend = x_seg - np.mean(x_seg)
    y_detrend = y_seg - np.mean(y_seg)

    # 2. 2048-point FFT: In MATLAB, fft(v, M) takes the first M samples when len(v) >= M
    x_input = x_detrend[:m] if len(x_detrend) >= m else np.pad(x_detrend, (0, m - len(x_detrend)))
    y_input = y_detrend[:m] if len(y_detrend) >= m else np.pad(y_detrend, (0, m - len(y_detrend)))

    fft_x = np.fft.fft(x_input, n=m)
    fft_y = np.fft.fft(y_input, n=m)

    # 3. FFT shift
    shift_x = np.fft.fftshift(fft_x)
    shift_y = np.fft.fftshift(fft_y)

    # 4. Retain positive-frequency half (1024 bins each)
    # In MATLAB: xf(end/2+1:end) corresponds to indices 1025:2048 (length 1024)
    half_m = m // 2
    xf = np.abs(shift_x)[half_m:]
    yf = np.abs(shift_y)[half_m:]

    # 5. Q=10 Boundary Matching Factor
    # MATLAB: c = mean(xf((end-Q+1):end)) / mean(yf(1:Q))
    xf_boundary_mean = np.mean(xf[-q:])
    yf_boundary_mean = np.mean(yf[:q])
    c = xf_boundary_mean / (yf_boundary_mean + 1e-12)

    # 6. Concatenation
    # MATLAB: [xf ; c * yf] -> 2048 bins
    stitched_mag = np.concatenate([xf, c * yf], axis=0)

    # 7. Power conversion: Data = data.^2
    power_spectrum = stitched_mag ** 2

    return power_spectrum.astype(np.float32)


def normalize_global_max(
    feature_matrix: np.ndarray,
    global_max: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Apply global maximum normalization matching Matlab/Main_2_Data_labeling.m:
        Data = Data ./ max(max(Data));
        
    Args:
        feature_matrix: 2D array of shape (N_samples, 2048) or 1D vector.
        global_max: Precomputed global maximum. If None, calculated from feature_matrix.
        
    Returns:
        Tuple of (normalized_matrix, computed_global_max).
    """
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    if global_max is None:
        global_max = float(np.max(np.abs(feature_matrix)))
        if global_max < 1e-12:
            global_max = 1.0

    normalized = feature_matrix / global_max
    return normalized, global_max
