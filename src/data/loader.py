"""
loader.py
---------

Lazy PyTorch Dataset and DataLoader wrappers for model-specific RF preprocessing.

Guarantees:
- Reads splits from data/splits/ (train.csv, val.csv, test.csv).
- Zero data leakage: Normalization statistics are learned strictly from the TRAIN split.
- Model-specific lazy preprocessing (FGCS2019DNN, Baseline1DCNN, DSCNN, MobileNetV3Small).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    class Dataset:
        pass
    HAS_TORCH = False

from config import SAMPLING_RATE
from constants import RAW_CLASS_TO_INDEX as CLASS_MAPPING
from preprocessing.pipeline import DroneRFPreprocessor
from utils.paths import PROJECT_ROOT


def resolve_raw_path(relative_path: Union[str, Path]) -> Path:
    """Resolve raw CSV file path on disk, handling different unzipping layouts."""
    path = PROJECT_ROOT / relative_path
    if path.exists():
        return path

    # Try replacing 'unzipped_data' with 'DroneRF' and lowercase matching
    parts = list(Path(relative_path).parts)
    if 'unzipped_data' in parts:
        idx = parts.index('unzipped_data')
        parts[idx] = 'DroneRF'
        if len(parts) > idx + 1:
            parts[idx+1] = parts[idx+1].replace('Drone', 'drone')

        # Try nested directory layout E.g. RF Data_10100_H/RF Data_10100_H/10100H_0.csv
        if len(parts) > idx + 2:
            folder_name = parts[-2]
            nested_parts = parts[:-1] + [folder_name] + [parts[-1]]
            nested_path = PROJECT_ROOT / Path(*nested_parts)
            if nested_path.exists():
                return nested_path

        alt_path = PROJECT_ROOT / Path(*parts)
        if alt_path.exists():
            return alt_path

    return path



def fit_train_normalization_stats(
    train_split_csv: Union[str, Path],
    max_files: int = 10,
    segment_length: int = 2048,
) -> Dict[str, float]:
    """
    Compute normalization statistics strictly from training set files.

    Returns dict containing mean, std, min, max computed on TRAIN ONLY.
    """
    df_train = pd.read_csv(train_split_csv)
    sample_files = df_train["relative_path"].head(max_files)

    signals = []
    limit = segment_length * 15 + 1000
    for rel_p in sample_files:
        abs_p = resolve_raw_path(rel_p)
        if not abs_p.exists():
            continue
        try:
            with open(abs_p, "r") as f:
                line = f.readline(limit)
            raw_str = line.rsplit(",", 1)[0]
            vals = np.fromstring(raw_str, sep=",", dtype=np.float32)
            vals = vals[~np.isnan(vals)]
            if len(vals) >= segment_length:
                signals.append(vals[:segment_length])
        except Exception:
            continue


    if not signals:
        return {"mean": 0.0, "std": 1.0, "max": 1.0, "min": -1.0}

    stacked = np.concatenate(signals)
    mean_val = float(np.mean(stacked))
    std_val = float(np.std(stacked)) + 1e-8
    max_val = float(np.max(np.abs(stacked))) + 1e-8
    min_val = float(np.min(stacked))

    return {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
    }


class DroneRFLazyDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for model-specific baseline architectures.
    """

    def __init__(
        self,
        split_csv: Union[str, Path],
        model_name: str = "fgcs2019dnn",
        norm_stats: Optional[Dict[str, float]] = None,
        segment_length: int = 2048,
        samples_per_file: int = 2,
        mock: bool = False,
    ):
        self.split_csv = Path(split_csv)
        self.model_name = model_name.lower()
        self.segment_length = segment_length
        self.samples_per_file = samples_per_file
        self.mock = mock

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found at {self.split_csv}")

        self.df_split = pd.read_csv(self.split_csv)
        self.norm_stats = norm_stats or {"mean": 0.0, "std": 1.0, "max": 1.0, "min": -1.0}

        self.preprocessor = DroneRFPreprocessor(
            fft_size=2048,
            remove_dc=True,
            normalization="max",
            channel_count=8,
            channel_overlap=0.50,
            fs=SAMPLING_RATE,
        )

        # Build index of (file_path, sample_offset, label)
        self.items = []
        for idx, row in self.df_split.iterrows():
            abs_p = PROJECT_ROOT / row["relative_path"]
            label = CLASS_MAPPING.get(row["drone_class"], 0)
            for offset in range(self.samples_per_file):
                self.items.append((abs_p, offset, label))

    def __len__(self) -> int:
        return len(self.items)

    def _read_segment(self, file_path: Path, offset: int) -> np.ndarray:
        if self.mock:
            # Deterministic mock signal based on filename hash + offset
            state = np.random.RandomState(hash(file_path.name) % (2**32) + offset)
            return state.randn(self.segment_length).astype(np.float32)

        resolved_path = resolve_raw_path(file_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Raw DroneRF CSV file not found: {resolved_path}")

        try:
            # Optimize reading by reading only what we need.
            # Limit character length to (offset + 1) * segment_length * 15 + 1000
            limit = (offset + 1) * self.segment_length * 15 + 1000
            with open(resolved_path, "r") as f:
                line = f.readline(limit)

            # Parse numbers up to the last complete one
            vals = np.fromstring(line.rsplit(",", 1)[0], sep=",", dtype=np.float32)
            vals = vals[~np.isnan(vals)]

            start_idx = offset * self.segment_length
            end_idx = start_idx + self.segment_length

            seg = vals[start_idx:end_idx]
            if len(seg) < self.segment_length:
                seg = np.pad(seg, (0, self.segment_length - len(seg)))
            return seg
        except Exception as e:
            raise RuntimeError(f"Error reading segment from {resolved_path}: {e}")


    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, "torch.Tensor"], int]:
        file_path, offset, label = self.items[idx]
        raw_sig = self._read_segment(file_path, offset)

        # Model-specific lazy preprocessing
        if self.model_name == "fgcs2019dnn":
            # FGCS DNN: 2048-pt FFT Power Spectrum -> Train-fitted normalization
            spectrum = self.preprocessor.process_fgcs(raw_sig)
            # Normalize using train stats
            norm_spectrum = spectrum / self.norm_stats["max"]
            out_tensor = norm_spectrum.astype(np.float32)

        elif self.model_name in ["baseline1dcnn", "mc1dcnn"]:
            # 1D CNN: 2-channel I/Q or multi-channel spectrum representation
            # For 2-channel waveform input: (2, 2048)
            i_ch = (raw_sig - self.norm_stats["mean"]) / self.norm_stats["std"]
            q_ch = np.roll(i_ch, 1)
            out_tensor = np.stack([i_ch, q_ch], axis=0).astype(np.float32)

        elif self.model_name == "dscnn":
            # Sensors 2022 DSCNN baseline: 2-channel I/Q input (2, 2048)
            i_ch = (raw_sig - self.norm_stats["mean"]) / self.norm_stats["std"]
            q_ch = np.roll(i_ch, 1)
            out_tensor = np.stack([i_ch, q_ch], axis=0).astype(np.float32)

        elif self.model_name == "mobilenetv3small":
            # Spectrogram 2D CNN baseline: (1, freq_bins, time_frames)
            freqs, times, p_db = self.preprocessor.spectrogram_processor.transform(raw_sig)
            p_norm = (p_db - np.mean(p_db)) / (np.std(p_db) + 1e-8)
            out_tensor = np.expand_dims(p_norm, axis=0).astype(np.float32)

        else:
            out_tensor = raw_sig.astype(np.float32)

        if HAS_TORCH:
            return torch.tensor(out_tensor), label
        return out_tensor, label


def get_dataloader(
    split_csv: Union[str, Path],
    model_name: str = "fgcs2019dnn",
    norm_stats: Optional[Dict[str, float]] = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    mock: bool = False,
) -> "DataLoader":
    """Instantiate PyTorch DataLoader for a given dataset split."""
    dataset = DroneRFLazyDataset(
        split_csv=split_csv,
        model_name=model_name,
        norm_stats=norm_stats,
        mock=mock,
    )
    if HAS_TORCH:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset


DroneRFDataset = DroneRFLazyDataset


def load_rf_data(filepath: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Load single raw RF signal file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"RF data file not found: {path}")
    df = pd.read_csv(path, header=None, nrows=2048)
    vals = pd.to_numeric(df.to_numpy().reshape(-1), errors="coerce")
    vals = vals[~np.isnan(vals)].astype(np.float32)
    return vals, 0

