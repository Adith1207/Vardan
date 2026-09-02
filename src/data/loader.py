"""
loader.py
---------

Lazy PyTorch Dataset and DataLoader wrappers for model-specific RF preprocessing.

Guarantees:
- Reads splits from data/splits/ (train.csv, val.csv, test.csv).
- Zero data leakage: Normalization statistics are learned strictly from the TRAIN split.
- Configurable raw dataset root resolution (CLI arg -> DRONERF_RAW_DIR env var -> /kaggle/input -> local project root).
- Model-specific lazy preprocessing (FGCS2019DNN, Baseline1DCNN, DSCNN, MobileNetV3Small, VardhanRFNet).
- Bounded on-demand file reading without loading the complete dataset into RAM.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from utils.paths import PROJECT_ROOT, RAW_DATA_DIR


def _search_file_in_base(base_dir: Path, relative_path: Path) -> Optional[Path]:
    """Helper to locate a raw CSV within a base directory, handling common naming variations."""
    # 1. Direct path check
    direct = base_dir / relative_path
    if direct.exists() and direct.is_file():
        return direct

    # 2. Strip leading repo prefixes (e.g. data/raw/DroneRF/unzipped_data/ -> AR Drone/...)
    parts = list(relative_path.parts)
    prefix_markers = ["unzipped_data", "DroneRF", "dronerf", "raw"]
    
    sub_parts = None
    for marker in prefix_markers:
        if marker in parts:
            idx = parts.index(marker)
            sub_parts = parts[idx + 1:]
            break

    candidate_subpaths: List[Path] = []
    if sub_parts:
        candidate_subpaths.append(Path(*sub_parts))
    candidate_subpaths.append(relative_path)
    if len(parts) >= 3:
        # e.g. class_folder / experiment_folder / filename.csv
        candidate_subpaths.append(Path(*parts[-3:]))
    if len(parts) >= 2:
        candidate_subpaths.append(Path(*parts[-2:]))

    for subp in candidate_subpaths:
        target = base_dir / subp
        if target.exists() and target.is_file():
            return target

        # Check lowercase 'drone' vs 'Drone'
        subp_str = str(subp)
        alt1 = base_dir / subp_str.replace("Drone", "drone")
        if alt1.exists() and alt1.is_file():
            return alt1
        alt2 = base_dir / subp_str.replace("drone", "Drone")
        if alt2.exists() and alt2.is_file():
            return alt2

        # Check nested directory layout (e.g. RF Data_10100_H/RF Data_10100_H/10100H_0.csv)
        p_list = list(subp.parts)
        if len(p_list) >= 2:
            nested = p_list[:-1] + [p_list[-2]] + [p_list[-1]]
            nested_path = base_dir / Path(*nested)
            if nested_path.exists() and nested_path.is_file():
                return nested_path

    return None


def resolve_raw_path(
    relative_path: Union[str, Path],
    raw_data_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve raw CSV file path on disk with strict resolution priority:
    1. Explicit raw_data_dir argument (if provided)
    2. Environment variable DRONERF_RAW_DIR or KAGGLE_INPUT_DIR (if set)
    3. Automatic search under /kaggle/input/ (Kaggle environment)
    4. Local PROJECT_ROOT / RAW_DATA_DIR fallback
    """
    rel_p = Path(relative_path)
    if rel_p.is_absolute() and rel_p.exists():
        return rel_p

    candidate_roots: List[Path] = []

    # Priority 1: Explicit argument
    if raw_data_dir is not None:
        candidate_roots.append(Path(raw_data_dir))

    # Priority 2: Environment variables
    env_raw = os.environ.get("DRONERF_RAW_DIR")
    if env_raw:
        candidate_roots.append(Path(env_raw))
    env_kaggle = os.environ.get("KAGGLE_INPUT_DIR")
    if env_kaggle:
        candidate_roots.append(Path(env_kaggle))

    # Priority 3: Kaggle standard input directory
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists() and kaggle_input.is_dir():
        candidate_roots.append(kaggle_input)
        # Add any subdirectories in /kaggle/input (e.g. /kaggle/input/dronerf)
        try:
            for child in kaggle_input.iterdir():
                if child.is_dir():
                    candidate_roots.append(child)
                    # Check one level deeper for nested dataset root
                    for subchild in child.iterdir():
                        if subchild.is_dir():
                            candidate_roots.append(subchild)
        except Exception:
            pass

    # Priority 4: Local project paths
    candidate_roots.extend([
        PROJECT_ROOT,
        RAW_DATA_DIR,
        PROJECT_ROOT / "data" / "raw" / "DroneRF",
        PROJECT_ROOT / "data" / "raw" / "DroneRF" / "unzipped_data",
    ])

    # Search each candidate root
    for root in candidate_roots:
        if root.exists():
            resolved = _search_file_in_base(root, rel_p)
            if resolved is not None:
                return resolved

    # Fallback to default relative to project root
    return PROJECT_ROOT / rel_p


def fit_train_normalization_stats(
    train_split_csv: Union[str, Path],
    max_files: int = 10,
    segment_length: int = 2048,
    raw_data_dir: Optional[Union[str, Path]] = None,
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
        abs_p = resolve_raw_path(rel_p, raw_data_dir=raw_data_dir)
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
    Loads raw CSV slices on-demand without loading the full 45 GB dataset into RAM.
    """

    def __init__(
        self,
        split_csv: Union[str, Path],
        model_name: str = "fgcs2019dnn",
        norm_stats: Optional[Dict[str, float]] = None,
        segment_length: int = 2048,
        samples_per_file: int = 2,
        raw_data_dir: Optional[Union[str, Path]] = None,
        mock: bool = False,
    ):
        self.split_csv = Path(split_csv)
        self.model_name = model_name.lower()
        self.segment_length = segment_length
        self.samples_per_file = samples_per_file
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        self.mock = mock

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found at {self.split_csv}")

        self.df_split = pd.read_csv(self.split_csv)
        self.norm_stats = norm_stats or {"mean": 0.0, "std": 1.0, "max": 1.0, "min": -1.0}

        self.preprocessor = DroneRFPreprocessor(
            fft_size=self.segment_length,
            remove_dc=True,
            normalization="max",
            channel_count=8,
            channel_overlap=0.50,
            fs=SAMPLING_RATE,
        )

        # Build lazy index: (relative_file_path, sample_offset, class_label)
        self.items = []
        for idx, row in self.df_split.iterrows():
            rel_p = row["relative_path"]
            label = CLASS_MAPPING.get(row["drone_class"], 0)
            for offset in range(self.samples_per_file):
                self.items.append((rel_p, offset, label))

    def __len__(self) -> int:
        return len(self.items)

    def _read_segment(self, file_path: Union[str, Path], offset: int) -> np.ndarray:
        if self.mock:
            # Deterministic mock signal based on filename hash + offset
            file_name = Path(file_path).name
            state = np.random.RandomState(hash(file_name) % (2**32) + offset)
            return state.randn(self.segment_length).astype(np.float32)

        resolved_path = resolve_raw_path(file_path, raw_data_dir=self.raw_data_dir)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Raw DroneRF CSV file not found: '{file_path}' (resolved to '{resolved_path}'). "
                f"Please pass --raw_data_dir or set DRONERF_RAW_DIR."
            )

        try:
            # Bounded character read to avoid loading full 90 MB CSV into RAM
            limit = (offset + 1) * self.segment_length * 15 + 1000
            with open(resolved_path, "r") as f:
                line = f.readline(limit)

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

        # Model-specific lazy preprocessing representations
        if self.model_name == "fgcs2019dnn":
            # FGCS DNN: DC-removed 2048-pt FFT Power Spectrum (Max-magnitude normalized) -> (2048,)
            spectrum = self.preprocessor.process_fgcs(raw_sig)
            out_tensor = spectrum.astype(np.float32)

        elif self.model_name in ["baseline1dcnn", "mc1dcnn", "1dcnn", "dscnn", "vardhan", "vardhanrfnet"]:
            # 2-channel I/Q time-domain waveform representation -> (2, 2048)
            i_ch = (raw_sig - self.norm_stats["mean"]) / self.norm_stats["std"]
            q_ch = np.roll(i_ch, 1)
            out_tensor = np.stack([i_ch, q_ch], axis=0).astype(np.float32)

        elif self.model_name == "mobilenetv3small":
            # 2D STFT Spectrogram Matrix -> (1, freq_bins=65, time_frames=61)
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
    pin_memory: bool = False,
    samples_per_file: int = 2,
    segment_length: int = 2048,
    raw_data_dir: Optional[Union[str, Path]] = None,
    mock: bool = False,
) -> "DataLoader":
    """Instantiate PyTorch DataLoader for a given dataset split with full parameter passthrough."""
    dataset = DroneRFLazyDataset(
        split_csv=split_csv,
        model_name=model_name,
        norm_stats=norm_stats,
        segment_length=segment_length,
        samples_per_file=samples_per_file,
        raw_data_dir=raw_data_dir,
        mock=mock,
    )
    if HAS_TORCH:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
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

