"""
fgcs_faithful_loader.py
-----------------------

Dataset loader and pairing manager for faithful FGCS reproduction (EXP_FGCS_FAITHFUL).

Features:
- Discovers and pairs all 227 synchronized (L, H) DroneRF recording pairs.
- Slices 100 non-overlapping 100,000-sample segments per pair -> 22,700 total segments.
- Bounded on-demand CSV parsing to avoid loading 90 MB CSVs entirely into memory.
- Integrates with process_faithful_fgcs_segment and global max normalization.
- Uses faithful 4-class label ordering: 0: Background, 1: Bebop, 2: AR, 3: Phantom.
"""

import os
import re
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

from preprocessing.fgcs_faithful import (
    FAITHFUL_FGCS_CLASS_TO_INDEX,
    FAITHFUL_FGCS_INDEX_TO_CLASS,
    BUI_TO_CLASS,
    parse_dronerf_filename,
    process_faithful_fgcs_segment,
    normalize_global_max,
)
from utils.paths import PROJECT_ROOT, RAW_DATA_DIR


def discover_and_pair_dronerf_files(
    raw_data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Scan raw data directory to find and pair all synchronized L and H files.
    
    Pairs are matched strictly on (BUI, file_segment_num).
    Example: '00000L_5.csv' pairs with '00000H_5.csv'.
    
    Returns:
        pd.DataFrame with 227 paired recording entries.
    """
    if raw_data_dir is None:
        raw_data_dir = RAW_DATA_DIR

    raw_path = Path(raw_data_dir)
    search_dirs = [
        raw_path,
        raw_path / "DroneRF",
        raw_path / "DroneRF" / "DroneRF",
        raw_path / "DroneRF" / "unzipped_data",
        raw_path / "unzipped_data",
        PROJECT_ROOT / "data" / "raw" / "DroneRF" / "unzipped_data",
        PROJECT_ROOT / "data" / "raw" / "DroneRF" / "DroneRF",
        Path("/kaggle/input/datasets/subashsanthanamk/dronerf-raw-dataset/DroneRF"),
    ]

    # Collect all CSV paths
    found_csvs: Dict[str, Path] = {}
    for base in search_dirs:
        if base.exists() and base.is_dir():
            for root, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith(".csv"):
                        # Normalize filename key
                        f_name = f.strip()
                        if f_name not in found_csvs:
                            found_csvs[f_name] = Path(root) / f

    # If no unpacked CSVs found, also check dronerf_metadata.csv for paths
    if not found_csvs:
        meta_path = PROJECT_ROOT / "data" / "metadata" / "dronerf_metadata.csv"
        if meta_path.exists():
            df_meta = pd.read_csv(meta_path)
            for _, row in df_meta.iterrows():
                rel_p = Path(row["relative_path"])
                found_csvs[rel_p.name] = rel_p

    # Organize into L and H buckets keyed by (bui, file_segment_num)
    l_files: Dict[Tuple[str, int], Dict] = {}
    h_files: Dict[Tuple[str, int], Dict] = {}

    for fname, p in found_csvs.items():
        parsed = parse_dronerf_filename(fname)
        if parsed is None:
            continue
        key = (parsed["bui"], parsed["file_segment_num"])
        info = {
            "path": p,
            "filename": fname,
            "bui": parsed["bui"],
            "file_segment_num": parsed["file_segment_num"],
            "drone_class": parsed["drone_class"],
            "faithful_label": parsed["faithful_label"],
        }
        if parsed["receiver"] == "L":
            l_files[key] = info
        elif parsed["receiver"] == "H":
            h_files[key] = info

    # Build paired records
    paired_rows: List[Dict] = []
    all_keys = sorted(set(l_files.keys()).union(set(h_files.keys())))

    for bui, seg_num in all_keys:
        key = (bui, seg_num)
        l_info = l_files.get(key)
        h_info = h_files.get(key)

        drone_cls, label_idx = BUI_TO_CLASS.get(bui, ("Unknown", -1))
        pair_id = f"{bui}_{seg_num}"

        paired_rows.append({
            "pair_id": pair_id,
            "bui": bui,
            "file_segment_num": seg_num,
            "drone_class": drone_cls,
            "faithful_label": label_idx,
            "has_l": l_info is not None,
            "has_h": h_info is not None,
            "is_paired": (l_info is not None) and (h_info is not None),
            "l_filename": l_info["filename"] if l_info else None,
            "h_filename": h_info["filename"] if h_info else None,
            "l_path": str(l_info["path"]) if l_info else None,
            "h_path": str(h_info["path"]) if h_info else None,
        })

    df_pairs = pd.DataFrame(paired_rows)
    return df_pairs


def build_faithful_manifest(
    raw_data_dir: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    segments_per_pair: int = 100,
    segment_length: int = 100000,
) -> pd.DataFrame:
    """
    Build a 22,700-segment manifest (227 pairs * 100 segments) with metadata.
    
    Args:
        raw_data_dir: Base directory for raw files.
        output_csv: Optional output path to save manifest CSV.
        segments_per_pair: Number of segments per recording pair (default: 100).
        segment_length: Samples per segment (default: 100,000).
        
    Returns:
        pd.DataFrame containing segment-level manifest.
    """
    df_pairs = discover_and_pair_dronerf_files(raw_data_dir=raw_data_dir)
    
    segment_rows: List[Dict] = []
    for _, pair in df_pairs.iterrows():
        pair_id = pair["pair_id"]
        bui = pair["bui"]
        file_seg_num = pair["file_segment_num"]
        drone_cls = pair["drone_class"]
        label_idx = pair["faithful_label"]
        l_path = pair["l_path"]
        h_path = pair["h_path"]

        for seg_offset in range(segments_per_pair):
            st_sample = seg_offset * segment_length
            fi_sample = st_sample + segment_length
            seg_uid = f"{pair_id}#seg_{seg_offset:02d}"

            segment_rows.append({
                "segment_unique_id": seg_uid,
                "pair_id": pair_id,
                "bui": bui,
                "file_segment_num": file_seg_num,
                "segment_offset": seg_offset,
                "sample_start": st_sample,
                "sample_end": fi_sample,
                "drone_class": drone_cls,
                "faithful_label": label_idx,
                "l_path": l_path,
                "h_path": h_path,
            })

    df_manifest = pd.DataFrame(segment_rows)
    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_manifest.to_csv(out_p, index=False)

    return df_manifest


class FGCSFaithfulLazyDataset(Dataset):
    """
    Lazy PyTorch Dataset for faithful FGCS DroneRF preprocessing.
    
    Loads paired (L, H) 100,000-sample segments on demand, computes the stitched 2048-pt
    power spectrum, and returns normalized feature vectors.
    """

    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        segment_length: int = 100000,
        global_max: Optional[float] = None,
        mock: bool = False,
    ):
        if isinstance(manifest, (str, Path)):
            self.df_manifest = pd.read_csv(manifest)
        else:
            self.df_manifest = manifest.copy()

        self.segment_length = segment_length
        self.global_max = global_max
        self.mock = mock

    def __len__(self) -> int:
        return len(self.df_manifest)

    def _read_file_chunk(self, file_path: Union[str, Path], offset: int) -> np.ndarray:
        """Read a 100,000-sample slice from a DroneRF single-line CSV with bounded reading."""
        if self.mock:
            # Deterministic synthetic chunk based on hash
            fname = Path(file_path).name if file_path else "mock"
            rng = np.random.RandomState(abs(hash(fname) + offset) % (2**31))
            return rng.randn(self.segment_length).astype(np.float32)

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"DroneRF file not found: {path}")

        # Bounded read: each float is ~10 chars + comma
        limit = (offset + 1) * self.segment_length * 15 + 10000
        with open(path, "r") as f:
            line = f.readline(limit)

        vals = np.fromstring(line.rsplit(",", 1)[0], sep=",", dtype=np.float32)
        vals = vals[~np.isnan(vals)]

        start_idx = offset * self.segment_length
        end_idx = start_idx + self.segment_length
        seg = vals[start_idx:end_idx]

        if len(seg) < self.segment_length:
            seg = np.pad(seg, (0, self.segment_length - len(seg)))
        return seg

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, "torch.Tensor"], int]:
        row = self.df_manifest.iloc[idx]
        offset = int(row["segment_offset"])
        label = int(row["faithful_label"])
        l_path = row["l_path"]
        h_path = row["h_path"]

        # Read 100k samples from both L and H
        x_seg = self._read_file_chunk(l_path, offset)
        y_seg = self._read_file_chunk(h_path, offset)

        # Process through faithful FGCS pipeline -> (2048,) power spectrum
        power_spectrum = process_faithful_fgcs_segment(
            x_seg=x_seg,
            y_seg=y_seg,
            q=10,
            m=2048,
        )

        # Global max normalization if provided
        if self.global_max is not None:
            power_spectrum, _ = normalize_global_max(power_spectrum, global_max=self.global_max)

        if HAS_TORCH:
            return torch.tensor(power_spectrum, dtype=torch.float32), label
        return power_spectrum, label


def get_fgcs_faithful_dataloader(
    manifest: Union[pd.DataFrame, str, Path],
    batch_size: int = 10,
    shuffle: bool = False,
    num_workers: int = 0,
    global_max: Optional[float] = None,
    mock: bool = False,
) -> "DataLoader":
    """Instantiate DataLoader for faithful FGCS dataset."""
    dataset = FGCSFaithfulLazyDataset(
        manifest=manifest,
        global_max=global_max,
        mock=mock,
    )
    if HAS_TORCH:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    return dataset
