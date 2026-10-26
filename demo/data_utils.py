"""Data utilities for discovering, pairing, and extracting real DroneRF signals for the demo."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add src to sys.path for repository modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.fgcs_faithful_loader import discover_and_pair_dronerf_files, read_raw_signal_10m
from demo.config import (
    CLASS_MAPPING,
    RAW_DATA_DIR,
    SAMPLING_RATE_HZ,
    SEGMENT_LENGTH,
)


def _has_readable_l_signal(row: pd.Series) -> bool:
    """Return True if the low-band (L) recording of a pair exists on disk (CSV or RAR)."""
    for key in ("l_rar", "l_path"):
        value = row.get(key)
        if isinstance(value, str) and value and Path(value).is_file():
            return True
    return False


class DemoDataLoader:
    """Discovers DroneRF recordings, extracts real 2048-sample slices, and applies exact normalization."""

    def __init__(self, raw_data_dir: Optional[Union[str, Path]] = None):
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else RAW_DATA_DIR
        self._df_pairs: Optional[pd.DataFrame] = None
        # Last full 10M-sample recording read, so moving the segment slider does not re-read the file
        self._signal_cache: Optional[Tuple[Tuple, np.ndarray]] = None

    def get_paired_manifest(self) -> pd.DataFrame:
        """Discover and return all paired DroneRF recordings (cached in memory).

        Only pairs whose L-band signal is actually present on disk are kept. When the raw
        files are missing, the faithful loader falls back to data/metadata/dronerf_metadata.csv,
        which lists recordings that cannot be read.
        """
        if self._df_pairs is None:
            df_pairs = discover_and_pair_dronerf_files(raw_data_dir=self.raw_data_dir)
            if not df_pairs.empty:
                df_pairs = df_pairs[df_pairs.apply(_has_readable_l_signal, axis=1)].reset_index(drop=True)
            if df_pairs.empty:
                raise FileNotFoundError(
                    f"No DroneRF dataset archives or CSVs found under '{self.raw_data_dir}'. "
                    "Ensure DroneRF archives or CSV files are placed in data/raw/DroneRF/."
                )
            self._df_pairs = df_pairs
        return self._df_pairs

    def get_sample_choices(self) -> List[Dict[str, Union[str, int]]]:
        """Return a clean list of selectable recordings across all 4 classes."""
        df = self.get_paired_manifest()
        choices = []
        for _, row in df.iterrows():
            label = int(row["faithful_label"])
            cls_name = CLASS_MAPPING.get(label, row["drone_class"])
            pair_id = row["pair_id"]
            bui = row["bui"]
            seg_num = row["file_segment_num"]
            display_name = f"[{cls_name}] Pair {pair_id} (BUI: {bui}, File {seg_num})"
            choices.append({
                "display": display_name,
                "pair_id": pair_id,
                "label": label,
                "class_name": cls_name,
                "row": row.to_dict(),
            })
        return choices

    def _read_l_signal(
        self,
        l_path: Optional[str],
        l_rar: Optional[str],
        l_inner: Optional[str],
    ) -> np.ndarray:
        """Read the full 10M-sample L-band signal, reusing the most recently read recording."""
        key = (l_path, l_rar, l_inner)
        if self._signal_cache is None or self._signal_cache[0] != key:
            # Read 10,000,000 floats from genuine DroneRF recording (zero synthetic fallback)
            raw_signal = read_raw_signal_10m(l_path, rar_path=l_rar, inner_file=l_inner)
            if raw_signal is None or len(raw_signal) == 0:
                raise RuntimeError(f"Failed to read signal data from {l_path} (inner: {l_inner}).")
            self._signal_cache = (key, raw_signal)
        return self._signal_cache[1]

    def load_real_segment(
        self,
        pair_row: Dict,
        segment_index: int = 0,
    ) -> Dict[str, Union[np.ndarray, int, str, float]]:
        """Load a genuine 2048-sample slice from a real DroneRF signal.

        Args:
            pair_row: Dictionary row containing metadata ('l_path', 'l_rar', 'l_inner', etc.).
            segment_index: Segment offset index (0 to 99). Each segment begins at index * 100,000.

        Returns:
            Dictionary with raw waveform, normalized waveform, ground truth, and signal statistics.
        """
        if not (0 <= segment_index < 100):
            raise ValueError(f"segment_index must be between 0 and 99, got {segment_index}")

        l_path = pair_row.get("l_path")
        l_rar = pair_row.get("l_rar")
        l_inner = pair_row.get("l_inner")

        raw_signal = self._read_l_signal(l_path, l_rar, l_inner)

        # Slicing matches training logic: start = segment_index * 100,000
        start_sample = segment_index * 100_000
        end_sample = start_sample + SEGMENT_LENGTH

        if end_sample > len(raw_signal):
            raise IndexError(
                f"Requested window [{start_sample}:{end_sample}] exceeds signal length {len(raw_signal)}."
            )

        raw_slice = raw_signal[start_sample:end_sample].astype(np.float32)

        # Exact per-segment normalization used during training
        mu = float(np.mean(raw_slice))
        sigma = float(np.std(raw_slice)) + 1e-8
        normalized_slice = (raw_slice - mu) / sigma

        label = int(pair_row["faithful_label"])
        class_name = CLASS_MAPPING.get(label, pair_row.get("drone_class", "Unknown"))

        return {
            "pair_id": pair_row["pair_id"],
            "bui": pair_row["bui"],
            "file_segment_num": pair_row["file_segment_num"],
            "source_file": l_inner or Path(l_path).name,
            "segment_index": segment_index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "ground_truth_label": label,
            "ground_truth_name": class_name,
            "raw_waveform": raw_slice,
            "normalized_waveform": normalized_slice,
            "mean": mu,
            "std": sigma,
            "min_val": float(np.min(raw_slice)),
            "max_val": float(np.max(raw_slice)),
            "sampling_rate_hz": SAMPLING_RATE_HZ,
        }
