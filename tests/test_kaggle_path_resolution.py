"""
test_kaggle_path_resolution.py
------------------------------

Regression & contract tests verifying that raw DroneRF CSV paths resolve correctly
across diverse layout variations:
1. Exact Kaggle dataset layout with nested experiment directories and lowercase "drone":
   DroneRF/AR drone/RF Data_10100_H/RF Data_10100_H/10100H_0.csv
2. Directory naming typos:
   Background RF activites/RF Data_00000_H1/00000H1_0.csv
3. Deeply nested Kaggle mount roots:
   /kaggle/input/datasets/subashsanthanamk/dronerf-raw-dataset/DroneRF/...
4. Bepop drone and Phantom drone layouts.
5. Priority order (explicit raw_data_dir -> env var -> kaggle input -> local fallback).
6. End-to-end DroneRFLazyDataset slice reading on Kaggle layout.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from data.loader import DroneRFLazyDataset, clear_raw_path_cache, resolve_raw_path


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure in-memory path index cache is reset before each test."""
    clear_raw_path_cache()
    yield
    clear_raw_path_cache()


def test_kaggle_nested_and_lowercase_resolution():
    """Test resolution of Kaggle's exact nested experiment folder and lowercase 'drone'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kaggle_root = Path(tmpdir) / "datasets" / "subashsanthanamk" / "dronerf-raw-dataset" / "DroneRF"
        
        # Create exact Kaggle structure
        fake_ar_csv = kaggle_root / "AR drone" / "RF Data_10100_H" / "RF Data_10100_H" / "10100H_0.csv"
        fake_ar_csv.parent.mkdir(parents=True, exist_ok=True)
        fake_ar_csv.write_text("0.1,0.2,0.3,0.4,0.5")

        rel_metadata_path = "data/raw/DroneRF/unzipped_data/AR Drone/RF Data_10100_H/10100H_0.csv"

        # Case A: User passes the exact DroneRF subfolder as raw_data_dir
        resolved_a = resolve_raw_path(rel_metadata_path, raw_data_dir=kaggle_root)
        assert resolved_a == fake_ar_csv, f"Expected {fake_ar_csv}, got {resolved_a}"

        # Case B: User passes the parent Kaggle input root
        resolved_b = resolve_raw_path(rel_metadata_path, raw_data_dir=Path(tmpdir))
        assert resolved_b == fake_ar_csv, f"Expected {fake_ar_csv}, got {resolved_b}"


def test_kaggle_background_typo_and_other_classes_resolution():
    """Test resolution of typo variations like 'Background RF activites' and Bepop / Phantom."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kaggle_root = Path(tmpdir) / "DroneRF"
        
        # 1. Background with typo 'activites'
        fake_bg_csv = kaggle_root / "Background RF activites" / "RF Data_00000_H1" / "00000H1_0.csv"
        fake_bg_csv.parent.mkdir(parents=True, exist_ok=True)
        fake_bg_csv.write_text("0.1,0.2,0.3")

        # 2. Bepop drone
        fake_bepop_csv = kaggle_root / "Bepop drone" / "RF Data_10000_H" / "10000H_0.csv"
        fake_bepop_csv.parent.mkdir(parents=True, exist_ok=True)
        fake_bepop_csv.write_text("0.4,0.5,0.6")

        # 3. Phantom drone
        fake_phantom_csv = kaggle_root / "Phantom drone" / "RF Data_11000_L1" / "11000L_0.csv"
        fake_phantom_csv.parent.mkdir(parents=True, exist_ok=True)
        fake_phantom_csv.write_text("0.7,0.8,0.9")

        # Resolving Background
        bg_rel = "data/raw/DroneRF/unzipped_data/Backround RF activities/RF Data_00000_H1/00000H1_0.csv"
        res_bg = resolve_raw_path(bg_rel, raw_data_dir=kaggle_root)
        assert res_bg == fake_bg_csv, f"Expected {fake_bg_csv}, got {res_bg}"

        # Resolving Bepop
        bepop_rel = "data/raw/DroneRF/unzipped_data/Bepop drone/RF Data_10000_H/10000H_0.csv"
        res_bepop = resolve_raw_path(bepop_rel, raw_data_dir=kaggle_root)
        assert res_bepop == fake_bepop_csv, f"Expected {fake_bepop_csv}, got {res_bepop}"

        # Resolving Phantom
        phantom_rel = "data/raw/DroneRF/unzipped_data/Phantom drone/RF Data_11000_L1/11000L_0.csv"
        res_phantom = resolve_raw_path(phantom_rel, raw_data_dir=kaggle_root)
        assert res_phantom == fake_phantom_csv, f"Expected {fake_phantom_csv}, got {res_phantom}"


def test_priority_hierarchy():
    """Verify explicit argument takes precedence over DRONERF_RAW_DIR env var."""
    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        file1 = Path(dir1) / "AR drone" / "10100H_0.csv"
        file1.parent.mkdir(parents=True, exist_ok=True)
        file1.write_text("file1")

        file2 = Path(dir2) / "AR drone" / "10100H_0.csv"
        file2.parent.mkdir(parents=True, exist_ok=True)
        file2.write_text("file2")

        rel_p = "data/raw/DroneRF/unzipped_data/AR Drone/RF Data_10100_H/10100H_0.csv"

        os.environ["DRONERF_RAW_DIR"] = str(dir2)
        try:
            # Explicit dir1 should win over env var dir2
            resolved = resolve_raw_path(rel_p, raw_data_dir=dir1)
            assert resolved == file1
        finally:
            del os.environ["DRONERF_RAW_DIR"]


def test_lazy_dataset_reads_from_kaggle_layout():
    """Test that DroneRFLazyDataset reads valid 2048-sample slices from Kaggle-style paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kaggle_root = Path(tmpdir) / "DroneRF"
        fake_csv = kaggle_root / "AR drone" / "RF Data_10100_H" / "RF Data_10100_H" / "10100H_0.csv"
        fake_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Write 5000 comma-separated float numbers
        numbers = [f"{0.01 * i:.4f}" for i in range(5000)]
        fake_csv.write_text(",".join(numbers))

        # Create tiny split manifest
        split_csv = Path(tmpdir) / "train_split.csv"
        split_csv.write_text(
            "relative_path,drone_class,recording_id\n"
            "data/raw/DroneRF/unzipped_data/AR Drone/RF Data_10100_H/10100H_0.csv,AR Drone,AR Drone_10100_H\n"
        )

        dataset = DroneRFLazyDataset(
            split_csv=split_csv,
            model_name="vardhan",
            segment_length=2048,
            samples_per_file=2,
            raw_data_dir=kaggle_root,
            mock=False,
        )

        assert len(dataset) == 2
        x_tensor, label = dataset[0]
        assert x_tensor.shape == (2, 2048)
        assert label == 1  # AR Drone label index
        assert torch.isfinite(x_tensor).all()
