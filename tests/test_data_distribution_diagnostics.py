"""
test_data_distribution_diagnostics.py
-------------------------------------

Focused unit and regression tests for scripts/analyze_data_distribution.py:
- Frequency band derivation (2.4 GHz vs 5.8 GHz)
- Recording component parsing (class, experiment, receiver)
- Split manifest loading and total file counts (308 Train, 73 Val, 73 Test)
- Class and frequency distribution summaries
- Detection of the severe Phantom drone frequency shift (5.8 GHz in Train vs 2.4 GHz in Val/Test)
- Sampling diagnostics: start index progression and non-duplicate slice verification
- End-to-end artifact generation (JSON and CSV)
"""

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import pytest

from scripts.analyze_data_distribution import (
    REPRESENTATIVE_DIAGNOSTIC_FILES,
    analyze_distribution_shifts,
    derive_frequency_band,
    generate_class_split_matrix,
    generate_frequency_receiver_matrix,
    generate_recording_summary,
    generate_split_summary,
    load_and_combine_splits,
    parse_recording_components,
    run_full_distribution_analysis,
    run_sampling_diagnostics,
)
from utils.paths import DATA_DIR


def test_derive_frequency_band():
    """Verify frequency band mapping for High and Low band receivers."""
    assert derive_frequency_band("H") == "5.8 GHz (High)"
    assert derive_frequency_band("H1") == "5.8 GHz (High)"
    assert derive_frequency_band("H2") == "5.8 GHz (High)"
    assert derive_frequency_band("L") == "2.4 GHz (Low)"
    assert derive_frequency_band("L1") == "2.4 GHz (Low)"
    assert derive_frequency_band("L2") == "2.4 GHz (Low)"


def test_parse_recording_components():
    """Verify recording_id string splitting."""
    cls, exp, rec = parse_recording_components("AR Drone_10100_H")
    assert cls == "AR Drone"
    assert exp == "10100"
    assert rec == "H"

    cls, exp, rec = parse_recording_components("Backround RF activities_0_H1")
    assert cls == "Backround RF activities"
    assert exp == "0"
    assert rec == "H1"


def test_load_and_combine_splits():
    """Verify loading and merging of train, val, and test splits."""
    splits_dir = DATA_DIR / "splits"
    df = load_and_combine_splits(splits_dir)

    assert len(df) == 454
    assert set(df["split"].unique()) == {"train", "val", "test"}
    assert "frequency_band" in df.columns
    assert "recording_id" in df.columns

    # Check file counts per split
    assert len(df[df["split"] == "train"]) == 308
    assert len(df[df["split"] == "val"]) == 73
    assert len(df[df["split"] == "test"]) == 73


def test_split_and_class_summaries():
    """Verify summary metrics and Class x Split matrix generation."""
    splits_dir = DATA_DIR / "splits"
    df = load_and_combine_splits(splits_dir)

    summary = generate_split_summary(df, samples_per_file=50)
    assert summary["train"]["file_count"] == 308
    assert summary["train"]["sample_count"] == 308 * 50
    assert summary["val"]["file_count"] == 73
    assert summary["test"]["file_count"] == 73
    assert summary["total"]["file_count"] == 454

    matrix = generate_class_split_matrix(df, samples_per_file=50)
    assert len(matrix) == 4
    class_map = {row["canonical_name"]: row for row in matrix}
    assert class_map["no_drone"]["total_files"] == 82
    assert class_map["ar_drone"]["total_files"] == 162
    assert class_map["bebop_drone"]["total_files"] == 168
    assert class_map["phantom_drone"]["total_files"] == 42


def test_phantom_distribution_shift_diagnostic():
    """Verify that the diagnostic explicitly identifies the Phantom frequency shift."""
    splits_dir = DATA_DIR / "splits"
    df = load_and_combine_splits(splits_dir)

    shifts = analyze_distribution_shifts(df)
    phantom_diag = shifts["phantom_drone_diagnostic"]

    assert phantom_diag["is_severe_shift"] is True
    assert phantom_diag["train_contains_5_8_ghz"] is True
    assert phantom_diag["train_contains_2_4_ghz"] is False
    assert phantom_diag["val_contains_2_4_ghz"] is True
    assert phantom_diag["test_contains_2_4_ghz"] is True


def test_sampling_diagnostics_offsets_and_uniqueness():
    """Verify that 5 sampled segments have consecutive offsets and are non-identical."""
    summary_list, df_sampling = run_sampling_diagnostics(
        representative_files=REPRESENTATIVE_DIAGNOSTIC_FILES,
        segment_length=2048,
        samples_per_file=5,
        mock=True,
    )

    assert len(summary_list) == 4
    assert len(df_sampling) == 20  # 4 files * 5 samples

    for s_info in summary_list:
        assert s_info["start_indices"] == [0, 2048, 4096, 6144, 8192]
        assert s_info["has_duplicate_segments"] is False
        assert s_info["min_pairwise_abs_diff"] > 0.1

    assert df_sampling["is_finite"].all()


def test_end_to_end_diagnostic_artifact_generation():
    """Verify end-to-end execution of run_full_distribution_analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "diagnostics"
        diag_data, df_samp = run_full_distribution_analysis(
            splits_dir=DATA_DIR / "splits",
            samples_per_file=50,
            segment_length=2048,
            diagnostic_samples=5,
            output_dir=out_dir,
            mock=True,
        )

        assert (out_dir / "data_distribution.json").exists()
        assert (out_dir / "sampling_diagnostics.csv").exists()
        assert diag_data["split_summary"]["total"]["file_count"] == 454
        assert len(df_samp) == 20
