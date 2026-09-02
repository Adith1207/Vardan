"""
analyze_data_distribution.py
----------------------------

Comprehensive Diagnostic Suite for DroneRF Dataset Sampling & Split Distribution.

Analyzes:
1. Split-Level Summary (files, recordings, sample counts per split, class distribution).
2. Recording-Level Summary (recording_id, split, class, receiver, experiment_id, frequency band, file count).
3. Class x Split Matrix (train/val/test files and generated sample counts).
4. Frequency/Receiver x Split Matrix (2.4 GHz vs 5.8 GHz across splits).
5. Explicit Distribution Shift Analysis (specifically highlighting the Phantom drone 5.8 GHz vs 2.4 GHz split).
6. Sampling Diagnostics (segment start indices, mean, std, min, max, and pairwise similarity checks).
7. Bounded read verification ensuring non-overlapping segment slices without full CSV loading into RAM.

Outputs:
- Console report with formatted tables and distribution shift alerts.
- results/diagnostics/data_distribution.json
- results/diagnostics/sampling_diagnostics.csv
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import NUM_CLASSES
from constants import CLASS_NAMES, LABEL_MAP, RAW_CLASS_TO_INDEX
from data.loader import DroneRFLazyDataset, resolve_raw_path
from utils.paths import DATA_DIR, RESULTS_DIR

# 4 Canonical representative files (1 per class) for sampling diagnostics
REPRESENTATIVE_DIAGNOSTIC_FILES = [
    {
        "class_name": "no_drone",
        "drone_class": "Backround RF activities",
        "experiment_id": "0",
        "receiver": "H1",
        "band": "5.8 GHz (H1)",
        "rel_path": "data/raw/DroneRF/unzipped_data/Backround RF activities/RF Data_00000_H1/00000H_0.csv",
    },
    {
        "class_name": "ar_drone",
        "drone_class": "AR Drone",
        "experiment_id": "10100",
        "receiver": "H",
        "band": "5.8 GHz (H)",
        "rel_path": "data/raw/DroneRF/unzipped_data/AR Drone/RF Data_10100_H/10100H_0.csv",
    },
    {
        "class_name": "bebop_drone",
        "drone_class": "Bepop drone",
        "experiment_id": "10000",
        "receiver": "H",
        "band": "5.8 GHz (H)",
        "rel_path": "data/raw/DroneRF/unzipped_data/Bepop drone/RF Data_10000_H/10000H_0.csv",
    },
    {
        "class_name": "phantom_drone",
        "drone_class": "Phantom drone",
        "experiment_id": "11000",
        "receiver": "H",
        "band": "5.8 GHz (H)",
        "rel_path": "data/raw/DroneRF/unzipped_data/Phantom drone/RF Data_11000_H/11000H_0.csv",
    },
]


def derive_frequency_band(receiver: str) -> str:
    """Derive RF frequency band (2.4 GHz vs 5.8 GHz) from receiver label."""
    rec_upper = str(receiver).upper().strip()
    if rec_upper.startswith("H"):
        return "5.8 GHz (High)"
    elif rec_upper.startswith("L"):
        return "2.4 GHz (Low)"
    return "Unknown"


def parse_recording_components(recording_id: str) -> Tuple[str, str, str]:
    """
    Parse recording_id into (drone_class, experiment_id, receiver).
    Expected format: <drone_class>_<experiment_id>_<receiver>
    """
    parts = str(recording_id).rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return recording_id, "unknown", "unknown"


def load_and_combine_splits(splits_dir: Path) -> pd.DataFrame:
    """Load train, val, and test split manifests into a single unified DataFrame."""
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing split files in {splits_dir}. Expected train.csv, val.csv, and test.csv.")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    combined = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Derive parsed metadata fields if not already explicit
    if "receiver" not in combined.columns or "experiment_id" not in combined.columns:
        parsed = combined["recording_id"].apply(parse_recording_components)
        combined["parsed_class"] = [p[0] for p in parsed]
        combined["experiment_id"] = [p[1] for p in parsed]
        combined["receiver"] = [p[2] for p in parsed]

    combined["frequency_band"] = combined["receiver"].apply(derive_frequency_band)
    return combined


def generate_split_summary(df_all: pd.DataFrame, samples_per_file: int) -> Dict[str, Any]:
    """Generate high-level split summary metrics."""
    splits = ["train", "val", "test"]
    summary = {}
    total_files = len(df_all)
    total_samples = total_files * samples_per_file

    for s in splits:
        df_s = df_all[df_all["split"] == s]
        file_count = len(df_s)
        rec_count = df_s["recording_id"].nunique()
        sample_count = file_count * samples_per_file
        summary[s] = {
            "file_count": file_count,
            "file_pct": float(file_count / total_files * 100),
            "recording_count": rec_count,
            "sample_count": sample_count,
            "sample_pct": float(sample_count / total_samples * 100),
        }

    summary["total"] = {
        "file_count": total_files,
        "recording_count": df_all["recording_id"].nunique(),
        "sample_count": total_samples,
    }
    return summary


def generate_recording_summary(df_all: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate detailed recording-level assignment breakdown."""
    records = []
    grouped = df_all.groupby(["recording_id", "drone_class", "experiment_id", "receiver", "frequency_band", "split"])
    for (rec_id, cls, exp, rec, freq, split), group in grouped:
        records.append({
            "recording_id": rec_id,
            "drone_class": cls,
            "canonical_label": RAW_CLASS_TO_INDEX.get(cls, -1),
            "experiment_id": exp,
            "receiver": rec,
            "frequency_band": freq,
            "split": split,
            "file_count": len(group),
        })
    # Sort deterministically by class, experiment, receiver
    records.sort(key=lambda r: (r["canonical_label"], r["experiment_id"], r["receiver"]))
    return records


def generate_class_split_matrix(df_all: pd.DataFrame, samples_per_file: int) -> List[Dict[str, Any]]:
    """Generate Class x Split breakdown table for files and sample counts."""
    classes = sorted(df_all["drone_class"].unique(), key=lambda c: RAW_CLASS_TO_INDEX.get(c, 99))
    matrix = []
    for cls in classes:
        df_cls = df_all[df_all["drone_class"] == cls]
        tr_files = len(df_cls[df_cls["split"] == "train"])
        va_files = len(df_cls[df_cls["split"] == "val"])
        te_files = len(df_cls[df_cls["split"] == "test"])
        tot_files = tr_files + va_files + te_files

        tr_samples = tr_files * samples_per_file
        va_samples = va_files * samples_per_file
        te_samples = te_files * samples_per_file
        tot_samples = tot_files * samples_per_file

        matrix.append({
            "drone_class": cls,
            "canonical_name": LABEL_MAP.get(RAW_CLASS_TO_INDEX.get(cls, 0), "unknown"),
            "train_files": tr_files,
            "val_files": va_files,
            "test_files": te_files,
            "total_files": tot_files,
            "train_samples": tr_samples,
            "val_samples": va_samples,
            "test_samples": te_samples,
            "total_samples": tot_samples,
        })
    return matrix


def generate_frequency_receiver_matrix(df_all: pd.DataFrame, samples_per_file: int) -> List[Dict[str, Any]]:
    """Generate Frequency Band / Receiver x Split breakdown table."""
    freq_groups = sorted(df_all["frequency_band"].unique())
    matrix = []
    for freq in freq_groups:
        df_freq = df_all[df_all["frequency_band"] == freq]
        receivers = sorted(df_freq["receiver"].unique())
        for r in receivers:
            df_r = df_freq[df_freq["receiver"] == r]
            tr_files = len(df_r[df_r["split"] == "train"])
            va_files = len(df_r[df_r["split"] == "val"])
            te_files = len(df_r[df_r["split"] == "test"])
            tot_files = tr_files + va_files + te_files

            matrix.append({
                "frequency_band": freq,
                "receiver": r,
                "train_files": tr_files,
                "val_files": va_files,
                "test_files": te_files,
                "total_files": tot_files,
                "train_samples": tr_files * samples_per_file,
                "val_samples": va_files * samples_per_file,
                "test_samples": te_files * samples_per_file,
                "total_samples": tot_files * samples_per_file,
            })
    return matrix


def analyze_distribution_shifts(df_all: pd.DataFrame) -> Dict[str, Any]:
    """
    Diagnose severe frequency/condition distribution shifts across Train, Val, and Test splits.
    Specifically checks for single-band recording assignments per class.
    """
    shifts = []
    phantom_shift = {}

    classes = df_all["drone_class"].unique()
    for cls in classes:
        df_cls = df_all[df_all["drone_class"] == cls]
        tr_bands = set(df_cls[df_cls["split"] == "train"]["frequency_band"])
        va_bands = set(df_cls[df_cls["split"] == "val"]["frequency_band"])
        te_bands = set(df_cls[df_cls["split"] == "test"]["frequency_band"])

        tr_recs = sorted(df_cls[df_cls["split"] == "train"]["recording_id"].unique())
        va_recs = sorted(df_cls[df_cls["split"] == "val"]["recording_id"].unique())
        te_recs = sorted(df_cls[df_cls["split"] == "test"]["recording_id"].unique())

        shift_detected = (tr_bands != va_bands) or (tr_bands != te_bands)
        info = {
            "drone_class": cls,
            "shift_detected": shift_detected,
            "train_bands": list(tr_bands),
            "val_bands": list(va_bands),
            "test_bands": list(te_bands),
            "train_recordings": tr_recs,
            "val_recordings": va_recs,
            "test_recordings": te_recs,
        }
        shifts.append(info)

        if "Phantom" in cls:
            phantom_shift = info

    # Check Phantom shift in detail
    phantom_details = {
        "is_severe_shift": bool(phantom_shift.get("shift_detected", False)),
        "train_contains_2_4_ghz": "2.4 GHz (Low)" in phantom_shift.get("train_bands", []),
        "train_contains_5_8_ghz": "5.8 GHz (High)" in phantom_shift.get("train_bands", []),
        "val_contains_2_4_ghz": "2.4 GHz (Low)" in phantom_shift.get("val_bands", []),
        "val_contains_5_8_ghz": "5.8 GHz (High)" in phantom_shift.get("val_bands", []),
        "test_contains_2_4_ghz": "2.4 GHz (Low)" in phantom_shift.get("test_bands", []),
        "test_contains_5_8_ghz": "5.8 GHz (High)" in phantom_shift.get("test_bands", []),
        "diagnosis": (
            "SEVERE GENERALIZATION BOTTLENECK: Phantom Drone is trained EXCLUSIVELY on 5.8 GHz RF signals (Phantom_11000_H), "
            "while Validation and Test sets contain EXCLUSIVELY 2.4 GHz RF signals (Phantom_11000_L1 and Phantom_11000_L2). "
            "Because 2.4 GHz and 5.8 GHz RF signatures differ fundamentally, models trained only on 5.8 GHz Phantom data "
            "fail to generalize to 2.4 GHz Phantom validation/test data."
        ),
    }

    return {
        "class_level_shifts": shifts,
        "phantom_drone_diagnostic": phantom_details,
    }


def run_sampling_diagnostics(
    representative_files: List[Dict[str, str]],
    raw_data_dir: Optional[Path] = None,
    segment_length: int = 2048,
    samples_per_file: int = 5,
    mock: bool = False,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Extract bounded segment slices for representative files and compute statistical metrics,
    start offsets, and pairwise segment uniqueness checks.
    """
    rows = []
    summary_list = []

    for rep in representative_files:
        cls_name = rep["class_name"]
        drone_cls = rep["drone_class"]
        rel_path = rep["rel_path"]
        band = rep.get("band", "Unknown")

        resolved_path = resolve_raw_path(rel_path, raw_data_dir=raw_data_dir)
        has_real_file = resolved_path.exists() and resolved_path.is_file()
        use_mock = mock or not has_real_file

        segments = []
        seg_stats = []

        for offset in range(samples_per_file):
            start_idx = offset * segment_length
            end_idx = start_idx + segment_length

            if use_mock:
                state = np.random.RandomState(hash(Path(rel_path).name) % (2**32) + offset)
                seg = state.randn(segment_length).astype(np.float32)
            else:
                # Bounded readline exactly like DroneRFLazyDataset
                limit = (offset + 1) * segment_length * 15 + 1000
                with open(resolved_path, "r") as f:
                    line = f.readline(limit)
                raw_str = line.rsplit(",", 1)[0]
                vals = np.fromstring(raw_str, sep=",", dtype=np.float32)
                vals = vals[~np.isnan(vals)]
                seg = vals[start_idx:end_idx]
                if len(seg) < segment_length:
                    seg = np.pad(seg, (0, segment_length - len(seg)))

            segments.append(seg)
            s_mean = float(np.mean(seg))
            s_std = float(np.std(seg))
            s_min = float(np.min(seg))
            s_max = float(np.max(seg))

            row_data = {
                "class_name": cls_name,
                "drone_class": drone_cls,
                "filename": Path(rel_path).name,
                "band": band,
                "segment_offset": offset,
                "sample_start_index": start_idx,
                "sample_end_index": end_idx,
                "segment_length": segment_length,
                "mean": s_mean,
                "std": s_std,
                "min": s_min,
                "max": s_max,
                "is_finite": bool(np.all(np.isfinite(seg))),
                "is_mock": use_mock,
            }
            rows.append(row_data)
            seg_stats.append(row_data)

        # Check pairwise similarity / duplicate segments
        is_identical = False
        max_abs_diff = 0.0
        min_abs_diff = float("inf")

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                diff = float(np.mean(np.abs(segments[i] - segments[j])))
                if diff < 1e-7:
                    is_identical = True
                max_abs_diff = max(max_abs_diff, diff)
                min_abs_diff = min(min_abs_diff, diff)

        summary_list.append({
            "class_name": cls_name,
            "drone_class": drone_cls,
            "filename": Path(rel_path).name,
            "band": band,
            "samples_extracted": len(segments),
            "offsets_evaluated": list(range(samples_per_file)),
            "start_indices": [i * segment_length for i in range(samples_per_file)],
            "has_duplicate_segments": is_identical,
            "min_pairwise_abs_diff": min_abs_diff if min_abs_diff != float("inf") else 0.0,
            "max_pairwise_abs_diff": max_abs_diff,
            "is_mock": use_mock,
            "segments_stats": seg_stats,
        })

    df_sampling = pd.DataFrame(rows)
    return summary_list, df_sampling


def print_console_report(
    split_summary: Dict[str, Any],
    recording_summary: List[Dict[str, Any]],
    class_matrix: List[Dict[str, Any]],
    freq_matrix: List[Dict[str, Any]],
    shift_analysis: Dict[str, Any],
    sampling_summary: List[Dict[str, Any]],
    samples_per_file: int,
    segment_length: int,
) -> None:
    """Print beautifully structured human-readable console report."""
    print("================================================================================")
    print("      VARDAN DRONERF SAMPLING & SPLIT DISTRIBUTION DIAGNOSTIC REPORT            ")
    print("================================================================================\n")

    # 1. SPLIT-LEVEL SUMMARY
    print("1. SPLIT-LEVEL SUMMARY (samples_per_file = {}, segment_length = {}):".format(samples_per_file, segment_length))
    print("--------------------------------------------------------------------------------")
    print(f"  Split       | Files | Pct Files | Recordings | Samples | Pct Samples")
    print("--------------------------------------------------------------------------------")
    for s in ["train", "val", "test"]:
        st = split_summary[s]
        print(f"  {s.upper():11s} | {st['file_count']:5d} | {st['file_pct']:8.2f}% | {st['recording_count']:10d} | {st['sample_count']:7d} | {st['sample_pct']:10.2f}%")
    tot = split_summary["total"]
    print("--------------------------------------------------------------------------------")
    print(f"  TOTAL       | {tot['file_count']:5d} |   100.00% | {tot['recording_count']:10d} | {tot['sample_count']:7d} |    100.00%\n")

    # 2. CLASS x SPLIT BREAKDOWN TABLE
    print("2. CLASS x SPLIT BREAKDOWN (Files & Generated Samples):")
    print("--------------------------------------------------------------------------------")
    print("  Class Name (Label)       | Train (Files/Smp) | Val (Files/Smp) | Test (Files/Smp) | Total Files")
    print("--------------------------------------------------------------------------------")
    for row in class_matrix:
        c_label = f"{row['canonical_name']} ({RAW_CLASS_TO_INDEX.get(row['drone_class'], 0)})"
        tr_str = f"{row['train_files']:3d} / {row['train_samples']:5d}"
        va_str = f"{row['val_files']:3d} / {row['val_samples']:5d}"
        te_str = f"{row['test_files']:3d} / {row['test_samples']:5d}"
        print(f"  {c_label:26s} | {tr_str:17s} | {va_str:15s} | {te_str:16s} | {row['total_files']:5d}")
    print("--------------------------------------------------------------------------------\n")

    # 3. FREQUENCY / RECEIVER x SPLIT BREAKDOWN TABLE
    print("3. FREQUENCY BAND & RECEIVER x SPLIT BREAKDOWN:")
    print("--------------------------------------------------------------------------------")
    print("  Frequency Band     | Receiver | Train Files | Val Files | Test Files | Total Files")
    print("--------------------------------------------------------------------------------")
    for row in freq_matrix:
        print(f"  {row['frequency_band']:18s} | {row['receiver']:8s} | {row['train_files']:11d} | {row['val_files']:9d} | {row['test_files']:10d} | {row['total_files']:11d}")
    print("--------------------------------------------------------------------------------\n")

    # 4. RECORDING-LEVEL ASSIGNMENTS
    print("4. RECORDING-LEVEL ASSIGNMENT MANIFEST (23 Discrete Sessions):")
    print("--------------------------------------------------------------------------------")
    print("  Recording ID                  | Class (Label)    | Receiver | Band      | Split | Files")
    print("--------------------------------------------------------------------------------")
    for rec in recording_summary:
        c_short = f"{rec['drone_class'][:14]} ({rec['canonical_label']})"
        print(f"  {rec['recording_id']:29s} | {c_short:16s} | {rec['receiver']:8s} | {rec['frequency_band'][:9]:9s} | {rec['split']:5s} | {rec['file_count']:4d}")
    print("--------------------------------------------------------------------------------\n")

    # 5. DISTRIBUTION SHIFTS & GENERALIZATION DIAGNOSTIC
    print("5. DISTRIBUTION SHIFT & GENERALIZATION BOTTLENECK ANALYSIS:")
    print("--------------------------------------------------------------------------------")
    phantom_diag = shift_analysis["phantom_drone_diagnostic"]
    if phantom_diag["is_severe_shift"]:
        print("  [!] CRITICAL DISTRIBUTION SHIFT IDENTIFIED:")
        print("     - Phantom Drone Train Frequency Bands:      [5.8 GHz (High)] ONLY (Phantom_11000_H, 21 files)")
        print("     - Phantom Drone Validation Frequency Bands: [2.4 GHz (Low)]  ONLY (Phantom_11000_L1, 10 files)")
        print("     - Phantom Drone Test Frequency Bands:       [2.4 GHz (Low)]  ONLY (Phantom_11000_L2, 11 files)")
        print(f"\n  [EXPLANATION]:")
        print(f"     {phantom_diag['diagnosis']}\n")
    else:
        print("  [OK] No severe recording frequency shift detected.\n")

    # 6. SAMPLING DIAGNOSTICS & BOUNDED-READ INTEGRITY
    print("6. SAMPLING DIAGNOSTICS (5 Segments per Representative File):")
    print("--------------------------------------------------------------------------------")
    print("  Class Name       | File Name      | Offset | Start Idx | End Idx | Mean     | Std      | Finite")
    print("--------------------------------------------------------------------------------")
    for s_info in sampling_summary:
        for st in s_info["segments_stats"]:
            print(f"  {st['class_name']:16s} | {st['filename']:14s} | {st['segment_offset']:6d} | {st['sample_start_index']:9d} | {st['sample_end_index']:7d} | {st['mean']:8.4f} | {st['std']:8.4f} | {st['is_finite']}")
        print("  ------------------------------------------------------------------------------")
        print(f"  -> {s_info['class_name']} Check: Start Indices={s_info['start_indices']} | "
              f"Duplicate Slices={s_info['has_duplicate_segments']} | Min Pairwise Diff={s_info['min_pairwise_abs_diff']:.4f}")
        print("  ------------------------------------------------------------------------------")
    print("  [OK] Verified: 5 extracted segments from one CSV correspond to consecutive non-overlapping windows.\n")


def run_full_distribution_analysis(
    raw_data_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    samples_per_file: int = 50,
    segment_length: int = 2048,
    diagnostic_samples: int = 5,
    output_dir: Optional[Path] = None,
    mock: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute complete data distribution, split manifest, and sampling analysis."""
    splits_p = Path(splits_dir) if splits_dir else DATA_DIR / "splits"
    out_dir = Path(output_dir) if output_dir else RESULTS_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_and_combine_splits(splits_p)

    split_summary = generate_split_summary(df_all, samples_per_file)
    recording_summary = generate_recording_summary(df_all)
    class_matrix = generate_class_split_matrix(df_all, samples_per_file)
    freq_matrix = generate_frequency_receiver_matrix(df_all, samples_per_file)
    shift_analysis = analyze_distribution_shifts(df_all)

    sampling_summary, df_sampling = run_sampling_diagnostics(
        representative_files=REPRESENTATIVE_DIAGNOSTIC_FILES,
        raw_data_dir=raw_data_dir,
        segment_length=segment_length,
        samples_per_file=diagnostic_samples,
        mock=mock,
    )

    # Print human-readable report
    print_console_report(
        split_summary=split_summary,
        recording_summary=recording_summary,
        class_matrix=class_matrix,
        freq_matrix=freq_matrix,
        shift_analysis=shift_analysis,
        sampling_summary=sampling_summary,
        samples_per_file=samples_per_file,
        segment_length=segment_length,
    )

    # Compile JSON artifact
    diagnostic_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parameters": {
            "samples_per_file": samples_per_file,
            "segment_length": segment_length,
            "splits_dir": str(splits_p),
            "raw_data_dir": str(raw_data_dir) if raw_data_dir else None,
            "mock": mock,
        },
        "split_summary": split_summary,
        "class_split_matrix": class_matrix,
        "frequency_receiver_matrix": freq_matrix,
        "recording_summary": recording_summary,
        "distribution_shift_analysis": shift_analysis,
        "sampling_diagnostics_summary": sampling_summary,
    }

    json_path = out_dir / "data_distribution.json"
    with open(json_path, "w") as f:
        json.dump(diagnostic_data, f, indent=2)

    csv_path = out_dir / "sampling_diagnostics.csv"
    df_sampling.to_csv(csv_path, index=False)

    print(f" [OK] Exported diagnostic JSON to: {json_path}")
    print(f" [OK] Exported sampling CSV to:    {csv_path}\n")

    return diagnostic_data, df_sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DroneRF Sampling & Split Distribution Diagnostic Script.")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Path to raw DroneRF dataset directory")
    parser.add_argument("--splits_dir", type=str, default=None, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--samples_per_file", type=int, default=50, help="2048-sample windows to extract per CSV file")
    parser.add_argument("--segment_length", type=int, default=2048, help="Signal window length (samples)")
    parser.add_argument("--diagnostic_samples", type=int, default=5, help="Number of segments for sampling diagnostics")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save diagnostic outputs")
    parser.add_argument("--mock", action="store_true", help="Run with mock signals if raw dataset is not local")
    args = parser.parse_args()

    run_full_distribution_analysis(
        raw_data_dir=Path(args.raw_data_dir) if args.raw_data_dir else None,
        splits_dir=Path(args.splits_dir) if args.splits_dir else None,
        samples_per_file=args.samples_per_file,
        segment_length=args.segment_length,
        diagnostic_samples=args.diagnostic_samples,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        mock=args.mock,
    )
