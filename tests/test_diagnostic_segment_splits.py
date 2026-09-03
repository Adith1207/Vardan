"""
test_diagnostic_segment_splits.py
---------------------------------

Unit and contract tests for the diagnostic randomized segment-level split generation utility:
- Zero exact 2048-sample window leakage across Train, Val, and Test
- Intentional recording/file overlap presence (verifying segment-level protocol)
- Complete 4-class representation across all splits
- Exact segment summation (Train + Val + Test == Total Segments = 2270)
- Determinism with random_seed=42
- DataLoader compatibility across all 6 model representations
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.create_diagnostic_segment_splits import generate_diagnostic_segment_splits
from src.data.loader import get_dataloader
from src.utils.paths import DATA_DIR


def test_diagnostic_segment_split_generation_and_counts():
    """Verify segment split generation, exact summation, and zero window leakage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "diag_split"
        df_tr, df_va, df_te, meta = generate_diagnostic_segment_splits(
            metadata_path=DATA_DIR / "metadata" / "dronerf_metadata.csv",
            output_dir=out_dir,
            samples_per_file=5,
            segment_length=2048,
            random_seed=42,
        )

        total_segs = 454 * 5  # 2270
        assert len(df_tr) == 1586
        assert len(df_va) == 342
        assert len(df_te) == 342
        assert len(df_tr) + len(df_va) + len(df_te) == total_segs

        # 1. Zero exact window leakage
        tr_uids = set(df_tr["segment_unique_id"])
        va_uids = set(df_va["segment_unique_id"])
        te_uids = set(df_te["segment_unique_id"])

        assert len(tr_uids.intersection(va_uids)) == 0, "Train and Val share exact window!"
        assert len(tr_uids.intersection(te_uids)) == 0, "Train and Test share exact window!"
        assert len(va_uids.intersection(te_uids)) == 0, "Val and Test share exact window!"

        # 2. Recording overlap is present (intentional for segment-level diagnosis)
        tr_recs = set(df_tr["recording_id"])
        va_recs = set(df_va["recording_id"])
        te_recs = set(df_te["recording_id"])

        assert len(tr_recs.intersection(va_recs)) == 23, "Train and Val should share recording sessions in segment split"
        assert len(tr_recs.intersection(te_recs)) == 23, "Train and Test should share recording sessions in segment split"

        # 3. All 4 classes represented in every split
        for split_df in [df_tr, df_va, df_te]:
            assert split_df["drone_class"].nunique() == 4
            assert set(split_df["drone_class"].unique()) == {
                "AR Drone",
                "Backround RF activities",
                "Bepop drone",
                "Phantom drone",
            }


def test_diagnostic_segment_split_determinism():
    """Verify that seed=42 produces bitwise identical splits across repeated calls."""
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        df_tr1, df_va1, df_te1, _ = generate_diagnostic_segment_splits(
            output_dir=Path(tmp1),
            samples_per_file=5,
            random_seed=42,
        )
        df_tr2, df_va2, df_te2, _ = generate_diagnostic_segment_splits(
            output_dir=Path(tmp2),
            samples_per_file=5,
            random_seed=42,
        )

        pd.testing.assert_frame_equal(df_tr1, df_tr2)
        pd.testing.assert_frame_equal(df_va1, df_va2)
        pd.testing.assert_frame_equal(df_te1, df_te2)


def test_dataloader_compatibility_with_diagnostic_segment_splits():
    """Verify DataLoader reads exact segment items from diagnostic segment splits for all models."""
    diag_dir = DATA_DIR / "splits" / "diagnostic_segment"
    train_csv = diag_dir / "train.csv"

    expected_shapes = {
        "fgcs2019dnn": (4, 2048),
        "baseline1dcnn": (4, 8, 256),
        "dscnn": (4, 1, 2048),
        "compressed_sensing": (4, 1, 1024),
        "mobilenetv3small": (4, 1, 65, 61),
        "vardhan": (4, 1, 2048),
    }

    for model_name, expected_shape in expected_shapes.items():
        loader = get_dataloader(
            split_csv=train_csv,
            model_name=model_name,
            batch_size=4,
            mock=True,
        )
        assert len(loader.dataset) == 1586, f"Expected 1586 items in dataset, got {len(loader.dataset)}"
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == expected_shape, f"Model {model_name} produced batch {batch_x.shape}, expected {expected_shape}"
        assert batch_y.shape == (4,)
        assert torch.isfinite(batch_x).all()


def test_preflight_checks_both_split_types():
    """Verify scripts/train_baselines.py run_preflight_checks supports both split types."""
    from scripts.train_baselines import run_preflight_checks

    # 1. Primary split check
    passed_prim, stats_prim = run_preflight_checks(
        DATA_DIR / "splits" / "train.csv",
        DATA_DIR / "splits" / "val.csv",
        DATA_DIR / "splits" / "test.csv",
        mock=True,
    )
    assert passed_prim is True
    assert "mean" in stats_prim

    # 2. Diagnostic segment split check
    passed_diag, stats_diag = run_preflight_checks(
        DATA_DIR / "splits" / "diagnostic_segment" / "train.csv",
        DATA_DIR / "splits" / "diagnostic_segment" / "val.csv",
        DATA_DIR / "splits" / "diagnostic_segment" / "test.csv",
        mock=True,
    )
    assert passed_diag is True
    assert "mean" in stats_diag
