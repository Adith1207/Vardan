"""
run_smoke_test.py
------------------

Comprehensive Verification & Tiny Smoke Test Suite for Vardan Counter-UAS Baselines and Proposed Architecture.

Tasks Executed:
- Task 1: Verify Dataset Pipeline (Train, Val, Test DataLoaders for all 5 models).
- Task 2: Verify Model Factory & Computational Parameter Profiling.
- Task 3: Training Pipeline, Raw Logits, & Loss Verification for 5 models.
- Task 4: Tiny Smoke Test (2 epochs, 2 batches/epoch per model, zero NaN/Inf).
- Task 5: Common Evaluation Metrics & Confusion Matrix.
- Task 6: Inference Latency Benchmark across all 5 architectures.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.bench import benchmark_inference_latency, profile_model_footprint
from data.loader import fit_train_normalization_stats, get_dataloader
from evaluation.metrics import calculate_metrics, generate_confusion_matrix
from models.model_factory import get_model
from models.trainer import BaselineTrainer, set_reproducible_seed
from utils.paths import DATA_DIR


def run_full_smoke_test_suite(
    raw_data_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    mock: bool = True,
):
    print("=================================================================")
    print("      Vardan Baseline & Research Architecture Smoke Test Suite   ")
    print("=================================================================\n")

    set_reproducible_seed(42)

    splits_p = Path(splits_dir) if splits_dir else DATA_DIR / "splits"
    train_csv = splits_p / "train.csv"
    val_csv = splits_p / "val.csv"
    test_csv = splits_p / "test.csv"

    # Fit train-only normalization stats
    train_stats = fit_train_normalization_stats(train_csv, max_files=10, raw_data_dir=raw_data_dir)

    models_specs = [
        ("FGCS2019DNN", "fgcs2019dnn", (2048,)),
        ("Baseline1DCNN", "baseline1dcnn", (8, 256)),
        ("DSCNN", "dscnn", (1, 2048)),
        ("CompressiveSensing", "compressed_sensing", (1, 1024)),
        ("MobileNetV3Small", "mobilenetv3small", (1, 65, 61)),
        ("VardhanRFNet", "vardhan", (1, 2048)),
    ]

    # -------------------------------------------------------------------
    # TASK 1: VERIFY DATASET PIPELINE
    # -------------------------------------------------------------------
    print("TASK 1: Verify Dataset Pipeline DataLoaders:")
    for model_title, model_key, expected_in_shape in models_specs:
        print(f"\n   --- Representation: {model_title} ({model_key}) ---")
        for split_name, split_path in [("Train", train_csv), ("Val", val_csv), ("Test", test_csv)]:
            loader = get_dataloader(
                split_csv=split_path,
                model_name=model_key,
                norm_stats=train_stats,
                batch_size=4,
                shuffle=False,
                samples_per_file=2,
                raw_data_dir=raw_data_dir,
                mock=mock,
            )
            x_batch, y_batch = next(iter(loader))

            x_np = x_batch.numpy()
            y_np = y_batch.numpy()

            nan_count = int(np.isnan(x_np).sum())
            inf_count = int(np.isinf(x_np).sum())
            valid_labels = set(y_np).issubset({0, 1, 2, 3})

            assert nan_count == 0 and inf_count == 0, f"NaN/Inf detected in {split_name} {model_key}!"
            assert valid_labels, f"Invalid label detected in {split_name} {model_key}!"

            print(f"   [{split_name:5s}] Samples={len(loader.dataset):4d} | Batch Size=4 | "
                  f"Input Shape={str(tuple(x_batch.shape)):15s} | Dtype={x_batch.dtype} | "
                  f"Range=[{x_np.min():.2f}, {x_np.max():.2f}] | NaN={nan_count} | Inf={inf_count} | Labels Valid={valid_labels}")

    # -------------------------------------------------------------------
    # TASK 2: VERIFY MODEL FACTORY & PARAMETERS
    # -------------------------------------------------------------------
    print("\n\nTASK 2: Verify Model Factory & Parameter Counts:")
    for model_title, model_key, in_shape in models_specs:
        m = get_model(model_key, num_classes=4)
        prof = profile_model_footprint(m)
        print(f"   - Model Key: '{model_key:16s}' | Class: {m.__class__.__name__:16s} | "
              f"Total Params: {prof['total_parameters']:8,d} | Trainable: {prof['trainable_parameters']:8,d} | "
              f"Est. Size: {prof['estimated_size_mb']:.4f} MB")

    # -------------------------------------------------------------------
    # TASK 3: TINY SMOKE TEST (2 EPOCHS, 2 BATCHES/EPOCH PER MODEL)
    # -------------------------------------------------------------------
    print("\n\nTASK 3: Tiny Smoke Test (2 Epochs, 2 Batches/Epoch):")
    smoke_results = []

    for model_title, model_key, expected_in_shape in models_specs:
        train_loader = get_dataloader(train_csv, model_name=model_key, norm_stats=train_stats, batch_size=4, samples_per_file=2, raw_data_dir=raw_data_dir, mock=mock)
        val_loader = get_dataloader(val_csv, model_name=model_key, norm_stats=train_stats, batch_size=4, samples_per_file=2, raw_data_dir=raw_data_dir, mock=mock)

        model = get_model(model_key, num_classes=4)
        trainer = BaselineTrainer(model=model, model_name=model_key, learning_rate=1e-3, seed=42)
        res = trainer.run_smoke_test(train_loader, val_loader, epochs=2, batches_per_epoch=2)
        smoke_results.append(res)

    # -------------------------------------------------------------------
    # TASK 4: COMMON EVALUATION METRICS VERIFICATION
    # -------------------------------------------------------------------
    print("\n\nTASK 4: Common Evaluation Metrics Test:")
    y_true_test = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred_test = np.array([0, 1, 2, 3, 0, 1, 2, 0])
    test_metrics = calculate_metrics(y_true_test, y_pred_test)
    cm = generate_confusion_matrix(y_true_test, y_pred_test, num_classes=4)

    print(f"   - Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   - Test Macro F1:  {test_metrics['f1_macro']:.4f}")
    print(f"   - Confusion Matrix (4x4):\n{cm}")

    # -------------------------------------------------------------------
    # TASK 5: COMPUTATIONAL BENCHMARKS (LATENCY PROFILE)
    # -------------------------------------------------------------------
    print("\n\nTASK 5: Computational Benchmarks (CPU Inference Latency):")
    for model_title, model_key, in_shape in models_specs:
        m = get_model(model_key, num_classes=4)
        lat = benchmark_inference_latency(m, input_shape=in_shape, num_runs=50, device="cpu")
        prof = profile_model_footprint(m)
        print(f"   - {model_title:16s} | Latency: {lat['mean_latency_ms']:.3f} ms ± {lat['std_latency_ms']:.3f} ms | Params: {prof['total_parameters']:,d}")

    print("\n=================================================================")
    print(" [OK] All Tasks & Smoke Tests Executed Successfully!")
    print("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test runner for Vardan baseline and custom models.")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Path to raw DroneRF dataset directory")
    parser.add_argument("--splits_dir", type=str, default=None, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--mock", action="store_true", default=True, help="Run with mock signals")
    args = parser.parse_args()

    run_full_smoke_test_suite(
        raw_data_dir=Path(args.raw_data_dir) if args.raw_data_dir else None,
        splits_dir=Path(args.splits_dir) if args.splits_dir else None,
        mock=args.mock,
    )
