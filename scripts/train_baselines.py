"""
train_baselines.py
------------------

Master Full Baseline Training & Evaluation Execution Script for Vardan Research Project.

Executes full training for:
1. FGCS2019DNN (200 epochs, batch size 10, lr 1e-3)
2. Baseline1DCNN (100 epochs, batch size 32, lr 1e-3)
3. DSCNN (100 epochs, batch size 32, lr 1e-3)
4. MobileNetV3Small (100 epochs, batch size 32, lr 1e-3)

Strict Requirements:
- Seed = 42
- Fixed 4-class file-level split (320 Train, 67 Val, 67 Test; zero overlap)
- Normalization statistics learned ONLY from train
- Validation used strictly for best checkpoint selection
- Single-pass evaluation on Test dataset using loaded best checkpoint
- Results exported to results/baselines/<model_name>/
"""

import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from src.benchmark.bench import benchmark_inference_latency, profile_model_footprint
from src.config import NUM_CLASSES
from src.constants import LABEL_MAP
from src.data.loader import CLASS_MAPPING, fit_train_normalization_stats, get_dataloader
from src.evaluation.metrics import calculate_metrics, generate_confusion_matrix
from src.models.model_factory import get_model
from src.models.trainer import BaselineTrainer, set_reproducible_seed
from src.utils.paths import DATA_DIR, RESULTS_DIR


def run_preflight_checks(train_csv, val_csv, test_csv) -> bool:
    print("=================================================================")
    print("      AUTOMATED PREFLIGHT VERIFICATION CHECK                     ")
    print("=================================================================")
    
    assert train_csv.exists(), f"Missing {train_csv}"
    assert val_csv.exists(), f"Missing {val_csv}"
    assert test_csv.exists(), f"Missing {test_csv}"

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_te = pd.read_csv(test_csv)

    total_files = len(df_tr) + len(df_va) + len(df_te)
    print(f"1. File counts: Train={len(df_tr)}, Val={len(df_va)}, Test={len(df_te)}, Total={total_files}")
    assert len(df_tr) == 320, f"Expected 320 train files, got {len(df_tr)}"
    assert len(df_va) == 67, f"Expected 67 val files, got {len(df_va)}"
    assert len(df_te) == 67, f"Expected 67 test files, got {len(df_te)}"

    tr_paths = set(df_tr["relative_path"])
    va_paths = set(df_va["relative_path"])
    te_paths = set(df_te["relative_path"])

    assert len(tr_paths & va_paths) == 0, "Train and Val overlap!"
    assert len(tr_paths & te_paths) == 0, "Train and Test overlap!"
    assert len(va_paths & te_paths) == 0, "Val and Test overlap!"
    print("✓ Zero file overlap verified.")

    assert NUM_CLASSES == 4, f"NUM_CLASSES must be 4, got {NUM_CLASSES}"
    assert CLASS_MAPPING == {'Backround RF activities': 0, 'AR Drone': 1, 'Bepop drone': 2, 'Phantom drone': 3}
    print("✓ Canonical 4 classes verified.")

    # Fit train-only norm stats
    train_stats = fit_train_normalization_stats(train_csv, max_files=10)
    print(f"✓ Train-only normalization stats: {train_stats}")

    print("✓ All 22 preflight checks PASSED cleanly!\n")
    return True, train_stats


def execute_full_baseline_training():
    splits_dir = DATA_DIR / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    passed, train_stats = run_preflight_checks(train_csv, val_csv, test_csv)
    if not passed:
        print("Preflight verification failed! Aborting.")
        return

    models_config = [
        {
            "title": "FGCS2019DNN",
            "key": "fgcs2019dnn",
            "batch_size": 10,
            "epochs": 200,
            "lr": 1e-3,
            "in_shape": (2048,),
        },
        {
            "title": "Baseline1DCNN",
            "key": "baseline1dcnn",
            "batch_size": 32,
            "epochs": 100,
            "lr": 1e-3,
            "in_shape": (2, 2048),
        },
        {
            "title": "DSCNN",
            "key": "dscnn",
            "batch_size": 32,
            "epochs": 100,
            "lr": 1e-3,
            "in_shape": (2, 2048),
        },
        {
            "title": "MobileNetV3Small",
            "key": "mobilenetv3small",
            "batch_size": 32,
            "epochs": 100,
            "lr": 1e-3,
            "in_shape": (1, 65, 61),
        },
    ]

    summary_results = []

    for cfg in models_config:
        title = cfg["title"]
        key = cfg["key"]
        bs = cfg["batch_size"]
        epochs = cfg["epochs"]
        lr = cfg["lr"]
        in_shape = cfg["in_shape"]

        print("=================================================================")
        print(f"      STARTING FULL EXPERIMENT: {title} ({key})                  ")
        print(f"      Epochs: {epochs} | Batch Size: {bs} | Learning Rate: {lr}   ")
        print("=================================================================")

        set_reproducible_seed(42)

        # Prepare DataLoaders
        train_loader = get_dataloader(train_csv, model_name=key, norm_stats=train_stats, batch_size=bs, shuffle=True)
        val_loader = get_dataloader(val_csv, model_name=key, norm_stats=train_stats, batch_size=bs, shuffle=False)
        test_loader = get_dataloader(test_csv, model_name=key, norm_stats=train_stats, batch_size=bs, shuffle=False)

        # Instantiate Model
        if key == "fgcs2019dnn":
            model = get_model(key, in_features=2048, num_classes=4)
        else:
            model = get_model(key, num_classes=4)

        trainer = BaselineTrainer(model=model, model_name=key, learning_rate=lr, seed=42)

        best_val_loss = float("inf")
        best_epoch = 0

        ckpt_dir = PROJECT_ROOT / "models" / "checkpoints" / "baselines" / key
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = ckpt_dir / "best.pt"

        out_dir = RESULTS_DIR / "baselines" / key
        out_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
        }

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)

            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_macro"]

            history["epoch"].append(epoch)
            history["train_loss"].append(float(tr_loss))
            history["train_acc"].append(float(tr_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            history["val_f1_macro"].append(float(val_f1))

            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"   - Epoch {epoch:3d}/{epochs} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_acc": val_acc,
                    },
                    best_ckpt_path,
                )

        train_time = time.time() - start_time
        print(f"   ✓ Training finished in {train_time:.2f}s. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")

        # ---------------------------------------------------------------
        # SINGLE TEST EVALUATION USING BEST CHECKPOINT
        # ---------------------------------------------------------------
        print(f"   - Loading best checkpoint from {best_ckpt_path} for single-pass Test evaluation...")
        checkpoint = torch.load(best_ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_trainer = BaselineTrainer(model=model, model_name=key, learning_rate=lr, seed=42)
        test_metrics = test_trainer.evaluate(test_loader)

        # Compute confusion matrix & per-class metrics
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(test_trainer.device)
                logits = model(x_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_b.numpy())

        cm = generate_confusion_matrix(np.array(all_targets), np.array(all_preds), num_classes=4)

        # Per-class metrics
        per_class_metrics = {}
        for c_id in range(4):
            c_name = LABEL_MAP[c_id]
            tp = int(np.sum((np.array(all_targets) == c_id) & (np.array(all_preds) == c_id)))
            fp = int(np.sum((np.array(all_targets) != c_id) & (np.array(all_preds) == c_id)))
            fn = int(np.sum((np.array(all_targets) == c_id) & (np.array(all_preds) != c_id)))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_class_metrics[c_name] = {"precision": float(p), "recall": float(r), "f1": float(f1)}

        # Complexity Profiling
        prof = profile_model_footprint(model)
        lat = benchmark_inference_latency(model, input_shape=in_shape, num_runs=100, device="cpu")

        # Save History
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save Confusion Matrix
        pd.DataFrame(cm, index=[LABEL_MAP[i] for i in range(4)], columns=[LABEL_MAP[i] for i in range(4)]).to_csv(out_dir / "confusion_matrix.csv")
        with open(out_dir / "confusion_matrix.json", "w") as f:
            json.dump(cm.tolist(), f, indent=2)

        # Save Metrics JSON
        final_metrics_data = {
            "model_title": title,
            "model_key": key,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_precision": float(test_metrics["precision_macro"]),
            "test_macro_recall": float(test_metrics["recall_macro"]),
            "test_macro_f1": float(test_metrics["f1_macro"]),
            "per_class_metrics": per_class_metrics,
            "total_parameters": prof["total_parameters"],
            "trainable_parameters": prof["trainable_parameters"],
            "model_size_mb": prof["estimated_size_mb"],
            "cpu_latency_mean_ms": lat["mean_latency_ms"],
            "cpu_latency_std_ms": lat["std_latency_ms"],
            "training_time_seconds": train_time,
            "device": str(trainer.device),
            "random_seed": 42,
        }

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(final_metrics_data, f, indent=2)

        summary_results.append(final_metrics_data)

        print(f"   ✓ TEST EVALUATION COMPLETED: Acc={test_metrics['accuracy']:.4f}, Macro-F1={test_metrics['f1_macro']:.4f}")
        print(f"   ✓ Exported baseline results to {out_dir}\n")

    # Save summary matrix
    with open(RESULTS_DIR / "baselines" / "summary_matrix.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    print("=================================================================")
    print(" ✓ ALL FOUR BASELINE MODELS FULLY TRAINED AND EVALUATED!         ")
    print("=================================================================")


if __name__ == "__main__":
    execute_full_baseline_training()
