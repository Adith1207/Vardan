"""
train_baselines.py
------------------

Master Baseline Training & Evaluation Execution Script for Vardan Research Project.

Executes training for:
1. FGCS2019DNN (200 epochs, batch size 10, lr 1e-3)
2. Baseline1DCNN (100 epochs, batch size 32, lr 1e-3)
3. DSCNN (100 epochs, batch size 32, lr 1e-3)
4. MobileNetV3Small (100 epochs, batch size 32, lr 1e-3)
5. VardhanRFNet (100 epochs, batch size 32, lr 1e-3)

Strict Requirements:
- Seed = 42
- Fixed 4-class recording-level split (308 Train, 73 Val, 73 Test; zero recording overlap)
- Normalization statistics learned ONLY from train split
- Validation used strictly for best checkpoint selection (best.pt)
- Latest checkpoint (last.pt) saved for seamless resumption across sessions/accounts
- Single-pass evaluation on Test dataset using loaded best checkpoint
- Results exported to results/baselines/<model_name>/ with run_config.json
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from benchmark.bench import benchmark_inference_latency, profile_model_footprint
from config import NUM_CLASSES
from constants import CLASS_NAMES, LABEL_MAP, RAW_CLASS_TO_INDEX
from data.loader import CLASS_MAPPING, fit_train_normalization_stats, get_dataloader
from evaluation.metrics import calculate_metrics, generate_confusion_matrix
from models.model_factory import get_model
from models.trainer import BaselineTrainer, set_reproducible_seed
from utils.paths import DATA_DIR, RESULTS_DIR


def run_preflight_checks(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    raw_data_dir: Optional[Path] = None,
    mock: bool = False,
) -> Tuple[bool, Dict[str, float]]:
    """Verify dataset split manifests, class mappings, and compute train-only normalization stats."""
    print("=================================================================")
    print("      AUTOMATED PREFLIGHT VERIFICATION CHECK                     ")
    print("=================================================================")

    assert train_csv.exists(), f"Missing split manifest: {train_csv}"
    assert val_csv.exists(), f"Missing split manifest: {val_csv}"
    assert test_csv.exists(), f"Missing split manifest: {test_csv}"

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_te = pd.read_csv(test_csv)

    total_files = len(df_tr) + len(df_va) + len(df_te)
    print(f"1. Split counts: Train={len(df_tr)}, Val={len(df_va)}, Test={len(df_te)}, Total={total_files}")
    assert len(df_tr) == 308, f"Expected 308 train files, got {len(df_tr)}"
    assert len(df_va) == 73, f"Expected 73 val files, got {len(df_va)}"
    assert len(df_te) == 73, f"Expected 73 test files, got {len(df_te)}"

    # Recording-level isolation check
    tr_recs = set(df_tr["recording_id"]) if "recording_id" in df_tr.columns else set()
    va_recs = set(df_va["recording_id"]) if "recording_id" in df_va.columns else set()
    te_recs = set(df_te["recording_id"]) if "recording_id" in df_te.columns else set()

    if tr_recs:
        assert len(tr_recs & va_recs) == 0, "Recording overlap between Train and Val!"
        assert len(tr_recs & te_recs) == 0, "Recording overlap between Train and Test!"
        assert len(va_recs & te_recs) == 0, "Recording overlap between Val and Test!"
        print("[OK] Zero recording overlap verified.")

    # File path isolation check
    tr_paths = set(df_tr["relative_path"])
    va_paths = set(df_va["relative_path"])
    te_paths = set(df_te["relative_path"])

    assert len(tr_paths & va_paths) == 0, "File overlap between Train and Val!"
    assert len(tr_paths & te_paths) == 0, "File overlap between Train and Test!"
    assert len(va_paths & te_paths) == 0, "File overlap between Val and Test!"
    print("[OK] Zero file overlap verified.")

    assert NUM_CLASSES == 4, f"NUM_CLASSES must be 4, got {NUM_CLASSES}"
    assert CLASS_MAPPING == RAW_CLASS_TO_INDEX, "CLASS_MAPPING mismatch with constants.py!"
    print("[OK] Canonical 4 classes verified.")

    # Fit train-only normalization stats
    train_stats = fit_train_normalization_stats(train_csv, max_files=10, raw_data_dir=raw_data_dir)
    print(f"[OK] Train-only normalization stats computed: {train_stats}")

    print("[OK] All preflight checks PASSED cleanly!\n")
    return True, train_stats


def train_single_model(
    model_cfg: dict,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    train_stats: dict,
    checkpoints_dir: Path,
    output_dir: Path,
    raw_data_dir: Optional[Path] = None,
    samples_per_file: int = 50,
    resume_path: Optional[Path] = None,
    save_every: int = 5,
    mock: bool = False,
) -> dict:
    """Execute training, checkpointing, and test evaluation for one model."""
    title = model_cfg["title"]
    key = model_cfg["key"]
    bs = model_cfg["batch_size"]
    epochs = model_cfg["epochs"]
    lr = model_cfg["lr"]
    opt_type = model_cfg.get("optimizer", "Adam")
    wd = model_cfg.get("weight_decay", 0.0)
    in_shape = model_cfg["in_shape"]

    print("=================================================================")
    print(f"      EXPERIMENT: {title} ({key})                                ")
    print(f"      Epochs: {epochs} | Batch Size: {bs} | Optimizer: {opt_type} | LR: {lr} (WD: {wd})")
    print("=================================================================")

    set_reproducible_seed(42)

    # Prepare DataLoaders
    train_loader = get_dataloader(
        train_csv,
        model_name=key,
        norm_stats=train_stats,
        batch_size=bs,
        shuffle=True,
        samples_per_file=samples_per_file,
        raw_data_dir=raw_data_dir,
        mock=mock,
    )
    val_loader = get_dataloader(
        val_csv,
        model_name=key,
        norm_stats=train_stats,
        batch_size=bs,
        shuffle=False,
        samples_per_file=samples_per_file,
        raw_data_dir=raw_data_dir,
        mock=mock,
    )
    test_loader = get_dataloader(
        test_csv,
        model_name=key,
        norm_stats=train_stats,
        batch_size=bs,
        shuffle=False,
        samples_per_file=samples_per_file,
        raw_data_dir=raw_data_dir,
        mock=mock,
    )

    # Instantiate Model
    model = get_model(key, num_classes=4)
    trainer = BaselineTrainer(
        model=model,
        model_name=key,
        learning_rate=lr,
        weight_decay=wd,
        optimizer_type=opt_type,
        seed=42,
    )

    model_ckpt_dir = checkpoints_dir / key
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = model_ckpt_dir / "best.pt"
    last_ckpt_path = model_ckpt_dir / "last.pt"

    model_out_dir = output_dir / key
    model_out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1

    # Resume training if requested
    if resume_path is not None:
        resume_p = Path(resume_path)
        print(f"   - Resuming training from checkpoint: {resume_p}")
        ckpt_data = trainer.load_checkpoint(resume_p, resume_training=True)
        start_epoch = ckpt_data.get("epoch", 0) + 1
        print(f"   [OK] Checkpoint loaded. Resuming at epoch {start_epoch}/{epochs} (Best Val Loss: {trainer.best_val_loss:.4f})")

    # Construct run configuration
    run_config = {
        "model_title": title,
        "model_key": key,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "raw_data_dir": str(raw_data_dir) if raw_data_dir else None,
        "splits_dir": str(train_csv.parent),
        "train_csv": train_csv.name,
        "val_csv": val_csv.name,
        "test_csv": test_csv.name,
        "samples_per_file": samples_per_file,
        "segment_length": 2048,
        "batch_size": bs,
        "learning_rate": lr,
        "optimizer": opt_type,
        "weight_decay": wd,
        "loss": "CrossEntropyLoss",
        "total_epochs": epochs,
        "start_epoch": start_epoch,
        "resumed_from": str(resume_path) if resume_path else None,
        "random_seed": 42,
        "device": str(trainer.device),
        "mock": mock,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Save run_config.json immediately
    with open(model_out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    if start_epoch > 1 and trainer.history:
        history = trainer.history
        for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro"]:
            if k not in history:
                history[k] = []
        num_past = max(len(history.get("train_loss", [])), len(history.get("val_loss", [])))
        if len(history["epoch"]) < num_past:
            history["epoch"] = list(range(1, num_past + 1))
        for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro"]:
            while len(history[k]) < num_past:
                history[k].append(0.0)
    else:
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
        }
    trainer.history = history

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
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

        if epoch % 10 == 0 or epoch == start_epoch or epoch == epochs:
            print(f"   - Epoch {epoch:3d}/{epochs} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Best checkpoint update
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.best_val_acc = val_acc
            trainer.best_val_f1 = val_f1
            trainer.best_epoch = epoch

            trainer.save_checkpoint(
                best_ckpt_path,
                epoch=epoch,
                val_metrics=val_metrics,
                is_best=True,
                extra_config=run_config,
            )

        # Periodic latest checkpoint save for resumption
        if epoch % save_every == 0 or epoch == epochs:
            trainer.save_checkpoint(
                last_ckpt_path,
                epoch=epoch,
                val_metrics=val_metrics,
                is_best=False,
                extra_config=run_config,
            )

    train_time = time.time() - start_time
    print(f"   [OK] Training complete in {train_time:.2f}s. Best Epoch: {trainer.best_epoch} (Val Loss: {trainer.best_val_loss:.4f})")

    # -------------------------------------------------------------------
    # TEST EVALUATION USING BEST CHECKPOINT
    # -------------------------------------------------------------------
    target_eval_ckpt = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
    print(f"   - Loading best checkpoint from {target_eval_ckpt} for Test evaluation...")
    test_trainer = BaselineTrainer(model=get_model(key, num_classes=4), model_name=key, learning_rate=lr, seed=42)
    test_trainer.load_checkpoint(target_eval_ckpt, resume_training=False)
    test_metrics = test_trainer.evaluate(test_loader)

    # Compute confusion matrix
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(test_trainer.device)
            logits = test_trainer.model(x_b)
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
    prof = profile_model_footprint(test_trainer.model)
    lat = benchmark_inference_latency(test_trainer.model, input_shape=in_shape, num_runs=50, device="cpu")

    # Save History
    pd.DataFrame(history).to_csv(model_out_dir / "history.csv", index=False)
    with open(model_out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save Confusion Matrix
    pd.DataFrame(cm, index=[LABEL_MAP[i] for i in range(4)], columns=[LABEL_MAP[i] for i in range(4)]).to_csv(model_out_dir / "confusion_matrix.csv")
    with open(model_out_dir / "confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f, indent=2)

    # Save Final Metrics
    final_metrics_data = {
        "model_title": title,
        "model_key": key,
        "best_epoch": trainer.best_epoch,
        "best_val_loss": float(trainer.best_val_loss),
        "best_val_accuracy": float(trainer.best_val_acc),
        "best_val_f1_macro": float(trainer.best_val_f1),
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

    with open(model_out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics_data, f, indent=2)

    print(f"   [OK] TEST EVALUATION COMPLETED: Acc={test_metrics['accuracy']:.4f}, Macro-F1={test_metrics['f1_macro']:.4f}")
    print(f"   [OK] Exported baseline results to {model_out_dir}\n")

    return final_metrics_data


def execute_baseline_suite(
    models_to_run: Optional[List[str]] = None,
    raw_data_dir: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    samples_per_file: int = 50,
    override_epochs: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    override_lr: Optional[float] = None,
    resume_path: Optional[Path] = None,
    save_every: int = 5,
    mock: bool = False,
):
    """Execute training pipeline across specified baseline models."""
    splits_p = Path(splits_dir) if splits_dir else DATA_DIR / "splits"
    out_p = Path(output_dir) if output_dir else RESULTS_DIR / "baselines"
    ckpt_p = Path(checkpoints_dir) if checkpoints_dir else PROJECT_ROOT / "models" / "checkpoints" / "baselines"

    train_csv = splits_p / "train.csv"
    val_csv = splits_p / "val.csv"
    test_csv = splits_p / "test.csv"

    passed, train_stats = run_preflight_checks(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        raw_data_dir=raw_data_dir,
        mock=mock,
    )
    if not passed:
        print("Preflight verification failed! Aborting.")
        return

    all_models_config = [
        {
            "title": "FGCS2019DNN",
            "key": "fgcs2019dnn",
            "batch_size": override_batch_size or 10,
            "epochs": override_epochs or 200,
            "lr": override_lr or 1e-3,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "in_shape": (2048,),
        },
        {
            "title": "Baseline1DCNN",
            "key": "baseline1dcnn",
            "batch_size": override_batch_size or 32,
            "epochs": override_epochs or 100,
            "lr": override_lr or 1e-3,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "in_shape": (8, 256),
        },
        {
            "title": "DSCNN",
            "key": "dscnn",
            "batch_size": override_batch_size or 32,
            "epochs": override_epochs or 100,
            "lr": override_lr or 1e-3,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "in_shape": (1, 2048),
        },
        {
            "title": "MobileNetV3Small",
            "key": "mobilenetv3small",
            "batch_size": override_batch_size or 32,
            "epochs": override_epochs or 100,
            "lr": override_lr or 1e-3,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "in_shape": (1, 65, 61),
        },
        {
            "title": "VardhanRFNet",
            "key": "vardhan",
            "batch_size": override_batch_size or 32,
            "epochs": override_epochs or 100,
            "lr": override_lr or 1e-3,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "in_shape": (1, 2048),
        },
    ]

    selected_keys = [k.lower().strip() for k in models_to_run] if models_to_run else ["all"]
    if "all" in selected_keys:
        active_configs = all_models_config
    else:
        active_configs = [c for c in all_models_config if c["key"] in selected_keys]

    if not active_configs:
        print(f"No valid models selected from: {models_to_run}. Available: {[c['key'] for c in all_models_config]}")
        return

    summary_results = []

    for cfg in active_configs:
        res = train_single_model(
            model_cfg=cfg,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            train_stats=train_stats,
            checkpoints_dir=ckpt_p,
            output_dir=out_p,
            raw_data_dir=raw_data_dir,
            samples_per_file=samples_per_file,
            resume_path=resume_path,
            save_every=save_every,
            mock=mock,
        )
        summary_results.append(res)

    # Save summary matrix across executed models
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "summary_matrix.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    print("=================================================================")
    print(f" [OK] ALL {len(active_configs)} MODELS EXECUTED SUCCESSFULLY!      ")
    print("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master baseline training & evaluation runner for Vardan.")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated model keys or 'all' (options: fgcs2019dnn, baseline1dcnn, dscnn, mobilenetv3small, vardhan, all)")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Path to raw DroneRF dataset directory")
    parser.add_argument("--splits_dir", type=str, default=None, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save metrics, histories, and matrices")
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="Directory to save model checkpoint weights")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override default batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override default learning rate")
    parser.add_argument("--samples_per_file", type=int, default=50, help="2048-sample windows to extract per CSV file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--save_every", type=int, default=5, help="Save frequency for latest checkpoint (epochs)")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic signals for testing without raw dataset")
    args = parser.parse_args()

    models_list = [m.strip() for m in args.models.split(",") if m.strip()]

    execute_baseline_suite(
        models_to_run=models_list,
        raw_data_dir=Path(args.raw_data_dir) if args.raw_data_dir else None,
        splits_dir=Path(args.splits_dir) if args.splits_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        checkpoints_dir=Path(args.checkpoints_dir) if args.checkpoints_dir else None,
        samples_per_file=args.samples_per_file,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_lr=args.lr,
        resume_path=Path(args.resume) if args.resume else None,
        save_every=args.save_every,
        mock=args.mock,
    )
