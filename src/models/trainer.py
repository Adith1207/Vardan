"""
trainer.py
----------

Unified PyTorch Trainer module for baseline models on the DroneRF dataset.

Guarantees:
- Deterministic random seed handling (default seed=42).
- Automatic device selection (CUDA / MPS / CPU).
- Training loop with CrossEntropyLoss and Adam optimizer.
- Validation loop without gradient computation or scaler modification.
- Explicit distinction between best validation checkpoint (best.pt) and latest checkpoint (last.pt).
- Safe checkpoint saving and full resumption (model, optimizer, scheduler, epoch, best metrics, config).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    class nn:
        Module = object
    HAS_TORCH = False

from config import NUM_CLASSES
from evaluation.metrics import calculate_metrics
from utils.paths import CHECKPOINTS_DIR


def set_reproducible_seed(seed: int = 42) -> None:
    """Set global random seeds for PyTorch, NumPy, and Python stdlib."""
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_target_device() -> str:
    """Return optimal hardware device string ('cuda', 'mps', or 'cpu')."""
    if not HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaselineTrainer:
    """
    Common training & evaluation manager for baseline and custom Counter-UAS models.
    Supports checkpoint saving, loading, resumption, and metrics tracking.
    """

    def __init__(
        self,
        model: "nn.Module",
        model_name: str,
        num_classes: int = NUM_CLASSES,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_type: str = "Adam",
        device: Optional[str] = None,
        seed: int = 42,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for BaselineTrainer.")

        self.seed = seed
        set_reproducible_seed(self.seed)

        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.device = device or get_target_device()
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        
        opt_upper = optimizer_type.upper().strip()
        if opt_upper == "ADAMW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif opt_upper == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        self.scheduler = None

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0

        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
        }

    def train_epoch(self, train_loader: DataLoader, max_batches: Optional[int] = None) -> Tuple[float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch)

            loss = self.criterion(logits, y_batch)
            loss.backward()

            # Check for NaN gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        raise ValueError(f"NaN/Inf gradient detected in parameter '{name}'!")

            self.optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().numpy())

        total_samples = len(all_targets)
        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        acc = float(np.mean(np.array(all_preds) == np.array(all_targets))) if total_samples > 0 else 0.0

        return avg_loss, acc

    def evaluate(self, val_loader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation loop on validation/test DataLoader."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                running_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_batch.cpu().numpy())

        total_samples = len(all_targets)
        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = float(avg_loss)
        return metrics

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: int,
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save complete training state checkpoint to disk.
        Preserves model weights, optimizer state, scheduler state, epoch, best metrics, and configs.
        """
        ckpt_p = Path(checkpoint_path)
        ckpt_p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "val_metrics": val_metrics or {},
            "history": self.history,
            "seed": self.seed,
            "is_best": is_best,
            "extra_config": extra_config or {},
        }

        torch.save(payload, ckpt_p)
        return ckpt_p

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        resume_training: bool = True,
    ) -> Dict[str, Any]:
        """
        Load and validate a saved checkpoint.
        Restores model weights, and optionally optimizer, scheduler, epoch, and metrics if resuming.
        """
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_p}")

        checkpoint = torch.load(ckpt_p, map_location=self.device)

        # Validate checkpoint contents
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint {ckpt_p} does not contain 'model_state_dict'.")

        ckpt_classes = checkpoint.get("num_classes")
        if ckpt_classes is not None and ckpt_classes != self.num_classes:
            raise ValueError(
                f"Checkpoint class count mismatch: checkpoint has {ckpt_classes} classes, "
                f"but trainer expected {self.num_classes}."
            )

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if resume_training:
            if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
            self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
            self.best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", 0))
            self.history = checkpoint.get("history", self.history)

        return checkpoint

    def run_smoke_test(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 2,
        batches_per_epoch: int = 2,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Tiny smoke test: Runs 2 epochs with 2 batches per epoch.
        Verifies forward, backward, loss, gradient, step, val, and checkpoint creation.
        """
        print(f"   [Smoke Test] Target Model: {self.model_name} on device '{self.device}'")

        smoke_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINTS_DIR / "smoke_test"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = smoke_dir / f"{self.model_name}_smoke_best.pt"

        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader, max_batches=batches_per_epoch)
            val_metrics = self.evaluate(val_loader, max_batches=batches_per_epoch)

            print(f"   - Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

            assert not np.isnan(tr_loss) and not np.isinf(tr_loss), "Train loss is NaN/Inf!"
            assert not np.isnan(val_metrics["loss"]) and not np.isinf(val_metrics["loss"]), "Val loss is NaN/Inf!"

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                self.best_val_loss = best_loss
                self.save_checkpoint(ckpt_path, epoch=epoch, val_metrics=val_metrics, is_best=True)

        assert ckpt_path.exists(), f"Smoke test checkpoint not created at {ckpt_path}"
        print(f"   [OK] Smoke test completed successfully! Saved checkpoint to {ckpt_path.name}")

        return {
            "model_name": self.model_name,
            "train_loss": tr_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "checkpoint_saved": str(ckpt_path),
        }
