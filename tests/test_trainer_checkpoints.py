"""
test_trainer_checkpoints.py
---------------------------

Unit and contract tests for BaselineTrainer checkpoint creation, loading, resumption,
best vs last isolation, and run config persistence.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.model_factory import get_model
from models.trainer import BaselineTrainer


@pytest.fixture
def synthetic_data():
    """Create a tiny synthetic dataset of (x, y) pairs."""
    torch.manual_seed(42)
    x = torch.randn(20, 2, 2048)
    y = torch.randint(0, 4, (20,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return train_loader, val_loader


def test_checkpoint_save_and_load(synthetic_data):
    """Test checkpoint creation, content validation, and exact state restoration."""
    train_loader, val_loader = synthetic_data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_path = tmp_path / "test_ckpt.pt"

        model = get_model("vardhan", num_classes=4)
        trainer = BaselineTrainer(model=model, model_name="vardhan", learning_rate=1e-3, seed=42)

        # Run 1 epoch
        tr_loss, tr_acc = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        trainer.best_val_loss = val_metrics["loss"]
        trainer.best_val_acc = val_metrics["accuracy"]
        trainer.best_epoch = 1

        # Save checkpoint
        saved_p = trainer.save_checkpoint(
            ckpt_path,
            epoch=1,
            val_metrics=val_metrics,
            is_best=True,
            extra_config={"test_key": "test_val"},
        )
        assert saved_p.exists()

        # Load into fresh trainer
        fresh_model = get_model("vardhan", num_classes=4)
        fresh_trainer = BaselineTrainer(model=fresh_model, model_name="vardhan", learning_rate=1e-3, seed=42)

        ckpt_data = fresh_trainer.load_checkpoint(saved_p, resume_training=True)

        # Verify restoration
        assert ckpt_data["epoch"] == 1
        assert fresh_trainer.best_val_loss == trainer.best_val_loss
        assert fresh_trainer.best_val_acc == trainer.best_val_acc
        assert fresh_trainer.best_epoch == 1
        assert ckpt_data["extra_config"]["test_key"] == "test_val"

        # Verify model parameters match exactly
        for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
            assert torch.allclose(p1, p2)


def test_checkpoint_resumption_epoch_progression(synthetic_data):
    """Test that resuming from epoch N continues training seamlessly from epoch N+1."""
    train_loader, val_loader = synthetic_data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_path = tmp_path / "epoch_2.pt"

        model = get_model("vardhan", num_classes=4)
        trainer = BaselineTrainer(model=model, model_name="vardhan", learning_rate=1e-3, seed=42)

        # Simulate 2 completed epochs
        for ep in range(1, 3):
            trainer.train_epoch(train_loader)
            v = trainer.evaluate(val_loader)
            trainer.history["train_loss"].append(0.5)
            trainer.history["val_loss"].append(v["loss"])
            trainer.save_checkpoint(ckpt_path, epoch=ep, val_metrics=v, is_best=False)

        # Resume in new trainer
        new_model = get_model("vardhan", num_classes=4)
        new_trainer = BaselineTrainer(model=new_model, model_name="vardhan", learning_rate=1e-3, seed=42)

        loaded = new_trainer.load_checkpoint(ckpt_path, resume_training=True)
        start_epoch = loaded["epoch"] + 1

        assert start_epoch == 3, f"Expected start_epoch=3, got {start_epoch}"
        assert len(new_trainer.history["train_loss"]) == 2


def test_best_vs_last_checkpoint_isolation(synthetic_data):
    """Test that a worsening validation metric does NOT overwrite best.pt."""
    train_loader, val_loader = synthetic_data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        best_ckpt = tmp_path / "best.pt"
        last_ckpt = tmp_path / "last.pt"

        model = get_model("vardhan", num_classes=4)
        trainer = BaselineTrainer(model=model, model_name="vardhan", learning_rate=1e-3, seed=42)

        # Epoch 1: Good loss (0.50)
        trainer.best_val_loss = 0.50
        trainer.best_epoch = 1
        trainer.save_checkpoint(best_ckpt, epoch=1, val_metrics={"loss": 0.50}, is_best=True)
        trainer.save_checkpoint(last_ckpt, epoch=1, val_metrics={"loss": 0.50}, is_best=False)

        # Epoch 2: Worse loss (0.80) -> Only update last.pt, do NOT update best.pt
        worse_loss = 0.80
        trainer.save_checkpoint(last_ckpt, epoch=2, val_metrics={"loss": worse_loss}, is_best=False)

        # Check best.pt still has epoch 1 and 0.50 loss
        best_data = torch.load(best_ckpt)
        assert best_data["epoch"] == 1
        assert best_data["best_val_loss"] == 0.50

        # Check last.pt has epoch 2 and latest state
        last_data = torch.load(last_ckpt)
        assert last_data["epoch"] == 2


def test_incompatible_checkpoint_fails_clearly():
    """Test that attempting to load a checkpoint with mismatched num_classes raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_path = tmp_path / "incompatible.pt"

        # Save checkpoint with num_classes = 10
        torch.save(
            {
                "epoch": 1,
                "model_name": "vardhan",
                "num_classes": 10,
                "model_state_dict": {},
            },
            ckpt_path,
        )

        model = get_model("vardhan", num_classes=4)
        trainer = BaselineTrainer(model=model, model_name="vardhan", num_classes=4)

        with pytest.raises(ValueError, match="Checkpoint class count mismatch"):
            trainer.load_checkpoint(ckpt_path)
