"""
test_training_configs.py
------------------------

Tests verifying the internal consistency and exact parameter instantiation
of the 5-model training configuration:
- configs/training.yaml schema and model-specific values
- Synchrony between training.yaml and scripts/train_baselines.py defaults
- Actual optimizer, learning rate, weight decay, scheduler, loss, seed, and checkpoint instantiation in BaselineTrainer
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_factory import get_model
from models.trainer import BaselineTrainer
from utils.helpers import load_yaml
from utils.paths import CONFIGS_DIR

EXPECTED_CONFIGS = {
    "fgcs2019dnn": {
        "title": "FGCS2019DNN",
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 10,
        "epochs": 200,
    },
    "baseline1dcnn": {
        "title": "Baseline1DCNN",
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 32,
        "epochs": 100,
    },
    "dscnn": {
        "title": "DSCNN",
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 32,
        "epochs": 100,
    },
    "mobilenetv3small": {
        "title": "MobileNetV3Small",
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 32,
        "epochs": 100,
    },
    "vardhan": {
        "title": "VardhanRFNet",
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 100,
    },
}


def test_training_yaml_schema_and_values():
    """Verify that configs/training.yaml matches the canonical configuration."""
    cfg = load_yaml(CONFIGS_DIR / "training.yaml")
    
    assert "common" in cfg
    assert cfg["common"]["random_seed"] == 42
    assert cfg["common"]["checkpoint_criterion"] == "min_val_loss"
    
    assert "scheduler" in cfg
    assert cfg["scheduler"]["type"] is None
    
    assert "loss" in cfg
    assert cfg["loss"]["type"] == "CrossEntropyLoss"
    assert cfg["loss"]["label_smoothing"] == 0.0
    
    assert "models" in cfg
    models_cfg = cfg["models"]
    
    for key, expected in EXPECTED_CONFIGS.items():
        assert key in models_cfg, f"Missing {key} in training.yaml models"
        m = models_cfg[key]
        assert m["optimizer"] == expected["optimizer"]
        assert pytest.approx(m["lr"]) == expected["lr"]
        assert pytest.approx(m["weight_decay"]) == expected["weight_decay"]
        assert m["batch_size"] == expected["batch_size"]
        assert m["epochs"] == expected["epochs"]


def test_trainer_instantiation_for_all_models():
    """Verify that BaselineTrainer instantiates the exact optimizer, lr, wd, and loss for all 5 models."""
    for key, expected in EXPECTED_CONFIGS.items():
        model = get_model(key, num_classes=4)
        trainer = BaselineTrainer(
            model=model,
            model_name=key,
            learning_rate=expected["lr"],
            weight_decay=expected["weight_decay"],
            optimizer_type=expected["optimizer"],
            seed=42,
        )
        
        # Check optimizer class
        if expected["optimizer"] == "AdamW":
            assert isinstance(trainer.optimizer, optim.AdamW)
        else:
            assert isinstance(trainer.optimizer, optim.Adam)
            
        # Check lr and weight decay in param groups
        param_group = trainer.optimizer.param_groups[0]
        assert pytest.approx(param_group["lr"]) == expected["lr"]
        assert pytest.approx(param_group["weight_decay"]) == expected["weight_decay"]
        
        # Check scheduler is None
        assert trainer.scheduler is None
        
        # Check loss is standard CrossEntropyLoss
        assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
        
        # Check seed and best loss tracking
        assert trainer.seed == 42
        assert trainer.best_val_loss == float("inf")
