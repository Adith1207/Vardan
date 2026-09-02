"""Model factory for instantiating baseline and custom models."""

from typing import Any, Dict
import torch.nn as nn
from config import NUM_CLASSES
from .baselines import Baseline1DCNN, DSCNN, FGCS2019DNN, MobileNetV3Small
from .vardhan import VardhanRFNet


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Instantiate and return the requested model with parameter safety.

    Args:
        model_name: The name or alias of the model to instantiate.
        kwargs: Hyperparameters to pass to the model constructor.

    Returns:
        An instance of a PyTorch nn.Module.
    """
    num_classes = kwargs.pop("num_classes", NUM_CLASSES)
    name = model_name.lower().strip()

    if name in ["fgcs2019dnn", "fgcs_dnn", "baseline_dnn", "allahham2019"]:
        in_features = kwargs.pop("in_features", 2048)
        return FGCS2019DNN(in_features=in_features, num_classes=num_classes)

    elif name in ["baseline1dcnn", "1dcnn", "mc1dcnn", "baseline_mc1dcnn", "ezuma2020"]:
        in_channels = kwargs.pop("in_channels", 2)
        seq_length = kwargs.pop("seq_length", 2048)
        return Baseline1DCNN(in_channels=in_channels, num_classes=num_classes, seq_length=seq_length)

    elif name in ["dscnn", "tinyml", "baseline_cnn", "medaiyese2022"]:
        in_channels = kwargs.pop("in_channels", 2)
        seq_length = kwargs.pop("seq_length", 2048)
        return DSCNN(in_channels=in_channels, num_classes=num_classes, seq_length=seq_length)

    elif name in ["mobilenetv3small", "mobilenetv3", "spectrogram_cnn", "howard2019"]:
        return MobileNetV3Small(num_classes=num_classes)

    elif name in ["vardhan", "vardhanrfnet", "vardhan_rf"]:
        in_channels = kwargs.pop("in_channels", 2)
        seq_length = kwargs.pop("seq_length", 2048)
        return VardhanRFNet(in_channels=in_channels, num_classes=num_classes, seq_length=seq_length)

    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Supported models: fgcs2019dnn, baseline1dcnn, dscnn, mobilenetv3small, vardhan."
        )

