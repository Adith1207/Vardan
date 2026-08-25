"""Model factory for instantiating baseline and custom models."""

import torch.nn as nn
from config import NUM_CLASSES
from .baselines import Baseline1DCNN, DSCNN, MobileNetV3Small, FGCS2019DNN
from .vardhan import VardhanRFNet


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Instantiate and return the requested model.

    Args:
        model_name: The name of the model to instantiate.
        kwargs: Hyperparameters to pass to the model constructor.

    Returns:
        An instance of a PyTorch nn.Module.
    """
    if "num_classes" not in kwargs:
        kwargs["num_classes"] = NUM_CLASSES

    name = model_name.lower()
    if name == "fgcs2019dnn":
        return FGCS2019DNN(**kwargs)
    elif name in ["baseline1dcnn", "1dcnn"]:
        return Baseline1DCNN(**kwargs)
    elif name == "dscnn":
        return DSCNN(**kwargs)
    elif name == "mobilenetv3small":
        return MobileNetV3Small(**kwargs)
    elif name in ["vardhan", "vardhanrfnet"]:
        return VardhanRFNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

