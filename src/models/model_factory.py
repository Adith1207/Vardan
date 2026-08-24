"""Model factory for instantiating baseline and custom models."""

import torch.nn as nn
from .baselines import Baseline1DCNN, DSCNN, MobileNetV3Small, FGCS2019DNN

def get_model(model_name: str, **kwargs) -> nn.Module:
    """Instantiate and return the requested model.

    Args:
        model_name: The name of the model to instantiate.
        kwargs: Hyperparameters to pass to the model constructor.

    Returns:
        An instance of a PyTorch nn.Module.
    """
    if model_name.lower() == "fgcs2019dnn":
        return FGCS2019DNN(**kwargs)
    elif model_name.lower() == "baseline1dcnn":
        return Baseline1DCNN(**kwargs)
    elif model_name.lower() == "dscnn":
        return DSCNN(**kwargs)
    elif model_name.lower() == "mobilenetv3small":
        return MobileNetV3Small(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
