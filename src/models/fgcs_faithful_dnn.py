"""
fgcs_faithful_dnn.py
--------------------

Deep Neural Network architectures for faithful FGCS reproduction (EXP_FGCS_FAITHFUL).

Supports both:
1. 'code' mode (Actual released Classification.py in Al-Sad/DroneRF):
   - 3 hidden layers of 128 units (int(number_inner_neurons / 2) where number_inner_neurons=256)
   - Linear(2048 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> num_classes) -> Sigmoid
2. 'paper' mode (Al-Sa'd FGCS 2019 Table 2 / Section 4.1 text):
   - 3 hidden layers of 256, 128, 64 units
   - Linear(2048 -> 256) -> ReLU -> Linear(256 -> 128) -> ReLU -> Linear(128 -> 64) -> ReLU -> Linear(64 -> num_classes) -> Sigmoid
"""

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            pass


class FGCSFaithfulDNN(nn.Module):
    """
    Faithful DNN classifier for Al-Sa'd DroneRF RF spectrum classification.
    """

    def __init__(
        self,
        in_features: int = 2048,
        num_classes: int = 4,
        architecture_mode: str = "code",
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.architecture_mode = architecture_mode.lower().strip()

        if not HAS_TORCH:
            return

        if self.architecture_mode == "code":
            # Exact architecture from released Classification.py: 3 hidden layers of 128 units
            self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )
        elif self.architecture_mode == "paper":
            # Architecture described in FGCS 2019 paper text: 256 -> 128 -> 64
            self.net = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )
        else:
            raise ValueError(f"Unknown architecture_mode: '{architecture_mode}'. Expected 'code' or 'paper'.")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits: bool = True):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, in_features).
            return_logits: If True, returns unnormalized logits. If False, applies Sigmoid.
        """
        if not HAS_TORCH:
            return x
        logits = self.net(x)
        if return_logits:
            return logits
        return self.sigmoid(logits)
