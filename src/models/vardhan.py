"""Custom proposed neural network architecture for drone detection."""

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    HAS_TORCH = True
except ImportError:
    class nn_Module:  # type: ignore
        pass
    nn = nn_Module  # type: ignore
    nn.Module = nn_Module  # type: ignore
    HAS_TORCH = False


class VardhanRFNet(nn.Module):
    """Vardhan proposed lightweight neural network for RF-based Counter-UAS.

    Incorporates dilated convolutions for receptive field expansion, multi-scale
    feature aggregation, and global pooling for parameter efficiency.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 5, seq_length: int = 2048):
        super().__init__()
        if not HAS_TORCH:
            return
            
        # Multiscale RF feature extractor
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1, dilation=2)  # Dilated
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2)
        )
        
        # Output channels from combined branches: 32 + 32 = 64
        self.transition = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        
        # High level classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_length).

        Returns:
            Class logits of shape (batch_size, num_classes).
        """
        if not HAS_TORCH:
            return x
            
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        feat = torch.cat([b1, b2], dim=1)
        feat = self.transition(feat)
        feat = self.global_pool(feat).squeeze(-1)
        return self.classifier(feat)
