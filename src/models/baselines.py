"""Baseline model architectures for RF and Spectrogram classification."""

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


class Baseline1DCNN(nn.Module):
    """Standard 1D Convolutional Neural Network baseline for raw RF waveform input."""

    def __init__(self, in_channels: int = 2, num_classes: int = 5, seq_length: int = 2048):
        super().__init__()
        if not HAS_TORCH:
            return
            
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=11, stride=2, padding=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate pool size dynamically
        dummy_input = torch.zeros(1, in_channels, seq_length)
        with torch.no_grad():
            dummy_out = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(dummy_input))))))
            self.flat_features = dummy_out.numel()
            
        self.fc1 = nn.Linear(self.flat_features, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass. Shape of x: (batch_size, in_channels, seq_length)."""
        if not HAS_TORCH:
            return x
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


class DSCNN(nn.Module):
    """Depthwise Separable Convolutional Neural Network baseline (TinyML optimized)."""

    def __init__(self, in_channels: int = 2, num_classes: int = 5, seq_length: int = 2048):
        super().__init__()
        if not HAS_TORCH:
            return
            
        # Standard first conv
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=11, stride=2, padding=5)
        self.relu1 = nn.ReLU()
        
        # Depthwise Separable Conv: Depthwise + Pointwise
        self.depthwise = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2, groups=32)
        self.pointwise = nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.MaxPool1d(kernel_size=4)
        
        dummy_input = torch.zeros(1, in_channels, seq_length)
        with torch.no_grad():
            dummy_out = self.pool(self.relu2(self.pointwise(self.depthwise(self.relu1(self.conv1(dummy_input))))))
            self.flat_features = dummy_out.numel()
            
        self.fc = nn.Linear(self.flat_features, num_classes)

    def forward(self, x):
        """Forward pass."""
        if not HAS_TORCH:
            return x
        x = self.relu1(self.conv1(x))
        x = self.depthwise(x)
        x = self.relu2(self.pointwise(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MobileNetV3Small(nn.Module):
    """Lightweight MobileNetV3-based classifier for 2D spectrogram inputs."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        if not HAS_TORCH:
            return
            
        # Simplified MobileNetV3 small representation
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, groups=16),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """Forward pass. Shape of x: (batch_size, 1, freq_bins, time_frames)."""
        if not HAS_TORCH:
            return x
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
