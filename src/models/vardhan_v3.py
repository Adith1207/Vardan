"""
vardhan_v3.py
-------------

Implementation of VARDHAN-v3 (Multi-Scale Dual-Domain RF-Net).
A lightweight, RF-specific neural network architecture for UAV radio frequency classification.

Architecture Summary:
1. Input: Single-channel raw RF time-domain waveform tensor of shape (B, 1, 2048).
2. Dual-Domain Representations:
   a. Time-Domain: Raw 2048-sample voltage slice directly processed by a multi-scale dilated TCN.
   b. Dual Spectral Domain: Deterministic 2048-point rFFT (DC bin removed, positive bins 1..1024 retained)
      yielding a 3-channel spectral tensor (B, 3, 1024):
        - Channel 0: Log Power Spectrum: log(1 + 1000 * |X|^2 / 2048)
        - Channel 1: Normalized Real Fourier Component: Re(X) / sqrt(2048)
        - Channel 2: Normalized Imaginary Fourier Component: Im(X) / sqrt(2048)
3. Multi-Scale Temporal Backbone (MS-TCN):
   - Stem: Conv1d(1->32, k=15, s=2, p=7) + BatchNorm1d + GELU.
   - 4 MultiScaleTemporalBlocks with parallel depthwise kernel branches (k=3, 7, 15, 31),
     dilation schedule d in {1, 2, 4, 8}, Squeeze-and-Excitation (SE) channel attention,
     and residual projections. Channels: 32 -> 48 -> 64 -> 80 -> 80.
   - Learnable self-attention pooling (AttentionPool1d: 80 -> 40 -> 1) producing h_T in R^(B, 80).
4. Multi-Scale Dual Spectral Backbone:
   - Stem: Conv1d(3->32, k=11, s=2, p=5) + BatchNorm1d + GELU.
   - 4 MultiScaleSpectralBlocks with dual depthwise kernel branches (k=5, 11),
     dilation schedule d in {1, 2, 4, 8}, SE channel attention, and residual projections.
     Channels: 32 -> 48 -> 64 -> 80 -> 80.
   - Learnable self-attention pooling (AttentionPool1d: 80 -> 40 -> 1) producing h_F in R^(B, 80).
5. Bi-Directional Cross-Domain Gated Fusion & Head:
   - Temporal gate: g_T = Sigmoid(Linear(80->40) -> GELU -> Linear(40->80)) applied to h_F.
   - Spectral gate: g_F = Sigmoid(Linear(80->40) -> GELU -> Linear(40->80)) applied to h_T.
   - Modulated features: h_T_mod = h_T * g_T, h_F_mod = h_F * g_F.
   - Concatenation: [h_T, h_F, h_T_mod, h_F_mod] in R^(B, 320).
   - Classifier: Linear(320->64) + BatchNorm1d(64) + GELU + Dropout(0.2) + Linear(64->4).

Total Trainable Parameters: Exactly 119,806.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class SqueezeExcitation1d(nn.Module):
    """Squeeze-and-Excitation (SE) channel attention block for 1D representations."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: channel-wise multiplication x * SE(x)."""
        return x * self.fc(x)


class MultiScaleTemporalBlock(nn.Module):
    """Multi-Scale Depthwise-Separable Residual Block for 1D Temporal Waveforms.

    Splits input channels across 4 parallel depthwise convolution branches with
    different kernel sizes (k=3, 7, 15, 31) to simultaneously capture micro-transients,
    carrier cycles, symbol preambles, and packet envelopes.
    Includes Squeeze-and-Excitation channel attention and residual skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        if in_channels % 4 != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by 4.")
        c_b = in_channels // 4

        # 4 Parallel Depthwise Convolutions
        self.dw3 = nn.Conv1d(
            c_b,
            c_b,
            kernel_size=3,
            stride=stride,
            padding=1 * dilation,
            dilation=dilation,
            groups=c_b,
            bias=False,
        )
        self.dw7 = nn.Conv1d(
            c_b,
            c_b,
            kernel_size=7,
            stride=stride,
            padding=3 * dilation,
            dilation=dilation,
            groups=c_b,
            bias=False,
        )
        self.dw15 = nn.Conv1d(
            c_b,
            c_b,
            kernel_size=15,
            stride=stride,
            padding=7 * dilation,
            dilation=dilation,
            groups=c_b,
            bias=False,
        )
        self.dw31 = nn.Conv1d(
            c_b,
            c_b,
            kernel_size=31,
            stride=stride,
            padding=15 * dilation,
            dilation=dilation,
            groups=c_b,
            bias=False,
        )

        # Pointwise Convolution, Normalization & Activation
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.se = SqueezeExcitation1d(out_channels, reduction=4)

        # Residual Skip Connection
        if in_channels != out_channels or stride != 1:
            self.res = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through parallel depthwise convs, pointwise conv, SE, and residual."""
        c = x.shape[1] // 4
        out_dw = torch.cat(
            [
                self.dw3(x[:, 0:c]),
                self.dw7(x[:, c:2 * c]),
                self.dw15(x[:, 2 * c:3 * c]),
                self.dw31(x[:, 3 * c:4 * c]),
            ],
            dim=1,
        )
        out = self.se(self.act(self.bn(self.pw(out_dw))))
        return out + self.res(x)


class MultiScaleSpectralBlock(nn.Module):
    """Multi-Scale Spectral Residual Block for 1D Fourier Representations.

    Splits input channels across 2 parallel depthwise convolution branches (k=5, 11)
    to simultaneously capture narrow-band carrier tones and wider sub-channel spectral envelopes.
    Includes Squeeze-and-Excitation channel attention and residual skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by 2.")
        c_b = in_channels // 2

        # 2 Parallel Depthwise Convolutions
        self.dw5 = nn.Conv1d(
            c_b,
            c_b,
            kernel_size=5,
            stride=stride,
            padding=2 * dilation,
            dilation=dilation,
            groups=c_b,
            bias=False,
        )
        self.dw11 = nn.Conv1d(
            c_b,
            c_b,
            kernel_size=11,
            stride=stride,
            padding=5 * dilation,
            dilation=dilation,
            groups=c_b,
            bias=False,
        )

        # Pointwise Convolution, Normalization & Activation
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.se = SqueezeExcitation1d(out_channels, reduction=4)

        # Residual Skip Connection
        if in_channels != out_channels or stride != 1:
            self.res = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through parallel spectral depthwise convs, pointwise conv, SE, and residual."""
        c = x.shape[1] // 2
        out_dw = torch.cat([self.dw5(x[:, :c]), self.dw11(x[:, c:])], dim=1)
        out = self.se(self.act(self.bn(self.pw(out_dw))))
        return out + self.res(x)


class AttentionPool1d(nn.Module):
    """Learnable self-attention pooling over the temporal or spectral sequence dimension."""

    def __init__(self, in_dim: int = 80, hidden_dim: int = 40):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: computes attention weights and aggregates features as a weighted sum.

        Args:
            x: Input tensor of shape (B, in_dim, T).

        Returns:
            Pooled feature vector of shape (B, in_dim).
        """
        weights = self.attn(x)  # (B, 1, T)
        return (x * weights).sum(dim=-1)  # (B, in_dim)


class CrossDomainGatedFusion(nn.Module):
    """Bi-Directional Cross-Domain Gated Fusion and Classifier Head.

    Enables cross-domain feature modulation:
      - Temporal features h_T are gated by spectral context h_F.
      - Spectral features h_F are gated by temporal context h_T.
    The original and modulated features are concatenated and projected to class logits.
    """

    def __init__(
        self,
        dim_t: int = 80,
        dim_f: int = 80,
        fused_dim: int = 64,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        mid_t = dim_t // 2
        mid_f = dim_f // 2

        # Cross-Domain Gating Networks
        self.gate_t = nn.Sequential(
            nn.Linear(dim_f, mid_t),
            nn.GELU(),
            nn.Linear(mid_t, dim_t),
            nn.Sigmoid(),
        )
        self.gate_f = nn.Sequential(
            nn.Linear(dim_t, mid_f),
            nn.GELU(),
            nn.Linear(mid_f, dim_f),
            nn.Sigmoid(),
        )

        # Classification Head: 4 * 80 = 320 -> 64 -> 4
        in_dim = dim_t + dim_f + dim_t + dim_f
        self.fc1 = nn.Linear(in_dim, fused_dim)
        self.bn = nn.BatchNorm1d(fused_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(fused_dim, num_classes)

    def forward(self, h_t: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        """Forward pass through cross-domain gating and classifier head."""
        g_t = self.gate_t(h_f)
        g_f = self.gate_f(h_t)
        h_t_mod = h_t * g_t
        h_f_mod = h_f * g_f
        cat = torch.cat([h_t, h_f, h_t_mod, h_f_mod], dim=-1)  # (B, 320)
        z = self.drop(self.act(self.bn(self.fc1(cat))))         # (B, 64)
        return self.fc2(z)                                      # (B, num_classes)


class VardhanV3(nn.Module):
    """VARDHAN-v3: Multi-Scale Dual-Domain RF Network.

    Processes raw single-channel 2048-sample RF waveforms through two complementary backbones:
      1. Multi-Scale Temporal Backbone (MS-TCN) capturing local transient edges, carrier cycles,
         symbol frames, and burst envelopes via parallel depthwise convolutions (k=3, 7, 15, 31).
      2. Multi-Scale Dual Spectral Backbone processing a 3-channel Fourier representation
         (Log Power Spectrum, Normalized Real FFT, Normalized Imaginary FFT) via parallel
         spectral depthwise convolutions (k=5, 11).
    Features are pooled via learnable attention and fused through bi-directional cross-domain gating.

    Total Trainable Parameters: Exactly 119,806.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # =====================================================================
        # 1. Multi-Scale Temporal Backbone (Input: (B, 1, 2048))
        # =====================================================================
        self.time_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),  # (B, 32, 1024)
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.t_b1 = MultiScaleTemporalBlock(32, 48, stride=2, dilation=1)      # (B, 48, 512)
        self.t_b2 = MultiScaleTemporalBlock(48, 64, stride=2, dilation=2)      # (B, 64, 256)
        self.t_b3 = MultiScaleTemporalBlock(64, 80, stride=2, dilation=4)      # (B, 80, 128)
        self.t_b4 = MultiScaleTemporalBlock(80, 80, stride=1, dilation=8)      # (B, 80, 128)
        self.t_pool = AttentionPool1d(80, hidden_dim=40)                       # (B, 80)

        # =====================================================================
        # 2. Multi-Scale Dual Spectral Backbone (Input: (B, 3, 1024))
        # =====================================================================
        self.freq_stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=11, stride=2, padding=5, bias=False),  # (B, 32, 512)
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.f_b1 = MultiScaleSpectralBlock(32, 48, stride=2, dilation=1)      # (B, 48, 256)
        self.f_b2 = MultiScaleSpectralBlock(48, 64, stride=2, dilation=2)      # (B, 64, 128)
        self.f_b3 = MultiScaleSpectralBlock(64, 80, stride=1, dilation=4)      # (B, 80, 128)
        self.f_b4 = MultiScaleSpectralBlock(80, 80, stride=1, dilation=8)      # (B, 80, 128)
        self.f_pool = AttentionPool1d(80, hidden_dim=40)                       # (B, 80)

        # =====================================================================
        # 3. Bi-Directional Cross-Domain Gated Fusion & Head
        # =====================================================================
        self.head = CrossDomainGatedFusion(
            dim_t=80,
            dim_f=80,
            fused_dim=64,
            num_classes=num_classes,
            dropout=0.2,
        )

    def extract_spectral_representation(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Extract deterministic 3-channel dual Fourier representation from raw waveform.

        Steps:
          1. Per-sample DC mean detrending: x - mean(x)
          2. Real FFT (n=2048) -> 1025 complex bins
          3. Drop DC bin (k=0) -> retain 1024 positive-frequency bins (k=1..1024)
          4. Channel 0: Log Power Spectrum = log(1 + 1000 * |X|^2 / 2048)
          5. Channel 1: Normalized Real FFT = Re(X) / sqrt(2048)
          6. Channel 2: Normalized Imaginary FFT = Im(X) / sqrt(2048)

        Returns:
            Tensor of shape (B, 3, 1024).
        """
        # 1. Detrend mean per sample
        x_detrend = x_raw - x_raw.mean(dim=-1, keepdim=True)

        # 2. Real FFT
        fft_complex = torch.fft.rfft(x_detrend.squeeze(1), n=2048)  # (B, 1025)

        # 3. Drop DC bin (k=0), retain positive bins k=1..1024
        fft_pos = fft_complex[:, 1:]  # (B, 1024)

        # 4. Channel 0: Log Power Spectrum
        power = (torch.abs(fft_pos) ** 2) / 2048.0
        p_log = torch.log1p(1000.0 * power).unsqueeze(1)  # (B, 1, 1024)

        # 5. Channel 1: Normalized Real Component
        re_fft = (fft_pos.real / (2048.0 ** 0.5)).unsqueeze(1)  # (B, 1, 1024)

        # 6. Channel 2: Normalized Imaginary Component
        im_fft = (fft_pos.imag / (2048.0 ** 0.5)).unsqueeze(1)  # (B, 1, 1024)

        return torch.cat([p_log, re_fft, im_fft], dim=1)  # (B, 3, 1024)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through temporal and dual-spectral backbones, attention pooling, and fusion head.

        Args:
            x: Raw time-domain waveform tensor of shape (B, 1, 2048).
            return_embeddings: If True, returns dictionary of intermediate branch embeddings.

        Returns:
            logits: Class logits of shape (B, num_classes).
        """
        # Extract 3-channel spectral representation
        x_freq = self.extract_spectral_representation(x)  # (B, 3, 1024)

        # 1. Temporal Backbone
        t_feat = self.time_stem(x)
        t_feat = self.t_b1(t_feat)
        t_feat = self.t_b2(t_feat)
        t_feat = self.t_b3(t_feat)
        t_feat = self.t_b4(t_feat)
        h_t = self.t_pool(t_feat)  # (B, 80)

        # 2. Spectral Backbone
        f_feat = self.freq_stem(x_freq)
        f_feat = self.f_b1(f_feat)
        f_feat = self.f_b2(f_feat)
        f_feat = self.f_b3(f_feat)
        f_feat = self.f_b4(f_feat)
        h_f = self.f_pool(f_feat)  # (B, 80)

        # 3. Cross-Domain Gated Fusion & Classifier
        logits = self.head(h_t, h_f)  # (B, num_classes)

        if return_embeddings:
            return logits, {
                "h_temporal": h_t,
                "h_spectral": h_f,
                "x_spectral_tensor": x_freq,
            }
        return logits
