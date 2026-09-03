"""
vardhan_v2a.py
--------------

Implementation of VARDHAN-v2A (Tri-Branch Multi-Representation RF-Net).
A lightweight, RF-specific architecture for UAV radio frequency classification.

Architecture Summary:
1. Temporal Branch: Multi-scale dilated depthwise-separable residual CNN with
   an effective receptive field of 1,035 samples (25.88 us at 40 MSps).
2. Fine-Grained Spectral Branch: Deterministic 2048-point DFT (dropping DC,
   retaining positive 1024 bins spanning 0-20 MHz) processed via 1D convolutions.
3. Coarse Spectral-Band Branch: 8 uniform computational sub-bands (~2.5 MHz each)
   derived from the same 1024-bin FFT representation.
4. Attention Pooling: Learnable self-attention pooling (AttentionPool1d) on all branches.
5. Lightweight Fusion & Classifier Head: Linear(192->64) + BatchNorm1d + GELU + Dropout + Linear(64->4).

Total Trainable Parameters: 69,559.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Depthwise-separable residual block with dilated convolution for 1D sequences."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        # 1. Depthwise Convolution
        self.dw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )

        # 2. Pointwise Convolution
        self.pw = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        # 3. Normalization & Activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        # 4. Residual Projection
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
        """Forward pass: Depthwise -> Pointwise -> BN -> GELU + Residual."""
        out = self.act(self.bn(self.pw(self.dw(x))))
        return out + self.res(x)


class AttentionPool1d(nn.Module):
    """Learnable self-attention pooling over the sequence/spectral dimension.

    Computes normalized attention weights alpha_t via a 2-layer 1x1 convolution
    and aggregates features as a weighted sum: sum(alpha_t * x_t).
    Parameter count: in_dim * hidden + hidden + hidden * 1 + 1 = 2,065 (for 64->32->1).
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim, T).

        Returns:
            Pooled feature vector of shape (B, in_dim).
        """
        weights = self.attn(x)  # (B, 1, T)
        return (x * weights).sum(dim=-1)  # (B, in_dim)


class VardhanV2A(nn.Module):
    """VARDHAN-v2A: Tri-Branch Multi-Representation RF Network.

    Processes raw single-file 2048-sample RF waveforms through three complementary
    representations:
      1. Temporal branch (raw waveform) -> (B, 64)
      2. Fine-grained spectral branch (1024-bin FFT) -> (B, 64)
      3. Coarse spectral-band branch (8x128 sub-bands) -> (B, 64)
    Fused via lightweight concatenation and a Linear-BN-GELU classifier head.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # =====================================================================
        # 1. Temporal Branch (Input: (B, 1, 2048))
        # =====================================================================
        self.time_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),  # (B, 32, 1024)
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.time_block1 = TemporalBlock(32, 48, kernel_size=7, stride=2, dilation=1)  # (B, 48, 512)
        self.time_block2 = TemporalBlock(48, 64, kernel_size=7, stride=2, dilation=2)  # (B, 64, 256)
        self.time_block3 = TemporalBlock(64, 64, kernel_size=7, stride=2, dilation=4)  # (B, 64, 128)
        self.time_block4 = TemporalBlock(64, 64, kernel_size=7, stride=1, dilation=8)  # (B, 64, 128)
        self.time_pool = AttentionPool1d(64, 32)  # (B, 64)

        # =====================================================================
        # 2. Fine-Grained Spectral Branch (Input: (B, 1, 1024))
        # =====================================================================
        self.freq_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=2, padding=5, bias=False),  # (B, 32, 512)
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.freq_block1 = TemporalBlock(32, 48, kernel_size=5, stride=2, dilation=1)  # (B, 48, 256)
        self.freq_block2 = TemporalBlock(48, 64, kernel_size=5, stride=2, dilation=2)  # (B, 64, 128)
        self.freq_block3 = TemporalBlock(64, 64, kernel_size=5, stride=1, dilation=4)  # (B, 64, 128)
        self.freq_pool = AttentionPool1d(64, 32)  # (B, 64)

        # =====================================================================
        # 3. Coarse Spectral-Band Branch (Input: (B, 8, 128))
        # =====================================================================
        self.mb_stem = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=7, stride=1, padding=3, bias=False),  # (B, 32, 128)
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.mb_block1 = TemporalBlock(32, 48, kernel_size=5, stride=2, dilation=1)  # (B, 48, 64)
        self.mb_block2 = TemporalBlock(48, 64, kernel_size=5, stride=1, dilation=2)  # (B, 64, 64)
        self.mb_pool = AttentionPool1d(64, 32)  # (B, 64)

        # =====================================================================
        # 4. Lightweight Fusion & Classifier Head (Input: (B, 192))
        # =====================================================================
        self.fusion_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def extract_spectral_representations(
        self, x_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract fine-grained (B, 1, 1024) and coarse sub-band (B, 8, 128) spectral views.

        Process:
          1. Per-sample mean detrending: x - mean(x)
          2. Real FFT (n=2048) -> 1025 complex bins
          3. Drop DC bin (k=0) -> 1024 positive-frequency bins (k=1..1024, 0-20 MHz)
          4. Power spectrum: |X|^2 / 2048
          5. Log scaling: log(1 + 1000 * P)
          6. Fine view: (B, 1, 1024)
          7. Coarse view: reshape to (B, 8, 128) (8 uniform ~2.5 MHz computational sub-bands)
        """
        # 1. Detrend mean per sample
        x_detrend = x_raw - x_raw.mean(dim=-1, keepdim=True)

        # 2. Real FFT
        fft_complex = torch.fft.rfft(x_detrend.squeeze(1), n=2048)  # (B, 1025)

        # 3. Drop DC bin (k=0), retain positive bins k=1..1024
        fft_pos = fft_complex[:, 1:]  # (B, 1024)

        # 4. Power spectrum
        power = (torch.abs(fft_pos) ** 2) / 2048.0  # (B, 1024)

        # 5. Log compression
        p_log = torch.log1p(1000.0 * power)  # (B, 1024)

        # 6. Fine-grained view: (B, 1, 1024)
        freq_input = p_log.unsqueeze(1)

        # 7. Coarse sub-band view: (B, 8, 128)
        mb_input = p_log.view(-1, 8, 128)

        return freq_input, mb_input

    def forward(
        self, x: torch.Tensor, return_embeddings: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through all three branches, attention pooling, and fusion head.

        Args:
            x: Raw time-domain waveform tensor of shape (B, 1, 2048).
            return_embeddings: If True, also returns dictionary of branch embeddings.

        Returns:
            logits: Class logits of shape (B, num_classes).
        """
        # Deterministic spectral extraction
        freq_input, mb_input = self.extract_spectral_representations(x)

        # 1. Temporal branch
        t_feat = self.time_stem(x)
        t_feat = self.time_block1(t_feat)
        t_feat = self.time_block2(t_feat)
        t_feat = self.time_block3(t_feat)
        t_feat = self.time_block4(t_feat)
        h_t = self.time_pool(t_feat)  # (B, 64)

        # 2. Fine-grained spectral branch
        f_feat = self.freq_stem(freq_input)
        f_feat = self.freq_block1(f_feat)
        f_feat = self.freq_block2(f_feat)
        f_feat = self.freq_block3(f_feat)
        h_f = self.freq_pool(f_feat)  # (B, 64)

        # 3. Coarse spectral-band branch
        m_feat = self.mb_stem(mb_input)
        m_feat = self.mb_block1(m_feat)
        m_feat = self.mb_block2(m_feat)
        h_m = self.mb_pool(m_feat)  # (B, 64)

        # 4. Feature concatenation & fusion
        h_cat = torch.cat([h_t, h_f, h_m], dim=-1)  # (B, 192)
        logits = self.fusion_head(h_cat)  # (B, num_classes)

        if return_embeddings:
            return logits, {
                "h_temporal": h_t,
                "h_spectral": h_f,
                "h_multiband": h_m,
                "h_fused": h_cat,
            }
        return logits
