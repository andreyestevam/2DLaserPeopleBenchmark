"""
lfe_model.py
PyTorch implementation of the Laser Feature Extractor (LFE) from:
    Amodeo et al. "FROG: a new people detection dataset for knee-high 2D range finders"
    Frontiers in Robotics and AI, 2025.

Architecture:
    - 3 residual blocks of depthwise separable 1D convolutions
    - Downsampling by 2 then by 3 (total 6x)
    - Global feature aggregator at each block
    - U-Net style decoder for segmentation output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution (Chollet 2017)."""

    def __init__(self, in_ch, out_ch, kernel_size, padding='same'):
        super().__init__()
        pad = kernel_size // 2 if padding == 'same' else 0
        self.depthwise  = nn.Conv1d(in_ch, in_ch,  kernel_size,
                                     padding=pad, groups=in_ch, bias=False)
        self.pointwise  = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn         = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class GlobalFeatureAggregator(nn.Module):
    """
    Global maxpool → broadcast and concatenate back to every position.
    Gives every local position access to global context.
    """

    def forward(self, x):
        # x: (B, C, L)
        g = x.max(dim=-1, keepdim=True).values   # (B, C, 1)
        g = g.expand_as(x)                        # (B, C, L)
        return torch.cat([x, g], dim=1)           # (B, 2C, L)


class ResidualBlock(nn.Module):
    """
    One LFE residual block:
        3 depthwise-separable convolutions (kernels 9, 7, 5)
        optional global feature aggregator after first conv
        residual skip connection
    """

    def __init__(self, in_ch, out_ch=32, use_global=False):
        super().__init__()
        self.use_global = use_global
        self.gfa        = GlobalFeatureAggregator() if use_global else None

        # After GFA the channel count doubles
        mid_ch = out_ch * 2 if use_global else out_ch

        self.conv1 = DepthwiseSeparableConv1d(in_ch,  out_ch, 9)
        self.conv2 = DepthwiseSeparableConv1d(mid_ch, out_ch, 7)
        self.conv3 = DepthwiseSeparableConv1d(out_ch, out_ch, 5)

        # 1x1 projection for residual if channel sizes differ
        self.proj  = nn.Conv1d(in_ch, out_ch, 1, bias=False) \
                     if in_ch != out_ch else nn.Identity()
        self.bn    = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        residual = self.proj(x)

        out = self.conv1(x)
        if self.use_global:
            out = self.gfa(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return F.relu(self.bn(out + residual))


# ─────────────────────────────────────────────
# Encoder (LFE backbone)
# ─────────────────────────────────────────────

class LFEEncoder(nn.Module):
    """
    LFE backbone — produces feature maps at three resolutions:
        scale0: original resolution  (B, 32, 720)
        scale1: downsampled by 2     (B, 32, 360)
        scale2: downsampled by 6     (B, 32, 120)
    """

    CHANNELS = 32

    def __init__(self):
        super().__init__()
        C = self.CHANNELS

        self.block0 = ResidualBlock(1,  C, use_global=True)   # input: 1 channel
        self.pool0  = nn.MaxPool1d(2)                          # 720 → 360

        self.block1 = ResidualBlock(C, C, use_global=True)
        self.pool1  = nn.MaxPool1d(3)                          # 360 → 120

        self.block2 = ResidualBlock(C, C, use_global=False)

    def forward(self, x):
        # x: (B, 1, 720)
        s0 = self.block0(x)          # (B, 32, 720)
        s1 = self.block1(self.pool0(s0))   # (B, 32, 360)
        s2 = self.block2(self.pool1(s1))   # (B, 32, 120)
        return s0, s1, s2


# ─────────────────────────────────────────────
# Decoder (for segmentation head)
# ─────────────────────────────────────────────

class LFEDecoder(nn.Module):
    """
    U-Net style decoder — mirrors the encoder, upsampling back to
    original resolution using skip connections.
    """

    def __init__(self):
        super().__init__()
        C = LFEEncoder.CHANNELS

        self.up1    = nn.Upsample(scale_factor=3, mode='linear', align_corners=False)
        self.block1 = ResidualBlock(C * 2, C)   # cat with s1

        self.up2    = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.block2 = ResidualBlock(C * 2, C)   # cat with s0

    def forward(self, s0, s1, s2):
        x = self.up1(s2)                         # (B, 32, 360)
        x = torch.cat([x, s1], dim=1)            # (B, 64, 360)
        x = self.block1(x)                       # (B, 32, 360)

        x = self.up2(x)                          # (B, 32, 720)
        x = torch.cat([x, s0], dim=1)            # (B, 64, 720)
        x = self.block2(x)                       # (B, 32, 720)
        return x


# ─────────────────────────────────────────────
# Full segmentation model
# ─────────────────────────────────────────────

class LFESegmentation(nn.Module):
    """
    Full LFE model for the segmentation pretraining task.
    Input:  (B, 1, 720) normalised range scan
    Output: (B, 720)    logits — sigmoid gives per-point person probability
    """

    def __init__(self):
        super().__init__()
        self.encoder = LFEEncoder()
        self.decoder = LFEDecoder()
        self.head    = nn.Conv1d(LFEEncoder.CHANNELS, 1, 1)   # pointwise → 1 logit

    def forward(self, x):
        s0, s1, s2 = self.encoder(x)
        features   = self.decoder(s0, s1, s2)      # (B, 32, 720)
        logits     = self.head(features).squeeze(1) # (B, 720)
        return logits

    def predict_proba(self, x):
        """Convenience: returns sigmoid probabilities instead of logits."""
        return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────
# Quick architecture sanity check
# ─────────────────────────────────────────────
if __name__ == '__main__':
    model = LFESegmentation()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LFESegmentation — total parameters: {total_params:,}")

    dummy = torch.randn(4, 1, 720)   # batch of 4 scans
    out   = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")   # expect (4, 720)
    assert out.shape == (4, 720), "Shape mismatch!"
    print("Architecture check passed.")