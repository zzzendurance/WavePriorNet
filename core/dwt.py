"""
Pure-PyTorch 2D Haar Wavelet Transform (DWT) and Inverse (IDWT).
Drop-in replacement for DWT_IDWT.DWT_IDWT_layer so the project has no
external wavelet dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT_2D(nn.Module):
    """2D Haar DWT: splits [B, C, H, W] into four sub-bands of size [B, C, H/2, W/2]."""

    def __init__(self, wavename='haar'):
        super().__init__()
        # Haar analysis filters (unnormalised)
        self.register_buffer('ll', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5)
        self.register_buffer('lh', torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5)
        self.register_buffer('hl', torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5)
        self.register_buffer('hh', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5)

    def _apply_filter(self, x, filt):
        # filt: [2, 2] → reshape to [1, 1, 2, 2] for depthwise conv
        C = x.shape[1]
        k = filt.view(1, 1, 2, 2).expand(C, 1, 2, 2)
        return F.conv2d(x, k, stride=2, groups=C)

    def forward(self, x):
        ll = self._apply_filter(x, self.ll)
        lh = self._apply_filter(x, self.lh)
        hl = self._apply_filter(x, self.hl)
        hh = self._apply_filter(x, self.hh)
        return ll, lh, hl, hh


class IDWT_2D(nn.Module):
    """2D Haar IDWT: reconstructs [B, C, H, W] from four sub-bands."""

    def __init__(self, wavename='haar'):
        super().__init__()
        # Haar synthesis filters (transpose of analysis)
        self.register_buffer('ll', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5)
        self.register_buffer('lh', torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5)
        self.register_buffer('hl', torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5)
        self.register_buffer('hh', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5)

    def _apply_filter_t(self, x, filt):
        C = x.shape[1]
        k = filt.view(1, 1, 2, 2).expand(C, 1, 2, 2)
        return F.conv_transpose2d(x, k, stride=2, groups=C)

    def forward(self, ll, lh, hl, hh):
        return (self._apply_filter_t(ll, self.ll)
                + self._apply_filter_t(lh, self.lh)
                + self._apply_filter_t(hl, self.hl)
                + self._apply_filter_t(hh, self.hh))
