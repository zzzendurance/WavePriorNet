"""
WavePriorNet  –  Stage 1
========================
Wavelet Spatial-Temporal Transformer with Physical Color Prior
for Video Dehazing on REVIDE_indoor.

Modules taken verbatim (or minimally adapted) from:
  - SVINet    : WaveletTransformerBlock, WaveletAttention, Encoder
  - DehazeFormer: RLN (detach_grad=False), SKFusion, TransformerBlock pattern
  - MAP-Net   : PhysicalPriorHead (t/A estimation), GMRA fusion pattern,
                residual prediction (out = img + delta), aux_j/aux_i outputs
  - color-alignment: Chamfer color loss in train.py

Forward signature
-----------------
  model(hazy, color_ref)  →  dict with keys:
      'out'    : [B*T, 3, H, W]   main prediction
      'aux_j'  : list[Tensor]     multi-stage scene predictions (training only)
      'aux_i'  : list[Tensor]     multi-stage physical reconstructions (training only)
      'img_in' : [B*T, 3, H, W]   normalized input (for aux_i loss target, training only)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.dwt import DWT_2D, IDWT_2D


# ============================================================
# 1.  RLN  –  copied verbatim from DehazeFormer/models/dehazeformer.py
#     Key fix vs old version: detach_grad=False by default so
#     gradients flow through meta1/meta2 (std/mean → rescale/rebias).
# ============================================================

class RLN(nn.Module):
    """Revised LayerNorm (DehazeFormer, verbatim).
    detach_grad=False  →  gradients flow through std/mean into meta nets.
    """
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super().__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias   = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        nn.init.trunc_normal_(self.meta1.weight, std=0.02)
        nn.init.constant_(self.meta1.bias, 1)
        nn.init.trunc_normal_(self.meta2.weight, std=0.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std  = torch.sqrt((x - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized = (x - mean) / std
        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)
        out = normalized * self.weight + self.bias
        return out, rescale, rebias


# ============================================================
# 2.  SKFusion  –  copied verbatim from DehazeFormer
#     Used for skip-connection fusion in the decoder.
# ============================================================

class SKFusion(nn.Module):
    """Selective Kernel Fusion (DehazeFormer, verbatim)."""
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1).view(B, self.height, C, H, W)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        return torch.sum(in_feats * attn, dim=1)


# ============================================================
# 3.  Basic blocks
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        return x + self.body(x)


class Deconv(nn.Module):
    """Bilinear upsample + conv (avoids checkerboard)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, 1, padding)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


# ============================================================
# 4.  Encoder  –  from SVINet (model_SVINet_10.py), verbatim
# ============================================================

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base=64, out_ch=128):
        super().__init__()
        self.encoder_l1 = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(base),
            ResBlock(base),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_l2 = nn.Sequential(
            nn.Conv2d(base, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(out_ch),
            ResBlock(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, frame):
        return self.encoder_l2(self.encoder_l1(frame))


# ============================================================
# 5.  WaveletAttention + WaveletTransformerBlock
#     from SVINet (model_SVINet_10.py), verbatim.
#     RLN pattern follows DehazeFormer TransformerBlock:
#       norm → attn → x = x*rescale + rebias → residual → FFN
# ============================================================

class WaveletAttention(nn.Module):
    """Multi-head wavelet attention (SVINet, verbatim)."""

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding   = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.output_linear_LL = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.output_linear_LH = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.output_linear_HL = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.output_linear_HH = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.output_linear_L3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        # learnable weight (SVINet Attention.self.w)
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x_LL, x_LH, x_HL, x_HH, x_L3, b, c):
        bt, _, h, w = x_LL.size()
        t   = bt // b
        d_k = c // len(self.patchsize)

        output_LL, output_LH, output_HL, output_HH, output_L3 = [], [], [], [], []

        _query = self.query_embedding(x_LL)
        _key   = self.key_embedding(x_LL)
        _value = self.value_embedding(x_LL)

        for (width, height), query, key, value, LH, HL, HH, L3 in zip(
                self.patchsize,
                torch.chunk(_query, len(self.patchsize), dim=1),
                torch.chunk(_key,   len(self.patchsize), dim=1),
                torch.chunk(_value, len(self.patchsize), dim=1),
                torch.chunk(x_LH,  len(self.patchsize), dim=1),
                torch.chunk(x_HL,  len(self.patchsize), dim=1),
                torch.chunk(x_HH,  len(self.patchsize), dim=1),
                torch.chunk(x_L3,  len(self.patchsize), dim=1)):

            out_w, out_h = w // width, h // height

            def to_seq(x):
                x = x.view(b, t, d_k, out_h, height, out_w, width)
                return x.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                    b, t * out_h * out_w, d_k * height * width)

            query = to_seq(query)
            key   = to_seq(key)
            value = to_seq(value)
            LH    = to_seq(LH)
            HL    = to_seq(HL)
            HH    = to_seq(HH)
            L3    = to_seq(L3)

            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
            p_attn = F.softmax(scores, dim=-1)

            def from_seq(y):
                y = y.view(b, t, out_h, out_w, d_k, height, width)
                return y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)

            output_LL.append(from_seq(torch.matmul(p_attn, value)))
            output_LH.append(from_seq(torch.matmul(p_attn, LH)))
            output_HL.append(from_seq(torch.matmul(p_attn, HL)))
            output_HH.append(from_seq(torch.matmul(p_attn, HH)))
            output_L3.append(from_seq(torch.matmul(p_attn, L3)))

        x_LL = self.output_linear_LL(torch.cat(output_LL, 1))
        x_LH = self.output_linear_LH(torch.cat(output_LH, 1))
        x_HL = self.output_linear_HL(torch.cat(output_HL, 1))
        x_HH = self.output_linear_HH(torch.cat(output_HH, 1))
        x_L3 = self.output_linear_L3(torch.cat(output_L3, 1))
        return x_LL, x_LH, x_HL, x_HH, x_L3


class FFN(nn.Module):
    """Feed-forward network (SVINet, verbatim)."""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class WaveletTransformerBlock(nn.Module):
    """
    SVINet WaveletTransformerBlock with DehazeFormer RLN pattern.

    DehazeFormer TransformerBlock (dehazeformer.py L261-273):
        identity = x
        x, rescale, rebias = norm1(x)
        x = attn(x)
        x = x * rescale + rebias          ← rescale/rebias applied AFTER attn
        x = identity + x                  ← residual
        x = x + ffn(x)
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attn    = WaveletAttention(patchsize, d_model=hidden)
        self.ffn     = FFN(hidden)
        self.ffn_l3  = FFN(hidden)
        self.dwt     = DWT_2D()
        self.idwt    = IDWT_2D()
        self.rln     = RLN(hidden, detach_grad=False)   # gradients flow through
        self.down    = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up      = Deconv(hidden, hidden, kernel_size=1, padding=0)
        self.fusion  = nn.Conv2d(hidden * 2, hidden, 1, 1, 0, bias=True)

    def forward(self, x_dict):
        x, b, c = x_dict['x'], x_dict['b'], x_dict['c']

        # --- DehazeFormer pattern: norm → attn → rescale/rebias → residual ---
        identity = x
        x_norm, rescale, rebias = self.rln(x)

        ll, lh, hl, hh = self.dwt(x_norm)
        x_down = self.down(x_norm)

        y_LL, y_LH, y_HL, y_HH, y_L3 = self.attn(ll, lh, hl, hh, x_down, b, c)

        Y = self.idwt(ll + y_LL, lh + y_LH, hl + y_HL, hh + y_HH)
        Y = Y * rescale + rebias          # rescale after attn (DehazeFormer pattern)
        Y = identity + Y                  # residual
        Y = Y + self.ffn(Y)               # FFN

        Y_L3    = (x_down + y_L3) + self.ffn_l3(x_down + y_L3)
        Y_up    = self.up(Y_L3)
        y_final = self.fusion(torch.cat([Y_up, Y], dim=1))

        return {'x': y_final, 'b': b, 'c': c}


# ============================================================
# 6.  PhysicalPriorHead  –  from MAP-Net PriorDecodeLayer
#     Estimates transmission map t and atmospheric light A
#     from the color-reference feature.
#     Output: t [B*T,1,H,W], A [B*T,1,1,1], prior_feat [B*T,C,H,W]
# ============================================================

class PhysicalPriorHead(nn.Module):
    """
    Simplified MAP-Net PriorDecodeLayer (mapnet_net.py L18-94).
    Estimates t (transmission) and A (atmospheric light) from prior features.
    """

    def __init__(self, channels, num_trans_bins=32):
        super().__init__()
        self.num_trans_bins = num_trans_bins

        # transmission head: softmax classification → weighted sum → scalar map
        self.head_t = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, num_trans_bins, 1, 1, 0),
        )
        # atmospheric light head: global pooling → scalar
        self.head_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        """
        feat: [B*T, C, H, W]
        returns:
            t    : [B*T, 1, H, W]   transmission map in (0,1)
            A    : [B*T, 1, 1, 1]   atmospheric light scalar
            feat : [B*T, C, H, W]   unchanged (pass-through for GMRA)
        """
        logit_t = self.head_t(feat)
        prob_t  = torch.softmax(logit_t, dim=1)   # [BT, bins, H, W]
        BT, d, h, w = prob_t.shape
        # discrete values in (0, 1)
        values = torch.linspace(0., 1., d, device=feat.device).view(1, d, 1, 1)
        t = (prob_t * values).sum(dim=1, keepdim=True)   # [BT, 1, H, W]
        A = self.head_a(feat)                             # [BT, 1, 1, 1]
        return t, A, feat


# ============================================================
# 7.  PriorBranch  –  encoder + PhysicalPriorHead
# ============================================================

class PriorBranch(nn.Module):
    def __init__(self, in_ch=3, hidden=128, num_trans_bins=32):
        super().__init__()
        self.encoder    = Encoder(in_ch=in_ch, base=64, out_ch=hidden)
        self.refine     = nn.Sequential(ResBlock(hidden), ResBlock(hidden))
        self.prior_head = PhysicalPriorHead(hidden, num_trans_bins)

    def forward(self, color_ref):
        """color_ref: [B*T, 3, H, W]"""
        feat = self.refine(self.encoder(color_ref))
        t, A, feat = self.prior_head(feat)
        return feat, t, A


# ============================================================
# 8.  GMRA Fusion  –  from MAP-Net SceneDecodeLayer (mapnet_net.py L197-221)
#     attn = attn_j + beta * attn_p   (per-pixel, nr=1 here)
#     Adapted to single-frame (nr=1) to avoid STDA dependency.
# ============================================================

class GMRAFusion(nn.Module):
    """
    MAP-Net GMRA (mapnet_net.py L197-221), adapted for nr=1 (single-range).
    attn = attn_scene + beta * attn_prior
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.aggre_beta = nn.Parameter(torch.ones(1))
        # guide_conv: cat(prior_feat, scene_feat) → scene_feat  (MAP-Net L117)
        self.guide_conv = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        # aggre_conv: cat(aggregated, query) → output  (MAP-Net L129)
        self.aggre_conv = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(hidden),
            ResBlock(hidden),
        )

    def forward(self, feat_scene, feat_prior):
        """
        feat_scene: [BT, C, H, W]
        feat_prior: [BT, C, H, W]
        """
        # prior guide (MAP-Net L146-149)
        feat_j = self.guide_conv(torch.cat([feat_prior, feat_scene], dim=1))

        # GMRA attention with nr=1: when sequence length=1, softmax([scalar])=1,
        # so attn@v = v = feat_j. Compute attention weight as channel-wise dot product
        # instead of [BT*H*W, 1, 1] matmul to avoid the 1.25GB allocation.
        scale = feat_j.shape[1] ** -0.5
        attn_j = (feat_j * feat_j).sum(dim=1, keepdim=True) * scale        # [BT,1,H,W]
        attn_p = (feat_j * feat_prior).sum(dim=1, keepdim=True) * scale     # [BT,1,H,W]
        # sigmoid replaces the degenerate softmax(scalar) which was always 1
        gate = torch.sigmoid(attn_j + self.aggre_beta * attn_p)             # [BT,1,H,W]
        out  = feat_j * gate                                                 # [BT,C,H,W]

        return self.aggre_conv(torch.cat([out, feat_j], dim=1))


# ============================================================
# 9.  Decoder  –  MAP-Net MAPUpsampler style
#     Key: conv_last initialized to 0 for residual prediction.
#     Uses SKFusion for skip connections (DehazeFormer).
# ============================================================

class Decoder(nn.Module):
    """
    Decoder with:
    - 2× bilinear upsample (stride-2 encoder → stride-1 output)
    - SKFusion skip connection (DehazeFormer)
    - conv_last init=0 for residual prediction (MAP-Net MAPUpsampler L95-96)
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.up = Deconv(hidden, hidden, 3, 1)
        self.skip_fusion = SKFusion(hidden, height=2)

        self.body = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(hidden),
            ResBlock(hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(64),
            ResBlock(64),
        )
        self.conv_hr   = nn.Conv2d(64, 64, 3, 1, 1)
        self.lrelu     = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # MAP-Net: init conv_last to 0 so initial output is pure residual=0
        nn.init.constant_(self.conv_last.weight, 0)
        nn.init.constant_(self.conv_last.bias,   0)

    def forward(self, feat, skip=None):
        """
        feat : [BT, hidden, H/2, W/2]
        skip : [BT, hidden, H/2, W/2]  (encoder skip, fused before upsample)
        """
        if skip is not None:
            feat = self.skip_fusion([feat, skip])      # fuse at H/2, same spatial size
        x = self.up(feat)                              # [BT, hidden, H, W]
        x = self.body(x)
        x = self.lrelu(self.conv_hr(x))
        return self.conv_last(x)                       # residual delta


# ============================================================
# 10.  WavePriorNet  –  full model
# ============================================================

class WavePriorNet(nn.Module):
    """
    Full WavePriorNet.

    Residual prediction (MAP-Net pattern):
        pred = hazy_in + decoder_delta
    where hazy_in is the input hazy frame (normalized to [0,1]).

    During training, also outputs:
        aux_j: list of per-stage scene predictions
        aux_i: list of per-stage physical reconstructions  J*t + A*(1-t)
    for multi-stage supervision (MAP-Net map.py L91-98).
    """

    def __init__(
        self,
        num_frames: int = 5,
        hidden: int = 128,
        stack_num: int = 6,
        num_trans_bins: int = 32,
        patchsize=((32, 32), (16, 16), (8, 8), (4, 4)),
    ):
        super().__init__()
        self.num_frames = num_frames

        # --- hazy encoder (SVINet) ---
        self.hazy_encoder = Encoder(in_ch=3, base=64, out_ch=hidden)

        # --- color prior branch (MAP-Net style) ---
        self.prior_branch = PriorBranch(in_ch=3, hidden=hidden, num_trans_bins=num_trans_bins)

        # --- GMRA fusion (MAP-Net style) ---
        self.gmra = GMRAFusion(hidden=hidden)

        # --- wavelet transformer backbone (SVINet + DehazeFormer RLN) ---
        self.transformer = nn.Sequential(*[
            WaveletTransformerBlock(patchsize, hidden=hidden)
            for _ in range(stack_num)
        ])

        # --- auxiliary scene prediction heads (MAP-Net aux_j style) ---
        # one head at the bottleneck (after transformer)
        self.aux_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 3, 1, 1, 0),
        )

        # --- decoder (MAP-Net residual style + DehazeFormer SKFusion) ---
        self.decoder = Decoder(hidden=hidden)

    def forward(self, hazy, color_ref):
        """
        hazy      : [B, T, 3, H, W]  values in [0, 1]
        color_ref : [B, T, 3, H, W]  values in [0, 1]

        Returns dict:
          'out'    : [B*T, 3, H, W]   final prediction, clamped to [0,1]
          'aux_j'  : [[B*T,3,H,W]]    auxiliary scene pred (training only)
          'aux_i'  : [[B*T,3,H,W]]    auxiliary physical recon (training only)
          'img_in' : [B*T,3,H,W]      input hazy (for aux_i loss target)
        """
        B, T, C, H, W = hazy.size()
        BT = B * T

        hazy_flat  = hazy.view(BT, C, H, W)       # keep for residual add
        color_flat = color_ref.view(BT, C, H, W)

        # --- encode ---
        feat_scene           = self.hazy_encoder(hazy_flat)         # [BT, ch, H/2, W/2]
        feat_prior, t, A     = self.prior_branch(color_flat)        # [BT,ch,H/2,W/2], t, A

        # --- GMRA fusion ---
        feat = self.gmra(feat_scene, feat_prior)                    # [BT, ch, H/2, W/2]

        # --- wavelet transformer ---
        _, C1, H2, W2 = feat.size()
        feat = self.transformer({'x': feat, 'b': B, 'c': C1})['x'] # [BT, ch, H/2, W/2]

        # --- auxiliary predictions (MAP-Net aux_j / aux_i pattern) ---
        aux_j_list, aux_i_list = [], []
        if self.training:
            # aux_j: scene prediction from bottleneck feature
            # upsample to full res first (aux head operates at H/2, we interp to H)
            delta_aux = self.aux_head(feat)                         # [BT, 3, H/2, W/2]
            delta_aux = F.interpolate(delta_aux, size=(H, W), mode='bilinear', align_corners=False)
            aux_j = (hazy_flat + delta_aux).clamp(0, 1)            # residual (MAP-Net L381)

            # aux_i: physical reconstruction  J*t + A*(1-t)  (MAP-Net L395)
            t_up = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False).clamp(0, 1)
            aux_i = aux_j * t_up + A * (1 - t_up)

            aux_j_list.append(aux_j)
            aux_i_list.append(aux_i)

        # --- decode (residual, MAP-Net style) ---
        delta = self.decoder(feat, skip=feat_scene)                 # [BT, 3, H, W]
        out   = (hazy_flat + delta).clamp(0, 1)                    # MAP-Net L381

        result = {'out': out, 'img_in': hazy_flat}
        if self.training:
            result['aux_j'] = aux_j_list
            result['aux_i'] = aux_i_list
        return result
