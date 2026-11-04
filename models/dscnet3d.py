"""3D dynamic snake convolution blocks for the dense model's U-Net (4 Nov 2026).

Pure-PyTorch port of DSCNet (Qi et al., ICCV 2023, arXiv 2307.08388; code YaoleiQi/DSCNet,
DSCNet_3D_opensource/Code/Kipa/DSCNet/S3_DSConv.py and S3_DSCNet.py), used to replace every
double-conv block of the MONAI BasicUNet while keeping its levels, widths and skips.
- DSConv3d: a 1-D kernel of K taps along one axis (morph 0/1/2 = W/H/D). A 3x3x3 conv -> BN ->
  tanh predicts, per tap, offsets in [-1, 1] along the two perpendicular axes; offsets accumulate
  outwards from the centre tap (pinned at 0 for every axis), are scaled by extend_scope, and the
  input is sampled with trilinear grid_sample at the bent positions (coordinates clamped to the
  volume, as in the released code). The K samples are combined by a learned linear map (the
  released (1,1,K)-kernel conv with stride K), then GroupNorm(out/4) and ReLU.
- DSCBlock: the released block: standard 3x3x3 conv branch + three DSConv3d (W, H, D) on the same
  input, each with GroupNorm(out/4) and ReLU, concatenated and fused by a 3x3x3 conv + GN + ReLU.
  Optional gradient checkpointing (recompute in backward) keeps batch 8 within one GPU.
Fixes relative to the released 3D code: the kernel length is used (it was hard-coded to 3), and
the D-direction centre tap is pinned in both perpendicular axes (one axis was zeroed twice).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _gn(ch):
    return nn.GroupNorm(max(1, ch // 4), ch)


class DSConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=9, morph=0, extend_scope=1.0):
        super().__init__()
        self.k, self.morph, self.scope = kernel_size, morph, extend_scope
        self.offset_conv = nn.Conv3d(in_ch, 2 * kernel_size, 3, padding=1)
        self.offset_norm = nn.BatchNorm3d(2 * kernel_size)
        self.combine = nn.Conv3d(in_ch * kernel_size, out_ch, 1)
        self.norm = _gn(out_ch)

    def snake(self, offset):
        """(B, K, ...) per-tap offsets -> cumulative offsets from the centre tap (centre = 0)."""
        c = self.k // 2
        right = torch.cumsum(offset[:, c + 1:], 1)
        left = torch.flip(torch.cumsum(torch.flip(offset[:, :c], [1]), 1), [1])
        return torch.cat([left, torch.zeros_like(offset[:, c:c + 1]), right], 1) * self.scope

    def forward(self, x):
        B, C, D, H, W = x.shape
        K, dev = self.k, x.device
        off = torch.tanh(self.offset_norm(self.offset_conv(x)))  # (B, 2K, D, H, W)
        o1, o2 = self.snake(off[:, :K]), self.snake(off[:, K:])  # (B, K, D, H, W) each
        zd, yh, xw = torch.meshgrid(torch.arange(D, device=dev), torch.arange(H, device=dev),
                                    torch.arange(W, device=dev), indexing='ij')
        base = torch.stack([zd, yh, xw], 0).float()[None, :, None]  # (1, 3, 1, D, H, W) as (d, h, w)
        tap = torch.arange(K, device=dev, dtype=torch.float32) - K // 2
        along = {0: 2, 1: 1, 2: 0}[self.morph]  # morph 0 runs along W, 1 along H, 2 along D
        perp = [a for a in (0, 1, 2) if a != along]
        pos = base.expand(B, 3, K, D, H, W).clone()
        pos[:, along] += tap.view(1, K, 1, 1, 1)
        pos[:, perp[0]] += o1
        pos[:, perp[1]] += o2
        size = torch.tensor([D, H, W], device=dev, dtype=torch.float32).view(1, 3, 1, 1, 1, 1)
        pos = torch.minimum(pos.clamp_min(0), size - 1)
        grid = (pos / (size - 1).clamp_min(1) * 2 - 1).flip(1)  # grid_sample wants (x=w, y=h, z=d)
        grid = grid.permute(0, 3, 4, 5, 2, 1).reshape(B, D, H, W * K, 3)
        s = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        s = s.view(B, C, D, H, W, K).permute(0, 1, 5, 2, 3, 4).reshape(B, C * K, D, H, W)
        return F.relu(self.norm(self.combine(s)))


class DSCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=9, extend_scope=1.0, checkpointing=True):
        super().__init__()
        self.std = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), _gn(out_ch), nn.ReLU(inplace=True))
        self.snakes = nn.ModuleList(DSConv3d(in_ch, out_ch, kernel_size, m, extend_scope) for m in (0, 1, 2))
        self.fuse = nn.Sequential(nn.Conv3d(4 * out_ch, out_ch, 3, padding=1), _gn(out_ch), nn.ReLU(inplace=True))
        self.checkpointing = checkpointing

    def forward(self, x):
        # Checkpoint each snake branch separately: only one branch's K-times-expanded samples are
        # alive at a time during the backward recomputation. (Its BatchNorm running statistics are
        # updated twice per step as a result; the batch statistics used for training are unchanged.)
        ckpt = self.checkpointing and self.training and torch.is_grad_enabled()
        branches = [checkpoint(s, x, use_reentrant=False) if ckpt else s(x) for s in self.snakes]
        return self.fuse(torch.cat([self.std(x)] + branches, 1))


def _channels(two_conv):
    convs = [m for m in two_conv.modules() if isinstance(m, nn.Conv3d)]
    return convs[0].in_channels, convs[-1].out_channels


def dscify_basic_unet(unet, kernel_size=9, extend_scope=1.0, checkpointing=True):
    """Replace every TwoConv of a 3D MONAI BasicUNet (conv_0, the Down and UpCat blocks) by a DSCBlock."""
    make = lambda tc: DSCBlock(*_channels(tc), kernel_size, extend_scope, checkpointing)
    unet.conv_0 = make(unet.conv_0)
    for name in ('down_1', 'down_2', 'down_3', 'down_4', 'upcat_4', 'upcat_3', 'upcat_2', 'upcat_1'):
        block = getattr(unet, name)
        block.convs = make(block.convs)
    return unet
