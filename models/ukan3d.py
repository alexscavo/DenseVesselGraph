"""3D tokenized-KAN blocks for the dense model's U-Net (3 Nov 2026).

Port of U-KAN (Li et al. 2024, arXiv 2406.02918; code CUHK-AIM-Group/U-KAN, Seg_UKAN/archs.py) to
3D, used to replace only the two deepest encoder and the two first decoder stages of the MONAI
BasicUNet, as U-KAN does (conv stages at high resolution, tokenized KAN at the deepest ones).
- KANLinear: B-spline KAN layer as in U-KAN's kan.py (efficient-kan formulation): output =
  SiLU(x) W_base + B-spline basis(x) W_spline, cubic splines, grid size 5 on [-1, 1].
- KANBlock: tokens + KANLayer(LayerNorm(tokens)), KANLayer = 3 x (KANLinear -> depthwise 3x3x3
  conv -> BN -> ReLU), as in the official code (pre-norm residual).
- TokKANDown replaces a BasicUNet Down: stride-2 3x3x3 conv tokenizer + LayerNorm (U-KAN's
  PatchEmbed), KANBlock, LayerNorm.
- TokKANUp replaces a BasicUNet UpCat: conv + trilinear 2x upsampling, skip fusion by
  concatenation + 1x1x1 projection (the paper's Eqs. 8-9, and what MONAI UpCat does) or by
  addition (the released code), KANBlock, LayerNorm.
- kan=False gives the MLP control: identical blocks with nn.Linear instead of KANLinear (U-KAN's
  own KAN-vs-MLP ablation).
Channels and resolutions match the replaced blocks, so BasicUNet.forward is unchanged.
Deviation from the code: spline weights start from small uniform noise (no curve2coeff fit).
"""
import torch
import torch.nn.functional as F
from torch import nn


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer('grid', grid.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5 * scale_base)
        with torch.no_grad():
            self.spline_weight.uniform_(-scale_noise / grid_size, scale_noise / grid_size)
        nn.init.kaiming_uniform_(self.spline_scaler, a=5 ** 0.5 * scale_spline)

    def b_splines(self, x):
        """Cox-de Boor recursion: (N, in) -> (N, in, grid_size + spline_order)."""
        g = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - g[:, :-(k + 1)]) / (g[:, k:-1] - g[:, :-(k + 1)]) * bases[..., :-1]
                     + (g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:-k]) * bases[..., 1:])
        return bases

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).float()  # splines in fp32 under autocast
        with torch.autocast(x.device.type, enabled=False):
            base = F.linear(F.silu(x), self.base_weight.float())
            spline = F.linear(self.b_splines(x).reshape(len(x), -1),
                              (self.spline_weight * self.spline_scaler.unsqueeze(-1)).reshape(self.out_features, -1).float())
        return (base + spline).reshape(*shape[:-1], self.out_features)


class DWBnRelu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, tokens, size):
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, *size)
        return F.relu(self.bn(self.dw(x))).flatten(2).transpose(1, 2)


class KANBlock(nn.Module):
    def __init__(self, dim, layers=3, kan=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.ModuleList((KANLinear(dim, dim) if kan else nn.Linear(dim, dim)) for _ in range(layers))
        self.dw = nn.ModuleList(DWBnRelu(dim) for _ in range(layers))

    def forward(self, tokens, size):
        y = self.norm(tokens)
        for fc, dw in zip(self.fc, self.dw):
            y = dw(fc(y), size)
        return tokens + y


class TokKANDown(nn.Module):
    """Drop-in for monai BasicUNet `Down`: (B, in, S) -> (B, out, S/2)."""

    def __init__(self, in_ch, out_ch, kan=True):
        super().__init__()
        self.embed = nn.Conv3d(in_ch, out_ch, 3, 2, 1)
        self.embed_norm = nn.LayerNorm(out_ch)
        self.block = KANBlock(out_ch, kan=kan)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.embed(x)
        size = x.shape[2:]
        t = self.norm(self.block(self.embed_norm(x.flatten(2).transpose(1, 2)), size))
        return t.transpose(1, 2).reshape(x.shape[0], -1, *size)


class TokKANUp(nn.Module):
    """Drop-in for monai BasicUNet `UpCat`: forward(x, skip) with x at S/2, skip at S -> (B, out, S)."""

    def __init__(self, in_ch, skip_ch, out_ch, kan=True, skip='cat'):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))
        self.skip = skip
        self.fuse = nn.Conv3d(out_ch + skip_ch, out_ch, 1) if skip == 'cat' else nn.Conv3d(skip_ch, out_ch, 1)
        self.embed_norm = nn.LayerNorm(out_ch)
        self.block = KANBlock(out_ch, kan=kan)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(self.conv(x), size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = self.fuse(torch.cat([skip, x], 1)) if self.skip == 'cat' else x + self.fuse(skip)
        size = x.shape[2:]
        t = self.norm(self.block(self.embed_norm(x.flatten(2).transpose(1, 2)), size))
        return t.transpose(1, 2).reshape(x.shape[0], -1, *size)


def kanify_basic_unet(unet, features, kan=True, skip='cat'):
    """Replace down_3, down_4 (encoder) and upcat_4, upcat_3 (decoder) of a 3D BasicUNet in place.
    features = BasicUNet features (f0, f1, f2, f3, f4, f_up1); kan=False: MLP control."""
    f = features
    unet.down_3 = TokKANDown(f[2], f[3], kan)
    unet.down_4 = TokKANDown(f[3], f[4], kan)
    unet.upcat_4 = TokKANUp(f[4], f[3], f[3], kan, skip)
    unet.upcat_3 = TokKANUp(f[3], f[2], f[2], kan, skip)
    return unet

