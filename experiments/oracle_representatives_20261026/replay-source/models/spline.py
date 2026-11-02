"""Fixed cubic spline geometry for an optional pairwise vessel head."""
from __future__ import annotations

import torch
from torch import nn


def cubic_basis(controls: int, samples: int = 64):
    if controls < 4 or samples < 2:
        raise ValueError('Cubic splines require >=4 controls and >=2 samples')
    knots = torch.tensor([0.] * 4 + [i/(controls-3) for i in range(1, controls-3)] + [1.] * 4,
                         dtype=torch.float64)
    t = torch.linspace(0, 1, samples, dtype=torch.float64)
    values = ((t[:, None] >= knots[:-1]) & (t[:, None] < knots[1:])).double()
    for degree in range(1, 4):
        width = len(knots)-degree-1
        left = knots[degree:degree+width]-knots[:width]
        right = knots[degree+1:degree+1+width]-knots[1:width+1]
        values = ((t[:, None]-knots[:width])/left.clamp_min(1e-15)*values[:, :width]*(left>0)
                  + (knots[degree+1:degree+1+width]-t[:, None])/right.clamp_min(1e-15)*values[:, 1:width+1]*(right>0))
    values[-1] = 0; values[-1, -1] = 1
    greville = torch.stack([knots[i+1:i+4].mean() for i in range(controls)])
    return values.float(), greville.float()


def canonical_reverse(endpoints_mm):
    """Lexicographic physical orientation, without sorting on unstable query IDs."""
    a, b = endpoints_mm[:, 0].detach(), endpoints_mm[:, 1].detach()
    reverse = a[:, 0] > b[:, 0]
    tied = a[:, 0] == b[:, 0]
    for axis in (1, 2):
        reverse = reverse | (tied & (a[:, axis] > b[:, axis]))
        tied = tied & (a[:, axis] == b[:, axis])
    return reverse


class SplineHead(nn.Module):
    """Predict physical residuals from pair tokens and explicit endpoint geometry.

    Inputs/outputs are normalized DHW coordinates; metric maps them to physical
    millimetres. No sigmoid/clipping: internal controls may lie outside a patch.
    """
    def __init__(self, pair_width, hidden_width, controls=8, samples=64, residual_scale='mm', curve_evidence=None, family='bspline', token_inputs='full', local_geometry=None):
        super().__init__()
        self.controls = int(controls)
        self.residual_scale = residual_scale
        if family not in ('bspline', 'catmull_rom'):
            raise ValueError('Unknown spline family')
        self.family = family
        if token_inputs not in ('full', 'zero'):
            raise ValueError('token_inputs must be full or zero')
        self.token_inputs = token_inputs
        basis, greville = cubic_basis(self.controls, samples)
        if family == 'catmull_rom':
            greville = torch.linspace(0, 1, self.controls)
        self.register_buffer('basis', basis)
        self.register_buffer('greville', greville)
        self.network = nn.Sequential(nn.Linear(pair_width+10, hidden_width), nn.ReLU(),
                                     nn.Linear(hidden_width, hidden_width), nn.ReLU(),
                                     nn.Linear(hidden_width, 3*(self.controls-2)))
        # Zero residual starts at an exactly linearly parameterized chord.
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
        self.curve_evidence = None
        self.local_geometry = None
        if local_geometry and local_geometry.get('enabled', False):
            from .local_spline_geometry import LocalSplineGeometry
            with torch.random.fork_rng(devices=[]):
                self.local_geometry = LocalSplineGeometry(controls,
                    channels=local_geometry.get('channels', 64), samples=local_geometry.get('samples', 32),
                    scale=local_geometry.get('scale', 1.), image_mode=local_geometry.get('image_mode', 'full'))
        if curve_evidence and curve_evidence.get('enabled', False):
            from .curve_evidence import CurveEvidence
            with torch.random.fork_rng(devices=[]):
                self.curve_evidence = CurveEvidence(
                    channels=curve_evidence.get('channels', 8), samples=curve_evidence.get('samples', 32),
                    scale=curve_evidence.get('scale', 1.), chunk_size=curve_evidence.get('chunk_size', 256))

    def forward(self, pair_features, endpoints, metric, *, local_features=None, input_shape=None):
        if self.token_inputs == 'zero':
            pair_features = torch.zeros_like(pair_features)
        physical = endpoints @ metric.T
        chord = physical[:, 1]-physical[:, 0]
        length = chord.norm(dim=-1, keepdim=True)
        features = torch.cat((pair_features, physical.flatten(1), chord, length), dim=-1)
        residual = self.network(features).reshape(-1, self.controls-2, 3)
        if self.residual_scale == 'relative':
            residual = residual * length[:, None].clamp_min(1e-6)
        t = self.greville[None, 1:-1, None]
        interior = physical[:, :1]*(1-t) + physical[:, 1:]*t + residual
        controls = torch.cat((physical[:, :1], interior, physical[:, 1:]), dim=1) @ torch.linalg.inv(metric).T
        # Copy endpoints exactly; inverse-affine roundoff must not disconnect edges.
        controls = torch.cat((endpoints[:, :1], controls[:, 1:-1], endpoints[:, 1:]), dim=1)
        if self.local_geometry is not None:
            if local_features is None or input_shape is None:
                raise ValueError('Local spline geometry requires CNN features and input shape')
            if self.family != 'bspline':
                raise ValueError('Local geometry diagnostic requires B-splines')
            base_curves = self.basis @ controls
            correction = self.local_geometry(local_features, base_curves, endpoints, metric, input_shape)
            internal = controls[:, 1:-1] + correction @ torch.linalg.inv(metric).T
            controls = torch.cat((endpoints[:, :1], internal, endpoints[:, 1:]), dim=1)
        if self.family == 'catmull_rom':
            from .catmull_rom import catmull_rom
            t = torch.linspace(0, 1, self.basis.shape[0], device=controls.device, dtype=controls.dtype)
            curves = catmull_rom(controls @ metric.T, t) @ torch.linalg.inv(metric).T
            curves = torch.cat((endpoints[:, :1], curves[:, 1:-1], endpoints[:, 1:]), dim=1)
            return controls, curves
        return controls, self.basis @ controls
