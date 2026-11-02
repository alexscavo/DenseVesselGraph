"""Optional local CNN evidence for B-spline geometry, independent of edge scoring."""
import torch
from torch import nn
from .curve_evidence import sample_along_curves


class LocalSplineGeometry(nn.Module):
    def __init__(self, controls, channels=64, samples=32, scale=1., image_mode='full'):
        super().__init__()
        if image_mode not in ('full', 'zero'):
            raise ValueError('Local geometry image_mode must be full or zero')
        self.samples, self.scale, self.image_mode = int(samples), float(scale), image_mode
        self.sequence = nn.Sequential(nn.Conv1d(channels+1, 16, 3, padding=1), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(16*self.samples+10, 64), nn.GELU(),
                                    nn.Linear(64, 3*(controls-2)))
        nn.init.zeros_(self.scorer[-1].weight); nn.init.zeros_(self.scorer[-1].bias)

    def sampled_features(self, feature_volume, curves, input_shape):
        # SENet's pre-pool conv has stride 2, padding 3, kernel 7: feature
        # centers correspond to raw voxel indices 0,2,4,... (no origin shift).
        indices = torch.linspace(0, curves.shape[1]-1, self.samples, device=curves.device).round().long()
        path = curves[:, indices].detach()  # No feedback through sampling coordinates.
        shape = path.new_tensor(feature_volume.shape[-3:])
        input_shape = path.new_tensor(input_shape)
        coordinates = path * (input_shape / (2*shape))
        features = feature_volume.detach()
        if self.image_mode == 'zero': features = torch.zeros_like(features)
        features = torch.cat((features, torch.ones_like(features[:1])), 0)
        return sample_along_curves(features, coordinates)

    def forward(self, feature_volume, curves, endpoints, metric, input_shape):
        sequence = self.sequence(self.sampled_features(feature_volume, curves, input_shape))
        physical = endpoints @ metric.T
        chord = physical[:, 1]-physical[:, 0]
        geometry = torch.cat((physical.flatten(1), chord, chord.norm(dim=-1, keepdim=True)), -1)
        return self.scale*self.scorer(torch.cat((sequence.flatten(1), geometry), -1)).reshape(len(curves), -1, 3)
