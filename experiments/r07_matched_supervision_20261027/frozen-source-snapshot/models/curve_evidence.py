"""Differentiable MRI evidence along predicted candidate vessel curves."""
import torch
from torch import nn
from torch.nn import functional as F


def sample_along_curves(features, curves):
    """C,D,H,W features; E,S,3 curves use voxel-index/axis-size DHW."""
    shape = curves.new_tensor(features.shape[-3:])
    # align_corners=False maps voxel centers to (2*i+1)/size-1.
    grid = (2 * curves + 1 / shape - 1)[..., [2, 1, 0]]
    sampled = F.grid_sample(features[None], grid[None, :, :, None],
                            mode='bilinear', padding_mode='zeros', align_corners=False)
    return sampled[0, :, :, :, 0].permute(1, 0, 2)


class CurveEvidence(nn.Module):
    """Undirected edge-logit correction; initialization preserves old scores."""
    def __init__(self, channels=8, samples=32, scale=1., chunk_size=256):
        super().__init__()
        self.samples, self.scale, self.chunk_size = int(samples), float(scale), int(chunk_size)
        self.encoder = nn.Sequential(nn.Conv3d(1, channels, 3, padding=1), nn.GELU(),
                                     nn.Conv3d(channels, channels, 3, padding=1), nn.GELU())
        self.sequence = nn.Sequential(nn.Conv1d(channels + 2, 16, 3, padding=1), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(51, 32), nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.scorer[-1].weight); nn.init.zeros_(self.scorer[-1].bias)

    def encode(self, image):
        # Only MRI input: intensity and valid-support channels accompany learned features.
        return torch.cat((self.encoder(image), image, torch.ones_like(image)), dim=1)

    def forward(self, features, curves, metric):
        indices = torch.linspace(0, curves.shape[1] - 1, self.samples, device=curves.device).round().long()
        path = curves[:, indices]
        evidence = sample_along_curves(features, path)
        physical = path @ metric.T
        arc = (physical[:, 1:] - physical[:, :-1]).norm(dim=-1).sum(-1)
        chord = (physical[:, -1] - physical[:, 0]).norm(dim=-1)
        geometry = torch.stack((arc, chord, arc / chord.clamp_min(1e-4)), -1)

        def score(sequence):
            h = self.sequence(sequence)
            pooled = torch.cat((h.mean(-1), h.amax(-1), h.amin(-1), geometry), -1)
            return self.scorer(pooled)
        return self.scale * .5 * (score(evidence) + score(evidence.flip(-1)))


def curve_relation_logits(relation_head, spline_head, objects, relations, nodes, pairs, metric, features):
    """Same candidate construction/scoring for training positives/negatives and inference."""
    from .spline import canonical_reverse
    chunks = []
    for pairs in pairs.split(spline_head.curve_evidence.chunk_size):
        if not len(pairs):
            continue
        pairs = pairs.clone()
        reverse = canonical_reverse(nodes[pairs] @ metric.T)
        pairs[reverse] = pairs[reverse][:, [1, 0]]
        parts = [objects[pairs[:, 0]], objects[pairs[:, 1]]]
        if relations.numel():
            parts.append(relations.reshape(1, -1).expand(len(pairs), -1))
        forward = torch.cat(parts, -1)
        backward = torch.cat([parts[1], parts[0], *parts[2:]], -1)
        _, curves = spline_head(forward, nodes[pairs], metric)
        base = .5 * (relation_head(forward) + relation_head(backward))
        chunks.append(base + spline_head.curve_evidence(features, curves, metric))
    return torch.cat(chunks) if chunks else nodes.new_empty((0, 2))
