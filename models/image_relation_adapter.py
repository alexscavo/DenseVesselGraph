"""Symmetric zero-initialized relation residuals from frozen tokens/CNN maps."""
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ['token_geometry', 'endpoints', 'chord', 'corridor']


def sample_early(features, points, input_shape=(64, 64, 64)):
    """Normalized DHW -> stride-2 convolution centres -> grid_sample WHD."""
    shape = points.new_tensor(input_shape)
    size = points.new_tensor(features.shape[-3:])
    grid = 2*(points*shape/2+.5)/size-1
    original = points.shape[:-1]
    grid = grid[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(features[None], grid, align_corners=False, padding_mode='border')
    return sampled[0, :, :, 0, 0].T.reshape(*original, features.shape[0])


def connection_points(nodes, pairs, metric, radius_mm=4., samples=16):
    """Canonical GT-independent 16 x 3 x 3 physical corridor, including chord."""
    from .spline import canonical_reverse
    endpoints = nodes[pairs] @ metric.T
    reverse = canonical_reverse(endpoints)
    endpoints = endpoints.clone(); endpoints[reverse] = endpoints[reverse][:, [1, 0]]
    direction = endpoints[:, 1]-endpoints[:, 0]
    length = direction.norm(dim=-1, keepdim=True)
    fallback = direction.new_tensor([1., 0., 0.]).expand_as(direction)
    direction = torch.where(length > 1e-8, direction/length.clamp_min(1e-8), fallback)
    axis = F.one_hot(direction.abs().argmin(-1), 3).to(direction)
    u = torch.linalg.cross(direction, axis); u = F.normalize(u, dim=-1)
    v = torch.linalg.cross(direction, u)
    offsets = torch.cartesian_prod(*[direction.new_tensor([-1., 0., 1.])]*2)*radius_mm
    t = torch.linspace(0, 1, samples, device=nodes.device, dtype=nodes.dtype)
    centres = endpoints[:, :1]*(1-t[None, :, None])+endpoints[:, 1:]*t[None, :, None]
    transverse = offsets[None, :, :1]*u[:, None]+offsets[None, :, 1:]*v[:, None]
    return (centres[:, :, None]+transverse[:, None]) @ torch.linalg.inv(metric).T


class ImageRelationAdapter(nn.Module):
    def __init__(self, token_width=552, relation_tokens=2, arm='token_geometry', channels=64, projection=8):
        super().__init__()
        assert arm in ARMS
        self.arm, self.projection = arm, projection
        self.image_projection = nn.Sequential(nn.Linear(channels, projection), nn.GELU())
        self.image_width = 2*27*projection+16*9*projection
        self.network = nn.Sequential(nn.Linear(token_width*(2+relation_tokens)+9+self.image_width, 128),
                                     nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('endpoint_offsets', torch.cartesian_prod(*[torch.tensor([-4., 0., 4.])]*3), persistent=False)

    def forward(self, tokens, nodes, confidence, metric, features, pairs, shape=(64, 64, 64)):
        # Inputs are frozen by construction; detach also prevents accidental upstream gradients.
        tokens, nodes, confidence, metric = [x.detach() for x in [tokens, nodes, confidence, metric]]
        a, b = pairs.T
        token = torch.cat([(tokens[a]+tokens[b])/2, (tokens[a]-tokens[b]).abs(),
                           tokens[256:].flatten()[None].expand(len(pairs), -1)], -1)
        ends = nodes[pairs]; physical = ends @ metric.T
        geometry = torch.cat([ends.mean(1), (ends[:, 0]-ends[:, 1]).abs(),
                              (physical[:, 0]-physical[:, 1]).norm(dim=-1, keepdim=True),
                              confidence[pairs].amin(1)[:, None], confidence[pairs].amax(1)[:, None]], -1)
        image = nodes.new_zeros((len(pairs), self.image_width))
        if self.arm != 'token_geometry':
            points = nodes[:, None]+self.endpoint_offsets[None]/nodes.new_tensor(shape)
            endpoint = self.image_projection(sample_early(features.detach(), points, shape)).flatten(1)
            endpoint = torch.cat([(endpoint[a]+endpoint[b])/2, (endpoint[a]-endpoint[b]).abs()], -1)
            context = nodes.new_zeros((len(pairs), 16, 9, self.projection))
            if self.arm in ('chord', 'corridor'):
                points = connection_points(nodes, pairs, metric)
                if self.arm == 'chord':
                    centre = self.image_projection(sample_early(features.detach(), points[:, :, 4], shape))
                    context[:, :, 4] = centre
                else:
                    context = self.image_projection(sample_early(features.detach(), points, shape))
            image = torch.cat([endpoint, context.flatten(1)], -1)
        return self.network(torch.cat([token, geometry, image], -1)).flatten()


def residual_logits(original, residual):
    """Add residual to the binary log-odds, preserving the original at zero."""
    return original+torch.stack([-residual/2, residual/2], -1)
