"""2D equivalents of the existing local A-xyz and cubic spline modules.

Nodes/controls use normalized XY=(column/width,row/height). Metric maps XY to
physical coordinates; unknown spacing is explicitly measured in pixels.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.spline import cubic_basis


def sample_local(features, points, *, stride=2.):
    """Input pixel centers XY -> early-CNN feature centers -> grid_sample."""
    size = points.new_tensor([features.shape[-1], features.shape[-2]])
    grid = 2*(points/stride+.5)/size-1
    return F.grid_sample(features, grid, padding_mode='border', align_corners=False)


class LocalCoordinateRefinement2D(nn.Module):
    def __init__(self, width, channels=64, radius_pixels=4., image_mode='full'):
        super().__init__()
        self.radius = float(radius_pixels)
        self.image_mode = image_mode
        self.register_buffer('offsets', torch.cartesian_prod(torch.tensor([-1.,0.,1.]),
            torch.tensor([-1.,0.,1.])), persistent=False)
        self.network=nn.Sequential(nn.Linear(width+channels*9+2,128),nn.ReLU(),nn.Linear(128,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self, tokens, nodes, features, input_shape):
        size=nodes.new_tensor([input_shape[-1],input_shape[-2]])
        points=nodes[...,:2].detach()[:,:,None]*size+self.offsets[None,None]*self.radius
        local=sample_local(features.detach(),points).permute(0,2,3,1).flatten(2)
        if self.image_mode=='zero': local=torch.zeros_like(local)
        delta=self.network(torch.cat((tokens,local,nodes[...,:2].detach()),-1))
        xy=(nodes[...,:2]+self.radius*delta.tanh()/size).clamp(0,1)
        return torch.cat((xy,nodes[...,2:]),-1)


class SplineHead2D(nn.Module):
    """Same endpoint-constrained physical chord residual objective as 3D."""
    curve_evidence=None
    local_geometry=None

    def __init__(self,pair_width,hidden_width,controls=8,samples=64):
        super().__init__()
        self.controls=int(controls)
        basis,greville=cubic_basis(controls,samples)
        self.register_buffer('basis',basis);self.register_buffer('greville',greville)
        self.network=nn.Sequential(nn.Linear(pair_width+7,hidden_width),nn.ReLU(),
            nn.Linear(hidden_width,hidden_width),nn.ReLU(),nn.Linear(hidden_width,2*(controls-2)))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,pair_features,endpoints,metric):
        physical=endpoints@metric.T
        chord=physical[:,1]-physical[:,0]
        residual=self.network(torch.cat((pair_features,physical.flatten(1),chord,
            chord.norm(dim=-1,keepdim=True)),-1)).reshape(-1,self.controls-2,2)
        t=self.greville[None,1:-1,None]
        interior=(1-t)*physical[:,:1]+t*physical[:,1:]+residual
        interior=interior@torch.linalg.inv(metric).T
        controls=torch.cat((endpoints[:,:1],interior,endpoints[:,1:]),1)
        return controls,self.basis@controls
