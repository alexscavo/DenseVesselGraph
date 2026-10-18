"""Optional local image evidence for node coordinates/confidence."""
import torch
from torch import nn
from torch.nn import functional as F

class LocalNodeRefinement(nn.Module):
    def __init__(self, token_width, channels=64, radius_vox=4., image_mode='full'):
        super().__init__()
        if image_mode not in ('full','zero'): raise ValueError(image_mode)
        self.image_mode=image_mode;self.radius_vox=float(radius_vox)
        offsets=torch.cartesian_prod(*[torch.tensor([-1.,0.,1.])]*3)
        self.register_buffer('offsets',offsets,persistent=False)
        self.network=nn.Sequential(nn.Linear(token_width+channels*27+3,128),nn.ReLU(),nn.Linear(128,5))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self, tokens, nodes, logits, features, input_shape):
        shape=nodes.new_tensor(input_shape)
        # Early 7x7 stride-2 convolution has feature centres at input indices 2*i.
        # Grid sample is W,H,D; normalized node positions are input index / shape.
        centres=nodes[...,:3].detach()*shape
        points=centres[:,:,None,:]+self.offsets[None,None,:,:]*self.radius_vox
        feature_shape=shape.new_tensor(features.shape[2:])
        grid=2*(points/2+.5)/feature_shape-1
        grid=grid[...,[2,1,0]].unsqueeze(3)
        local=F.grid_sample(features.detach(),grid,padding_mode='border',align_corners=False)
        local=local.squeeze(-1).permute(0,2,3,1).flatten(2)
        if self.image_mode=='zero':local=torch.zeros_like(local)
        delta=self.network(torch.cat([tokens,local,nodes[...,:3].detach()],dim=-1))
        xyz=(nodes[...,:3]+self.radius_vox*delta[...,:3].tanh()/shape).clamp(0,1)
        return torch.cat([xyz,nodes[...,3:]],dim=-1),logits+delta[...,3:]
