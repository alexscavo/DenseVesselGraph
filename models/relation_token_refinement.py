"""Small relation-only residuals; the caller keeps original tokens for other heads."""
import torch
from torch import nn
from torch.nn import functional as F


def sample_local_cnn(features, nodes, input_shape, radius_vox=4.):
    """The established layer0/relu1 3x3x3 sampler, centred on refined coordinates."""
    offsets=torch.cartesian_prod(*[nodes.new_tensor([-1.,0.,1.])]*3)
    shape=nodes.new_tensor(input_shape)
    points=nodes[...,:3].detach()[:,:,None,:]*shape+offsets[None,None]*radius_vox
    feature_shape=shape.new_tensor(features.shape[2:])
    grid=(2*(points/2+.5)/feature_shape-1)[...,[2,1,0]].unsqueeze(3)
    sampled=F.grid_sample(features,grid,padding_mode='border',align_corners=False)
    return sampled.squeeze(-1).permute(0,2,3,1).flatten(2)


class RelationTokenResidual(nn.Module):
    """Return z_rel without mutating z, coordinates, logits or shared relation tokens.

    R1 supplies a fixed repetition of z to the context projection. Thus both arms
    have exactly the same, active trainable layers, while only R2 sees image data.
    The repetition adds no information beyond z; its rank limit is documented.
    """
    def __init__(self, image_conditioned, width=552, channels=64, projection=64):
        super().__init__()
        self.image_conditioned=bool(image_conditioned);self.context_width=channels*27
        self.register_buffer('token_context_indices',torch.arange(self.context_width)%width,persistent=False)
        self.token_projection=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,projection),nn.ReLU())
        self.context_projection=nn.Sequential(nn.LayerNorm(self.context_width),nn.Linear(self.context_width,projection),nn.ReLU())
        self.output=nn.Linear(2*projection,width)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,z,local):
        context=local if self.image_conditioned else z[...,self.token_context_indices]
        return z+self.output(torch.cat([self.token_projection(z),self.context_projection(context)],-1))
