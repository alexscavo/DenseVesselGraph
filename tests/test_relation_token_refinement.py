import torch
from models.relation_token_refinement import RelationTokenResidual,sample_local_cnn


def test_identity_and_parameter_matched_active_control():
    z=torch.randn(2,5,552);local=torch.randn(2,5,1728)
    a=RelationTokenResidual(False);b=RelationTokenResidual(True)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    torch.testing.assert_close(a(z,local),z,rtol=0,atol=0)
    torch.testing.assert_close(b(z,local),z,rtol=0,atol=0)
    with torch.no_grad():
        a.output.weight.normal_(std=.01);b.load_state_dict(a.state_dict())
    torch.testing.assert_close(a(z,local),a(z,local+3),rtol=0,atol=0)
    assert not torch.equal(b(z,local),b(z,local.roll(1,1)))
    a(z,local).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in a.parameters())


def test_local_sampler_uses_established_feature_centres():
    d,h,w=torch.meshgrid(torch.arange(32.),torch.arange(32.),torch.arange(32.),indexing='ij')
    features=(2*d+6*h+10*w)[None,None]
    nodes=torch.tensor([[[.5,.5,.5]]])
    actual=sample_local_cnn(features,nodes,(64,64,64),radius_vox=4.)
    offsets=torch.cartesian_prod(*[torch.tensor([-1.,0.,1.])]*3)
    expected=((32+offsets*4)*torch.tensor([1.,3.,5.])).sum(-1)
    torch.testing.assert_close(actual[0,0],expected,atol=2e-4,rtol=0)
