"""Differentiable centripetal Catmull–Rom in physical coordinates."""
import torch


def catmull_rom(points, parameters):
    """Evaluate E,K,3 interpolation points at S (or E,S) parameters in [0,1].

    Reflected endpoint ghosts introduce no learned parameters. Knot intervals
    depend on physical distances; a 1e-8 mm floor handles coincident points.
    Interval selection is discrete; interpolation and knots retain gradients.
    """
    extended = torch.cat((2*points[:, :1]-points[:, 1:2], points,
                          2*points[:, -1:]-points[:, -2:-1]), dim=1)
    intervals = torch.linalg.vector_norm(extended[:, 1:]-extended[:, :-1], dim=-1).clamp_min(1e-8).sqrt()
    knots = torch.cat((torch.zeros_like(intervals[:, :1]), intervals[:, 1:-1].cumsum(-1)), -1)
    knots = knots / knots[:, -1:]
    t = parameters.expand(len(points), -1).contiguous()
    segment = (torch.searchsorted(knots.detach().contiguous(), t.detach(), right=True)-1).clamp(0, points.shape[1]-2)
    batch = torch.arange(len(points), device=points.device)[:, None]
    p0, p1, p2, p3 = (extended[batch, segment+i] for i in range(4))
    h0, h1, h2 = (intervals[batch, segment+i, None] for i in range(3))
    m1 = h1*((p1-p0)/h0-(p2-p0)/(h0+h1)+(p2-p1)/h1)
    m2 = h1*((p2-p1)/h1-(p3-p1)/(h1+h2)+(p3-p2)/h2)
    u = ((t-knots[batch, segment])/(knots[batch, segment+1]-knots[batch, segment]).clamp_min(torch.finfo(points.dtype).eps))[..., None]
    return ((2*u**3-3*u**2+1)*p1+(u**3-2*u**2+u)*m1
            +(-2*u**3+3*u**2)*p2+(u**3-u**2)*m2)
