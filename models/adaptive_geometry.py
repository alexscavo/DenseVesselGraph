"""Fixed-basis piecewise curves and a shared-trunk, K-specific geometry bank.

Selectors are optimized separately on detached features. Internal segment joins
are spline parameters, never object queries. Continuity follows knot multiplicity.
"""
import numpy as np
import torch
from torch import nn
from scipy.interpolate import BSpline
from .spline import SplineHead

CANDIDATES=(4,8,12,16,24)


def piecewise_knots(segments=4,continuity='C1'):
    if segments<1 or continuity not in ('C0','C1'):raise ValueError('Invalid segmentation')
    multiplicity=3 if continuity=='C0' else 2
    interior=[]
    for s in range(segments):
        interior.extend((s+np.arange(1,5)/5)/segments)
        if s+1<segments:interior.extend([(s+1)/segments]*multiplicity)
    return np.r_[np.zeros(4),interior,np.ones(4)]


def uniform_knots(k):return np.r_[np.zeros(4),np.arange(1,k-3)/(k-3),np.ones(4)]


def fit_basis(path,knots,samples=64):
    from data.spline_targets import resample
    k=len(knots)-4;guide=resample(np.asarray(path,float),257)
    basis=BSpline(knots,np.eye(k),3)(np.linspace(0,1,257))
    residual=guide-basis[:,:1]*guide[0]-basis[:,-1:]*guide[-1]
    controls=np.vstack([guide[0],np.linalg.lstsq(basis[:,1:-1],residual,rcond=None)[0],guide[-1]])
    return controls,BSpline(knots,controls,3)(np.linspace(0,1,samples))


class PiecewiseSplineHead(SplineHead):
    def __init__(self,pair_width,hidden_width,segments=4,continuity='C1',samples=64,**kwargs):
        knots=piecewise_knots(segments,continuity);k=len(knots)-4
        super().__init__(pair_width,hidden_width,k,samples,**kwargs)
        self.segments=segments;self.continuity=continuity
        self.register_buffer('knots',torch.tensor(knots,dtype=torch.float64))
        self.basis=torch.tensor(BSpline(knots,np.eye(k),3)(np.linspace(0,1,samples)),dtype=torch.float32)
        self.greville=torch.tensor([knots[i+1:i+4].mean() for i in range(k)],dtype=torch.float32)


class AdaptiveSplineBank(nn.Module):
    def __init__(self,pair_width,hidden_width,candidates=CANDIDATES,**kwargs):
        super().__init__();self.candidates=tuple(candidates)
        self.heads=nn.ModuleDict({str(k):SplineHead(pair_width,hidden_width,k,**kwargs) for k in candidates})
        shared=self.heads[str(candidates[0])]
        for head in list(self.heads.values())[1:]:
            for i in range(4):head.network[i]=shared.network[i]
            if shared.local_geometry is not None:
                head.local_geometry.sequence=shared.local_geometry.sequence
                for i in range(2):head.local_geometry.scorer[i]=shared.local_geometry.scorer[i]
        self.selector=nn.Sequential(nn.Linear(hidden_width,64),nn.ReLU(),nn.Linear(64,len(candidates)))
        self.selector.requires_grad_(False)

    curve_evidence=None

    @property
    def local_geometry(self):return self.heads[str(self.candidates[0])].local_geometry

    def features(self,pair_features,endpoints,metric):
        head=self.heads[str(self.candidates[0])];physical=endpoints@metric.T;chord=physical[:,1]-physical[:,0]
        features=torch.cat([pair_features,physical.flatten(1),chord,chord.norm(dim=-1,keepdim=True)],-1)
        return head.network[:4](features)

    def choose(self,pair_features,endpoints,metric):
        return self.selector(self.features(pair_features,endpoints,metric).detach()).argmax(-1)

    def forward_k(self,k,*args,**kwargs):return self.heads[str(k)](*args,**kwargs)


def simplify_piecewise_curve(curve,continuity='C1',tolerance=.5,maximum_segments=4):
    """Inference-only simplification preserving the chosen continuity family.

    Select fewer native segments by refitting the *predicted* curve. The check is
    numerical (0.1-mm sampling plus vertices), not an anatomical-error certificate.
    Fallback is the original predicted polyline, preserving all of its geometry.
    """
    from metrics.representation_geometry import point_to_polyline_distance,sample_polyline_by_arclength
    curve=np.asarray(curve,float)
    for segments in range(1,maximum_segments):
        knots=piecewise_knots(segments,continuity);controls,_=fit_basis(curve,knots)
        candidate=BSpline(knots,controls,3)(np.linspace(0,1,257))
        forward=np.vstack([sample_polyline_by_arclength(candidate,.1),candidate]);backward=np.vstack([sample_polyline_by_arclength(curve,.1),curve])
        error=max(point_to_polyline_distance(forward,curve).max(),point_to_polyline_distance(backward,candidate).max())
        if error+.05<=tolerance:return candidate,segments,len(controls)
    return curve,maximum_segments,len(piecewise_knots(maximum_segments,continuity))-4
