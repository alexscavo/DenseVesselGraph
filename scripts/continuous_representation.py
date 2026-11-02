"""Endpoint-constrained representation alternatives; all geometry is physical mm.

No detector or joint loss changes. Splitting is a bounded greedy algorithm with
adjacent merging, not a claim of globally minimal segmentation.
"""
import numpy as np
from data.spline_targets import fit_spline, resample
from scripts.analyze_spline_target_capacity import measure, spline_polyline
from metrics.representation_geometry import point_to_polyline_distance

KS=(4,8,12,16,24)
TOLERANCES=(.25,.5,1.)


def fit_options(path, cached8=None):
    return {k: (cached8 if k==8 and cached8 is not None else fit_spline(path,k)[0]) for k in KS}


def choose(options, reference, tolerance, measurements=None):
    """Smallest *independently refitted* K with certified numerical upper bound.

    If no option passes, return the least-error fit and mark failure, never
    silently report the maximum available K as meeting the tolerance.
    """
    scores=measurements or {k:measure(c,reference) for k,c in options.items()}
    valid=[k for k in sorted(options) if scores[k]['hausdorff_upper_mm']<=tolerance]
    k=valid[0] if valid else min(scores,key=lambda k:scores[k]['hausdorff_upper_mm'])
    return k,bool(valid),scores[k]


def split_path(path, tolerance, minimum_arc=2., maximum_segments=32):
    """Error-driven splits at original vertices, preserving endpoint connectivity.

    Split at largest dense-to-spline discrepancy within the minimum-arc limits;
    if the error is spline overshoot, bisect arc length. Merge adjacent pieces
    whenever the merged 8-control fit still passes. Every accepted piece has a
    conservative continuous symmetric error bound <= tolerance.
    """
    path=np.asarray(path,float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    cache={}
    def fit(a,b):
        key=(a,b)
        if key not in cache:
            c=fit_spline(path[a:b+1],8)[0];cache[key]=(c,measure(c,path[a:b+1]))
        return cache[key]
    segments=[(0,len(path)-1)]
    while len(segments)<maximum_segments:
        candidates=[]
        for i,(a,b) in enumerate(segments):
            c,m=fit(a,b)
            valid=np.arange(a+1,b)
            valid=valid[(arc[valid]-arc[a]>=minimum_arc)&(arc[b]-arc[valid]>=minimum_arc)]
            if m['hausdorff_upper_mm']>tolerance and len(valid):
                d=point_to_polyline_distance(path[valid],spline_polyline(c))
                j=int(valid[np.argmax(d)]) if d.max()>.05 else int(valid[np.argmin(abs(arc[valid]-(arc[a]+arc[b])/2))])
                candidates.append((m['hausdorff_upper_mm'],i,j))
        if not candidates:break
        _,i,j=max(candidates);a,b=segments[i];segments[i:i+1]=[(a,j),(j,b)]
    changed=True
    while changed:
        changed=False
        for i in range(len(segments)-1):
            a,b=segments[i][0],segments[i+1][1]
            if fit(a,b)[1]['hausdorff_upper_mm']<=tolerance:
                segments[i:i+2]=[(a,b)];changed=True;break
    controls=[fit(a,b)[0] for a,b in segments]
    scores=[fit(a,b)[1] for a,b in segments]
    assert all(np.array_equal(controls[i][-1],controls[i+1][0]) for i in range(len(controls)-1))
    return dict(controls=controls,intervals=segments,scores=scores,
                passed=all(m['hausdorff_upper_mm']<=tolerance for m in scores),fits=len(cache))


def compressed_curve(controls,tolerance=.5,candidates=KS):
    """Inference-only compression: consumes a predicted curve, never GT."""
    reference=spline_polyline(controls,tolerance=.005)
    options={k:fit_spline(reference,k)[0] for k in candidates if k<len(controls)}
    # Identity is exact, so it remains a safe fallback even at very small tolerances.
    options[len(controls)]=np.asarray(controls)
    scores={k:measure(c,reference) for k,c in options.items()}
    valid=[k for k in sorted(options) if scores[k]['hausdorff_upper_mm']+.005<=tolerance]
    k=valid[0] if valid else len(controls)
    return options[k],k


def curve_features(controls):
    """GT-independent, reversal-invariant predicted-curve features for B2 probe."""
    path=resample(spline_polyline(controls),32)
    def features(p):
        chord=p[-1]-p[0];length=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
        residual=p-((1-np.linspace(0,1,32)[:,None])*p[0]+np.linspace(0,1,32)[:,None]*p[-1])
        return np.r_[residual.ravel(),chord,length]
    return (features(path)+features(path[::-1]))/2
