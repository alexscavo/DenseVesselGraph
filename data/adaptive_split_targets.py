"""Read-only explicit subdivision of existing dense-path caches.

Original branch IDs and inverse subedge-count weights are retained for future
branch-normalized geometry supervision. This does not rerun target contraction.
"""
import numpy as np
from scripts.continuous_representation import split_path
from data.spline_targets import resample


def split_target(cache,tolerance=.5):
    metric=np.asarray(cache['metric'],float);inverse=np.linalg.inv(metric);nodes=list(np.asarray(cache['nodes'],float));boundary=list(cache.get('boundary',np.zeros(len(nodes),bool)))
    edges=[];controls=[];curves=[];guides=[];paths=[];offsets=[0];parent=[];weights=[];unresolved=[]
    from scripts.analyze_spline_target_capacity import spline_polyline
    for j,(edge,a,b) in enumerate(zip(cache['edges'],cache['path_offsets'][:-1],cache['path_offsets'][1:])):
        path=np.asarray(cache['full_paths'][a:b],float)@metric.T;pieces=split_path(path,tolerance)
        if not pieces['passed']:unresolved.append(j)
        left=int(edge[0]);n=len(pieces['controls'])
        for i,(c,(first,last)) in enumerate(zip(pieces['controls'],pieces['intervals'])):
            if i==n-1:right=int(edge[1])
            else:right=len(nodes);nodes.append(c[-1]@inverse.T);boundary.append(False)
            edges.append((left,right));controls.append(c@inverse.T);from scipy.interpolate import BSpline
            from models.adaptive_geometry import uniform_knots
            curves.append(BSpline(uniform_knots(8),c,3)(np.linspace(0,1,64))@inverse.T)
            original=path[first:last+1]@inverse.T;paths.append(original);guides.append(resample(original,64));offsets.append(offsets[-1]+len(original));parent.append(j);weights.append(1/n);left=right
    return dict(nodes=np.asarray(nodes,np.float32),edges=np.asarray(edges,np.int64).reshape(-1,2),controls=np.asarray(controls,np.float32).reshape(-1,8,3),curves=np.asarray(curves,np.float32).reshape(-1,64,3),guides=np.asarray(guides,np.float32).reshape(-1,64,3),metric=np.asarray(metric,np.float32),boundary=np.array(boundary,bool),full_paths=np.concatenate(paths) if paths else np.empty((0,3)),path_offsets=np.array(offsets),parent_branch=np.array(parent),branch_weights=np.array(weights,np.float32),unresolved_branches=np.array(unresolved,int))
