"""2D dense-path extraction, topology-preserving contraction, crop and K8 fit."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import skeletonize
from metrics.branch_connectivity import contract_degree_two


@dataclass
class Graph2D:
    nodes: np.ndarray
    edges: np.ndarray
    paths: list
    provenance: dict


def _simple_graph(branches,provenance,isolates=()):
    """Encode closed/parallel curves with extra degree-2 anchors, never delete them."""
    nodes=[];edges=[];paths=[];anchors={};used=set()
    def node(key,xy):
        if key not in anchors:anchors[key]=len(nodes);nodes.append(np.asarray(xy,float))
        return anchors[key]
    for start,end,path in branches:
        path=np.asarray(path,float)
        a=node(start,path[0]);b=node(end,path[-1])
        if a==b:
            # Simple adjacency cannot store self-loops. Preserve ring geometry.
            if len(path)<4:raise ValueError('Closed curve has insufficient dense points')
            cuts=[0,max(1,len(path)//3),max(2,2*len(path)//3),len(path)-1]
            ids=[a,node(('cycle',len(paths),1),path[cuts[1]]),node(('cycle',len(paths),2),path[cuts[2]]),a]
            for j in range(3):
                edges.append((ids[j],ids[j+1]));paths.append(path[cuts[j]:cuts[j+1]+1])
        elif tuple(sorted((a,b))) in used:
            k=max(1,min(len(path)-2,len(path)//2))
            if len(path)<3: path=np.stack((path[0],(path[0]+path[-1])/2,path[-1]));k=1
            middle=node(('parallel',len(paths)),path[k])
            edges.extend(((a,middle),(middle,b)));paths.extend((path[:k+1],path[k:]))
        else:edges.append((a,b));paths.append(path)
        used.update(tuple(sorted(e)) for e in edges[-3:])
    for key,xy in isolates:node(key,xy)
    return Graph2D(np.asarray(nodes,float).reshape(-1,2),np.asarray(edges,int).reshape(-1,2),paths,provenance)


def from_dense_graph(nodes,edges,*,provenance):
    nodes=np.asarray(nodes,float).reshape(-1,2);edges=np.asarray(edges,int).reshape(-1,2)
    graph=contract_degree_two(nodes,edges)
    branches=[(b.start,b.end,b.polyline) for b in graph.branches]
    isolates=[(i,nodes[i]) for i,role in graph.roles.items() if role=='isolated']
    return _simple_graph(branches,provenance,isolates)


def from_mask(mask,*,provenance,is_centerline=False):
    """8-connected skeleton; avoid diagonal corner shortcuts with an orthogonal route.

    No spur removal, component filtering or crossing reconnection is performed.
    Four-way crossings remain annotation-ambiguous and are explicitly flagged.
    """
    if np.asarray(mask).ndim!=2:raise ValueError('Expected a single binary mask plane')
    sk=np.asarray(mask,bool).copy() if is_centerline else skeletonize(np.asarray(mask,bool));rc=np.argwhere(sk)
    lookup={tuple(p):i for i,p in enumerate(rc)};edges=[]
    for i,(y,x) in enumerate(rc):
        for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
            q=(y+dy,x+dx)
            if q not in lookup:continue
            if dy and dx and ((y+dy,x) in lookup or (y,x+dx) in lookup):continue
            edges.append((i,lookup[q]))
    edges=np.asarray(edges,int).reshape(-1,2)
    degree=np.bincount(edges.ravel(),minlength=len(rc))
    junction=np.where(degree>=3)[0];junction_set=set(junction.tolist())
    parents={int(i):int(i) for i in junction}
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    for a,b in edges.tolist():
        if a in junction_set and b in junction_set:parents[root(a)]=root(b)
    groups={}
    for i in junction:groups.setdefault(root(int(i)),[]).append(int(i))
    provenance={**provenance,'graph_annotation':'derived_from_provided_centerline' if is_centerline else 'derived_from_mask_skeleton',
        'junction_pixels':int((degree>=3).sum()),'ambiguous_degree4plus_pixels':int((degree>=4).sum()),
        'adjacent_junction_clusters':sum(len(v)>1 for v in groups.values()),
        'junction_policy':'preserve adjacent junction identities; flag clusters for QC, no implicit merge',
        'isolated_skeleton_pixels':int((degree==0).sum()),'skeleton_pixels':len(rc),
        'requires_crossing_review':bool((degree>=4).any() or any(len(v)>1 for v in groups.values()))}
    return from_dense_graph(rc[:,[1,0]],edges,provenance=provenance)


def clip_segment(a,b,low,high):
    d=b-a;t0,t1=0.,1.
    for axis in range(2):
        if abs(d[axis])<1e-12:
            if a[axis]<low[axis] or a[axis]>high[axis]:return None
            continue
        t=(np.array([low[axis],high[axis]])-a[axis])/d[axis]
        t0=max(t0,float(t.min()));t1=min(t1,float(t.max()))
        if t1<t0:return None
    if t1-t0<=1e-12:return None
    return a+t0*d,a+t1*d


def crop_graph(graph,start_xy,shape_hw):
    """Clip every dense segment, including outside/outside and re-entering paths.

    Boundaries are pixel-center extents [0,W-1]×[0,H-1]. Crossing paths never
    merge by spatial proximity: source anchor identities remain authoritative.
    """
    start=np.asarray(start_xy,float);high=start+np.asarray(shape_hw)[::-1]-1
    branches=[]
    for edge_id,((a,b),path) in enumerate(zip(graph.edges,graph.paths)):
        fragments=[];current=[]
        for p,q in zip(path[:-1],path[1:]):
            piece=clip_segment(p,q,start,high)
            if piece is None:
                if current:fragments.append(np.asarray(current));current=[]
                continue
            left,right=piece
            if current and not np.allclose(current[-1],left,atol=1e-8,rtol=0):
                fragments.append(np.asarray(current));current=[]
            if not current:current.append(left)
            current.append(right)
        if current:fragments.append(np.asarray(current))
        for i,path2 in enumerate(fragments):
            ka=('source',int(a)) if np.allclose(path2[0],graph.nodes[a],atol=1e-8,rtol=0) else ('boundary',edge_id,i,0)
            kb=('source',int(b)) if np.allclose(path2[-1],graph.nodes[b],atol=1e-8,rtol=0) else ('boundary',edge_id,i,1)
            branches.append((ka,kb,path2-start))
    degrees=np.bincount(graph.edges.ravel(),minlength=len(graph.nodes))
    isolates=[(('source',i),p-start) for i,p in enumerate(graph.nodes) if degrees[i]==0 and np.all(p>=start) and np.all(p<=high)]
    return _simple_graph(branches,{**graph.provenance,'crop_start_xy':start.tolist(),
        'image_shape_hw':list(shape_hw),'boundary_convention':'pixel_centers_0_to_size_minus_1'},isolates)


def fit_controls(path,k=8):
    """Endpoint-constrained cubic least squares, parameterized by physical arc length."""
    path=np.asarray(path,float);lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
    keep=np.r_[True,lengths>1e-10];path=path[keep]
    if len(path)<2:raise ValueError('Cannot fit a zero-length branch')
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    knots=np.r_[np.zeros(4),np.arange(1,k-3)/(k-3),np.ones(4)]
    # Fit uniformly along the SAME piecewise-linear guide. Irregular source
    # vertices can leave a whole B-spline knot span unsampled even when len>=k;
    # unconstrained controls then explode. Uniform samples make this a stable
    # arc-length-weighted approximation without changing the reference graph.
    t=np.linspace(0,1,max(64,8*k))
    path=np.stack([np.interp(t*arc[-1],arc,path[:,a]) for a in range(2)],1)
    basis=BSpline.design_matrix(t,knots,3).toarray()
    endpoints=path[[0,-1]]
    residual=path-basis[:,[0]]*endpoints[0]-basis[:,[-1]]*endpoints[1]
    interior=np.linalg.lstsq(basis[:,1:-1],residual,rcond=None)[0]
    return np.concatenate((endpoints[:1],interior,endpoints[1:]))


def save_target(path,graph,shape_hw,*,spacing_xy=None,units=None):
    size=np.asarray(shape_hw)[::-1]
    if spacing_xy is None:
        if units not in (None,'pixel'):raise ValueError('Physical units require measured spacing')
        spacing_xy=[1.,1.];units='pixel'
    elif units not in ('mm','micron','pixel'):raise ValueError('Explicit spatial units required')
    spacing=np.asarray(spacing_xy,float)
    if units=='micron':spacing=spacing/1000.;units='mm'
    metric=np.diag(size*spacing)
    controls=np.stack([fit_controls(p*spacing) / spacing / size for p in graph.paths]) if graph.paths else np.empty((0,8,2))
    paths=np.concatenate(graph.paths)/size if graph.paths else np.empty((0,2))
    offsets=np.r_[0,np.cumsum([len(p) for p in graph.paths])]
    np.savez_compressed(path,nodes=graph.nodes/size,edges=graph.edges,controls=controls,
        metric=metric,full_paths=paths,path_offsets=offsets,shape_hw=shape_hw,units=np.array(units))
