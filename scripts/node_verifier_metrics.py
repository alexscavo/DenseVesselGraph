"""CPU-only one-to-one proposal diagnostics and annotation-relative accounting."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score, average_precision_score


def association(x, gt, tolerance=2.):
    d = np.linalg.norm(x[:, None] - gt[None], axis=-1)
    i, j = linear_sum_assignment(np.where(d <= tolerance, d, (min(d.shape)+1)*(tolerance+1)))
    return {int(a): int(b) for a, b in zip(i, j) if d[a, b] <= tolerance}


def supervision(x, gt):
    """Reliable spatial winners, worse duplicates, far negatives; gray zone ignored.

    Never uses Hungarian training assignments. A surplus candidate within 0.5 mm
    of its representative's localization error is an ambiguous alternative.
    Candidates near multiple GT anchors are ignored. No BRAVA training uses this.
    """
    d = np.linalg.norm(x[:, None] - gt[None], axis=-1)
    y = np.full(len(x), -1., dtype=np.float32)
    w = np.zeros(len(x), dtype=np.float32)
    if not len(gt):
        return np.zeros(len(x), np.float32), np.ones(len(x), np.float32)
    mapping = association(x, gt)
    y[d.min(1) > 3] = 0; w[d.min(1) > 3] = 1
    inverse = {j: i for i, j in mapping.items()}
    for i, j in mapping.items():
        if (d[i] <= 2).sum() == 1:
            y[i], w[i] = 1, 1
    for i in range(len(x)):
        j = int(d[i].argmin())
        if i not in mapping and j in inverse and (d[i] <= 2).sum() == 1 and d[i,j] > d[inverse[j],j] + .5:
            y[i], w[i] = 0, .25
    return y, w


def rank_nms(x, scores, radius):
    order = np.argsort(-scores, kind='stable')
    keep, removed = [], []
    for i in order:
        if radius and keep and np.any(np.linalg.norm(x[keep]-x[i], axis=1) <= radius): removed.append(int(i))
        else: keep.append(int(i))
    return np.array(keep, int), np.array(removed, int)


def select(c, scores, threshold=.5, radius=0, count=None):
    # A threshold sweep changes neither ranking nor NMS. Cache this work once
    # per score vector/radius, keyed by values rather than reusable object IDs.
    cache=c.setdefault('_ranks',{})
    key=(float(radius),scores.tobytes())
    if key not in cache:cache[key]=rank_nms(c['physical'], scores, radius)
    keep, removed = cache[key]
    if count is not None:
        return np.r_[keep, removed][:count]
    return keep[scores[keep] > threshold]


def components(n, edges):
    if not n: return 0, 0
    edges=np.asarray(edges,int).reshape(-1,2)
    a=csr_matrix((np.ones(len(edges)), (edges[:,0],edges[:,1])),shape=(n,n))
    b0=connected_components(a,directed=False,return_labels=False)
    return int(b0), int(len(edges)-n+b0)


def measure(c, ids, tolerance=2., graph=True):
    ids=np.asarray(ids,int);gt=c['truth'];x=c['physical']
    local=association(x[ids],gt,tolerance);mapping={int(ids[i]):j for i,j in local.items()}
    recovered=set(mapping.values());key=('_baseline',float(tolerance))
    if key not in c:c[key]=association(x[c['scores']>.5],gt,tolerance)
    baseline=c[key]
    dist=np.linalg.norm(x[ids,None]-gt[None],axis=-1)
    duplicate=sum(i not in local and bool((dist[i]<=tolerance).any()) for i in range(len(ids)))
    out=dict(nodes=len(ids),gt_nodes=len(gt),node_tp=len(mapping),duplicates=int(duplicate),
             rescued=len(recovered-set(baseline.values())),lost=len(set(baseline.values())-recovered),patches=1)
    if graph and 'pairs' in c:
        active=np.isin(c['pairs'],ids).all(1);pairs=c['pairs'][active];scores=c['relations'][active]
        links={tuple(sorted(map(int,e))) for e in c['gt_edges']}
        labels=np.array([a in mapping and b in mapping and tuple(sorted((mapping[a],mapping[b]))) in links for a,b in pairs],bool)
        eligible=sum(a in recovered and b in recovered for a,b in links)
        chosen=pairs[scores>.5];remap={int(q):i for i,q in enumerate(ids)}
        edges=np.array([[remap[int(a)],remap[int(b)]] for a,b in chosen],int).reshape(-1,2)
        p0,p1=components(len(ids),edges);g0,g1=components(len(gt),c['gt_edges'])
        out.update(edges=len(chosen),gt_edges=len(links),edge_tp=int(labels[scores>.5].sum()),eligible=eligible,beta0_error=abs(p0-g0),beta1_error=abs(p1-g1))
    return out


def aggregate(rows):
    total={k:float(sum(r.get(k,0) for r in rows)) for k in rows[0] if isinstance(rows[0][k],(int,float,np.number))}
    div=lambda a,b:a/b if b else 0.
    total.update(node_precision=div(total['node_tp'],total['nodes']),node_recall=div(total['node_tp'],total['gt_nodes']),node_f1=div(2*total['node_tp'],total['nodes']+total['gt_nodes']),duplicate_rate=div(total['duplicates'],total['nodes']))
    if 'edge_tp' in total:
        total.update(edge_precision=div(total['edge_tp'],total['edges']),edge_recall=div(total['edge_tp'],total['gt_edges']),edge_f1=div(2*total['edge_tp'],total['edges']+total['gt_edges']),conditional_recall=div(total['edge_tp'],total['eligible']),beta0_mae=div(total['beta0_error'],total['patches']),beta1_mae=div(total['beta1_error'],total['patches']))
    return total


def discrimination(caches, scores):
    labels=[];probs=[];kinds=[];reliable=[];reliable_p=[]
    for c,p in zip(caches,scores):
        mapping=association(c['physical'],c['truth']);d=np.linalg.norm(c['physical'][:,None]-c['truth'][None],axis=-1)
        kind=np.full(len(p),2);kind[d.min(1)<=2]=1;kind[list(mapping)]=0
        kinds.extend(kind);probs.extend(p);labels.extend(kind==0)
        y,w=supervision(c['physical'],c['truth']);reliable.extend(y[w>0]);reliable_p.extend(p[w>0])
    p=np.array(probs);y=np.array(labels);kind=np.array(kinds);ry=np.array(reliable);rp=np.array(reliable_p)
    auc=lambda a,b:float(roc_auc_score(a,b)) if len(np.unique(a))==2 else None
    out=dict(auc_correct_duplicate=auc(y[kind<2],p[kind<2]),auc_correct_false=auc(y[kind!=1],p[kind!=1]),canonical_ap=float(average_precision_score(y,p)),reliable_auc=auc(ry,rp),brier=float(np.mean((rp-ry)**2)),ignored_fraction=1-len(ry)/len(p),ece=0.)
    for lo in np.linspace(0,.9,10):
        m=(rp>=lo)&(rp<lo+.1 if lo<.9 else rp<=1)
        if m.any():out['ece']+=float(m.mean()*abs(rp[m].mean()-ry[m].mean()))
    for k,name in enumerate(['correct','duplicate','false']):out[name+'_mean_confidence']=float(p[kind==k].mean()) if (kind==k).any() else None
    return out
