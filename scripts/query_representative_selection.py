"""GT-free node decisions from frozen coordinates, confidence and symmetric pair scores."""
import numpy as np

def greedy_nms(x,score,radius=1.,target=None,pool=None):
    pool=np.arange(len(score)) if pool is None else np.asarray(pool,int)
    order=pool[np.argsort(-score[pool],kind='stable')];keep=[];removed=[]
    for q in order:
        if any(np.linalg.norm(x[q]-x[k])<=radius for k in keep):removed.append(int(q))
        else:keep.append(int(q))
    if target is not None:
        keep=keep[:target]
        if len(keep)<target:keep.extend([q for q in order if q not in keep][:target-len(keep)])
    return np.asarray(keep,int)

def relation_matrix(n,pairs,scores):
    a=np.zeros((n,n),float);i,j=np.asarray(pairs,int).T;a[i,j]=a[j,i]=scores
    np.fill_diagonal(a,0.);return a

def reference_context(x,p,r):
    """Fixed one-pass scaffold, no GT, no iterative self-support from replacements."""
    normal=np.flatnonzero(p>.5);seeds=greedy_nms(x,p,pool=normal)
    refs=seeds[p[seeds]>=.75][:12]
    distance=np.linalg.norm(x[:,None]-x[None],axis=-1)
    return normal,seeds,refs,distance

def relation_features(x,p,r,refs,distance):
    if len(refs)==0:return np.zeros(len(p)),np.zeros(len(p))
    eligible=distance[:,refs]>2.
    v=np.where(eligible,r[:,refs],0.)
    refv=v[refs];degree=(refv>.5).sum(1)
    # A reciprocal-rank test relative to other fixed reference representatives.
    sorted_reference=np.sort(refv,axis=0)[::-1]
    cutoff=sorted_reference[min(2,len(refs)-1)]
    reciprocal=eligible&(v>.5)&(v>=cutoff[None])
    weighted=v*p[refs][None]/np.sqrt(np.maximum(degree,1))[None]
    good=np.where(reciprocal,weighted,0.).max(1)
    bad=np.where(eligible&~reciprocal,weighted,0.).max(1)
    guarded=good-bad
    naive=v.sum(1)/3. # deliberately uncapped; can reward hubs and mutually false cliques.
    return naive,guarded

def local_representatives(x,p,r,seeds,refs,distance,weight):
    if not len(seeds):return seeds.copy()
    near=distance[:,seeds];owner=near.argmin(1);chosen=[]
    for j,q in enumerate(seeds):
        other=distance[q,seeds[seeds!=q]];radius=min(1.,.49*float(other.min())) if len(other) else 1.
        candidates=np.flatnonzero((p>=.1)&(near[:,j]<=radius)&(owner==j));assert q in candidates
        external=refs[distance[q,refs]>2.];positive=external[r[q,external]>.5];negative=external[r[q,external]<=.5]
        if len(positive):
            support=r[np.ix_(candidates,positive)].mean(1)
            penalty=np.sort(r[np.ix_(candidates,negative)],axis=1)[:,-min(2,len(negative)):].mean(1) if len(negative) else np.zeros(len(candidates))
            feature=support-penalty
        else:feature=np.zeros(len(candidates))
        score=p[candidates]+weight*feature
        best=candidates[np.lexsort((candidates,-p[candidates],-score))[0]];chosen.append(int(best))
    assert len(set(chosen))==len(seeds)
    return np.array(chosen,int)

def protected_nms(x,p,r,refs,distance):
    """A predicted short connection/different external profile vetoes suppression."""
    keep=[]
    for q in np.argsort(-p,kind='stable'):
        if p[q]<=.5:continue
        suppress=False
        for k in keep:
            if distance[q,k]>1.:continue
            external=refs[(distance[q,refs]>2.)&(distance[k,refs]>2.)]
            distinct_profile=len(external)>0 and np.mean(abs(r[q,external]-r[k,external]))>.25
            if r[q,k]<=.5 and not distinct_profile:suppress=True;break
        if not suppress:keep.append(int(q))
    return np.array(keep,int)

def select_queries(coordinates,node_scores,pairs,relation_scores,method='normal',weight=0.):
    """Only inference-time tensors enter this interface; no annotations or GT grouping."""
    x=np.asarray(coordinates);p=np.asarray(node_scores);r=relation_matrix(len(p),pairs,relation_scores)
    normal,seeds,refs,distance=reference_context(x,p,r)
    if method=='normal':return normal
    if method=='confidence_nms':return seeds
    if method=='protected_nms':return protected_nms(x,p,r,refs,distance)
    if method=='local_profile':return local_representatives(x,p,r,seeds,refs,distance,weight)
    naive,guarded=relation_features(x,p,r,refs,distance)
    if method not in ['naive_global','guarded_global']:raise ValueError(method)
    feature=naive if method=='naive_global' else guarded
    return greedy_nms(x,p+weight*feature,target=len(seeds),pool=np.flatnonzero(p>=.1))

def count_controls(coordinates,node_scores,count):
    pool=np.flatnonzero(node_scores>.5);assert count<=len(pool)
    confidence=pool[np.argsort(-node_scores[pool],kind='stable')][:count]
    spatial=greedy_nms(coordinates,node_scores,pool=pool,target=count)
    return confidence,spatial
