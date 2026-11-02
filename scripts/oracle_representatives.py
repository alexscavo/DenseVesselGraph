"""Fixed-cache representative interventions. GT is used only for offline diagnostics."""
import argparse, collections, functools, json, sys, csv, gzip
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, map_coordinates
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts.analyze_graph_errors import (cases, load_case, geometry, match_hungarian,
    evaluate_mapping, safe_selection, aggregate, write, read, csvwrite, sha, ORIGINAL)
from scripts.graph_error_counterfactuals import confidence_oracle
from scripts.node_verifier_metrics import association, rank_nms

OUT = Path(__file__).resolve().parents[1]/'experiments/oracle_representatives_20261026'

def selected_cases():
    return [c for c in cases() if c['dataset']=='SyntheticMRI' and c['arm']=='A-xyz' and c['phase']=='final']

@functools.lru_cache(maxsize=640)
def foreground(path):
    gtpath=Path(path); source=gtpath.parent.parent/'seg'/(gtpath.stem+'_seg.nii.gz')
    image=nib.load(source); mask=np.asarray(image.dataobj)>0
    # Same native array axes as MedPy; normalized graph coordinates use axis_size.
    gt=geometry(path); spacing=np.linalg.norm(gt['metric'],axis=0)/np.array(mask.shape)
    assert np.allclose(image.header.get_zooms()[:3],spacing)
    return mask.astype(np.float32),distance_transform_edt(~mask,sampling=spacing).astype(np.float32)

def support(c,gt,path):
    mask,dist=foreground(path); voxel=(c['physical']@np.linalg.inv(gt['metric'].T))*np.array(mask.shape)
    inside=((voxel>=0)&(voxel<=np.array(mask.shape)-1)).all(1)
    near=map_coordinates(mask,voxel.T,order=0,mode='constant',cval=0)
    soft=map_coordinates(mask,voxel.T,order=1,mode='constant',cval=0)
    dd=map_coordinates(dist,voxel.T,order=1,mode='nearest')
    # Out-of-volume queries have no mask support; distance is a lower bound.
    return dict(foreground=near,soft_support=soft,foreground_distance_mm=dd,in_volume=inside)

def matrices(c,gt):
    q=len(c['scores']); scores=np.full((q,q),np.nan)
    a,b=c['pairs'].T; scores[a,b]=scores[b,a]=c['relations']
    adjacency=np.zeros((len(c['truth']),)*2,bool);a,b=gt['edges'].T
    adjacency[a,b]=adjacency[b,a]=True
    return scores,adjacency

def graphrow(c,gt,tol,mapping,meta,condition):
    r,z=evaluate_mapping(c,gt,tol,list(mapping),mapping)
    r.update(meta,condition=condition,tolerance=tol,raw_node_tp=len(association(c['physical'],c['truth'],tol)))
    return r,z

def single_swap(base,g,query):
    """A one-vertex intervention is valid only if it does not reuse another query."""
    if query in set(base.values())-{base[g]}:return None
    result=dict(base);result[g]=query
    return result

def pair_interventions(c,gt,n,h,meta,tol):
    s,adj=matrices(c,gt);changed=[g for g in sorted(set(n)&set(h)) if n[g]!=h[g]];rows=[]
    for i,g in enumerate(changed):
        for v in changed[i+1:]:
            collision=n[g]==h[v] or h[g]==n[v]
            snn=float(s[n[g],n[v]]);scc=float(s[h[g],h[v]])
            scn=float(s[h[g],n[v]]) if not collision else None;snc=float(s[n[g],h[v]]) if not collision else None
            rows.append(dict(meta,tolerance=tol,label=int(adj[g,v]),nearest_score=snn,first_swap_score=scn,second_swap_score=snc,both_score=scc,query_collision=collision,score_interaction=scc-scn-snc+snn if not collision else None,decision_interaction=int(scc>.5)-int(scn>.5)-int(snc>.5)+int(snn>.5) if not collision else None))
    return rows

def repair_interactions():
    rows=[];swaps=[]
    for case in selected_cases():
        for record,c in load_case(case):
            gt=geometry(record['gt']);meta=dict(seed=case['seed'],split=record['split'],sample=record['sample'],subject=record['subject'])
            for tol in [1.,2.]:
                n={g:q for q,g in association(c['physical'],c['truth'],tol).items()};_,mp=confidence_oracle(c,tol);h={g:q for q,g in mp.items()}
                rows.extend(pair_interventions(c,gt,n,h,meta,tol))
                s,adj=matrices(c,gt)
                for g in sorted(set(n)&set(h)):
                    if n[g]==h[g] or single_swap(n,g,h[g]) is None:continue
                    neighbors=[v for v in n if v!=g];labels=np.array([adj[g,v] for v in neighbors]);sn=np.array([s[n[g],n[v]] for v in neighbors]);sc=np.array([s[h[g],n[v]] for v in neighbors])
                    swaps.append(dict(meta,tolerance=tol,gt_vertex=g,neighbors=len(neighbors),positives=int(labels.sum()),nearest_tp=int(((sn>.5)&labels).sum()),confidence_tp=int(((sc>.5)&labels).sum()),nearest_fp=int(((sn>.5)&~labels).sum()),confidence_fp=int(((sc>.5)&~labels).sum()),margin_nearest=float(sn[labels].mean()-sn[~labels].mean()) if labels.any() and (~labels).any() else None,margin_confidence=float(sc[labels].mean()-sc[~labels].mean()) if labels.any() and (~labels).any() else None))
    csvwrite(OUT/'interactions.csv.gz',rows)
    csvwrite(OUT/'single-swaps.csv.gz',swaps)

def analyze(limit=None):
    representatives=[]; pairs=[]; swaps=[]; graphs=[]; interactions=[]; audit=[]; chosen_replay=[]
    prefix='smoke-' if limit else ''
    stream=gzip.open(OUT/(prefix+'graph-pairs.csv.gz'),'wt',newline='')
    pairwriter=csv.DictWriter(stream,fieldnames=['seed','split','sample','subject','tolerance','condition','label','score']);pairwriter.writeheader()
    for case in selected_cases():
        replay_selection=collections.defaultdict(list)
        if limit:case=dict(case,records=[r for s in ['train','val'] for r in [v for v in case['records'] if v['split']==s][:limit]])
        for nr,(record,c) in enumerate(load_case(case)):
            gt=geometry(record['gt']); distance=np.linalg.norm(c['physical'][:,None]-c['truth'][None],axis=-1)
            hs,ht,error=match_hungarian(c,gt,case['config']);assert error<1e-4
            hmap=dict(zip(map(int,hs),map(int,ht))); s,adj=matrices(c,gt);sup=support(c,gt,record['gt'])
            degree=np.bincount(gt['edges'].ravel(),minlength=len(gt['nodes']))
            boundary=None
            with np.load(record['gt']) as f:boundary=f['boundary'] if 'boundary' in f else np.zeros(len(gt['nodes']),bool)
            meta=dict(seed=case['seed'],split=record['split'],sample=record['sample'],subject=record['subject'])
            for tol in [1.,2.]:
                nearest=association(c['physical'],c['truth'],tol);_,conf=confidence_oracle(c,tol)
                assert len(nearest)==len(conf)
                n={g:q for q,g in nearest.items()};h={g:q for q,g in conf.items()};common=sorted(set(n)&set(h))
                changed=[g for g in common if n[g]!=h[g]];stable=[g for g in common if n[g]==h[g]]
                for condition,mapping in [('nearest',nearest),('confidence',conf),('nearest_shared',{n[g]:g for g in common}),('confidence_shared',{h[g]:g for g in common})]:
                    r,z=graphrow(c,gt,tol,mapping,meta,condition)
                    r.update(common_gt=len(common),changed=len(changed),only_nearest=len(set(n)-set(h)),only_confidence=len(set(h)-set(n)))
                    graphs.append(r)
                    aa,bb=c['pairs'].T
                    pairwriter.writerows(dict(meta,tolerance=tol,condition=condition,label=int(z['labels'][i]),score=float(c['relations'][i])) for i in np.flatnonzero(z['active']))
                # Original inference, and one predeclared GT-free duplicate rule plus count control.
                ids=np.flatnonzero(c['scores']>.5)
                keep,_=rank_nms(c['physical'][ids],c['scores'][ids],1.)
                control=ids[np.argsort(-c['scores'][ids],kind='stable')[:len(keep)]]
                for condition,selected in [('normal',ids),('confidence_nms_1mm',ids[keep]),('matched_count_confidence',control)]:
                    local=association(c['physical'][selected],c['truth'],tol);mp={int(selected[q]):g for q,g in local.items()}
                    r,z=graphrow(c,gt,tol,mp,meta,condition)
                    # Graph includes selected queries that are not spatially associated.
                    r,z=evaluate_mapping(c,gt,tol,selected,mp);r.update(meta,tolerance=tol,condition=condition,raw_node_tp=len(nearest));graphs.append(r)
                    pairwriter.writerows(dict(meta,tolerance=tol,condition=condition,label=int(z['labels'][i]),score=float(c['relations'][i])) for i in np.flatnonzero(z['active']))
                for g in common:
                    qn,qh=n[g],h[g];row=dict(meta,tolerance=tol,gt_vertex=g,changed=qn!=qh,degree=int(degree[g]),boundary=bool(boundary[g]),raw_candidates=int((distance[:,g]<=tol).sum()),retained_candidates=int(((distance[:,g]<=tol)&(c['scores']>.5)).sum()),alternative_separation_mm=float(np.linalg.norm(c['physical'][qn]-c['physical'][qh])))
                    for name,q in [('nearest',qn),('confidence',qh)]:
                        row.update({name+'_query':q,name+'_error_mm':float(distance[q,g]),name+'_confidence':float(c['scores'][q]),name+'_hungarian_assigned':q in hmap,name+'_hungarian_same_gt':hmap.get(q)==g,name+'_near_gt_count':int((distance[q]<=tol).sum()),name+'_below_threshold':bool(c['scores'][q]<=.5)})
                        row.update({name+'_'+k:float(v[q]) for k,v in sup.items()})
                    representatives.append(row)
                    if qn==qh:continue
                    for neighbor in stable:
                        if neighbor==g:continue
                        d=float(np.linalg.norm(c['truth'][g]-c['truth'][neighbor]));label=int(adj[g,neighbor]);sn=float(s[qn,n[neighbor]]);sc=float(s[qh,n[neighbor]])
                        pairs.append(dict(meta,tolerance=tol,gt_vertex=g,neighbor=neighbor,category='positive' if label else 'near_negative' if d<5 else 'far_negative' if d>=20 else 'middle_negative',label=label,endpoint_distance_mm=d,nearest_score=sn,confidence_score=sc))
                    valid=single_swap(n,g,qh)
                    if valid is not None:
                        neighbors=[v for v in n if v!=g]
                        labels=np.array([adj[g,v] for v in neighbors]);sn=np.array([s[qn,n[v]] for v in neighbors]);sc=np.array([s[qh,n[v]] for v in neighbors])
                        swaps.append(dict(meta,tolerance=tol,gt_vertex=g,neighbors=len(neighbors),positives=int(labels.sum()),nearest_tp=int(((sn>.5)&labels).sum()),confidence_tp=int(((sc>.5)&labels).sum()),nearest_fp=int(((sn>.5)&~labels).sum()),confidence_fp=int(((sc>.5)&~labels).sum()),margin_nearest=float(sn[labels].mean()-sn[~labels].mean()) if labels.any() and (~labels).any() else None,margin_confidence=float(sc[labels].mean()-sc[~labels].mean()) if labels.any() and (~labels).any() else None))
                interactions.extend(pair_interventions(c,gt,n,h,meta,tol))
                if tol==2. and changed:
                    replay_selection[record['split']].append(dict(record=record,changed=len(changed)))
            if nr%160==0:print(case['seed'],nr,len(case['records']),flush=True)
        # Fixed first two changed-representative patches in each split, no outcome-based selection.
        for split,rr in replay_selection.items():chosen_replay.extend(dict(seed=case['seed'],**v) for v in rr[:2])
        audit.append(dict(seed=case['seed'],archive=case['archive'],sha256=sha(case['archive']),checkpoint=case['checkpoint'],config_sha256=sha(Path(case['checkpoint']).parent/'config.json'),hungarian_replay_max_mm=error,embedding_cached=False))
    stream.close()
    for name,rr in [('representatives',representatives),('stable-neighbor-pairs',pairs),('single-swaps',swaps),('graphs',graphs),('interactions',interactions)]:csvwrite(OUT/(prefix+name+'.csv.gz'),rr)
    write(OUT/(prefix+'audit.json'),audit);write(OUT/(prefix+'embedding-replay-selection.json'),chosen_replay)
    print('saved',len(representatives),len(pairs),len(swaps),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--repair-interactions',action='store_true');args=p.parse_args()
    if args.repair_interactions:repair_interactions()
    else:analyze(2 if args.smoke else None)
