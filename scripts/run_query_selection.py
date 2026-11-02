"""N08 fixed-cache CPU study. Training chooses settings; validation is read afterward."""
import argparse,gzip,io,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from scripts.query_representative_selection import select_queries,count_controls,relation_matrix
from scripts.analyze_graph_errors import (geometry,association,evaluate_mapping,aggregate,write,read,csvwrite,sha)

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/gt_free_query_selection_20261026'
ORIGINAL=ROOT.parent/'Graph-Native-Betti-Matching-Representation'
PRIOR=ROOT.parent/'gnbm-graph-errors-20261026/experiments/oracle_representatives_20261026'
FAMILIES=['naive_global','guarded_global','local_profile']
WEIGHTS=[.25,1.]

def cases():
    identity=read(PRIOR/'input-identities.json');plan=read(ORIGINAL/'experiments/local_refinement_long/plan.json');result=[]
    for seed in [364505,92741]:
        item=next(r for r in identity if r['seed']==seed);archive=Path(item['prediction_archive']);records=read(archive.parent/'records.json')
        result.append(dict(seed=seed,archive=archive,records=[dict(r,gt=str(Path(plan['dataset']['root'])/r['split']/'spline'/(r['sample']+'.npz'))) for r in records],identity=item))
    return result

def records(case,split=None,limit=None):
    rr=[r for r in case['records'] if split is None or r['split']==split]
    if limit:rr=rr[:limit]
    with zipfile.ZipFile(case['archive']) as z:
        for r in rr:
            with np.load(io.BytesIO(z.read(r['sample']+'.npz'))) as f:c={k:f[k] for k in f.files}
            yield r,c

def evaluate(c,gt,ids,tol,baseline=None):
    ids=np.asarray(ids,int);local=association(c['physical'][ids],c['truth'],tol);mp={int(ids[q]):g for q,g in local.items()}
    row,flags=evaluate_mapping(c,gt,tol,ids,mp);row['raw_node_tp']=len(association(c['physical'],c['truth'],tol))
    a,b=c['pairs'].T;fp=flags['chosen']&~flags['labels'];mm=flags['mm'];target=flags['target'];selected=flags['chosen'];x,y=gt['edges'].T
    hops=np.full(len(a),np.nan);hops[mm]=gt['hops'][target[a[mm]],target[b[mm]]]
    row['fp_gt_shortcut']=int((fp&mm&np.isfinite(hops)&(hops>=2)).sum())
    truthdist=np.linalg.norm(c['truth'][:,None]-c['truth'][None],axis=-1)
    row['fp_near_nonadjacent']=int((fp&mm&(truthdist[np.maximum(target[a],0),np.maximum(target[b],0)]<5)).sum())
    degree=np.bincount(c['pairs'][selected].ravel(),minlength=len(c['scores']))[ids]
    row.update(selected_original_rejected=int((c['scores'][ids]<=.5).sum()),pred_hubs_degree6=int((degree>=6).sum()),pred_degree_max=int(degree.max()) if len(degree) else 0)
    inverse={g:q for q,g in mp.items()};score=relation_matrix(len(c['scores']),c['pairs'],c['relations'])
    short=np.linalg.norm(c['truth'][x]-c['truth'][y],axis=-1)<5.
    row['gt_short_edges']=int(short.sum());row['short_edges_tp']=sum(i in inverse and j in inverse and score[inverse[i],inverse[j]]>.5 for i,j in gt['edges'][short])
    gdegree=np.bincount(gt['edges'].ravel(),minlength=len(c['truth']));np.fill_diagonal(truthdist,np.inf)
    close_junction=(gdegree>=3)&(truthdist.min(1)<2.)
    row['gt_close_junctions']=int(close_junction.sum());row['close_junctions_recovered']=int(close_junction[list(inverse)].sum())
    if baseline is not None:
        base=set(baseline['mapping'].values());current=set(mp.values());row['anchors_gained']=len(current-base);row['anchors_lost']=len(base-current)
    return row,flags

def training_selection(limit=None):
    rows=[]
    for case in cases():
        for rec,c in records(case,'train',limit):
            gt=geometry(rec['gt']);base=select_queries(c['physical'],c['scores'],c['pairs'],c['relations'],'confidence_nms');controls=count_controls(c['physical'],c['scores'],len(base))
            for family in FAMILIES:
                for weight in WEIGHTS:
                    ids=select_queries(c['physical'],c['scores'],c['pairs'],c['relations'],family,weight);r,_=evaluate(c,gt,ids,2.);r.update(seed=case['seed'],sample=rec['sample'],subject=rec['subject'],method=family,weight=weight);rows.append(r)
            for name,ids in [('confidence_nms',base),('count_confidence',controls[0])]:
                r,_=evaluate(c,gt,ids,2.);r.update(seed=case['seed'],sample=rec['sample'],subject=rec['subject'],method=name,weight=0.);rows.append(r)
    summaries=[]
    for seed in [364505,92741]:
        for name in FAMILIES+['confidence_nms','count_confidence']:
            for weight in WEIGHTS if name in FAMILIES else [0.]:
                rr=[r for r in rows if r['seed']==seed and r['method']==name and r['weight']==weight];summaries.append(dict(seed=seed,method=name,weight=weight,**aggregate([{k:v for k,v in r.items() if k!='weight'} for r in rr])))
    selected={};train_scores={}
    for family in FAMILIES:
        options=[]
        for weight in WEIGHTS:
            deltas=[];recall_ok=[]
            for seed in [364505,92741]:
                r=next(r for r in summaries if r['seed']==seed and r['method']==family and r['weight']==weight)
                controls=[r for r in summaries if r['seed']==seed and r['method'] in ['confidence_nms','count_confidence']]
                deltas.append(r['edge_f1']-max(z['edge_f1'] for z in controls));recall_ok.append(r['node_recall']>=max(z['node_recall'] for z in controls)-.01)
            options.append(dict(weight=weight,mean_delta=float(np.mean(deltas)),minimum_delta=float(min(deltas)),node_recall_guard=all(recall_ok)))
        winner=max(options,key=lambda r:(r['node_recall_guard'],r['mean_delta'],r['minimum_delta'],-r['weight']));selected[family]=winner['weight'];train_scores[family]=winner
    # Freeze one guarded primary before any validation ranking/results are computed.
    primary=max(['guarded_global','local_profile'],key=lambda name:(train_scores[name]['node_recall_guard'],train_scores[name]['mean_delta'],train_scores[name]['minimum_delta']))
    suffix='smoke-' if limit else ''
    write(OUT/(suffix+'training-selection.json'),dict(selected_weights=selected,primary=primary,selected_scores=train_scores,training_summaries=summaries,criterion='mean two-seed training edge F1 difference against stronger count-matched control; node-recall guard <=1 percentage point loss; ties prefer smaller weight',weights=WEIGHTS,radius_mm=1.,raw_candidate_min_confidence=.1,validation_used=False))
    csvwrite(OUT/(suffix+'training-candidates.csv.gz'),rows);return selected,primary

def rank_data(c,flags,reference,gt):
    inverse={g:q for q,g in flags['mapping'].items()};other={g:q for q,g in reference['mapping'].items()};common=sorted(set(inverse)&set(other));a,b=np.triu_indices(len(common),1)
    qa=np.array([inverse[g] for g in common],int);qb=np.array([other[g] for g in common],int);s=relation_matrix(len(c['scores']),c['pairs'],c['relations']);adj=np.zeros((len(c['truth']),)*2,bool);x,y=gt['edges'].T;adj[x,y]=adj[y,x]=True
    return adj[np.array(common,int)[a],np.array(common,int)[b]],s[qb[a],qb[b]],s[qa[a],qa[b]]

def run(limit=None):
    chosen,primary=training_selection(limit);rows=[];ranking={};suffix='smoke-' if limit else '';trace=gzip.open(OUT/(suffix+'selections.jsonl.gz'),'wt')
    identities=[]
    for case in cases():
        identities.append(case['identity']);assert sha(case['archive'])==case['identity']['prediction_archive_sha256']
        subject_names=sorted({r['subject'] for r in case['records']});subject_lookup={name:i for i,name in enumerate(subject_names)}
        for rec,c in records(case,limit=limit):
            gt=geometry(rec['gt']);assert np.allclose(gt['truth'],c['truth'],atol=2e-5);assert np.array_equal(gt['edges'],c['gt_edges'])
            selected={name:select_queries(c['physical'],c['scores'],c['pairs'],c['relations'],name,chosen.get(name,0.)) for name in ['normal','confidence_nms','protected_nms']+FAMILIES}
            # Every method receives both count-matched controls on every patch.
            for method in list(selected):
                if method in ['normal','confidence_nms']:continue
                top,spatial=count_controls(c['physical'],c['scores'],len(selected[method]));selected[method+'_count_confidence']=top;selected[method+'_count_spatial']=spatial
            for method,ids in selected.items():trace.write(json.dumps(dict(seed=case['seed'],sample=rec['sample'],split=rec['split'],method=method,query_ids=ids.tolist()))+'\n')
            for tol in [1.,2.]:
                base_r,base=evaluate(c,gt,selected['normal'],tol);nms_r,nms=evaluate(c,gt,selected['confidence_nms'],tol,base)
                for method,ids in selected.items():
                    r,flags=evaluate(c,gt,ids,tol,base);meta=dict(seed=case['seed'],sample=rec['sample'],subject=rec['subject'],split=rec['split'],tolerance=tol,method=method);r.update(meta);rows.append(r)
                    key=(case['seed'],rec['split'],tol,method);data=ranking.setdefault(key,{})
                    for pop,mask in [('conditional',flags['active']&flags['mm']),('all_retained',flags['active'])]:
                        for field,value in [('labels',flags['labels'][mask]),('scores',c['relations'][mask].astype(np.float32)),('subject',np.full(int(mask.sum()),subject_lookup[rec['subject']],np.int16))]:data.setdefault(pop+'_'+field,[]).append(value)
                    for refname,ref in [('normal',base),('nms',nms)]:
                        y,pb,ps=rank_data(c,flags,ref,gt)
                        for field,value in [('labels',y),('base',pb.astype(np.float32)),('policy',ps.astype(np.float32)),('subject',np.full(len(y),subject_lookup[rec['subject']],np.int16))]:data.setdefault('shared_'+refname+'_'+field,[]).append(value)
            if len(rows)%2000<30:print('evaluation',case['seed'],rec['split'],rec['sample'],len(rows),flush=True)
    trace.close();csvwrite(OUT/(suffix+'per-patch.csv.gz'),rows);manifest=[];(OUT/(suffix+'ranking')).mkdir(exist_ok=True)
    for (seed,split,tol,method),data in ranking.items():
        target=OUT/(suffix+'ranking')/f'{seed}-{split}-{tol:g}-{method}.npz';np.savez_compressed(target,**{k:np.concatenate(v) for k,v in data.items()});manifest.append(dict(seed=seed,split=split,tolerance=tol,method=method,path=str(target),subject_names=subject_names))
    write(OUT/(suffix+'ranking-manifest.json'),manifest);write(OUT/(suffix+'input-identities.json'),identities)
    write(OUT/(suffix+'complete.json'),dict(completed=True,cpu_only=True,gpu_inference=False,training=False,primary=primary,rows=len(rows)))
    print('N08 cache analysis complete',len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();run(2 if a.smoke else None)
