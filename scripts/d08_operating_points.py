"""CPU-only D08 confidence/edge calibration; preserve the cache's candidate limits.

Node calibration uses all 256 cached raw proposals. Edge calibration uses only
the original >0.5 node population, whose pair scores actually exist. No missing
pair score or spline is synthesized. All operating points use training patches.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
from metrics.branch_connectivity import branch_connectivity_f1

SOURCE=ROOT.parent/'gnbm-brava-long-20261027/experiments/brava_long_20261027'
OUT=ROOT/'experiments/d08_operating_points_20261028'
ARMS=('scratch','synthetic')
GRID=np.array([0.,.025,.05,.075,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1.])
SEED=281028


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def save(path,value):
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match_distances(distances,tolerance):
    """Exactly the D08 max-cardinality, then distance positional convention."""
    penalty=(min(distances.shape)+1)*(tolerance+1)
    r,c=linear_sum_assignment(np.where(distances<=tolerance,distances,penalty))
    return {int(a):int(b) for a,b in zip(r,c) if distances[a,b]<=tolerance}


def node_row(cache,gt,threshold,tolerance,distances=None):
    if distances is None:distances=np.linalg.norm((cache['raw_nodes'][:,None]-gt['nodes'][None])@gt['metric'].T,axis=-1)
    ids=np.flatnonzero(cache['raw_scores']>float(threshold))
    mapping=match_distances(distances[ids],tolerance)
    recovered=set(mapping.values());near=distances[ids]<=tolerance
    duplicate=[i for i in range(len(ids)) if i not in mapping and any(near[i,j] for j in recovered)]
    return dict(nodes=len(ids),gt_nodes=len(gt['nodes']),node_tp=len(mapping),
        duplicates=len(duplicate),high_confidence_duplicates=int((cache['raw_scores'][ids[duplicate]]>=.9).sum()),
        node_abs_count_error=abs(len(ids)-len(gt['nodes'])),
        localization_sum=float(sum(distances[ids[i],j] for i,j in mapping.items())),
        endpoint_recovered=sum(a in recovered and b in recovered for a,b in gt['edges']))


def node_aggregate(rows):
    n=sum(r['nodes'] for r in rows);g=sum(r['gt_nodes'] for r in rows);tp=sum(r['node_tp'] for r in rows)
    return dict(patches=len(rows),predicted_nodes=n,gt_nodes=g,node_tp=tp,node_fp=n-tp,node_fn=g-tp,
        node_precision=tp/n if n else None,node_recall=tp/g if g else None,
        node_f1=2*tp/(n+g) if n+g else 1.,duplicates=sum(r['duplicates'] for r in rows),
        high_confidence_duplicates=sum(r['high_confidence_duplicates'] for r in rows),
        node_count_bias=(n-g)/len(rows),node_count_mae=float(np.mean([r['node_abs_count_error'] for r in rows])),
        localization_mean_mm=sum(r['localization_sum'] for r in rows)/tp if tp else None,
        gt_edges_both_endpoints_recovered=sum(r['endpoint_recovered'] for r in rows))


def edge_thresholds(scores,labels,gt_count,precision_target):
    """Strict > in original score dtype; tied groups remain intact (R07 fix)."""
    if not len(scores) or not labels.any():
        return dict(train_max_f1=1.,train_preserve_precision=1.,precision_target_attainable=False)
    order=np.argsort(-scores,kind='stable');p=scores[order];y=labels[order]
    end=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1]
    tp=np.cumsum(y)[end];count=end+1
    cuts=np.nextafter(p[end],np.array(-np.inf,dtype=p.dtype))
    f1=2*tp/(count+gt_count);best=int(np.argmax(f1))
    eligible=np.flatnonzero((tp/count>=precision_target)&(tp>0))
    pick=int(eligible[np.argmax(tp[eligible])]) if len(eligible) else None
    return dict(train_max_f1=float(cuts[best]),
        train_preserve_precision=float(cuts[pick]) if pick is not None else 1.,
        precision_target_attainable=pick is not None)


def betti(n,edges):
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in edges:parent[find(int(a))]=find(int(b))
    b0=len({find(i) for i in range(n)})
    return b0,len(edges)-n+b0


def edge_labels(cache,gt,tolerance):
    d=np.linalg.norm((cache['nodes'][:,None]-gt['nodes'][None])@gt['metric'].T,axis=-1)
    mapping=match_distances(d,tolerance)
    pairs=cache['candidate_edges'];links={tuple(sorted(e)) for e in gt['edges'].tolist()}
    conditional=np.array([int(a) in mapping and int(b) in mapping for a,b in pairs],bool)
    labels=np.array([a in mapping and b in mapping and tuple(sorted((mapping[a],mapping[b]))) in links for a,b in pairs],bool)
    return mapping,labels,conditional


def edge_row(cache,gt,tolerance,threshold,*,branches=True):
    mapping,labels,conditional=edge_labels(cache,gt,tolerance)
    selected=cache['candidate_scores']>float(threshold);edges=cache['candidate_edges'][selected]
    tp=int(labels[selected].sum());fp=len(edges)-tp
    p0,p1=betti(len(cache['nodes']),edges);g0,g1=betti(len(gt['nodes']),gt['edges'])
    missing=sum(not conditional[i] for i in np.flatnonzero(selected))
    recovered=set(mapping.values());eligible=sum(a in recovered and b in recovered for a,b in gt['edges'])
    nearby=[]
    for i in np.flatnonzero(conditional&~labels):
        a,b=cache['candidate_edges'][i]
        if np.linalg.norm((gt['nodes'][mapping[a]]-gt['nodes'][mapping[b]])@gt['metric'].T)<5.:nearby.append(i)
    row=dict(nodes=len(cache['nodes']),edges=len(edges),gt_edges=len(gt['edges']),tp=tp,fp=fp,fn=len(gt['edges'])-tp,
        recoverable_edges=eligible,unmatched_fp=int(missing),near_negative_candidates=len(nearby),near_negative_fp=int(selected[nearby].sum()),
        beta0_error=abs(p0-g0),beta1_error=abs(p1-g1),component_bias=p0-g0,cycle_bias=p1-g1,
        gt_nodes=len(gt['nodes']),node_tp=len(mapping),node_abs_count_error=abs(len(cache['nodes'])-len(gt['nodes'])))
    if branches:
        b=branch_connectivity_f1(cache['nodes']@gt['metric'].T,edges,gt['nodes']@gt['metric'].T,gt['edges'],threshold_mm=1.)
        row.update(branch_tp=b['tp'],branch_fp=b['fp'],branch_fn=b['fn'])
    return row


def edge_aggregate(rows):
    sums={k:sum(r[k] for r in rows) for k in ['nodes','edges','gt_edges','tp','fp','fn','recoverable_edges','unmatched_fp','near_negative_candidates','near_negative_fp']}
    tp=sums['tp'];n=sums['edges'];g=sums['gt_edges']
    sums.update(patches=len(rows),precision=tp/n if n else None,recall=tp/g if g else None,
        edge_f1=2*tp/(n+g) if n+g else 1.,conditional_recall=tp/sums['recoverable_edges'] if sums['recoverable_edges'] else None,
        unmatched_fraction_of_fp=sums['unmatched_fp']/sums['fp'] if sums['fp'] else None,
        near_negative_fpr=sums['near_negative_fp']/sums['near_negative_candidates'] if sums['near_negative_candidates'] else None,
        component_mae=float(np.mean([r['beta0_error'] for r in rows])),cycle_mae=float(np.mean([r['beta1_error'] for r in rows])),
        component_bias=float(np.mean([r['component_bias'] for r in rows])),cycle_bias=float(np.mean([r['cycle_bias'] for r in rows])))
    if rows and 'branch_tp' in rows[0]:
        bt,bp,bn=(sum(r[k] for r in rows) for k in ['branch_tp','branch_fp','branch_fn'])
        sums.update(branch_tp=bt,branch_fp=bp,branch_fn=bn,branch_f1=2*bt/(2*bt+bp+bn) if 2*bt+bp+bn else 1.)
    return sums


def samples(arm,split):
    summary=json.loads((SOURCE/'large/predictions'/arm/'epoch-300/summary.json').read_text())
    records=[r for r in summary['records'] if r['split']==split]
    archive=SOURCE/'large/predictions'/arm/'epoch-300'/f'{split}.zip'
    with zipfile.ZipFile(archive) as source:
        for r in records:
            with np.load(io.BytesIO(source.read(r['entry']))) as z:cache={k:z[k].copy() for k in z.files}
            with np.load(r['gt']) as z:gt={k:z[k].copy() for k in ['nodes','edges','metric']}
            yield r,cache,gt


def select_node(candidates):
    # Highest cut is the conservative tie break. Training data alone choose it.
    return max(candidates,key=lambda r:(r['metrics']['node_f1'],r['threshold']))['threshold']


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'thresholds.json').exists():raise RuntimeError('Training thresholds already exist; reuse or select a new output directory')
    plan=json.loads((SOURCE/'plan.json').read_text())
    assert not set(plan['subject_groups']['train'])&set(plan['subject_groups']['val'])
    curves=[];settings={};baselines={}
    for arm in ARMS:
        node={t:{float(c):[] for c in GRID} for t in [1.,2.]};score_parts=[];label_parts=[];gt_edges=0;baseline=[]
        for r,cache,gt in samples(arm,'train'):
            assert len(cache['raw_nodes'])==len(cache['raw_scores'])==256
            np.testing.assert_array_equal(cache['query_ids'],np.flatnonzero(cache['raw_scores']>.5))
            np.testing.assert_array_equal(cache['nodes'],cache['raw_nodes'][cache['query_ids']])
            np.testing.assert_array_equal(cache['node_scores'],cache['raw_scores'][cache['query_ids']])
            assert len(cache['candidate_edges'])==len(cache['nodes'])*(len(cache['nodes'])-1)//2
            d=np.linalg.norm((cache['raw_nodes'][:,None]-gt['nodes'][None])@gt['metric'].T,axis=-1)
            for tolerance in node:
                for cut in GRID:node[tolerance][float(cut)].append(node_row(cache,gt,float(cut),tolerance,d))
            _,labels,_=edge_labels(cache,gt,2.)
            score_parts.append(cache['candidate_scores']);label_parts.append(labels);gt_edges+=len(gt['edges'])
            baseline.append(edge_row(cache,gt,2.,.5))
        for tolerance in node:
            for cut,rows in node[tolerance].items():curves.append(dict(arm=arm,tolerance_mm=tolerance,threshold=cut,metrics=node_aggregate(rows)))
        primary=[r for r in curves if r['arm']==arm and r['tolerance_mm']==2.]
        cut=select_node(primary)
        scores=np.concatenate(score_parts);labels=np.concatenate(label_parts);base=edge_aggregate(baseline)
        precision=base['precision'] if base['precision'] is not None else 0.
        calibration=edge_thresholds(scores,labels,gt_edges,precision)
        settings[arm]=dict(node_train_max_f1=cut,node_selection_tolerance_mm=2.,
            node_grid=GRID.tolist(),baseline_training_precision=precision,edge_selection_tolerance_mm=2.,
            edge=dict(fixed_0p5=.5,**calibration),node_graph_needs_uncached_queries=cut<.5,
            training_patches=len(baseline),training_subjects=len(plan['subject_groups']['train']))
        baselines[arm]=base
        print(json.dumps(dict(arm=arm,node_cut=cut,edge=calibration)),flush=True)
    manifest=dict(protocol='D08 CPU operating-point audit v1',epoch=300,selection_split='train160',
        threshold_selection_uses_validation=False,node_grid=GRID.tolist(),primary_tolerance_mm=2.,secondary_tolerance_mm=1.,
        baseline_node_cut=.5,strict_comparison='>',original_coordinate_and_score_caches_unchanged=True,
        limits='No raw-query embeddings/pair scores or new-edge spline controls in original D08 caches; joint lower-node graph replay is not fabricated',
        checkpoint_identities={a:json.loads((SOURCE/'large/runs'/a/'model-epoch-300.json').read_text()) for a in ARMS},
        source_paths={a:str(SOURCE/'large/predictions'/a/'epoch-300') for a in ARMS},
        source_plan_sha256=sha(SOURCE/'plan.json'),training_settings=settings)
    save(output/'protocol.json',manifest);save(output/'thresholds.json',settings)
    save(output/'training-node-curves.json',curves);save(output/'training-baselines.json',baselines)


def bootstrap(rows_a,rows_b,kind,subjects):
    a={s:[r for r in rows_a if r['subject']==s] for s in subjects};b={s:[r for r in rows_b if r['subject']==s] for s in subjects}
    rng=np.random.default_rng(SEED);keys=['node_f1','node_recall','node_precision'] if kind=='node' else ['edge_f1','precision','recall','branch_f1','cycle_mae','component_mae']
    values={k:[] for k in keys};aggregate=node_aggregate if kind=='node' else edge_aggregate
    for _ in range(1000):
        chosen=rng.choice(subjects,len(subjects),replace=True)
        aa=aggregate([r for s in chosen for r in a[s]]);bb=aggregate([r for s in chosen for r in b[s]])
        for k in keys:
            if aa.get(k) is not None and bb.get(k) is not None:values[k].append(bb[k]-aa[k])
    return {k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in values.items()}


def analyze(output):
    settings=json.loads((output/'thresholds.json').read_text());plan=json.loads((SOURCE/'plan.json').read_text())
    baseline=json.loads((SOURCE/'analysis/epoch-300.json').read_text())
    rows=[];summary={};pairs=[];parity=[];uncertainty={}
    for arm in ARMS:
        cuts=dict(normal_0p5=.5,training_node_f1=settings[arm]['node_train_max_f1'],all_raw=0.)
        policies={k:v for k,v in settings[arm]['edge'].items() if k in ['fixed_0p5','train_max_f1','train_preserve_precision']}
        for split in ['train','val']:
            rank={t:{k:[] for k in ['scores','labels','conditional']} for t in [1.,2.]}
            for r,cache,gt in samples(arm,split):
                d=np.linalg.norm((cache['raw_nodes'][:,None]-gt['nodes'][None])@gt['metric'].T,axis=-1)
                for tolerance in [1.,2.]:
                    for name,cut in cuts.items():
                        row=node_row(cache,gt,cut,tolerance,d)
                        rows.append(dict(arm=arm,split=split,sample_id=r['sample'],subject=r['subject'],kind='node',
                            policy=name,tolerance_mm=tolerance,threshold=cut,**row))
                    _,labels,conditional=edge_labels(cache,gt,tolerance)
                    rank[tolerance]['scores'].append(cache['candidate_scores']);rank[tolerance]['labels'].append(labels);rank[tolerance]['conditional'].append(conditional)
                    for name,cut in policies.items():
                        row=edge_row(cache,gt,tolerance,cut)
                        rows.append(dict(arm=arm,split=split,sample_id=r['sample'],subject=r['subject'],kind='edge',
                            policy=name,tolerance_mm=tolerance,threshold=cut,**row))
                # Categories are offline geometric diagnostics, never features.
                mapping=match_distances(d,2.);recovered=set(mapping.values())
                duplicate=[i for i in range(256) if i not in mapping and any(d[i,j]<=2. for j in recovered)]
                correct=list(mapping)
                pairs.append(dict(arm=arm,split=split,sample_id=r['sample'],subject=r['subject'],matched=len(correct),duplicates=len(duplicate),
                    correct_scores=cache['raw_scores'][correct].tolist(),duplicate_scores=cache['raw_scores'][duplicate].tolist(),
                    rejected_correct=int((cache['raw_scores'][correct]<=.5).sum())))
            for tolerance in [1.,2.]:
                for name in cuts:
                    group=[r for r in rows if r['arm']==arm and r['split']==split and r['kind']=='node' and r['policy']==name and r['tolerance_mm']==tolerance]
                    summary[f'{arm}/{split}/node/{name}/{tolerance}']=node_aggregate(group)
                for name in policies:
                    group=[r for r in rows if r['arm']==arm and r['split']==split and r['kind']=='edge' and r['policy']==name and r['tolerance_mm']==tolerance]
                    summary[f'{arm}/{split}/edge/{name}/{tolerance}']=edge_aggregate(group)
                scores=np.concatenate(rank[tolerance]['scores']);labels=np.concatenate(rank[tolerance]['labels']);cond=np.concatenate(rank[tolerance]['conditional'])
                rankings={}
                for population,mask in [('all_retained',np.ones(len(scores),bool)),('conditional',cond)]:
                    y=labels[mask];s=scores[mask]
                    rankings[population]=dict(candidates=len(y),positives=int(y.sum()),prevalence=float(y.mean()) if len(y) else None,
                        ap=float(average_precision_score(y,s)) if y.any() else None)
                summary[f'{arm}/{split}/ranking/{tolerance}']=rankings
                original=baseline['arms'][arm]['metrics'][split][str(tolerance)]
                actual=summary[f'{arm}/{split}/edge/fixed_0p5/{tolerance}'];node=summary[f'{arm}/{split}/node/normal_0p5/{tolerance}']
                for name in ['nodes','edges','tp','fp','fn','recoverable_edges']:assert actual[name]==original[name],(arm,split,tolerance,name)
                assert node['node_tp']==original['node_tp']
                assert abs(actual['edge_f1']-original['f1'])<1e-14
                if tolerance==2.:
                    common=baseline['arms'][arm]['common'][split]
                    for key in ['tp','fp','fn']:assert actual['branch_'+key]==common['common_branch_'+key+'_total']
                parity.append(dict(arm=arm,split=split,tolerance_mm=tolerance,baseline_counts_parity=True,baseline_branch_parity=True))
        for tolerance in [1.,2.]:
            def choose(kind,policy):return [r for r in rows if r['arm']==arm and r['split']=='val' and r['kind']==kind and r['policy']==policy and r['tolerance_mm']==tolerance]
            uncertainty[f'{arm}/node/{tolerance}']=bootstrap(choose('node','normal_0p5'),choose('node','training_node_f1'),'node',plan['subject_groups']['val'])
            for policy in ['train_max_f1','train_preserve_precision']:
                uncertainty[f'{arm}/edge/{policy}/{tolerance}']=bootstrap(choose('edge','fixed_0p5'),choose('edge',policy),'edge',plan['subject_groups']['val'])
        print('completed cache analysis',arm,flush=True)
    save(output/'summary.json',dict(completed=True,thresholds=settings,aggregates=summary,paired_subject_intervals=uncertainty,
        parity=parity,limitations=json.loads((output/'protocol.json').read_text())['limits'],
        no_gpu_inference=True,no_training=True))
    with gzip.open(output/'per-patch.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)
    save(output/'confidence-categories.json',pairs)
    with (output/'per-subject.csv').open('w',newline='') as stream:
        results=[]
        for key in sorted({(r['arm'],r['split'],r['subject'],r['kind'],r['policy'],r['tolerance_mm']) for r in rows}):
            arm,split,subject,kind,policy,t=key
            group=[r for r in rows if (r['arm'],r['split'],r['subject'],r['kind'],r['policy'],r['tolerance_mm'])==key]
            results.append(dict(arm=arm,split=split,subject=subject,kind=kind,policy=policy,tolerance_mm=t,
                **(node_aggregate(group) if kind=='node' else edge_aggregate(group))))
        writer=csv.DictWriter(stream,fieldnames=sorted({k for r in results for k in r}));writer.writeheader();writer.writerows(results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','analyze']);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();started=time.monotonic()
    (prepare if args.mode=='prepare' else analyze)(args.output)
    print(args.mode,'seconds',time.monotonic()-started,flush=True)


if __name__=='__main__':main()
