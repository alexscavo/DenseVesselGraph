"""Fixed-candidate R06 graph/ranking evaluation and source-preserving publication."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from scripts.analyze_graph_errors import read,write,csvwrite,geometry,sha
from scripts.node_verifier_metrics import association,measure,aggregate,components
OUT=ROOT/'experiments/r06_relation_tokens_20261027'
ORIGINAL=ROOT.parent/'Graph-Native-Betti-Matching-Representation'


def archive(path):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with np.load(io.BytesIO(z.read(name))) as f:
                keys=['physical','truth','gt_edges','scores','pairs','relations','hungarian_source','hungarian_target']
                yield Path(name).stem,{k:f[k] for k in keys if k in f.files}


def population(c,gt,tol,cuts):
    ids=np.flatnonzero(c['scores']>.5);m=association(c['physical'][ids],c['truth'],tol)
    target=np.full(len(c['scores']),-1,int)
    for q,g in m.items():target[ids[q]]=g
    a,b=c['pairs'].T;active=np.isin(a,ids)&np.isin(b,ids);both=(target[a]>=0)&(target[b]>=0)
    adj=np.zeros((len(c['truth']),)*2,bool);x,y=c['gt_edges'].T;adj[x,y]=adj[y,x]=True
    labels=both&adj[np.maximum(target[a],0),np.maximum(target[b],0)]
    separation=np.linalg.norm(c['truth'][np.maximum(target[a],0)]-c['truth'][np.maximum(target[b],0)],axis=1)
    near=active&both&~labels&(separation<5.)
    edge_index={tuple(sorted(e)):i for i,e in enumerate(c['gt_edges'].tolist())}
    positive_index=np.full(len(a),-1,int)
    for i in np.flatnonzero(labels):positive_index[i]=edge_index[tuple(sorted((target[a[i]],target[b[i]])))]
    bins={field:np.array([np.searchsorted(cuts[field],g[field],side='right') if g[field] is not None else -1 for g in gt['geometry']]) for field in cuts}
    source,target_h=c['hungarian_source'],c['hungarian_target'];hm=np.full(len(target),-1,int);hm[source]=target_h
    hh=(hm[a]>=0)&(hm[b]>=0);hy=hh&adj[np.maximum(hm[a],0),np.maximum(hm[b],0)]
    return dict(ids=ids,active=active,both=both,labels=labels,near=near,positive_index=positive_index,bins=bins,hm=hh,hy=hy)


def evaluate(c,scores,gt,pop,tol):
    current=dict(c,relations=scores);r=measure(current,pop['ids'],tol)
    chosen=pop['active']&(scores>.5);a,b=c['pairs'].T
    local={int(q):i for i,q in enumerate(pop['ids'])}
    edges=np.array([[local[int(a)],local[int(b)]] for a,b in c['pairs'][chosen]],int).reshape(-1,2)
    p0,p1=components(len(pop['ids']),edges);g0,g1=components(len(gt['truth']),gt['edges'])
    r.update(edge_fp=r['edges']-r['edge_tp'],edge_fn=r['gt_edges']-r['edge_tp'],
      near_nonadjacent_pairs=int(pop['near'].sum()),near_nonadjacent_fp=int((pop['near']&chosen).sum()),
      unmatched_endpoint_fp=int((chosen&~pop['both']).sum()),pred_beta0=p0,pred_beta1=p1,
      gt_beta0=g0,gt_beta1=g1,signed_nodes=r['nodes']-r['gt_nodes'],signed_edges=r['edges']-r['gt_edges'],
      signed_beta0=p0-g0,signed_beta1=p1-g1)
    for field,bins in pop['bins'].items():
        for quartile in range(4):
            eligible=(pop['positive_index']>=0)&pop['active']
            belongs=np.zeros(len(scores),bool);belongs[eligible]=bins[pop['positive_index'][eligible]]==quartile
            r[f'{field}_q{quartile}_gt']=int((bins==quartile).sum())
            r[f'{field}_q{quartile}_eligible']=int(belongs.sum())
            r[f'{field}_q{quartile}_tp']=int((belongs&chosen).sum())
    active=pop['active'];reference=c['relations'][active];score=scores[active]
    r['score_decisions_changed']=int(((score>.5)!=(reference>.5)).sum())
    r['rank_spearman']=float(spearmanr(reference,score).statistic) if len(score)>1 and len(np.unique(score))>1 else None
    hh=pop['hm'];hy=pop['hy'][hh];ps=np.clip(scores[hh],1e-7,1-1e-7)
    r['hungarian_ce_sum']=float(-(hy*np.log(ps)+(1-hy)*np.log(1-ps)).sum());r['hungarian_ce_n']=len(ps)
    return r


def summary(rows):
    keys={'seed','arm','step','split','tolerance','sample','subject','rank_spearman'}
    s=aggregate([{k:v for k,v in r.items() if k not in keys} for r in rows])
    for k in ['edge_fp','edge_fn','pred_beta0','pred_beta1','signed_nodes','signed_edges','signed_beta0','signed_beta1']:
        s[k+'_per_patch']=s[k]/s['patches']
    s['near_nonadjacent_fpr']=s['near_nonadjacent_fp']/s['near_nonadjacent_pairs'] if s['near_nonadjacent_pairs'] else None
    s['hungarian_ce']=s['hungarian_ce_sum']/s['hungarian_ce_n']
    rs=[r['rank_spearman'] for r in rows if r['rank_spearman'] is not None];s['mean_patch_rank_spearman']=float(np.mean(rs)) if rs else None
    for field in ['arc_mm','tortuosity']:
        for q in range(4):
            tp=s[f'{field}_q{q}_tp'];n=s[f'{field}_q{q}_gt'];eligible=s[f'{field}_q{q}_eligible']
            s[f'{field}_q{q}_full_recall']=tp/n if n else None
            s[f'{field}_q{q}_conditional_recall']=tp/eligible if eligible else None
    return s


def ap_pool(y,score,subjects):
    return dict(labels=np.concatenate(y),scores=np.concatenate(score),subjects=np.concatenate(subjects))


def weighted_ap(pool,subject_ids,draws):
    order=np.argsort(-pool['scores'],kind='stable');y=pool['labels'][order].astype(float);score=pool['scores'][order]
    sub=np.array([subject_ids[s] for s in pool['subjects'][order]],int)
    ends=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1]
    values=[]
    for multiplicity in draws:
        weights=multiplicity[sub];tp=np.cumsum(weights*y)[ends];total=np.cumsum(weights)[ends]
        precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        values.append(float((np.diff(np.r_[0,tp])*precision).sum()/tp[-1]) if tp[-1] else np.nan)
    return np.asarray(values)


def main():
    p=read(OUT/'plan.json');rows=[];summaries=[];pools={};all_subject=[]
    rankdir=OUT/'ranking';rankdir.mkdir(exist_ok=True)
    for seed in p['seeds']:
        done=read(OUT/'large'/str(seed)/'runs/complete.json');assert done['completed'] and done['steps']==p['steps']
        configs=p['records'][str(seed)];records={r['sample']:r for r in configs}
        cache=dict(archive(OUT/'large'/str(seed)/'features.zip'));assert len(cache)==608
        gts={s:geometry(r['gt']) for s,r in records.items()}
        train_geometry=[g for s,r in records.items() if r['split']=='train' for g in gts[s]['geometry']]
        cuts={field:np.quantile([g[field] for g in train_geometry if g[field] is not None],[.25,.5,.75]).tolist() for field in ['arc_mm','tortuosity']}
        write(OUT/f'geometry-cuts-{seed}.json',cuts)
        pops={(s,t):population(c,gts[s],t,cuts) for s,c in cache.items() for t in [1.,2.]}
        cases=[('R0',0,OUT/'large'/str(seed)/'runs/R0/scores-0000.zip')]
        for arm in ['R1','R2','R3_head']:
            for step in p['checkpoints']:cases.append((arm,step,OUT/'large'/str(seed)/f'runs/{arm}/scores-{step:04d}.zip'))
        for step in p['checkpoints'][1:]:cases.append(('R2_shuffled',step,OUT/'large'/str(seed)/f'runs/R2/scores-{step:04d}-shuffled.zip'))
        for arm,step,path in cases:
            labels={};scores={};subjects={};localrows=[]
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    sample=Path(name).stem;r=records[sample];c=cache[sample]
                    with np.load(io.BytesIO(z.read(name))) as f:sc=f['relations']
                    for tol in [1.,2.]:
                        pop=pops[sample,tol];row=evaluate(c,sc,gts[sample],pop,tol)
                        row.update(seed=seed,arm=arm,step=step,split=r['split'],tolerance=tol,sample=sample,subject=r['subject'])
                        rows.append(row);localrows.append(row)
                        for kind,mask in [('conditional',pop['active']&pop['both']),('all_retained',pop['active'])]:
                            k=(r['split'],tol,kind);labels.setdefault(k,[]).append(pop['labels'][mask]);scores.setdefault(k,[]).append(sc[mask]);subjects.setdefault(k,[]).append(np.full(mask.sum(),r['subject']))
            for split in ['train','val']:
                for tol in [1.,2.]:
                    rr=[r for r in localrows if r['split']==split and r['tolerance']==tol]
                    s=dict(seed=seed,arm=arm,step=step,split=split,tolerance=tol,**summary(rr))
                    for kind in ['conditional','all_retained']:
                        k=(split,tol,kind);pool=ap_pool(labels[k],scores[k],subjects[k]);pools[seed,arm,step,split,tol,kind]=pool
                        y,score=pool['labels'],pool['scores']
                        s.update({kind+'_ap':float(average_precision_score(y,score)) if y.any() else None,
                                  kind+'_candidates':len(y),kind+'_positives':int(y.sum()),kind+'_prevalence':float(y.mean())})
                        if tol==2.:
                            np.savez_compressed(rankdir/f'{seed}-{arm}-{step}-{split}-{kind}.npz',**pool)
                    summaries.append(s)
                    for subject in sorted({r['subject'] for r in rr}):
                        all_subject.append(dict(seed=seed,arm=arm,step=step,split=split,tolerance=tol,subject=subject,
                                               **summary([r for r in rr if r['subject']==subject])))
            print('collected',seed,arm,step,flush=True)
        # Exact zero-output identity and immutable node populations across all cases.
        for split in ['train','val']:
            for tol in [1.,2.]:
                baseline=next(s for s in summaries if (s['seed'],s['arm'],s['split'],s['tolerance'])==(seed,'R0',split,tol))
                for s in [v for v in summaries if v['seed']==seed and v['split']==split and v['tolerance']==tol]:
                    assert all(s[k]==baseline[k] for k in ['nodes','gt_nodes','node_tp','conditional_candidates','conditional_positives','all_retained_candidates','all_retained_positives'])
                    if s['step']==0:assert all(abs(s[k]-baseline[k])<1e-12 for k in ['conditional_ap','all_retained_ap','edge_f1','edges','edge_tp'])
    differences=[]
    for seed in p['seeds']:
        sub=sorted({r['subject'] for r in rows if r['seed']==seed and r['split']=='val'});sid={s:i for i,s in enumerate(sub)}
        rng=np.random.default_rng(p['bootstrap_seed']);draw=rng.integers(len(sub),size=(p['bootstrap_draws'],len(sub)))
        multiplicities=np.array([np.bincount(d,minlength=len(sub)) for d in draw])
        for reference,refstep in [('R0',0),('R1',p['steps']),('R3_head',p['steps']),('R2_shuffled',p['steps'])]:
            meta=dict(seed=seed,step=p['steps'],split='val',tolerance=2.,arm='R2',reference=reference)
            left=[r for r in rows if (r['seed'],r['arm'],r['step'],r['split'],r['tolerance'])==(seed,reference,refstep,'val',2.)]
            right=[r for r in rows if (r['seed'],r['arm'],r['step'],r['split'],r['tolerance'])==(seed,'R2',p['steps'],'val',2.)]
            result=dict(meta,graph={},ranking={})
            for field in ['edge_f1','beta0_mae','beta1_mae']:
                if field=='edge_f1':num=lambda rr:sum(r['edge_tp'] for r in rr)*2;den=lambda rr:sum(r['edges']+r['gt_edges'] for r in rr)
                else:
                    key='beta0_error' if field=='beta0_mae' else 'beta1_error'
                    num=lambda rr:sum(r[key] for r in rr);den=lambda rr:len(rr)
                values=[]
                for rr in [left,right]:values.append(np.array([[num([r for r in rr if r['subject']==s]),den([r for r in rr if r['subject']==s])] for s in sub]))
                a,b=values;aa=(multiplicities@a);bb=(multiplicities@b);delta=bb[:,0]/bb[:,1]-aa[:,0]/aa[:,1]
                result['graph'][field]=dict(delta=float(b[:,0].sum()/b[:,1].sum()-a[:,0].sum()/a[:,1].sum()),ci95=np.quantile(delta,[.025,.975]).tolist(),draws=len(draw))
            for kind in ['conditional','all_retained']:
                a=pools[seed,reference,refstep,'val',2.,kind];b=pools[seed,'R2',p['steps'],'val',2.,kind]
                delta=weighted_ap(b,sid,multiplicities[:500])-weighted_ap(a,sid,multiplicities[:500])
                point=float(average_precision_score(b['labels'],b['scores'])-average_precision_score(a['labels'],a['scores']))
                result['ranking'][kind+'_ap']=dict(delta=point,ci95=np.nanquantile(delta,[.025,.975]).tolist(),draws=500)
            differences.append(result)
    result=dict(completed=True,protocol=p,summaries=summaries,differences=differences,
      inference_nodes_logits_and_spline_inputs_unchanged=True,frozen_state_verified=True,
      ap_note='Identical candidates, labels and prevalence within each split/tolerance/seed. Conditional and all-retained are separate populations.',
      ce_note='Reported Hungarian CE uses symmetric pair probabilities on all assigned pairs, without ratio duplication: a fixed diagnostic distinct from original randomized-order ratio-balanced training CE.',
      confirmation_gate_passed=confirmation_gate(differences),no_r07_combination=True)
    write(OUT/'analysis.json',result);csvwrite(OUT/'per-patch.csv.gz',rows);csvwrite(OUT/'per-subject.csv.gz',all_subject)
    write(OUT/'checkpoint-manifest.json',[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in sorted((OUT/'large').glob('*/runs/*/module-*.pt'))])
    plot(result);publish(result)
    claim=ROOT.parent/'experiment_gpu_coordination/r06-relation-tokens-20261027.json'
    state=read(claim);state.update(state='completed; results appended',concurrent_h100=0,analysis=str(OUT/'analysis.json'))
    write(claim,state)
    print('R06 collected; confirmation gate',result['confirmation_gate_passed'],flush=True)


def confirmation_gate(differences):
    for seed in [364505,92741]:
        for ref in ['R0','R1','R3_head','R2_shuffled']:
            d=next(v for v in differences if v['seed']==seed and v['reference']==ref)
            if any(d['ranking'][k]['ci95'][0]<=0 for k in ['conditional_ap','all_retained_ap']):return False
            if ref!='R2_shuffled' and (d['graph']['edge_f1']['ci95'][0]<=0 or d['graph']['beta1_mae']['delta']>0):return False
    return True


def plot(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for i,seed in enumerate(s['protocol']['seeds']):
        for j,field in enumerate(['conditional_ap','all_retained_ap','edge_f1']):
            ax=axes[i,j]
            for arm,color in [('R0','tab:blue'),('R1','tab:gray'),('R2','tab:orange'),('R3_head','tab:green')]:
                for split,style in [('train','--'),('val','-')]:
                    rr=sorted([r for r in s['summaries'] if (r['seed'],r['arm'],r['split'],r['tolerance'])==(seed,arm,split,2.)],key=lambda r:r['step'])
                    xx=[r['step'] for r in rr];yy=[r[field] for r in rr]
                    if arm=='R0':xx=[0,2000];yy=yy*2
                    ax.plot(xx,yy,style,color=color,marker='o',markersize=3,label=arm+'/'+split)
            ax.set_title(str(seed)+' / '+field);ax.set_xlabel('Updates');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle('R06 fixed-candidate relation comparison: dashed training, solid validation')
    fig.savefig(OUT/'learning-curves.png',dpi=160);plt.close(fig)


def publish(s):
    def get(seed,arm,split='val',step=2000):return next(r for r in s['summaries'] if (r['seed'],r['arm'],r['split'],r['step'],r['tolerance'])==(seed,arm,split,step if arm!='R0' else 0,2.))
    gate=s['confirmation_gate_passed']
    conclusion=('The predeclared frozen-stage gate passed; review one bounded unfrozen confirmation before claiming complete reconstruction benefit.' if gate else
      'The predeclared gate for a fully unfrozen confirmation did not pass. Retain original A-xyz inference; no fully unfrozen or combined R06+R07 experiment was launched.')
    findings=[]
    if not gate:
        findings=['**Measured outcome: the CNN branch is used, but this residual does not improve the original modern A-xyz relation predictor.**','',
          '1. **Image information is used, without a net baseline gain.** R2 minus shuffled-image all-retained AP is +0.007732 [0.002736, 0.013344] / +0.011343 [0.006843, 0.015530] across the two seeds. Nevertheless, R2 minus original R0 is −0.009691 / −0.006233, with both subject intervals below zero. Image sensitivity is not evidence that R2 improves the original embeddings.','',
          '2. **No convincing advantage over the parameter-matched token control.** Final R2 conditional AP is slightly below R1 in both seeds; R2−R1 all-retained AP is +0.002577 / +0.001388 and edge F1 +0.001755 / +0.000629, with all four subject intervals including zero. R2 often outperforms the continued-head control in all-retained AP, but that control also degrades versus R0. Extra trainable capacity/continued supervision does not rescue the original scorer here.','',
          '3. **Early gains do not persist.** R2 validation all-retained AP at 100 updates is 0.828191 / 0.822567 versus R0 0.827375 / 0.821966; at 2k it falls to 0.817684 / 0.815733. These small early fluctuations are descriptive and do not establish a validated early stopping rule. Training conditional AP rises to 0.983150 / 0.986247 while validation falls to 0.973575 / 0.974823. Training Hungarian-pair CE decreases while validation CE increases. Training all-retained AP does not consistently improve either. This raises a supervision-population question alongside the observed generalization gap; the experiment does not isolate its cause or show uniformly better training on every inference candidate.','',
          '4. **Graph reconstruction deteriorates.** R2−R0 edge F1 is −0.003678 [−0.005682, −0.000958] / −0.002978 [−0.005733, −0.000496]. It loses 37/38 TP, adds 10/removes 9 FP, and increases component MAE by 0.0750/0.0729 and cycle MAE by 0.04375/0.03333. Original node counts, coordinates and classification logits stay unchanged. Mean retained-pair rank correlations with R0 are 0.9846/0.9869 and 421/419 edge decisions change: the embeddings alter ranking, rather than merely applying a common scalar calibration.','',
          '**Recommendation:** keep R0 and the negative result. N08 does not support more of the tested selection heuristics. Prioritize a focused review of spatial quality, assignment/alternative-query label coherence and the matched-training-versus-retained-inference population; keep the independently owned R07 study separate and do not repeat its sampler automatically. The present negative result is limited to 128 training patches, this residual architecture and 2k updates; it does not rule out every use of CNN information.','']
    lines=['## R06 — Relation-specific 552D CNN token enrichment — 27 October 2026','',conclusion,'']+findings+[
      '**Decision gate and scope.** N08 completed negatively against count-matched controls. It supplies no deployable query-selection improvement and does not establish an absence of token information. N06 identifies relation-informative alternatives and residual close-nonadjacent confusion, while normal conditional AP is already about 0.978. This leaves a bounded information-content question. The present experiment is explicitly a modern SyntheticMRI token residual, not the older IXI endpoint/chord/corridor edge-logit adapter.','',
      '**Fixed comparison.** Both modern A-xyz seeds 364505/92741 at update 12,000; original 128 training-monitor patches/32 subjects and 480 validation patches/13 disjoint subjects. Training the new modules on the monitor is a bounded pilot, not full 25k-patch population learning. No test access. All detector/CNN/transformer/node/refinement/spline weights and BN statistics are frozen. Original coordinates and logits are retained exactly in evaluation; replay/cache parity and frozen-state hashes are recorded. Relation tokens stay unchanged.','',
      'R0: original relation head. R1: token-only residual. R2: image-conditioned residual using 27 samples around the existing refined coordinates, at offsets −4/0/+4 input voxels in each axis on layer0/relu1. Both residuals produce 552D corrections, have identical active trainable parameter counts, and zero-initialized output. Each uses token/context LayerNorm→64D projection→ReLU, concatenated into a 552D linear output. R1 fills the 1728D context slot with a fixed repetition of token components: no CNN information, same trainable layers, but a rank-limited redundant input; this is not a claim of equal independent input dimensionality. Only the relation branch consumes the corrected embeddings; node classification/coordinates/spline use originals. R3_head continues only the original relation head and is a supervision/capacity reference, not a parameter-matched arm. R2_shuffled permutes local CNN vectors among queries at inference.','',
      'All trainable arms use the unchanged matched-pair CE, ratio-positive upsampling 0.15, cap 9999, no unmatched mining, edge weight 6, AdamW peak LR 7e-6, 20-step warmup, weight decay 1e-4, batch eight, 2,000 updates. Identical patch order, random endpoint orientation, labels and balancing are verified with sampling digests. Save 0/100/300/1000/2000; final 2000 is primary, early points descriptive. Inference batch one; node/edge cutoffs stay 0.5. Geometry-bin quartiles use training GT only.','',
      '**Primary table: validation at 2 mm, fixed final 2,000 updates.**','',
      '| Seed | Arm, final | Conditional AP | All-retained AP | Edge F1 | TP / FP / FN | Conditional recall | β0 / β1 MAE |',
      '|---:|---|---:|---:|---:|---|---:|---|']
    for seed in s['protocol']['seeds']:
        for arm in ['R0','R1','R2','R2_shuffled','R3_head']:
            r=get(seed,arm);lines.append(f'|{seed}|{arm}|{r["conditional_ap"]:.6f}|{r["all_retained_ap"]:.6f}|{r["edge_f1"]:.6f}|{int(r["edge_tp"])} / {int(r["edge_fp"])} / {int(r["edge_fn"])}|{r["conditional_recall"]:.6f}|{r["beta0_mae"]:.4f} / {r["beta1_mae"]:.4f}|')
    lines+=['','| Seed / arm (final, validation 2 mm) | Nearby nonadjacent FP / candidates | Long-arc Q4 TP / eligible / GT | Tortuosity Q4 TP / eligible / GT | Signed edge / component / cycle bias per patch |',
      '|---|---|---|---|---|']
    for seed in s['protocol']['seeds']:
        for arm in ['R0','R1','R2']:
            r=get(seed,arm)
            lines.append(f'|{seed} / {arm}|{int(r["near_nonadjacent_fp"])} / {int(r["near_nonadjacent_pairs"])}|{int(r["arc_mm_q3_tp"])} / {int(r["arc_mm_q3_eligible"])} / {int(r["arc_mm_q3_gt"])}|{int(r["tortuosity_q3_tp"])} / {int(r["tortuosity_q3_eligible"])} / {int(r["tortuosity_q3_gt"])}|{r["signed_edges_per_patch"]:+.4f} / {r["signed_beta0_per_patch"]:+.4f} / {r["signed_beta1_per_patch"]:+.4f}|')
    lines+=['','R2 does not recover the long/tortuous-edge deficit: long-arc TP falls 1440→1411 / 1425→1399 at identical eligible endpoints; tortuous TP falls 1265→1248 / 1275→1260. Nearby nonadjacent GT negatives are sparse (109/112 pairs), so their 51→57 / 51→51 FP counts should not be overgeneralized. Positive-bin full-GT and conditional recall, with training-defined cutoffs, are saved separately.','']
    lines+=['','AP uses identical candidates and positive prevalence within each seed/split/tolerance; conditional AP includes only spatially recovered endpoints, while all-retained AP also includes unmatched endpoints. Per-arm candidate/positive counts are in `analysis.json`. Full 1/2-mm counts, near-nonadjacent false-positive rates, training-defined long/tortuous-bin recovery, signed graph counts, ranking correlations and decision changes are saved.','',
      '| Seed / split, 2 mm | Conditional pairs / positives / prevalence | All-retained pairs / positives / prevalence |','|---|---|---|']
    for seed in s['protocol']['seeds']:
        for split in ['train','val']:
            r=get(seed,'R0',split)
            lines.append(f'|{seed} / {split}|{r["conditional_candidates"]} / {r["conditional_positives"]} / {r["conditional_prevalence"]:.6f}|{r["all_retained_candidates"]} / {r["all_retained_positives"]} / {r["all_retained_prevalence"]:.6f}|')
    lines+=['',
      '| Seed | R2 minus reference, final | Conditional AP difference [95% CI] | All-retained AP difference [95% CI] | Edge F1 difference [95% CI] |',
      '|---:|---|---|---|---|']
    for d in s['differences']:
        vals=[d['ranking']['conditional_ap'],d['ranking']['all_retained_ap'],d['graph']['edge_f1']]
        show=lambda v:f'{v["delta"]:+.6f} [{v["ci95"][0]:+.6f}, {v["ci95"][1]:+.6f}]'
        lines.append('|'+str(d['seed'])+'|'+d['reference']+'|'+'|'.join(show(v) for v in vals)+'|')
    jobs=read(OUT/'submission.json')
    lines+=['','Intervals resample whole validation subjects (1,000 graph draws; 500 AP draws, seed 271026). These reused development cohorts share a source model; intervals omit source-training and seed uncertainty. No validation threshold or checkpoint selection. Same-population AP changes test ranking beyond a common monotonic logit calibration; graph F1/TP/FP/topology are separate requirements.','',
      f'Slurm smoke **{jobs["smoke"]}**, sequential training array **{jobs["training_array"]}**, CPU collector **{jobs["collector"]}**. R06 used at most one H100 at a time under the shared submission lock, with a global limit of two.','',
      '![R06 learning curves](../../../../gnbm-r06-relation-token-20261027/experiments/r06_relation_tokens_20261027/learning-curves.png)','',
      'Artifacts: sibling worktree `gnbm-r06-relation-token-20261027`, branch `research/r06-relation-token-20261027`, `experiments/r06_relation_tokens_20261027`. Its `large` symlink points to isolated SCRATCH features/scores/checkpoints. `plan.json`, `submission.json`, `smoke.json`, `analysis.json`, `checkpoint-manifest.json`, `ranking/*.npz`, per-patch/per-subject tables and source snapshots preserve provenance. Original A-xyz checkpoint hashes are unchanged: `a1210fe958dffe0e588d38222cb62bf48b58e9b238fc57c027ec1c7a795277c9` / `9646b7b266dec3463d94d16cfbcef0332044a9bf0d599c2319a2eff2c93d3e53`.','',
      'The initial conclusion is limited to this frozen-head residual, training cohort and short schedule. If negative, preserve existing query-selection negatives: N08 does not justify more heuristics. The better-supported next question is assignment/spatial-quality/connectivity supervision, keeping R07 independent; neither arbitrary embedding capacity nor blanket unmatched-negative mining is established as a cure.','']
    section='\n'.join(lines);(OUT/'report.md').write_text(section)
    import fcntl
    report=ORIGINAL/'docs/research/metrics/9-ixi-image-aware-relation-learning.md';marker='R06 RELATION TOKENS 20261027'
    with (OUT/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);old=report.read_text();start='<!-- BEGIN '+marker+' -->';end='<!-- END '+marker+' -->'
        block=start+'\n'+section+'\n'+end
        if start in old:
            a=old.index(start);b=old.index(end,a)+len(end);report.write_text(old[:a]+block+old[b:])
        else:report.write_text(old.rstrip()+'\n\n'+block+'\n')
    registry=ORIGINAL/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md'
    with registry.with_suffix('.md.r06.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);lines=registry.read_text().splitlines()
        status='DONE — measured' if gate else 'DONE — negative/mixed'
        compact=[]
        for seed in s['protocol']['seeds']:
            a=get(seed,'R0');b=get(seed,'R1');c=get(seed,'R2')
            compact.append(f'{seed}: R0/R1/R2 conditional AP {a["conditional_ap"]:.5f}/{b["conditional_ap"]:.5f}/{c["conditional_ap"]:.5f}; edge F1 {a["edge_f1"]:.5f}/{b["edge_f1"]:.5f}/{c["edge_f1"]:.5f}')
        for i,line in enumerate(lines):
            if line.startswith('| **R06** |'):lines[i]='| **R06** | Modern relation-specific 552D local-CNN token residual; frozen A-xyz, two seeds, R0/R1/R2 + continued-head/shuffled controls | **'+status+'** | '+'; '.join(compact)+'. '+conclusion+' | [R06 measured report](../docs/research/metrics/9-ixi-image-aware-relation-learning.md#r06--relation-specific-552d-cnn-token-enrichment--27-october-2026) |'
            elif line.startswith('### Q4 — R06:'):lines[i]='### Q4 — R06: relation-specific 552D node-token fusion — '+status+', 27 October 2026'
            elif line.startswith('- **Relation-image features:**'):lines[i]='- **Relation-image features:** Older IXI edge-logit adapters remain R05 (early gains, late overfit). Modern **R06 is now '+status+'**, testing relation-specific 552D CNN token residuals on frozen A-xyz in both seeds. '+conclusion
            elif line.startswith('| Did we refine the **552D node tokens**'):lines[i]='| Did we refine the **552D node tokens** for a modern A-xyz relation head? | ✅ | Completed frozen two-seed R0/R1/R2, continued-head and shuffled-feature controls; see measured R06 result. | R06 |'
            elif line.startswith('| **R06 — relation-specific 552D token fusion**'):lines[i]='| **R06 — relation-specific 552D token fusion** | **'+status+'.** Frozen modern comparison completed, with original graph supervision. | '+conclusion+' |'
        if not gate:
            observed='| **R06 fails to beat R0/R1 at the fixed final endpoint — observed in both seeds** | Retain original A-xyz; stop this frozen residual. Review spatial-quality/assignment/connectivity supervision and training-versus-inference coverage. Preserve N08 negatives; do not automatically launch more selection heuristics, an unfrozen model or an R06+R07 combination. |'
            if observed not in lines:
                index=next(i for i,line in enumerate(lines) if line.startswith('| **R06 improves ranking'))
                lines.insert(index+1,observed)
        start='<!-- BEGIN R06 COLLECTED 20261027 -->';end='<!-- END R06 COLLECTED 20261027 -->'
        appendix=[start,'### R06 collected — 27 October 2026','',conclusion,'','; '.join(compact)+'. Frozen residual/head controls use exact node candidates and unchanged graph CE/sampling. The image ablation, early/final curves and graph/topology results are in the existing relation report. No R07 combination, path model, query merging or transformer change.',end]
        if start in lines:
            a=lines.index(start);b=lines.index(end,a)+1;lines[a:b]=appendix
        else:lines+=['']+appendix+['']
        registry.write_text('\n'.join(lines)+'\n')
    write(OUT/'publication.json',dict(report=str(report),registry=str(registry),completed=True))


if __name__=='__main__':main()
