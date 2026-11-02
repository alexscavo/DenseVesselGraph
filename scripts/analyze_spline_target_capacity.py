"""Physical GT-only spline adequacy audit; never reads test data or model weights."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.interpolate import BSpline
from metrics.representation_geometry import point_to_polyline_distance, sample_polyline_by_arclength, polyline_length

SPACING=.1
FLATNESS=.01
SEED=221039


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spline_polyline(controls,tolerance=FLATNESS):
    """Cubic Bezier subdivision; curve/polyline Hausdorff discrepancy <= tolerance."""
    controls=np.asarray(controls,dtype=float);k=len(controls)
    knots=np.r_[np.zeros(4),np.arange(1,k-3)/(k-3),np.ones(4)]
    spline=BSpline(knots,controls,3);derivative=spline.derivative()
    breaks=np.unique(knots);values=spline(breaks);slopes=derivative(breaks)
    output=[values[0]]
    def split(c,depth=0):
        flat=float(point_to_polyline_distance(c[1:3],c[[0,3]]).max())
        if flat<=tolerance:
            output.append(c[3]);return
        if depth>=30:raise ValueError('Spline tessellation exceeded recursion limit')
        ab=(c[:-1]+c[1:])/2;bc=(ab[:-1]+ab[1:])/2;mid=(bc[0]+bc[1])/2
        split(np.array([c[0],ab[0],bc[0],mid]),depth+1)
        split(np.array([mid,bc[1],ab[2],c[3]]),depth+1)
    for i,h in enumerate(np.diff(breaks)):
        split(np.array([values[i],values[i]+h*slopes[i]/3,values[i+1]-h*slopes[i+1]/3,values[i+1]]))
    return np.array(output)


def measure(controls,reference):
    c=spline_polyline(controls);reference=np.asarray(reference,float)
    a=sample_polyline_by_arclength(c,SPACING);b=sample_polyline_by_arclength(reference,SPACING)
    forward=point_to_polyline_distance(a,reference);backward=point_to_polyline_distance(b,c)
    # Include every vertex so sparse sharp corners are never omitted from maxima.
    fmax=max(float(forward.max()),float(point_to_polyline_distance(c,reference).max()))
    bmax=max(float(backward.max()),float(point_to_polyline_distance(reference,c).max()))
    hausdorff=max(fmax,bmax);length=polyline_length(reference);clength=polyline_length(c)
    chord=float(np.linalg.norm(reference[-1]-reference[0]))
    vectors=np.diff(reference,axis=0);norms=np.linalg.norm(vectors,axis=1);vectors=vectors[norms>1e-10]/norms[norms>1e-10,None]
    turning=float(np.arccos(np.clip((vectors[:-1]*vectors[1:]).sum(1),-1,1)).sum()) if len(vectors)>1 else 0.
    result=dict(length_mm=length,chord_mm=chord,tortuosity=length/chord if chord>1e-10 else None,
        total_turn_radians=turning,spline_length_mm=clength,length_ratio=clength/length if length else None,
        endpoint_max_mm=float(np.linalg.norm(controls[[0,-1]]-reference[[0,-1]],axis=1).max()),
        symmetric_mean_mm=float((forward.mean()+backward.mean())/2),
        spline_to_dense_mean_mm=float(forward.mean()),dense_to_spline_mean_mm=float(backward.mean()),
        spline_to_dense_p95_mm=float(np.quantile(forward,.95)),dense_to_spline_p95_mm=float(np.quantile(backward,.95)),
        spline_to_dense_max_sampled_mm=fmax,dense_to_spline_max_sampled_mm=bmax,hausdorff_sampled_mm=hausdorff,
        hausdorff_lower_mm=max(0.,hausdorff-FLATNESS),hausdorff_upper_mm=hausdorff+SPACING/2+FLATNESS,
        spline_samples=len(a),dense_samples=len(b),spline_polyline_points=len(c))
    for tol in [.5,1.,2.]:
        key=str(tol).replace('.','p');p=int((forward<=tol).sum());r=int((backward<=tol).sum())
        precision=p/len(a);recall=r/len(b)
        result.update({f'supported_spline_{key}':p,f'supported_dense_{key}':r,
            f'curve_precision_{key}':precision,f'curve_recall_{key}':recall,
            f'curve_f1_{key}':2*precision*recall/(precision+recall) if precision+recall else 0.})
    return result


def patch_rows(dataset,split,subject,sample,path):
    with np.load(path,allow_pickle=False) as f:
        metric=f['metric'].astype(float);controls=f['controls'].astype(float)@metric.T
        paths=f['full_paths'].astype(float)@metric.T;offset=f['path_offsets'];nodes=f['nodes'].astype(float)@metric.T
        assert controls.shape==(len(f['edges']),8,3) and np.isfinite(metric).all() and abs(np.linalg.det(metric))>1e-9
        assert np.isfinite(controls).all() and np.isfinite(paths).all()
        knots=np.r_[np.zeros(4),np.arange(1,5)/5,np.ones(4)]
        replay=BSpline(knots,np.moveaxis(controls,1,0),3)(np.linspace(0,1,64))
        np.testing.assert_allclose(np.moveaxis(replay,0,1),f['curves'].astype(float)@metric.T,atol=1e-4,rtol=1e-6)
        for edge,(a,b) in enumerate(zip(offset[:-1],offset[1:])):
            reference=paths[a:b];assert len(reference)>=2
            np.testing.assert_allclose(reference[[0,-1]],nodes[f['edges'][edge]],atol=1e-4)
            yield dict(dataset=dataset,split=split,subject=subject,sample_id=sample,edge=edge,controls=8,**measure(controls[edge],reference))


def worker(root,task,pilot=False):
    plan=json.loads((root/'plan.json').read_text());spec=plan['tasks'][task];source=plan['datasets'][spec['dataset']]
    assert digest(Path(source['root'])/'manifest.json')==source['manifest_sha256']
    ids=spec['ids'][:8] if pilot else spec['ids'];out=root/('pilot' if pilot else 'workers');out.mkdir(parents=True,exist_ok=True)
    name=f'{task:02d}';started=time.monotonic();count=0;patches=0
    with gzip.open(out/(name+'-edges.csv.gz'),'wt',newline='') as handle:
        writer=None
        for sample in ids:
            subject=source['subjects'][sample.split('_')[1]]
            path=Path(source['root'])/spec['split']/'spline'/(sample+'.npz')
            for row in patch_rows(spec['dataset'],spec['split'],subject,sample,path):
                if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);count+=1
            patches+=1
            if patches%128==0:print(spec['dataset'],spec['split'],patches,'/',len(ids),'edges',count,flush=True)
    seconds=time.monotonic()-started
    write(out/(name+'-summary.json'),dict(completed=True,task=task,dataset=spec['dataset'],split=spec['split'],patches=patches,edges=count,
        ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),seconds=seconds,seconds_per_patch=seconds/max(1,patches),pilot=pilot))
    print(out/(name+'-summary.json'),seconds,'seconds',flush=True)


def summary(group):
    h=group.hausdorff_sampled_mm.to_numpy();n=len(group)
    patch_max=group.groupby('sample_id').hausdorff_lower_mm.max()
    d=dict(edges=n,subjects=group.subject.nunique(),patches=group.sample_id.nunique(),
        fraction_patches_definite_exceed_1mm=float((patch_max>1.).mean()),fraction_patches_definite_exceed_2mm=float((patch_max>2.).mean()),
        length_median_mm=float(group.length_mm.median()),length_p95_mm=float(group.length_mm.quantile(.95)),
        max_error_median_mm=float(np.median(h)),max_error_p95_mm=float(np.quantile(h,.95)),max_error_p99_mm=float(np.quantile(h,.99)),
        max_error_worst_mm=float(h.max()),mean_symmetric_distance_mm=float(group.symmetric_mean_mm.mean()),
        length_ratio_median=float(group.length_ratio.median()))
    for tol in [.5,1.,2.]:
        key=str(tol).replace('.','p')
        p=group[f'supported_spline_{key}'].sum()/group.spline_samples.sum();r=group[f'supported_dense_{key}'].sum()/group.dense_samples.sum()
        d.update({f'fraction_definite_exceed_{key}':float((group.hausdorff_lower_mm>tol).mean()),
                  f'fraction_definite_within_{key}':float((group.hausdorff_upper_mm<=tol).mean()),
                  f'fraction_borderline_{key}':float(((group.hausdorff_lower_mm<=tol)&(group.hausdorff_upper_mm>tol)).mean()),
                  f'same_edge_curve_precision_{key}':float(p),f'same_edge_curve_recall_{key}':float(r),
                  f'same_edge_curve_f1_{key}':float(2*p*r/(p+r)) if p+r else 0.})
    return d


def collect(root,smoke=False):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from data.spline_targets import fit_spline
    plan=json.loads((root/'plan.json').read_text());parts=[]
    for task,spec in enumerate(plan['tasks']):
        m=json.loads((root/'workers'/f'{task:02d}-summary.json').read_text());assert m['completed'] and m['patches']==len(spec['ids']) and not m['pilot']
        assert m['ids_sha256']==hashlib.sha256('\n'.join(spec['ids']).encode()).hexdigest()
        parts.append(pd.read_csv(root/'workers'/f'{task:02d}-edges.csv.gz'))
    df=pd.concat(parts,ignore_index=True);assert not df.duplicated(['dataset','split','sample_id','edge']).any()
    out=root/'results';out.mkdir(exist_ok=True);rows=[];subjects=[];bins=[];intervals=[];rng=np.random.default_rng(SEED)
    for (dataset,split),g in df.groupby(['dataset','split']):
        rows.append(dict(dataset=dataset,split=split,**summary(g)))
        for subject,gg in g.groupby('subject'):subjects.append(dict(dataset=dataset,split=split,subject=subject,**summary(gg)))
        for lower,upper in [(0,10),(10,25),(25,50),(50,100),(100,np.inf)]:
            gg=g[(g.length_mm>=lower)&(g.length_mm<upper)]
            if len(gg):bins.append(dict(dataset=dataset,split=split,axis='length_mm',lower=lower,upper=None if np.isinf(upper) else upper,**summary(gg)))
        for lower,upper in [(1,1.1),(1.1,1.5),(1.5,2),(2,np.inf)]:
            gg=g[(g.tortuosity>=lower)&(g.tortuosity<upper)]
            if len(gg):bins.append(dict(dataset=dataset,split=split,axis='tortuosity',lower=lower,upper=None if np.isinf(upper) else upper,**summary(gg)))
        ss=[s for s in subjects if s['dataset']==dataset and s['split']==split]
        for key in ['fraction_definite_exceed_1p0','fraction_definite_exceed_2p0','mean_symmetric_distance_mm']:
            values=np.array([s[key] for s in ss]);draws=rng.integers(0,len(values),size=(1000,len(values)))
            intervals.append(dict(dataset=dataset,split=split,metric=key,subject_equal_weight_mean=float(values.mean()),subject_bootstrap_95pct=np.quantile(values[draws].mean(1),[.025,.975]).tolist(),subjects=len(values)))
    df.groupby(['dataset','split','subject','sample_id']).agg(edges=('edge','size'),max_sampled_mm=('hausdorff_sampled_mm','max'),max_lower_mm=('hausdorff_lower_mm','max'),max_upper_mm=('hausdorff_upper_mm','max'),mean_symmetric_mm=('symmetric_mean_mm','mean'),longest_branch_mm=('length_mm','max'),mean_length_ratio=('length_ratio','mean')).to_csv(out/'per-patch.csv.gz',compression='gzip')
    pd.DataFrame(rows).to_csv(out/'aggregate.csv',index=False);pd.DataFrame(subjects).to_csv(out/'per-subject.csv',index=False);pd.DataFrame(bins).to_csv(out/'length-and-tortuosity.csv',index=False)
    # Predeclared diagnostic: 64 worst branches + 64 subject-balanced random branches per dataset/split.
    alternatives=[];selected=[]
    for (dataset,split),g in df.groupby(['dataset','split']):
        worst=g.nlargest(64,'hausdorff_sampled_mm').copy();worst['selection']='worst_max_error'
        balanced=g.sample(frac=1,random_state=SEED).groupby('subject',sort=True).head(1).head(64).copy();balanced['selection']='one_seeded_branch_per_subject'
        selected.append(pd.concat([worst,balanced]).drop_duplicates(['sample_id','edge']))
    subset=pd.concat(selected)
    special=df[(df.dataset=='BRAVA')&(df.sample_id=='sample_000002_0222')].copy();special['selection']='requested_example'
    subset=pd.concat([subset,special]).drop_duplicates(['dataset','split','sample_id','edge'])
    for _,r in subset.iterrows():
        source=plan['datasets'][r.dataset];cache=Path(source['root'])/r.split/'spline'/(r.sample_id+'.npz')
        with np.load(cache) as f:
            edge=int(r.edge);a,b=f['path_offsets'][edge:edge+2];metric=f['metric'].astype(float);reference=f['full_paths'][a:b].astype(float)@metric.T
            cached=f['controls'][edge].astype(float)@metric.T
            for k in [8,12,16]:
                controls,_=fit_spline(reference,k,64)
                alternatives.append(dict(dataset=r.dataset,split=r.split,subject=r.subject,sample_id=r.sample_id,edge=edge,selection=r.selection,controls=k,
                    refit8_cache_control_max_difference_mm=float(np.linalg.norm(controls-cached,axis=1).max()) if k==8 else None,**measure(controls,reference)))
    alt=pd.DataFrame(alternatives);alt.to_csv(out/'diagnostic-refits.csv',index=False)
    worst=df.sort_values('hausdorff_sampled_mm',ascending=False).groupby(['dataset','split']).head(20)
    worst.to_csv(out/'worst-examples.csv',index=False);special.to_csv(out/'sample_000002_0222.csv',index=False)
    figure,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,dataset in zip(axes,['IXI','BRAVA']):
        g=df[(df.dataset==dataset)&(df.split=='val')];g=g.sample(min(15000,len(g)),random_state=SEED)
        ax.scatter(g.length_mm,g.hausdorff_sampled_mm,s=3,alpha=.3);ax.axhline(1,color='orange',label='1 mm');ax.axhline(2,color='red',label='2 mm');ax.set(xlabel='Annotated branch arc length (mm)',ylabel='Sampled symmetric maximum error (mm)',title=dataset+' validation',yscale='symlog');ax.legend()
    figure.savefig(out/'length-vs-error.png',dpi=170);plt.close(figure)
    lines=['# IXI and BRAVA — adequacy of actual 8-control spline targets','','GT-only analysis of the complete prepared training and validation populations; no test targets, GPU inference or training. Existing targets and graph anchors were not changed.','','## Metrics and numerical precision','','Distances are to the **corresponding original dense branch**, in physical millimetres, in both directions. This prevents nearby unrelated vessels from hiding errors. Shortcutting appears in dense→spline distances; overshoot appears in spline→dense distances. Uniform arc sampling is0.1 mm; all vertices are also included in maximum-distance checks. Cubic splines are adaptively tessellated to a0.01-mm Hausdorff error bound. The true continuous symmetric maximum is bounded by `[sampled maximum −0.01, sampled maximum +0.06]` mm (lower bound clipped to zero). Threshold exceedance is labelled definite only when the lower bound exceeds the threshold; borderline cases are reported separately. Mean/P95 and coverage are arc-sampled numerical approximations.','','Same-edge Curve F1 is a geometry-only diagnostic, not the whole-graph common-v2 score: each spline is evaluated only against its own branch. Topology/anchors are fixed GT, so connectivity scores would be tautological.','','## Actual cached K=8 targets','','|Dataset|Split|Patches|Branches|Median max mm|P95 max mm|Worst max mm|Definitely >1 mm|Definitely >2 mm|Same-edge Curve F1 at1 mm|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['dataset']}|{r['split']}|{r['patches']}|{r['edges']}|{r['max_error_median_mm']:.3f}|{r['max_error_p95_mm']:.3f}|{r['max_error_worst_mm']:.3f}|{100*r['fraction_definite_exceed_1p0']:.2f}%|{100*r['fraction_definite_exceed_2p0']:.2f}%|{r['same_edge_curve_f1_1p0']:.4f}|")
    lines+=['','|Dataset|Split|Patches with any definite >1-mm branch error|Patches with any definite >2-mm branch error|','|---|---|---:|---:|']
    for r in rows:lines.append(f"|{r['dataset']}|{r['split']}|{100*r['fraction_patches_definite_exceed_1mm']:.2f}%|{100*r['fraction_patches_definite_exceed_2mm']:.2f}%|")
    lines+=['','## Length dependence','','|Dataset|Split|Arc-length interval mm|Branches|P95 maximum mm|Definitely >1 mm|Definitely >2 mm|','|---|---|---|---:|---:|---:|---:|']
    for r in bins:
        if r['axis']=='length_mm':lines.append(f"|{r['dataset']}|{r['split']}|{r['lower']}–{r['upper'] or '∞'}|{r['edges']}|{r['max_error_p95_mm']:.3f}|{100*r['fraction_definite_exceed_1p0']:.2f}%|{100*r['fraction_definite_exceed_2p0']:.2f}%|")
    lines+=['','## Diagnostic additional controls','','Only the predefined subset below was refitted; it is enriched for difficult branches and does **not** estimate population prevalence. Fits use the unchanged production endpoint-constrained least-squares procedure with257 arc-length fitting samples and the same anchors/connectivity. K=8 replay differences are retained to distinguish fit reproduction from capacity changes. These are target fits, not trained model comparisons.','','|Dataset|Split|Controls|Diagnostic branches|Median maximum mm|P95 maximum mm|','|---|---|---:|---:|---:|---:|']
    for (dataset,split,k),g in alt.groupby(['dataset','split','controls']):lines.append(f'|{dataset}|{split}|{k}|{len(g)}|{g.hausdorff_sampled_mm.median():.3f}|{g.hausdorff_sampled_mm.quantile(.95):.3f}|')
    lines+=['','## Requested BRAVA example: sample_000002_0222','','This example is **already a64³ training patch**, not a whole-brain fit. Its longest branches remain long within the crop. All its edges are included in the diagnostic refits.','','|Edge|Original arc mm|Sampled maximum mm|Mean symmetric mm|Dense coverage at1 mm|','|---|---:|---:|---:|---:|']
    for _,r in special.sort_values('length_mm',ascending=False).head(8).iterrows():lines.append(f'|{int(r.edge)}|{r.length_mm:.2f}|{r.hausdorff_sampled_mm:.3f}|{r.symmetric_mean_mm:.3f}|{r.curve_recall_1p0:.4f}|')
    lines+=['','## Subject-level uncertainty','','Equal weight per subject;1,000 subject-bootstrap resamples. These values need not equal the edge-weighted population table.','','|Dataset|Split|Metric|Subject mean|95% interval|','|---|---|---|---:|---|']
    for r in intervals:
        lo,hi=r['subject_bootstrap_95pct'];lines.append(f"|{r['dataset']}|{r['split']}|{r['metric']}|{r['subject_equal_weight_mean']:.5f}|[{lo:.5f}, {hi:.5f}]|")
    lines+=['','## Interpretation and limitations','','Use the all-population error prevalence and long-branch strata together: good average coverage can coexist with severe local shortcuts. An edge exceeding a chosen maximum-error tolerance demonstrates approximation loss in that cached target before model learning. It does not prove that every8-control fit is inadequate or that extra controls will be easy for the network to learn. Reduced error after12/16-control refits supports additional geometric capacity on those branches, not improved learned graph reconstruction. A non-monotonic refit error is possible because this is least squares, not minimax fitting, and knot grids differ.','','Patches overlap and the same anatomical branch may recur; counts describe **patch-edge occurrences**, not independent anatomical vessels. Per-subject summaries and1,000-resample subject-bootstrap intervals avoid treating patches as independent subjects; they do not capture training-seed uncertainty. IXI extracted guides and BRAVA manual traces differ in provenance and possible annotation noise. Total turning angle depends on source discretization; use it alongside tortuosity and length. No vessel segmentation is read. No claim about optimal query budget or a changed architecture follows from this GT-only analysis.','','![Validation length versus maximum error](length-vs-error.png)','','## Reproduction','','`python3 scripts/analyze_spline_target_capacity.py collect --root '+str(root)+'`','','Lightweight results: `python3 scripts/analyze_spline_target_capacity.py show --root '+str(root)+'`.']
    if smoke:lines=['# SMOKE ONLY — reduced fixtures; not population research results','']+lines
    (out/'report.md').write_text('\n'.join(lines)+'\n');write(out/'summary.json',dict(completed=True,smoke=smoke,aggregates=rows,subject_bootstrap=intervals,patches=sum(len(t['ids']) for t in plan['tasks']),edges=len(df),test_evaluated=False,spacing_mm=SPACING,tessellation_bound_mm=FLATNESS))
    write(out/'subject-bootstrap.json',intervals)
    if smoke:
        print('Collector smoke passed:',out/'report.md',flush=True);return
    import shutil
    evidence=Path('docs/research/evidence/spline_target_capacity_20261022');evidence.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():shutil.copy2(p,evidence/p.name)
    doc=Path('docs/research/spline_learning/ixi-brava-control-capacity.md');doc.write_text((out/'report.md').read_text().replace('(length-vs-error.png)','(../evidence/spline_target_capacity_20261022/length-vs-error.png)'))
    marker='## IXI/BRAVA spline target capacity — 22 October 2026'
    review=Path('docs/research/ALL_EXPERIMENTS_AND_RESULTS.md');review.write_text(review.read_text().split(marker)[0].rstrip()+'\n\n'+marker+'\n\n'+doc.read_text())
    print(out/'report.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['worker','pilot','collect','show']);p.add_argument('--root',type=Path);p.add_argument('--task',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.root=a.root or Path(json.loads(Path('experiments/spline_target_capacity/submission.json').read_text())['output_root'])
    if a.mode in ['worker','pilot']:worker(a.root,a.task,a.mode=='pilot')
    elif a.mode=='collect':collect(a.root,a.smoke)
    else:
        path=a.root/'results/report.md'
        if not path.exists():raise SystemExit('Results pending: '+str(path))
        print(path.read_text())
