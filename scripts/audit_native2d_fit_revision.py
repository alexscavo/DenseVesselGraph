"""Fixed GT-geometry capacity sample and same corrected fit for prepared REAVER."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.interpolate import BSpline
from native2d.targets import fit_controls
from metrics.representation_geometry import sample_polyline_by_arclength,point_to_polyline_distance

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/native2d_20261028'
OUT=ROOT/'experiments/native2d_20261028_targetfix'

def capacity():
    rows=json.loads((OUT/'manifest.json').read_text())['samples'];pool=[]
    for i,row in enumerate(rows):pool.extend((i,j) for j in range(row['edges']))
    rng=np.random.default_rng(364505);selected=sorted(rng.choice(len(pool),256,replace=False).tolist())
    knots=np.r_[np.zeros(4),np.arange(1,5)/5,np.ones(4)];result=[]
    for idx in selected:
        i,j=pool[idx];row=rows[i]
        with np.load(OUT/row['target']) as z:
            m=z['metric'];path=z['full_paths'][z['path_offsets'][j]:z['path_offsets'][j+1]]@m.T
            controls=z['controls'][j]@m.T
        curve=BSpline(knots,controls,3)(np.linspace(0,1,257));reference=sample_polyline_by_arclength(path,.25)
        entry=dict(sample_id=row['sample_id'],edge=j)
        for name,line in [('K8',curve),('chord',path[[0,-1]])]:
            sample=sample_polyline_by_arclength(line,.25)
            d=point_to_polyline_distance(sample,path);reverse=point_to_polyline_distance(reference,line)
            entry[name]=dict(pred_samples=len(d),gt_samples=len(reverse),pred_supported=int((d<=1).sum()),
                gt_supported=int((reverse<=1).sum()),max_symmetric_distance=float(max(d.max(),reverse.max())))
        result.append(entry)
    summary={}
    for name in ['K8','chord']:
        p=sum(r[name]['pred_supported'] for r in result)/sum(r[name]['pred_samples'] for r in result)
        r=sum(r[name]['gt_supported'] for r in result)/sum(r[name]['gt_samples'] for r in result)
        summary[name]=dict(curve_f1=2*p*r/(p+r),precision=p,recall=r,max_distance_quantiles=np.quantile([x[name]['max_symmetric_distance'] for x in result],[.5,.9,.99,1]).tolist())
    (OUT/'gt-geometry-capacity.json').write_text(json.dumps(dict(purpose='GT geometry approximation only; not prediction performance',
        seed=364505,branches=256,units='pixel',tolerance=1.,sampling=.25,summary=summary,samples=result),indent=2)+'\n')
    print('GT capacity '+json.dumps(summary),flush=True)

def reaver():
    manifest=json.loads((BASE/'reaver-manifest.json').read_text());updated=[];directory=OUT/'reaver-diaphragm';directory.mkdir(exist_ok=True)
    maximum=0.
    for row in manifest['samples']:
        with np.load(BASE/row['target']) as z:target={k:z[k] for k in z.files}
        m=target['metric'];inverse=np.linalg.inv(m)
        paths=[target['full_paths'][a:b]@m.T for a,b in zip(target['path_offsets'][:-1],target['path_offsets'][1:])]
        target['controls']=np.stack([fit_controls(p)@inverse.T for p in paths]) if paths else np.empty((0,8,2))
        assert target['units'].item()=='micron' and np.isfinite(target['controls']).all()
        if len(paths):
            np.testing.assert_allclose(target['controls'][:,[0,-1]],target['nodes'][target['edges']],atol=1e-10)
            maximum=max(maximum,float(np.abs(target['controls']@m.T).max()))
        path=directory/Path(row['target']).name;np.savez_compressed(path,**target)
        with np.load(BASE/row['target']) as z:assert all(np.array_equal(target[k],z[k]) for k in target if k!='controls')
        new=copy.deepcopy(row);new['image']=str((BASE/row['image']).resolve());new['target']=str(path.relative_to(OUT));updated.append(new)
    manifest.update(samples=updated,target_fit='uniform64 arc-length resampling; all noncontrol tensors exactly preserved',
        max_abs_control_micron=maximum,training_submitted=False)
    (OUT/'reaver-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('REAVER revised '+json.dumps(dict(samples=len(updated),max_abs_control_micron=maximum)),flush=True)

if __name__=='__main__':
    capacity();reaver()
