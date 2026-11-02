"""Refit K8 targets only, preserving the frozen AV-WIDE population/graph guides."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.interpolate import BSpline
from native2d.targets import fit_controls

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/native2d_20261028'
OUT=ROOT/'experiments/native2d_20261028_targetfix'

def main():
    OUT.mkdir(exist_ok=True);(OUT/'prepared').mkdir(exist_ok=True)
    manifest=json.loads((BASE/'manifest.json').read_text());records=[];old_max=0.;new_max=0.;affected=[];branches=0
    for row in manifest['samples']:
        with np.load(BASE/row['target']) as z:target={k:z[k] for k in z.files}
        original=target['controls'];m=target['metric'];inverse=np.linalg.inv(m)
        paths=[target['full_paths'][a:b]@m.T for a,b in zip(target['path_offsets'][:-1],target['path_offsets'][1:])]
        controls=np.stack([fit_controls(p)@inverse.T for p in paths]) if paths else np.empty((0,8,2))
        old=float(np.abs(original@m.T).max()) if len(original) else 0.
        new=float(np.abs(controls@m.T).max()) if len(controls) else 0.
        old_max=max(old_max,old);new_max=max(new_max,new);branches+=len(paths)
        assert np.isfinite(controls).all() and new<1024
        if len(controls):np.testing.assert_allclose(controls[:,[0,-1]],target['nodes'][target['edges']],atol=1e-10)
        if old>512:affected.append(dict(sample_id=row['sample_id'],split=row['split'],old_control_max_abs_pixel=old,new_control_max_abs_pixel=new))
        target['controls']=controls
        path=OUT/'prepared'/Path(row['target']).name;np.savez_compressed(path,**target)
        with np.load(BASE/row['target']) as old_target:
            assert all(np.array_equal(target[k],old_target[k]) for k in target if k!='controls')
        newrow=copy.deepcopy(row);newrow['image']=str((BASE/row['image']).resolve());newrow['target']=str(path.relative_to(OUT));records.append(newrow)
    manifest['samples']=records;manifest['protocol']='native2d-wide-image-disjoint-v1-uniform-K8-fit'
    manifest['target_fit_revision']='uniform64 arc-length samples of identical dense polylines; unchanged nodes/edges/endpoints/reference paths/metric/splits'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ['av-wide-audit.json','splits.json']:(OUT/name).write_bytes((BASE/name).read_bytes())
    knots=np.r_[np.zeros(4),np.arange(1,5)/5,np.ones(4)]
    basis=BSpline.design_matrix(np.linspace(0,1,64),knots,3).toarray()[:,1:-1]
    audit=dict(passed=True,samples=len(records),branches=branches,old_max_abs_control_pixel=old_max,
        new_max_abs_control_pixel=new_max,old_targets_over_512px=affected,uniform_design_condition_number=float(np.linalg.cond(basis)),
        invariants='exact equality of nodes, edges, metric, full_paths, offsets, units, crops and splits; only fitted controls differ',
        original_experiment=str(BASE),new_experiment=str(OUT),budget='one fresh matched pair, same6000 updates; no continuation of old weights')
    (OUT/'target-fit-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
