"""Small provenance/alignment/target-contract audit; reuses prior full GT audits."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
from scipy.interpolate import BSpline
from models.checkpoint import load_checkpoint_file
from models.adaptive_geometry import piecewise_knots,fit_basis,uniform_knots
from scripts.multidataset_graph_evaluation import distance
from scripts.analyze_spline_target_capacity import write


def audit(root):
    plan=json.loads((root/'plan.json').read_text());out=dict(completed=False,datasets={},checkpoints={},new_piecewise_targets=[])
    for name,spec in plan['datasets'].items():
        base=Path(spec['root']);assert hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()==spec['manifest_sha256']
        train={s.split('_')[1] for s in spec['train_ids']};val={s.split('_')[1] for s in spec['val_ids']};assert not train&val
        images=[]
        for split,names in [('train',spec['monitor_train'][:8]),('val',spec['monitor_val'][:8])]:
            for sample in names:
                with np.load(base/split/'spline'/(sample+'.npz')) as f:
                    nodes=f['nodes'];edges=f['edges'];metric=f['metric'];assert np.isfinite(nodes).all() and abs(np.linalg.det(metric))>1e-9
                    for edge,a,b in zip(edges,f['path_offsets'][:-1],f['path_offsets'][1:]):np.testing.assert_allclose(f['full_paths'][a:b][[0,-1]],nodes[edge],atol=1e-6)
                    spacing=np.linalg.norm(metric,axis=0)/64
                image=nib.load(base/split/'raw'/(sample+'_data.nii.gz')).get_fdata();assert image.shape==(64,64,64) and np.isfinite(image).all()
                images.append(dict(sample_id=sample,split=split,mean=float(image.mean()),std=float(image.std()),p01=float(np.quantile(image,.01)),p99=float(np.quantile(image,.99)),spacing_mm=spacing.tolist(),nodes=len(nodes),edges=len(edges)))
        index=[r for r in csv.DictReader((base/'patch_index.csv').open()) if r['sample_id'] in set(spec['train_ids']+spec['val_ids'])]
        out['datasets'][name]=dict(training_patches=len(spec['train_ids']),validation_patches=len(spec['val_ids']),training_subjects=len(train),validation_subjects=len(val),max_nodes=spec['max_target_nodes'],negative_graph_patches=sum(int(r['branch_count'])==0 for r in index),image_checks=images)
    for name,spec in plan['checkpoints'].items():
        if name=='synthetic_straight':continue
        path=Path(spec['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
        cp=load_checkpoint_file(path);cfg=cp['config'];out['checkpoints'][name]=dict(path=str(path),sha256=spec['sha256'],queries=cfg['model']['decoder']['object_queries'],matcher=cfg['model']['matcher'],node_loss=cfg['loss']['node'],edge_loss=cfg['loss']['edge'],spline_loss=cfg['loss']['spline'],node_refinement=cfg['model'].get('node_refinement',{}),saved_steps=cp.get('steps',cp.get('epoch')))
        assert cfg['model']['spline']['local_geometry']['enabled']
        current_name='IXI' if name=='ixi_spline' else 'SyntheticMRI'
        population=Path(spec.get('training_population_manifest',Path(cfg['data']['datasets']['synthetic_mri']['root'])/'manifest.json'))
        previous=json.loads(population.read_text());old_train={x.split('_')[1] for x in previous['train_ids']};current_val={x.split('_')[1] for x in plan['datasets'][current_name]['val_ids']}
        assert not old_train&current_val,'Source checkpoint saw current validation subjects'
        out['checkpoints'][name].update(training_population_manifest=str(population),training_population_sha256=hashlib.sha256(population.read_bytes()).hexdigest(),source_training_current_validation_overlap=0)
        del cp
    # A new-basis readiness check, not a repeated GT audit or validation tuning.
    base=Path(plan['datasets']['BRAVA']['root'])
    for edge in [14,25,18,1]:
        with np.load(base/'train/spline/sample_000002_0222.npz') as f:
            a,b=f['path_offsets'][edge:edge+2];path=f['full_paths'][a:b].astype(float)@f['metric'].T
            basecurve=BSpline(uniform_knots(8),f['controls'][edge]@f['metric'].T,3)(np.linspace(0,1,257));baseline=distance(basecurve,path)
        for continuity in ['C0','C1']:
            knots=piecewise_knots(4,continuity);controls,_=fit_basis(path,knots);spline=BSpline(knots,controls,3);curve=spline(np.linspace(0,1,257));error=distance(curve,path)
            angles=[]
            for t in [.25,.5,.75]:
                d=spline.derivative();u=d(t-1e-9);v=d(t+1e-9)
                angles.append(float(np.degrees(np.arccos(np.clip(u@v/max(1e-20,np.linalg.norm(u)*np.linalg.norm(v)),-1,1)))))
                if continuity=='C1':np.testing.assert_allclose(u,v,atol=1e-4,rtol=1e-5)
            assert error[0]<baseline[0],'New piecewise basis failed difficult-branch readiness check'
            out['new_piecewise_targets'].append(dict(sample_id='sample_000002_0222',edge=edge,continuity=continuity,segments=4,controls=len(controls),mean_mm=error[0],sampled_max_mm=error[1],K8_mean_mm=baseline[0],max_join_angle_degrees=max(angles)))
    out['completed']=True;write(root/'audit.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.root)
