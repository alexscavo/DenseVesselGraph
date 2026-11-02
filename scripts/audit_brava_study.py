"""Read-only BRAVA development audit and annotation representation analysis."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from scripts.ixi_cached_mechanisms import save,table,sha


def adjacency(positions,edges):
    def point(p):return tuple(np.round(p,4))
    return {tuple(sorted((point(positions[a]),point(positions[b])))) for a,b in edges}


def source_graph(path):
    from scripts.audit_synthetic_mri_grid import SourceGraph
    g=SourceGraph.from_directory(path);lookup={int(v):i for i,v in enumerate(g.node_ids)}
    edges=np.array([[lookup[a],lookup[b]] for a,b in g.edges],int).reshape(-1,2)
    return g,g.node_positions,edges


def audit(root,limit=None):
    from scripts.prepare_brava import read_swc,swc_to_voxel
    from training.evaluation.metrics import graph_betti_numbers
    p=json.loads((root/'plan.json').read_text());dataset=Path(p['dataset']);prepared=dataset.parent
    assert (dataset/'READY').exists() and sha(dataset/'manifest.json')==p['manifest_sha256']
    manifest=json.loads((dataset/'manifest.json').read_text());qc=json.loads((prepared/'audit/complete.json').read_text())
    assert qc['completed'] and len(qc['checks'])==111
    assert manifest['segmentation_role']=='brain_mask_only_not_vessel_GT'
    subject_sets={s:{p['subjects_by_id'][n] for n in p['ids'][s]} for s in ['train','val','test']}
    assert [len(subject_sets[s]) for s in ['train','val','test']]==[39,8,9]
    assert not (subject_sets['train']&subject_sets['val'] or subject_sets['train']&subject_sets['test'] or subject_sets['val']&subject_sets['test'])
    out=root/('local-audit' if limit else 'audit');out.mkdir(parents=True,exist_ok=False)
    rows=[];volumes=[];sources={r['patient_id']:r for r in csv.DictReader((prepared/'sources/source_manifest.csv').open())}
    index={r['sample_id']:r for r in csv.DictReader((dataset/'patch_index.csv').open())}
    qcrows={r['subject']:r for r in qc['inventory']['subjects']}
    # No test images/targets are opened. Test membership alone is checked above.
    for source in sources.values():
        if source['patch_split']=='test':continue
        print('Checking SWC/graph geometry',source['subject'],flush=True)
        marker=json.loads((prepared/'volumes'/source['subject']/'complete.json').read_text())
        assert marker['completed'] and bool(marker.get('reflect_x',False))==(source['reflect_x']=='True')
        assert sha(Path(source['graph'])/'graph.vvg')==manifest['patients'][source['patient_id']]['graph_sha256']
        original=qcrows[source['subject']];a,_=read_swc(original['primary_swc'])
        q=swc_to_voxel(a[:,2:5],original['shape'],original['spacing'],reflect_x=original.get('reflect_x',False))
        expected=q@np.array(original['affine'])[:3,:3].T+np.array(original['affine'])[:3,3]
        base=prepared/'volumes'/source['subject']/'graphs'/marker['primary_variant']
        dense,positions,edges=source_graph(base/'dense');assert cKDTree(positions).query(expected)[0].max(initial=0)<1e-3
        allowed=adjacency(positions,edges);betti=graph_betti_numbers(len(positions),edges)
        for representation in ['dense','junction_only','adaptive']:
            g,n,e=source_graph(base/representation)
            assert graph_betti_numbers(len(n),e)==betti, ('Representation topology changed',source['subject'],representation)
            for line in g.edge_polylines:
                assert adjacency(line,np.column_stack([np.arange(len(line)-1),np.arange(1,len(line))]))<=allowed, 'Guide contains a segment absent from original SWC'
            volumes.append(dict(subject=source['subject'],split=source['patch_split'],representation=representation,nodes=len(n),edges=len(e),components=betti[0],cycles=betti[1]))
    for split in ['train','val']:
        names=p['smoke_ids'][split] if limit else p['ids'][split]
        for number,name in enumerate(names,1):
            gt=dataset/split/'spline'/(name+'.npz');record=index[name];source=sources[record['patient_id']]
            for directory,suffix in [('raw','_data.nii.gz'),('seg','_seg.nii.gz'),('vtp','_graph.vtp')]:
                # Discover graph naming from the native loader, which also checks caches.
                if directory=='vtp':assert (dataset/split/directory/(name+suffix)).is_file()
                else:assert (dataset/split/directory/(name+suffix)).is_file()
            with np.load(gt) as f:
                assert all(np.isfinite(f[k]).all() for k in ['nodes','controls','curves','guides','metric','full_paths'])
                n=f['nodes'];e=f['edges'];assert len(n)<=256 and f['controls'].shape==(len(e),8,3)
                assert len(f['path_offsets'])==len(e)+1 and f['path_offsets'][-1]==len(f['full_paths'])
                np.testing.assert_allclose(f['metric'],np.array(manifest['patients'][record['patient_id']]['affine'])[:3,:3]@np.diag([64]*3),atol=1e-4)
                for edge,(lo,hi) in zip(e,zip(f['path_offsets'][:-1],f['path_offsets'][1:])):
                    np.testing.assert_allclose(f['full_paths'][[lo,hi-1]],n[edge],atol=2e-5)
                assert len({tuple(sorted(x)) for x in e})==len(e)
                rows.append(dict(split=split,sample_id=name,subject=source['subject'],nodes=len(n),edges=len(e),dense_guide_points=len(f['full_paths']),**dict(zip(['components','cycles'],graph_betti_numbers(len(n),e)))))
            if number%1000==0:print('audited',split,number,'/',len(names),flush=True)
    # Load actual samples: this exercises VTP/cache correspondence, shared rotations,
    # physical metric updates and intensity processing without regenerating data.
    from data.loaders import build_evaluation_loader,build_data_loaders
    from scripts.brava_graph_study import configuration
    cfg=configuration(p,'standard')
    val=build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=p['smoke_ids']['val'])
    for i in range(len(val.dataset)):assert tuple(val.dataset[i][0][0].shape)==(1,64,64,64)
    train,_=build_data_loaders(cfg)
    for i in range(2):assert tuple(train.dataset.datasets[0][i][0][0].shape)==(1,64,64,64)
    table(out/'patch-statistics.csv',rows);table(out/'volume-representations.csv',volumes)
    save(out/'summary.json',dict(completed=True,counts=manifest['counts'],subject_counts={s:len(v) for s,v in subject_sets.items()},
        audited_development_patches=len(rows),volume_representation_rows=len(volumes),test_images_or_targets_opened=False,
        mask_role='brain_only_never_vessel_support',source_topology_and_SWC_segment_identity=True,
        max_nodes=max(r['nodes'] for r in rows),mean_nodes=float(np.mean([r['nodes'] for r in rows])),mean_edges=float(np.mean([r['edges'] for r in rows]))))
    print(out/'summary.json',flush=True)


def reference_polylines(graph, nodes, edges):
    """Convert source geometry to the existing metric's ordered edge paths."""
    paths=[]
    for edge, curve in zip(edges, graph.edge_polylines):
        curve=np.asarray(curve)
        if len(curve)==1:
            # SourceGraph deduplicates coincident SWC parent/child coordinates.
            # Retain the annotated link; expose both endpoints to the strict metric.
            np.testing.assert_allclose(nodes[edge],np.repeat(curve,2,axis=0),atol=1e-9,rtol=0)
            curve=nodes[edge].copy()
        paths.append(curve)
    assert len(paths)==len(edges)
    return paths


def representation_analysis(root):
    from data.spline_targets import fit_spline
    from metrics.common_vessel_metrics import evaluate_common_vessels,aggregate_common_vessels
    p=json.loads((root/'plan.json').read_text());dataset=Path(p['dataset']);prepared=dataset.parent
    out=root/'representations';out.mkdir(parents=True,exist_ok=True);rows=[]
    sources=list(csv.DictReader((prepared/'sources/source_manifest.csv').open()))
    for source in sources:
        if source['patch_split']=='test':continue
        subject_result=out/(source['subject']+'-metrics.json')
        if subject_result.exists():
            cached=json.loads(subject_result.read_text())
            assert len(cached)==4 and all(r['subject']==source['subject'] for r in cached)
            rows.extend(cached);continue
        first=len(rows)
        marker=json.loads((prepared/'volumes'/source['subject']/'complete.json').read_text())
        base=prepared/'volumes'/source['subject']/'graphs'/marker['primary_variant']
        dense,n,e=source_graph(base/'dense');paths=reference_polylines(dense,n,e)
        offsets=np.r_[0,np.cumsum([len(x) for x in paths])]
        reference=out/(source['subject']+'-reference.npz')
        np.savez_compressed(reference,nodes=n,edges=e,metric=np.eye(3),full_paths=np.concatenate(paths),path_offsets=offsets)
        for representation in ['dense','junction_only','adaptive','junction_spline_k8']:
            g,nodes,edges=source_graph(base/('junction_only' if representation=='junction_spline_k8' else representation))
            prediction=dict(nodes=nodes,edges=edges)
            if representation=='junction_spline_k8':
                controls=[fit_spline(line,8,64)[0] if np.linalg.norm(np.diff(line,axis=0),axis=1).sum()>1e-10 else np.repeat(line[:1],8,axis=0) for line in g.edge_polylines]
                prediction.update(spline_controls=np.array(controls).reshape(-1,8,3),spline_family='bspline')
            result=evaluate_common_vessels(prediction,reference)
            rows.append(dict(subject=source['subject'],split=source['patch_split'],representation=representation,
                object_nodes=len(nodes),edges=len(edges),internal_geometry_controls=6*len(edges) if representation=='junction_spline_k8' else 0,**result))
        save(subject_result,rows[first:])
        print('GT representation analysis',source['subject'],flush=True)
    save(out/'per-subject.json',rows);table(out/'per-subject.csv',rows);summary=[]
    for split in ['train','val']:
        for name in ['dense','junction_only','adaptive','junction_spline_k8']:
            group=[r for r in rows if r['split']==split and r['representation']==name]
            summary.append(dict(split=split,representation=name,mean_object_nodes=float(np.mean([r['object_nodes'] for r in group])),
                mean_edges=float(np.mean([r['edges'] for r in group])),**aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith('common_')} for r in group])))
    save(out/'summary.json',dict(completed=True,comparisons=summary,learned_model_efficiency_claim=False,test_evaluated=False))


def checkpoint_audit(root):
    import gc
    import torch
    from models import build_model
    from models.checkpoint import load_checkpoint_file
    from scripts.brava_graph_study import configuration,detector_identity,SEED
    from configs import validate_config
    p=json.loads((root/'plan.json').read_text());torch.set_num_threads(1);rows=[]
    for name,item in p['checkpoints'].items():
        cp=load_checkpoint_file(item['path']);cfg=cp['config'];model=build_model(cfg);model.load_state_dict(cp['net'],strict=True)
        assert cfg['training']['input']=='image' and cfg['model']['spline']['enabled']
        assert cfg['model']['spline'].get('family','bspline')=='bspline' and cfg['model']['spline']['controls']==8
        assert not any('brava' in v['root'].lower() for v in cfg['data']['datasets'].values())
        rows.append(dict(name=name,path=item['path'],strict_load=True,object_queries=model.obj_token,spline=cfg['model']['spline'],
            refinement=cfg['model'].get('node_refinement'),node_threshold=cfg['evaluation'].get('node_threshold'),edge_threshold=cfg['evaluation'].get('edge_threshold'),
            source_datasets=cfg['data']['datasets'],spline_loss_weight=cfg['loss']['weights']['spline']))
        print(name,'strict CPU load passed; queries',model.obj_token,flush=True)
        del model,cp;gc.collect()
    identities=[]
    for arm in ['standard','coordinate_refinement']:
        cfg=configuration(p,arm);validate_config(cfg);torch.manual_seed(SEED);model=build_model(cfg);identities.append(detector_identity(model));del model;gc.collect()
    assert identities[0]==identities[1]
    save(root/'checkpoint-inventory.json',dict(completed=True,models=rows,shared_fresh_state=identities[0]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--limit',action='store_true');p.add_argument('--representations',action='store_true');p.add_argument('--checkpoints',action='store_true');a=p.parse_args()
    if a.checkpoints:checkpoint_audit(a.root)
    elif a.representations:representation_analysis(a.root)
    else:audit(a.root,a.limit)
