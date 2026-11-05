#!/usr/bin/env python3
"""Shared junction/end/boundary graph and eight-control targets from full paths.

Preserve parent-volume splits. Select a fixed hash-ranked compatible population;
record all topology/capacity exclusions. Original images and graphs remain intact.
"""
import argparse,csv,hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import pyvista as pv
from data.spline_targets import crop_dense_paths,contract_spline_graph,fit_spline,resample
from scripts.audit_synthetic_mri_grid import SourceGraph,patch_voxel_cell_bounds


def rank(row):
    return hashlib.sha256(('364505/'+row['split']+'/'+row['sample_id']).encode()).digest()


def _betti(n, links):
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in links:parent[root(a)]=root(b)
    b0=len({root(i) for i in range(n)});return b0,len(links)-n+b0


def _check_lossless(dense,anchor_ids,links,paths_mm,metric):
    """Subdivision must keep beta_0, beta_1 and total centreline length of the cropped dense graph."""
    edges=dense.edges[dense.edges[:,0]!=dense.edges[:,1]]
    used=np.unique(edges) if len(edges) else np.zeros(0,int)
    # dense isolated points are anchors too (contractor keeps them): compare on the same vertex set
    iso=sorted(set(range(len(dense.positions)))-set(used.tolist()))
    remap={k:i for i,k in enumerate(used.tolist()+iso)}
    want=_betti(len(remap),[(remap[a],remap[b]) for a,b in edges.tolist()])
    got=_betti(len(anchor_ids),links)
    dense_len=float(np.linalg.norm(np.diff(dense.positions[edges]@metric.T,axis=1),axis=-1).sum()) if len(edges) else 0.
    graph_len=float(sum(np.linalg.norm(np.diff(p,axis=0),axis=1).sum() for p in paths_mm))
    if want!=got or abs(dense_len-graph_len)>1e-6*max(1.,dense_len):
        raise AssertionError(f'subdivision not lossless: betti {want} vs {got}, length {dense_len:.6f} vs {graph_len:.6f}')


def prepare(source, patches, output, limits, *, smoke=False, subdivide=False):
    config=json.loads((patches/'generation_config.json').read_text())
    assert config['patch_size']==[64,64,64]
    output.mkdir(parents=True,exist_ok=False)
    graph_root=source/'graphs'
    if (graph_root/'adaptive').is_dir():graph_root=graph_root/'adaptive'
    rows=[];missing_subjects={};source_counts={}
    for split in limits:
        index=patches/f'active_{split}_index.csv'
        if not index.exists():index=patches/'patch_index.csv'
        pool=[r for r in csv.DictReader(index.open()) if r['split']==split]
        source_counts[split]=len(pool)
        missing_subjects[split]=sorted({r['patient_id'] for r in pool if not all((graph_root/r['patient_id']/f).is_file() for f in ('nodes.csv','edges.csv','graph.vvg'))})
        if 'IXI' in source.parts and missing_subjects[split]:raise FileNotFoundError(f'Missing IXI graphs: {missing_subjects[split]}')
        pool=[r for r in pool if r['patient_id'] not in missing_subjects[split]]
        if limits[split] is not None:
            pool=sorted(pool,key=rank)[:max(limits[split]+100,int(limits[split]*(1.5 if 'IXI' in source.parts else 1.15))) if not smoke else 8]
        rows.extend(pool)
    subjects={s:{r['patient_id'] for r in rows if r['split']==s} for s in limits}
    assert all(not subjects[a]&subjects[b] for a in subjects for b in subjects if a<b)
    groups=defaultdict(list)
    for row in rows:groups[row['patient_id']].append(row)
    accepted=[];excluded=[];patients={};staging=output/'candidates'
    for patient,group in sorted(groups.items()):
        image=nib.load(source/'seg'/f'{patient}.nii.gz');affine=image.affine
        assert image.header.get_xyzt_units()[0]=='mm'
        graphdir=graph_root/patient
        full=SourceGraph.from_directory(graphdir).transformed(np.linalg.inv(affine))
        shape=np.asarray(config['patch_size']);pad=np.asarray(config['pad']);crop=np.asarray(config['crop_size'])
        metric=affine[:3,:3]@np.diag(shape);inverse=np.linalg.inv(metric)
        patients[patient]=dict(affine=affine.tolist(),graph_sha256=hashlib.sha256((graphdir/'graph.vvg').read_bytes()).hexdigest())
        for row in group:
            name,split=row['sample_id'],row['split'];start=np.array([int(row['start_'+a]) for a in ('d','h','w')])
            dense=crop_dense_paths(full,patch_voxel_cell_bounds(start,crop),retain_tangent_contacts=True,boundary_policy='last_inside_sample')
            graph,reason=contract_spline_graph((dense.positions-start+pad)/shape,dense.edges,max_anchors=256,closed_components=dense.closed_components,preserve_nodes=np.flatnonzero(dense.boundary),subdivide=subdivide)
            if reason:
                excluded.append(dict(sample_id=name,split=split,reason=reason));continue
            ids=sorted(graph.anchors);mapping={k:i for i,k in enumerate(ids)}
            inserted=np.array([k in getattr(graph,'inserted',()) for k in ids],dtype=bool)
            boundary=np.array([bool(dense.boundary[k]) if 0<=k<len(dense.boundary) else False for k in ids],dtype=bool)
            nodes=np.asarray([graph.anchors[i] for i in ids]).reshape(-1,3)
            links=[];controls=[];curves=[];guides=[];paths=[];offsets=[0]
            for branch in graph.branches:
                path=branch.polyline@metric.T;a,b=branch.start,branch.end
                if tuple(path[0])>tuple(path[-1]):path=path[::-1];a,b=b,a
                if np.linalg.norm(np.diff(path,axis=0),axis=1).sum()<1e-10:
                    reason='degenerate_branch';break
                c,curve=fit_spline(path,8,64)
                controls.append(c@inverse.T);curves.append(curve@inverse.T);guides.append(resample(path,64)@inverse.T)
                links.append((mapping[a],mapping[b]));paths.append(path@inverse.T);offsets.append(offsets[-1]+len(path))
            if reason:
                excluded.append(dict(sample_id=name,split=split,reason=reason));continue
            edges=np.asarray(links,dtype=np.int64).reshape(-1,2)
            if subdivide:
                try:_check_lossless(dense,ids,[tuple(e) for e in edges.tolist()],[p@metric.T for p in paths],affine[:3,:3])
                except AssertionError as error:
                    print(name,error,flush=True);excluded.append(dict(sample_id=name,split=split,reason='lossless_check_failed'));continue
            leaf=staging/split
            for folder in ('raw','seg','vtp','spline'):(leaf/folder).mkdir(parents=True,exist_ok=True)
            for folder,suffix in (('raw','data'),('seg','seg')):
                src=patches/split/folder/f'{name}_{suffix}.nii.gz'
                assert nib.load(src).shape==(64,64,64)
                (leaf/folder/src.name).symlink_to(src.resolve())
            mesh=pv.PolyData(nodes)
            if len(edges):mesh.lines=np.column_stack((np.full(len(edges),2),edges)).ravel()
            mesh.save(leaf/'vtp'/f'{name}_graph.vtp')
            np.savez_compressed(leaf/'spline'/f'{name}.npz',nodes=nodes.astype('float32'),edges=edges,
                controls=np.asarray(controls,dtype='float32').reshape(-1,8,3),curves=np.asarray(curves,dtype='float32').reshape(-1,64,3),
                guides=np.asarray(guides,dtype='float32').reshape(-1,64,3),metric=metric.astype('float32'),
                full_paths=(np.concatenate(paths) if paths else np.empty((0,3))).astype('float32'),path_offsets=np.asarray(offsets),boundary=boundary,inserted=inserted)
            accepted.append(dict(row,anchor_count=len(nodes),branch_count=len(edges),inserted_count=int(inserted.sum())))
        print(f'{patient}: accepted {len(accepted)}, excluded {len(excluded)}',flush=True)
    selected=[]
    for split,limit in limits.items():
        eligible=sorted([r for r in accepted if r['split']==split],key=rank)
        selected.extend(eligible if limit is None else eligible[:limit])
    counts=dict(Counter(r['split'] for r in selected))
    manifest=dict(source=str(source),patches=str(patches),requested=limits,counts=counts,seed=364505,patients=patients,
        attempted=len(rows),source_counts=source_counts,missing_graph_subjects=missing_subjects,excluded=excluded,exclusion_counts=dict(Counter(r['reason'] for r in excluded)),
        boundary_policy='last_inside_sample_voxel_cells_with_sparse_component_exact_fallback',
        guide='Extracted/generated centerline paths are uncertain geometric supervision, not perfect anatomy.',
        max_anchors=256,controls=8,loop_policy='subdivided: inserted degree-2 anchors (npz key inserted), no patch excluded for loops' if subdivide else 'patches with parallel/closed branches excluded',train_ids=sorted(r['sample_id'] for r in selected if r['split']=='train'),
        validation_ids=sorted(r['sample_id'] for r in selected if r['split']=='val'),test_ids=sorted(r['sample_id'] for r in selected if r['split']=='test'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for split,limit in limits.items():
        if not counts.get(split) or (limit is not None and counts[split]!=limit):raise ValueError(f'Insufficient compatible population: {counts}; no smaller fallback')
    for row in selected:
        name,split=row['sample_id'],row['split']
        for folder,suffix in [('raw','_data.nii.gz'),('seg','_seg.nii.gz'),('vtp','_graph.vtp'),('spline','.npz')]:
            dest=output/split/folder/f'{name}{suffix}';dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(staging/split/folder/dest.name)
    with (output/'patch_index.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(sorted(selected,key=lambda r:r['sample_id']))
    (output/'generation_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'source_manifest.json').write_text(json.dumps({'subjects':[dict(patient_id=k,image=str(source/'seg'/f'{k}.nii.gz')) for k in patients]},indent=2)+'\n')
    (output/'READY').write_text('complete\n');print(counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--patches',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--no-subdivide-loops',dest='subdivide_loops',action='store_false',help='legacy (caches before 5 Nov 2026): exclude patches with parallel/closed branches instead of subdividing them');p.add_argument('--subdivide-loops',dest='subdivide_loops',action='store_true',help='default since 5 Nov 2026: keep loop patches with breaking-point anchors');p.set_defaults(subdivide_loops=True);p.add_argument('--train-limit',type=int,default=10000);p.add_argument('--val-limit',type=int);p.add_argument('--test',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    limits={'train':2 if a.smoke else (a.train_limit if a.train_limit>0 else None),'val':2 if a.smoke else a.val_limit}  # --train-limit 0: all compatible patches
    if a.test:limits['test']=2 if a.smoke else None
    prepare(a.source,a.patches,a.output,limits,smoke=a.smoke,subdivide=a.subdivide_loops)
