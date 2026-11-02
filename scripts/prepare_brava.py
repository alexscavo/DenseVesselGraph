#!/usr/bin/env python3
"""Brava skull stripping, SWC graph import and IXI-compatible graph patches.

No vessel segmentations are created. The legacy seg slot carries the supplied
brain mask solely for loader compatibility; its semantic role is recorded.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scripts.ixi_vessel_graph import VesselGraph, _dense_branch_paths, optimal_keep_mask, write_source_graph

DEFAULT = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava')
TRACER_UNITS_PER_VOXEL = .62
ORIENTATION_CORRECTIONS = Path(__file__).resolve().parents[1]/'docs/research/evidence/brava_preparation/orientation-corrections.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(name):
    name=name.split('.')[0].removesuffix('_ColorCoded')
    m=re.fullmatch(r'(BG|BH|BI)0*(\d+)',name)
    return m[1]+f'{int(m[2]):04d}' if m else name


def inventory(source):
    corrections=json.loads(ORIENTATION_CORRECTIONS.read_text())['corrections']
    brains={p.name[:-7]:p for p in (source/'brains').glob('*.nii.gz')}
    masks={p.name.removeprefix('sub-').removesuffix('_space-MRA_desc-brain_mask.nii.gz'):p for p in (source/'mask_hull').glob('*.nii.gz')}
    assert brains.keys()==masks.keys(), 'Brain/mask inventory mismatch'
    swcs={}
    for p in sorted((source/'swc_files').glob('*.swc')):
        variants=swcs.setdefault(canonical(p.name),{})
        key='colorcoded' if '_ColorCoded' in p.name else 'original'
        assert key not in variants, ('Duplicate SWC subject/variant',p)
        variants[key]=str(p)
    assert set(brains)<=set(swcs), ('Missing traces',set(brains)-set(swcs))
    rows=[]
    invalid={}
    for subject,variants in swcs.items():
        for variant,path in variants.items():
            try:read_swc(path)
            except (ValueError,AssertionError) as error:invalid[path]=str(error)
    for i,name in enumerate(sorted(brains)):
        image=nib.load(brains[name]);mask=nib.load(masks[name])
        assert len(image.shape)==3 and image.shape==mask.shape
        np.testing.assert_allclose(image.affine,mask.affine,atol=1e-4)
        assert image.header.get_xyzt_units()[0]=='mm', (name,'Image units must establish millimetres')
        valid={v:p for v,p in swcs[name].items() if p not in invalid}
        assert valid, ('No complete valid SWC for image',name)
        rows.append(dict(index=i,patient_id=f'{i:06d}',subject=name,image=str(brains[name]),brain_mask=str(masks[name]),
            swcs=swcs[name],valid_swcs=valid,primary_swc=valid.get('original',valid.get('colorcoded')),
            shape=list(image.shape),affine=image.affine.tolist(),spacing=list(map(float,image.header.get_zooms()[:3])),
            reflect_x=bool(corrections.get(name,{}).get('reflect_x',False))))
        if name in corrections:
            for path in [str(brains[name]),str(masks[name]),*valid.values()]:
                assert corrections[name]['input_sha256'].get(path)==digest(Path(path)), ('Orientation correction source changed; review again',name,path)
    return dict(subjects=rows,swcs=swcs,orphan_swc_subjects=sorted(set(swcs)-set(brains)),
                image_subjects=len(rows),swc_subjects=len(swcs),swc_files=sum(map(len,swcs.values())),invalid_swcs=invalid)


def read_swc(path):
    a=np.loadtxt(path,comments='#',ndmin=2)
    assert a.shape[1]==7 and np.isfinite(a).all(), ('Invalid SWC',path)
    ids=a[:,0].astype(np.int64);parents=a[:,6].astype(np.int64)
    assert np.array_equal(ids,a[:,0]) and len(set(ids))==len(ids)
    assert np.array_equal(parents,a[:,6]) and (a[:,5]>=0).all()
    lookup={n:i for i,n in enumerate(ids)}; edges=[]
    for i,parent in enumerate(parents):
        if parent==-1:continue
        assert parent in lookup and lookup[parent]!=i, ('Invalid parent',path,parent)
        edges.append((lookup[parent],i))
    edges=np.asarray(edges,dtype=np.int64).reshape(-1,2)
    degree=np.bincount(edges.ravel(),minlength=len(a))
    graph=VesselGraph(a[:,2:5].copy(),degree,list(map(tuple,edges)),[a[[i,j],2:5].copy() for i,j in edges],a[:,5].copy())
    assert graph.betti()[1]==0, ('Cyclic parent links in SWC',path)
    return a,edges


def swc_to_voxel(points, shape, spacing, *, reflect_x=False):
    # Tracer coordinates use a fixed .62-unit lattice, independently of NIfTI
    # spacing (e.g. BG0006 has .5625-mm in-plane pixels). Keep the spacing
    # argument for existing callers; only the NIfTI affine defines world units.
    q=points[:,[0,2,1]]/TRACER_UNITS_PER_VOXEL
    q[:,1]=shape[1]-1-q[:,1]
    q[:,2]=shape[2]-1+q[:,2]
    if reflect_x:q[:,0]=shape[0]-1-q[:,0]
    return q


def dense_graph(a,edges,voxels,spacing):
    degree=np.bincount(edges.ravel(),minlength=len(a))
    return VesselGraph(voxels,degree,list(map(tuple,edges)),[voxels[[i,j]] for i,j in edges],
        a[:,5]/TRACER_UNITS_PER_VOXEL,skeleton_voxels=len(a))


def compact_graph(dense,spacing,radii_mm,adaptive):
    paths=_dense_branch_paths(dense)
    anchors=sorted(set(np.flatnonzero(dense.node_degrees!=2))|{p[0] for p in paths}|{p[-1] for p in paths})
    mapping={v:i for i,v in enumerate(anchors)}
    positions=[dense.node_positions[v] for v in anchors];radii=[dense.node_radii[v] for v in anchors]
    edges=[];curves=[]
    for path in paths:
        p=dense.node_positions[path]
        keep=optimal_keep_mask(p*np.asarray(spacing),np.maximum(.1,.75*radii_mm[path])) if adaptive else np.arange(len(p))[[0,-1]]
        retained=np.flatnonzero(keep) if adaptive else keep
        previous=mapping[path[0]];begin=0
        for end in retained[1:]:
            if end==len(path)-1:node=mapping[path[-1]]
            else:
                node=len(positions);positions.append(p[end]);radii.append(dense.node_radii[path[int(end)]])
            edges.append((previous,node));curves.append(p[begin:end+1]);previous=node;begin=int(end)
    degree=np.bincount(np.asarray(edges,dtype=int).ravel(),minlength=len(positions))
    graph=VesselGraph(np.asarray(positions),degree,edges,curves,np.asarray(radii),skeleton_voxels=dense.skeleton_voxels)
    assert graph.betti()==dense.betti(), ('Topology changed on contraction',graph.betti(),dense.betti())
    return graph


def alignment(row, image, mask, a):
    xyz=a[:,2:5];spacing=np.asarray(row['spacing']);shape=np.asarray(row['shape'])
    fixed=swc_to_voxel(xyz,shape,spacing,reflect_x=row.get('reflect_x',False));alternatives=[]
    # Check all reflections around the same axis mapping against the fixed convention.
    for flips in itertools.product([False,True],repeat=3):
        base=xyz[:,[0,2,1]].copy();base[:,2]*=-1;base/=TRACER_UNITS_PER_VOXEL
        q=np.where(flips,shape-1-base,base)
        score=map_coordinates(image,q.T,order=1,mode='constant',cval=0)
        alternatives.append(dict(flips=list(flips),median=float(np.median(score)),mean=float(score.mean())))
    values=map_coordinates(image,fixed.T,order=1,mode='constant',cval=0)
    brain=map_coordinates(mask.astype('uint8'),fixed.T,order=0,mode='constant',cval=0)
    inside=((fixed>=-.5)&(fixed<=shape-.5)).all(1)
    selected=next(r for r in alternatives if r['flips']==[row.get('reflect_x',False),True,True])
    assert inside.mean()>=.99, (row['subject'],'SWC outside image',float(inside.mean()))
    assert brain.mean()>=.95, (row['subject'],'SWC poorly supported by brain mask',float(brain.mean()))
    assert selected['median']>=.95*max(r['median'] for r in alternatives), (row['subject'],'Fixed orientation not supported by image intensity')
    return dict(subject=row['subject'],inside_fraction=float(inside.mean()),brain_mask_fraction=float(brain.mean()),
        centerline_median=float(np.median(values)),reflect_x=row.get('reflect_x',False),reflections=alternatives)


def overlay(path,image,q):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for axis,ax in enumerate(axes):
        keep=[i for i in range(3) if i!=axis]
        mip=image.max(axis=axis);lo,hi=np.percentile(mip,[2,99.5])
        ax.imshow(mip.T,origin='lower',cmap='gray',vmin=lo,vmax=hi)
        ax.scatter(q[:,keep[0]],q[:,keep[1]],s=1,c='red',alpha=.35)
        ax.set(title=f'Projection along array axis {axis}',xlabel=f'Axis {keep[0]} voxels',ylabel=f'Axis {keep[1]} voxels')
    fig.savefig(path,dpi=160);plt.close(fig)


def audit(source,output):
    assert not (output/'audit/complete.json').exists(), 'Existing completed preparation: use its records, no duplicate run'
    manifest=inventory(source);save(output/'inventory.json',manifest)
    native=output/'native_swc';native.mkdir(parents=True,exist_ok=True)
    for subject,variants in manifest['swcs'].items():
        for variant,path in variants.items():
            if path in manifest['invalid_swcs']:
                save(native/(subject+'-'+variant+'-invalid.json'),dict(path=path,sha256=digest(Path(path)),
                    error=manifest['invalid_swcs'][path],action='Original preserved; no parent IDs or edges invented; use complete paired variant'))
                continue
            a,edges=read_swc(path)
            np.savez_compressed(native/(subject+'-'+variant+'.npz'),ids=a[:,0].astype('int64'),types=a[:,1].astype('int64'),
                coordinates_swc=a[:,2:5],radii_swc=a[:,5],parents=a[:,6].astype('int64'),edges=edges)
    checks=[];failures=[];directory=output/'audit';directory.mkdir(exist_ok=True)
    for row in manifest['subjects']:
        image=np.asarray(nib.load(row['image']).dataobj,dtype=np.float32)
        mask=np.asarray(nib.load(row['brain_mask']).dataobj)>0
        assert np.isfinite(image).all()
        # Inspect every supplied valid variant and every subject; one bad scan
        # must not hide other failures until the next expensive submission.
        for variant,path in row['valid_swcs'].items():
            a,_=read_swc(path)
            try:
                check=alignment(row,image,mask,a);checks.append(dict(check,variant=variant))
                print('alignment',row['subject'],variant,check['inside_fraction'],check['brain_mask_fraction'],flush=True)
            except AssertionError as error:
                failures.append(dict(subject=row['subject'],variant=variant,error=str(error)))
                print('alignment FAILED',failures[-1],flush=True)
        if row['subject'] in ['BG0001','BG0002','BH0003','BH0040'] or row.get('reflect_x',False):
            a,_=read_swc(row['primary_swc'])
            overlay(directory/(row['subject']+'-alignment.png'),image,swc_to_voxel(a[:,2:5],row['shape'],row['spacing'],reflect_x=row.get('reflect_x',False)))
        save(directory/'checks.json',dict(checks=checks,failures=failures))
    save(directory/'failures.json',failures)
    assert not failures, ('Alignment failed; all affected subjects/variants recorded in audit/failures.json',failures)
    save(directory/'complete.json',dict(completed=True,checks=checks,
        transform='voxel=[swc_x/0.62, Ny-1-swc_z/0.62, Nz-1+swc_y/0.62]; world=NIfTI_affine(voxel)',
        tracer_units_per_voxel=TRACER_UNITS_PER_VOXEL,
        orientation_corrections_sha256=digest(ORIENTATION_CORRECTIONS),
        coordinate_warning='SWC intrinsic coordinates are not NIfTI world coordinates; default transform plus explicit hash-verified subject x reflections, never automatic image-fitted transforms.',
        inventory=manifest))


def subject(output,index,base=None):
    audit=json.loads((output/'audit/complete.json').read_text());assert audit['completed']
    row=audit['inventory']['subjects'][index];base=base or output
    directory=base/'volumes'/row['subject']
    if (directory/'complete.json').exists():
        marker=json.loads((directory/'complete.json').read_text())
        assert marker['completed'] and marker['image']==row['image'] and marker['image_sha256']==digest(Path(row['image']))
        assert marker['brain_mask_sha256']==digest(Path(row['brain_mask']))
        assert marker.get('reflect_x',False)==row.get('reflect_x',False), 'Cached graph orientation differs; do not reuse'
        for variant,details in marker['swcs'].items():assert details['swc_sha256']==digest(Path(row['valid_swcs'][variant]))
        return directory
    directory.mkdir(parents=True,exist_ok=False)
    image=nib.load(row['image']);mask_image=nib.load(row['brain_mask']);mask=np.asarray(mask_image.dataobj)>0
    data=np.asarray(image.dataobj);stripped=np.where(mask,data,0)
    brain=nib.Nifti1Image(stripped,image.affine,image.header.copy());brain.header.set_xyzt_units('mm')
    nib.save(brain,directory/'brain_skullstripped.nii.gz')
    (directory/'brain_mask.nii.gz').symlink_to(Path(row['brain_mask']).resolve())
    (directory/'brain_original.nii.gz').symlink_to(Path(row['image']).resolve())
    details={}
    for variant,path in row['valid_swcs'].items():
        a,edges=read_swc(path);voxels=swc_to_voxel(a[:,2:5],row['shape'],row['spacing'],reflect_x=row.get('reflect_x',False))
        dense=dense_graph(a,edges,voxels,row['spacing'])
        # SWC scalar radii become voxel radii first; use mean physical spacing
        # for the scalar adaptive tolerance on slightly anisotropic voxels.
        physical_radii=a[:,5]/TRACER_UNITS_PER_VOXEL*float(np.mean(row['spacing']))
        representations=dict(dense=dense,junction_only=compact_graph(dense,row['spacing'],physical_radii,False),
                             adaptive=compact_graph(dense,row['spacing'],physical_radii,True))
        for name,graph in representations.items():write_source_graph(directory/'graphs'/variant/name,graph,image.affine,image.shape)
        details[variant]=dict(swc=path,swc_sha256=digest(Path(path)),
            representations={k:dict(nodes=g.node_count,edges=g.edge_count,betti=list(g.betti())) for k,g in representations.items()})
    assert np.count_nonzero(stripped[~mask])==0
    np.testing.assert_array_equal(stripped[mask],data[mask])
    save(directory/'complete.json',dict(completed=True,subject=row['subject'],patient_id=row['patient_id'],
        image=row['image'],image_sha256=digest(Path(row['image'])),brain_mask=row['brain_mask'],brain_mask_sha256=digest(Path(row['brain_mask'])),
        affine=row['affine'],shape=row['shape'],swcs=details,reflect_x=row.get('reflect_x',False),
        policy='SWC parent links preserved; no skeletonization, pruning, merging, smoothing or vessel-mask containment',
        primary_variant='original' if 'original' in row['valid_swcs'] else 'colorcoded'))
    return directory


def stage(output,base=None,indices=None):
    from scripts.prepare_real_graph_patch_sources import assign_splits
    audit=json.loads((output/'audit/complete.json').read_text());assert audit['completed']
    base=base or output;rows=audit['inventory']['subjects']
    if indices is not None:rows=[rows[i] for i in indices]
    source=base/'sources';source.mkdir(parents=True,exist_ok=False);records=[]
    for row in rows:
        folder=base/'volumes'/row['subject'];marker=json.loads((folder/'complete.json').read_text());assert marker['completed']
        records.append(dict(patient_id=row['patient_id'],dataset='brava',modality='mra',source_split='train',subject=row['subject'],
            image=str(folder/'brain_skullstripped.nii.gz'),original_image=row['image'],brain_mask=row['brain_mask'],
            graph=str(folder/'graphs'/marker['primary_variant']/'adaptive'),reflect_x=row.get('reflect_x',False)))
    assignment=assign_splits(records,42) if indices is None else {r['patient_id']:['train','val','test'][i] for i,r in enumerate(records)}
    for r in records:
        for name,origin in [('raw',r['image']),('seg',r['brain_mask']),('brain_mask',r['brain_mask']),('graphs/adaptive',r['graph'])]:
            directory=source/name;directory.mkdir(parents=True,exist_ok=True)
            suffix='.nii.gz' if name!='graphs/adaptive' else ''
            (directory/(r['patient_id']+suffix)).symlink_to(Path(origin).resolve())
        r['patch_split']=assignment[r['patient_id']]
    with (source/'source_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    save(source/'source_manifest.json',dict(subjects=records,segmentation_role='brain_mask_only_not_vessel_GT',
        graph_policy='SWC radius-adaptive geometry simplification: max(0.1mm,0.75*physical radius); scalar radius uses mean physical spacing/0.62; no lumen criterion; unsmoothed original guides'))
    with (source/'patient_split.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['patient_id','split']);writer.writeheader();writer.writerows(dict(patient_id=k,split=v) for k,v in assignment.items())
    return source


def patches(source,output,workers):
    subprocess.run([sys.executable,'scripts/generate_synthetic_mri_dataset.py','--root',str(source),'--output-dir',str(output),
        '--split-output',str(source/'patient_split.csv'),'--patch-size','64','64','64','--pad','0','0','0',
        '--maximum-stride','40','40','40','--workers',str(workers),'--resume','--allow-existing-split-proportions'],check=True)
    from scripts.curate_vessel_training_patches import curate
    curate(output,apply=True,all_splits=True)
    config=json.loads((output/'generation_config.json').read_text());config.update(segmentation_role='brain_mask_only_not_vessel_GT',image_variant='skull_stripped')
    save(output/'brava_semantics.json',dict(segmentation_role=config['segmentation_role'],vessel_segmentations_created=False,
        normalization='Same existing IXI normalization, including float/P99.5 fallback if legacy MAD is zero',
        note='seg/ is the brain mask required by the legacy loader. Do not use it for vessel support, vessel Dice or mask-input vessel oracles.'))


def targets(source,patches_root,output,smoke=False):
    from scripts.prepare_transfer_splines import prepare
    prepare(source,patches_root,output,{'train':2,'val':2,'test':2} if smoke else {'train':None,'val':None,'test':None},smoke=smoke)
    semantics=json.loads((patches_root/'brava_semantics.json').read_text())
    save(output/'brava_semantics.json',semantics)
    manifest=json.loads((output/'manifest.json').read_text());manifest['ground_truth_source']='provided SWC parent-link annotations, not segmentation-derived graphs'
    manifest['segmentation_role']=semantics['segmentation_role'];save(output/'manifest.json',manifest)


def smoke(output):
    rows=json.loads((output/'inventory.json').read_text())['subjects']
    indices=[0,next(i for i,r in enumerate(rows) if r['subject'].startswith('BH')),len(rows)-1]
    base=output/'smoke';base.mkdir(exist_ok=False)
    (base/'volumes').mkdir()
    for i in indices:
        folder=subject(output,i)
        (base/'volumes'/rows[i]['subject']).symlink_to(folder.resolve())
    source=stage(output,base,indices);patches(source,base/'patches',2)
    targets(source,base/'patches',base/'targets',True)
    # The actual shared target path above is the smoke, not a second graph implementation.
    m=json.loads((base/'targets/manifest.json').read_text());assert m['counts']=={'train':2,'val':2,'test':2}
    for split,names in [('train',m['train_ids']),('val',m['validation_ids']),('test',m['test_ids'])]:
        for name in names:
            with np.load(base/'targets'/split/'spline'/(name+'.npz')) as c:
                assert len(c['nodes'])<=256 and np.isfinite(c['controls']).all() and np.isfinite(c['metric']).all()
    save(base/'complete.json',dict(completed=True,subjects=indices,target_counts=m['counts']))


def collect(output):
    manifest=json.loads((output/'transfer_splines_k8/manifest.json').read_text());assert (output/'transfer_splines_k8/READY').exists()
    inventory=json.loads((output/'inventory.json').read_text());details=[]
    for row in inventory['subjects']:
        m=json.loads((output/'volumes'/row['subject']/'complete.json').read_text());assert m['completed'];details.append(m)
    summary=dict(completed=True,output=str(output),images=len(details),native_swc_subjects=inventory['swc_subjects'],
        native_swc_files=inventory['swc_files'],orphan_swc_subjects=inventory['orphan_swc_subjects'],
        patch_counts=manifest['counts'],exclusion_counts=manifest['exclusion_counts'],
        no_vessel_segmentations=True,segmentation_role='brain_mask_only_not_vessel_GT',
        limitation='SWC parent-link forests do not establish absence of anatomical vascular cycles; trace coverage may be incomplete.')
    save(output/'summary.json',summary)
    evidence=Path('docs/research/evidence/brava_preparation');evidence.mkdir(parents=True,exist_ok=True)
    for p in [output/'summary.json',output/'inventory.json',output/'audit/complete.json']:
        shutil.copy2(p,evidence/('alignment.json' if p.name=='complete.json' else p.name))
    for p in (output/'audit').glob('*.png'):shutil.copy2(p,evidence/p.name)
    text='\n## Completed preparation\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n'
    doc=Path('docs/research/BRAVA_preparation.md');doc.write_text(doc.read_text().split('\n## Completed preparation')[0]+text)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['inventory','audit','subject','smoke','stage','patches','targets','collect','show'])
    p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.mode=='inventory':print(json.dumps(inventory(a.source),indent=2))
    elif a.mode=='audit':audit(a.source,a.output)
    elif a.mode=='subject':subject(a.output,a.index)
    elif a.mode=='smoke':smoke(a.output)
    elif a.mode=='stage':stage(a.output)
    elif a.mode=='patches':patches(a.output/'sources',a.output/'patches',a.workers)
    elif a.mode=='targets':targets(a.output/'sources',a.output/'patches',a.output/'transfer_splines_k8')
    elif a.mode=='collect':collect(a.output)
    else:
        path=a.output/'summary.json'
        if not path.exists():raise SystemExit(f'Preparation pending; no waiting. Expected {path}')
        print(path.read_text())
