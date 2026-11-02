#!/usr/bin/env python3
"""Native 2D long-study manifests; originals/test arrays are never modified."""
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from native2d.targets import from_dense_graph,crop_graph,save_target

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native2d_long_20261028'
DATA=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets')

def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def stats(rows):
    counts=np.array([r['nodes'] for r in rows if r['split']=='train'])
    return dict(samples={s:sum(r['split']==s for r in rows) for s in ['train','validation']},
        training_node_quantiles=dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(counts,[0,.5,.9,.95,.99,1]).tolist())),
        maximum_nodes=int(max(r['nodes'] for r in rows)),training_maximum_nodes=int(counts.max()))

def linked(name,source,units):
    directory=OUT/name;directory.mkdir(exist_ok=True)
    manifest=json.loads(source.read_text());rows=[]
    for row in manifest['samples']:
        r=dict(row)
        for key in ['image','target']:
            p=source.parent/r[key]
            if not p.exists() and ROOT.name in p.parts:
                p=ROOT/Path(*p.parts[p.parts.index(ROOT.name)+1:])
            assert p.is_file(),p
            r[key]=str(p.resolve())
        rows.append(r)
    write(directory/'manifest.json',dict(dataset=name,units=units,split_scope=manifest.get('split_scope','source-image disjoint; patient identity unknown'),samples=rows,source_manifest=str(source),no_test=True))
    write(directory/'audit.json',dict(passed=True,source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),units=units,**stats(rows)))

def edges_from_vtk(lines):
    edges=[];i=0
    while i<len(lines):
        n=int(lines[i]);p=lines[i+1:i+1+n];edges.extend(zip(p[:-1],p[1:]));i+=n+1
    return np.asarray(edges,int).reshape(-1,2)

def plant_edges(lines):
    # VTP exports sometimes repeat the exact same two-node segment. The
    # supervised relation is binary undirected adjacency, not record multiplicity.
    # Paths through distinct interior nodes are retained as distinct paths.
    raw=edges_from_vtk(lines)
    unique=np.unique(np.sort(raw,axis=1),axis=0)
    return unique,len(raw)-len(unique)

def encode(graph):
    assert np.isfinite(graph.nodes).all() and ((graph.nodes>=-1e-7)&(graph.nodes<=127+1e-7)).all()
    stream=io.BytesIO();save_target(stream,graph,[128,128]);data=stream.getvalue()
    with np.load(io.BytesIO(data)) as z:
        assert np.isfinite(z['controls']).all()
        if len(z['controls']):
            assert np.abs(z['controls']@z['metric'].T).max()<1024
            np.testing.assert_allclose(z['controls'][:,[0,-1]],z['nodes'][z['edges']],atol=1e-7)
    return data

def plants():
    directory=OUT/'plants';directory.mkdir(exist_ok=True)
    if (directory/'manifest.json').exists():return
    source=DATA/'plants_3d2cut';patches=source/'patches';membership={r['sample_id']:r['split'] for r in csv.DictReader((source/'splits.csv').open())}
    parents={s:{re.sub(r'_\d+$','',p.stem) for p in (patches/s/'raw').glob('*.png')} for s in ['train','val','test']}
    assert all(not parents[a]&parents[b] for a,b in [('train','val'),('train','test'),('val','test')])
    rows=[];alignment=[];duplicates=[];rejected=[];inspected=0;archive=directory/'targets.zip.tmp'
    with ZipFile(archive,'w',ZIP_STORED) as packed:
        for split in ['train','val']:
            for path in sorted((patches/split/'raw').glob('*.png')):
                inspected+=1
                parent=re.sub(r'_\d+$','',path.stem);assert membership[parent]==split
                graph_path=patches/split/'graphs'/(path.stem+'_gt_graph.vtp');mask_path=patches/split/'seg'/(path.stem+'_seg.png')
                assert graph_path.stat().st_size>0 and mask_path.stat().st_size>0
                with Image.open(path) as im:assert im.size==(128,128);im.verify()
                with Image.open(mask_path) as mask:assert mask.size==(128,128)
                mesh=pv.read(graph_path);points=np.asarray(mesh.points,float);assert np.allclose(points[:,2],0)
                assert np.isfinite(points).all()
                if len(points):assert points[:,:2].min()>=-1e-6 and points[:,:2].max()<=1+1e-6
                nodes=points[:,:2]*128;edges,repeated=plant_edges(mesh.lines)
                if repeated:duplicates.append(dict(sample=path.stem,repeated_identical_segments=repeated))
                if any(a==b for a,b in edges.tolist()):
                    rejected.append(dict(sample=path.stem,split=split,source=str(graph_path),reason='zero-length self-loop segment'));continue
                graph=from_dense_graph(nodes,edges,provenance=dict(source=str(graph_path),coordinate_order='normalized XY, confirmed by paired segmentation; no unconditional legacy swap',annotation='supplied sparse plant graph; piecewise-linear guides only'))
                graph=crop_graph(graph,[0,0],[128,128]);entry=split+'_'+path.stem+'.npz'
                try:encoded=encode(graph)
                except ValueError as error:
                    if str(error)!='Cannot fit a zero-length branch':raise
                    rejected.append(dict(sample=path.stem,split=split,source=str(graph_path),reason=str(error)));continue
                packed.writestr(entry,encoded)
                if len(rows)%256==0 and len(nodes):
                    mask=np.array(Image.open(mask_path));mask=mask.max(-1) if mask.ndim==3 else mask
                    dt=distance_transform_edt(mask==0);xy=np.clip(np.rint(nodes).astype(int),0,127)
                    alignment.append(dict(sample=path.stem,xy_mean_distance=float(dt[xy[:,1],xy[:,0]].mean()),swapped_mean_distance=float(dt[xy[:,0],xy[:,1]].mean())))
                rows.append(dict(sample_id='plants_'+path.stem,subject_id='plants_'+parent,split='train' if split=='train' else 'validation',image=str(path),target=entry,target_archive='targets.zip',nodes=len(graph.nodes),edges=len(graph.edges),units='pixel',source_graph=str(graph_path)))
                if len(rows)%1000==0:print('PLANTS '+str(len(rows)),flush=True)
    write(directory/'quarantine.json',dict(inspected=inspected,rejected=rejected,rule='exclude only zero-length branch/self-loop crops; no node merging; stop source if >5 percent excluded',fraction=len(rejected)/inspected))
    assert len(rows)+len(rejected)==32344
    if len(rejected)/inspected>.05:
        write(directory/'readiness.json',dict(passed=False,reason='zero-length annotation quarantine exceeds predefined 5 percent source gate',fraction=len(rejected)/inspected));return
    assert np.median([r['xy_mean_distance'] for r in alignment])<2
    archive.rename(directory/'targets.zip')
    write(directory/'manifest.json',dict(dataset='plants',units='pixel',samples=rows,no_test=True,split_scope='official source-image membership; all train/val/test parent IDs disjoint; plant-individual IDs unavailable',geometry_provenance='contracted supplied VTP segments, no invented dense manual centerline; uniform64 K8 fitting; clip128 boundary to127 pixel centers'))
    write(directory/'audit.json',dict(passed=True,**stats(rows),parents={s:len(v) for s,v in parents.items()},alignment=alignment,original_split_csv_sha256=hashlib.sha256((source/'splits.csv').read_bytes()).hexdigest(),coordinate_order='XY',annotation='plant graph labels; segmentation used for QC only',duplicate_segment_records=duplicates,edge_policy='binary undirected adjacency; exact repeated segment records collapsed, distinct paths retained',quarantined_crops=len(rejected),quarantine_fraction=len(rejected)/inspected,quarantine_manifest='quarantine.json'))

def csv_graph(folder):
    node_rows=list(csv.DictReader((folder/'nodes.csv').open()));edge_rows=list(csv.DictReader((folder/'edges.csv').open()))
    mapping={int(r['id']):i for i,r in enumerate(node_rows)}
    xyz=np.array([[float(r[k]) for k in ['x','y','z']] for r in node_rows])
    edges=np.array([[mapping[int(r['node1_id'])],mapping[int(r['node2_id'])]] for r in edge_rows],int).reshape(-1,2)
    assert all(a!=b for a,b in edges.tolist()) and np.isfinite(xyz).all()
    assert len(set(tuple(sorted(e)) for e in edges.tolist()))==len(edges)
    return xyz,edges

def octa():
    source=DATA/'octa-synth-packed';directory=OUT/'octasynth';directory.mkdir(exist_ok=True)
    if (directory/'manifest.json').exists():return
    if not (source/'splits.csv').exists():
        write(directory/'readiness.json',dict(passed=False,reason='awaiting octa-synth-packed upload'));return
    split_rows=list(csv.DictReader((source/'splits.csv').open()));rows=[];alignment=[];parents={'train':[],'val':[],'test':[]}
    labels={p.name:p for p in (source/'labels').glob('*.png')}
    with ZipFile(directory/'targets.zip.tmp','w',ZIP_STORED) as packed:
        for row in split_rows:
            split=row['split'];name=row['patient_id'];parent=name.removesuffix('data.png');parents[split].append(parent)
            if split=='test':continue
            path=source/'raw'/name;assert path.stat().st_size>0
            with Image.open(path) as im:image=np.array(im.convert('RGB'))
            height,width=image.shape[:2];xyz,edges=csv_graph(source/'graphs'/parent)
            # Source CSV x/y are row/column for these rendered images. Paired
            # masks resolve this convention: native XY=(CSV y,CSV x). Small
            # out-of-FOV nodes are clipped, never clamped into false junctions.
            nodes=xyz[:,[1,0]]*[width,height]
            graph=from_dense_graph(nodes,edges,provenance=dict(source=str(source/'graphs'/parent),dataset='OCTA-Synth',geometry='native XY=(CSV y,CSV x), verified against paired label; retain adjacency across crossings; z and radius not predicted',units='pixel'))
            # Resolve the actual paired label name, never use it as image input.
            candidates=[p for n,p in labels.items() if n.startswith(parent)]
            assert len(candidates)==1,(parent,candidates)
            mask=np.array(Image.open(candidates[0]));mask=mask.max(-1) if mask.ndim==3 else mask;assert mask.shape==(height,width)
            dt=distance_transform_edt(mask==0);xy=np.clip(np.rint(nodes).astype(int),[0,0],[width-1,height-1])
            alignment.append(dict(parent=parent,xy_mean_distance=float(dt[xy[:,1],xy[:,0]].mean()),uncorrected_mean_distance=float(dt[np.clip(xy[:,0],0,height-1),np.clip(xy[:,1],0,width-1)].mean()),outside_nodes=int(((nodes<0)|(nodes>np.array([width-1,height-1]))).any(1).sum()),shape=[height,width],z_min=xyz[:,2].min().item(),z_max=xyz[:,2].max().item()))
            for y in sorted(set([*range(0,height-127,128),height-128])):
                for x in sorted(set([*range(0,width-127,128),width-128])):
                    patch=crop_graph(graph,[x,y],[128,128]);key=parent+f'_x{x}_y{y}';entry=key+'.npz'
                    image_path=directory/(key+'.png');Image.fromarray(image[y:y+128,x:x+128]).save(image_path);packed.writestr(entry,encode(patch))
                    rows.append(dict(sample_id='octa_'+key,subject_id='octa_'+parent,split='train' if split=='train' else 'validation',image=str(image_path),target=entry,target_archive='targets.zip',nodes=len(patch.nodes),edges=len(patch.edges),units='pixel',source=str(path),crop_start_xy=[x,y]))
            print('OCTA '+parent,flush=True)
    assert all(not set(parents[a])&set(parents[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    assert np.median([a['xy_mean_distance'] for a in alignment])<2
    (directory/'targets.zip.tmp').rename(directory/'targets.zip')
    write(directory/'manifest.json',dict(dataset='octasynth',units='pixel',samples=rows,no_test=True,split_scope='official splits.csv source-image groups; synthetic subject independence beyond those IDs unknown',preprocessing='new deterministic128 grid from originals; correct outside/outside and entering/exiting dense-path clipping; original uploaded patches not silently reused',source=str(source)))
    write(directory/'audit.json',dict(passed=True,**stats(rows),parents={s:len(p) for s,p in parents.items()},alignment=alignment,split_csv_sha256=hashlib.sha256((source/'splits.csv').read_bytes()).hexdigest(),units='pixel; physical calibration not supplied',annotation='supplied projected synthetic vascular graph, retain adjacency; not independently hand-traced skeleton'))

def plan():
    sources={}
    for name in ['avwide','reaver','plants','octasynth']:
        p=OUT/name/'audit.json'
        if p.exists():sources[name]=json.loads(p.read_text())
    q=max(96,int(math.ceil(max(s['training_maximum_nodes'] for s in sources.values())*1.25/8)*8))
    assert all(s['maximum_nodes']<=q for s in sources.values()),'Validation exceeds training-derived shared capacity'
    write(OUT/'plan.json',dict(seed=364505,common_queries=q,batch_size=4,plant_batch_size=32,plant_epochs=50,vessel_epochs=100,thresholds=dict(node=.5,edge=.5),sources=sources,plant_readiness=json.loads((OUT/'plants/readiness.json').read_text()) if 'plants' not in sources else dict(passed=True),query_policy='shared capacity >=96, derived from maximum contracted training GT across available source/target datasets; transfers all query slots unchanged',design='B0/Axyz scratch plus B0/Axyz initialized from one common final50-epoch B0 plant checkpoint if source passes; target finetuning has no plant samples',stopping='fixed epochs; stop on nonfinite/error; no automatic extension, sweep or threshold change'))

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    old=ROOT/'experiments/native2d_20261028_targetfix'
    linked('avwide',old/'manifest.json','pixel');linked('reaver',old/'reaver-manifest.json','micron')
    octa();plants();plan()
