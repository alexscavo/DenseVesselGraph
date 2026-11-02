#!/usr/bin/env python3
"""Frozen image-disjoint AV-WIDE pilot and read-only secondary-source audit."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.io import loadmat, whosmat
from scipy.sparse.csgraph import connected_components
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from native2d.targets import from_dense_graph, from_mask, crop_graph, save_target

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/native2d_20261028'
DATA = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets')

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')

def topology(nodes, edges):
    n=len(nodes)
    a=sparse.coo_matrix((np.ones(2*len(edges)), (edges.ravel(), edges[:, ::-1].ravel())), shape=(n,n))
    c=connected_components(a, directed=False, return_labels=False) if n else 0
    return [int(c), int(len(edges)-n+c)]

def inspect_wide():
    prepared=OUT/'prepared'; prepared.mkdir(exist_ok=True)
    splits={'train':[1,*range(3,7),*range(8,20)], 'validation':[2,*range(20,25)], 'test':list(range(25,31)), 'excluded':[7]}
    write('splits.json', dict(images=splits, scope='source-image disjoint; patient/eye IDs unavailable',
        test_policy='25-30: MAT headers only, no array loading, crop generation or evaluation',
        reason='01 remains training and 02 validation as in previous development; numerical IDs otherwise fixed before outcomes'))
    samples=[]; audit=[]; panels=[]; hashes={}; header=[]
    for path in sorted((DATA/'AV Paper Online Files/AV-WIDE').glob('*.mat')):
        number=int(path.stem.rsplit('_',1)[1]); split=next(s for s,ids in splits.items() if number in ids)
        header.append(dict(file=path.name, variables=whosmat(path), split=split))
        if split=='test':continue
        raw=loadmat(path,simplify_cells=True); image=raw['I2']; g=raw['G']; nodes=np.asarray(g['V'],float)-1
        a=sparse.csr_matrix(g['A']); rows,cols=sparse.triu(a,k=1).nonzero(); edges=np.stack((rows,cols),1)
        assert (a!=a.T).nnz==0
        if split=='excluded':
            audit.append(dict(file=path.name,split=split,reason='diagonal self-loop without a geometric path',diagonal_node_ids=np.flatnonzero(a.diagonal()).tolist()))
            continue
        assert not a.diagonal().any()
        assert image.ndim==3 and image.shape[2]==3 and image.dtype==np.uint8
        assert np.isfinite(nodes).all()
        outside=int(((nodes<0)|(nodes>np.array(image.shape[:2])[::-1]-1)).any(1).sum())
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        assert digest not in hashes, 'Identical source images across split'
        hashes[digest]=path.name
        provenance=dict(dataset='AV-WIDE',source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            annotation='supplied AV-labelled dense graph; not independent manual centerline tracing',
            conversion='G.V XY MATLAB 1-based minus 1 exactly once',
            crossings='preserve G.A; never merge by proximity; no AV class prediction in this pilot', units='pixel')
        graph=from_dense_graph(nodes,edges,provenance=provenance)
        assert topology(nodes,edges)==topology(graph.nodes,graph.edges), 'Contraction changed topology'
        degree=np.bincount(edges.ravel(),minlength=len(nodes))
        unique=len(np.unique(nodes,axis=0))
        counts=[];boundary=0
        # A fixed spatial grid, with final row/column shifted to cover the image.
        # Keep ALL tiles, including empty targets. No graph-density selection.
        ys=sorted(set([*range(0,image.shape[0]-127,128),image.shape[0]-128]))
        xs=sorted(set([*range(0,image.shape[1]-127,128),image.shape[1]-128]))
        # Restrict paths by bounding boxes before exact per-segment clipping.
        bounds=np.array([[p.min(0),p.max(0)] for p in graph.paths])
        for y in ys:
            for x in xs:
                from native2d.targets import Graph2D
                keep=np.where(np.all(bounds[:,1]>=[x,y],axis=1)&np.all(bounds[:,0]<=[x+127,y+127],axis=1))[0]
                local=Graph2D(graph.nodes,graph.edges[keep],[graph.paths[i] for i in keep],graph.provenance)
                # Keep true isolates only: a reduced edge list would invent isolates.
                if len(keep)!=len(graph.edges):
                    used=np.unique(graph.edges[keep])
                    isolates=np.where(np.bincount(graph.edges.ravel(),minlength=len(graph.nodes))==0)[0]
                    used=np.union1d(used,isolates)
                    remap=np.full(len(graph.nodes),-1);remap[used]=np.arange(len(used))
                    local=Graph2D(graph.nodes[used],remap[graph.edges[keep]],[graph.paths[i] for i in keep],graph.provenance)
                patch=crop_graph(local,[x,y],[128,128])
                assert np.isfinite(patch.nodes).all() and ((patch.nodes>=-1e-7)&(patch.nodes<=127+1e-7)).all()
                for (u,v),p in zip(patch.edges,patch.paths):
                    np.testing.assert_allclose(p[[0,-1]],patch.nodes[[u,v]],atol=1e-6)
                boundary+=int(np.any(np.isclose(patch.nodes,0)|np.isclose(patch.nodes,127),axis=1).sum())
                key=f'wide{number:02d}_x{x}_y{y}'
                Image.fromarray(image[y:y+128,x:x+128]).save(prepared/f'{key}.png')
                save_target(prepared/f'{key}.npz',patch,[128,128])
                samples.append(dict(sample_id=key,subject_id=f'wide{number:02d}',subject_identity_scope='source image only',
                    split=split,image=f'prepared/{key}.png',target=f'prepared/{key}.npz',crop_start_xy=[x,y],
                    source=str(path),source_sha256=provenance['source_sha256'],units='pixel',nodes=len(patch.nodes),edges=len(patch.edges)))
                counts.append(len(patch.nodes))
        entry=dict(file=path.name,split=split,image_shape=list(image.shape),dense_nodes=len(nodes),dense_edges=len(edges),
            contracted_nodes=len(graph.nodes),contracted_edges=len(graph.edges),topology=topology(graph.nodes,graph.edges),
            duplicate_coordinate_nodes=len(nodes)-unique,degree4plus=int((degree>=4).sum()),outside_source_extent_nodes=outside,
            graph_fields=list(g),tiles=len(counts),tile_nodes_min=int(min(counts)),tile_nodes_max=int(max(counts)),boundary_nodes=boundary,
            subject_metadata='No patient/eye identifier in supplied README or graph fields')
        audit.append(entry)
        # Full-source overlay records correspondence and projected crossing connectivity.
        fig,ax=plt.subplots(figsize=(10,6));ax.imshow(image)
        for p in graph.paths:ax.plot(*p.T,color='cyan',lw=.45,alpha=.8)
        ax.scatter(*graph.nodes.T,s=2,c='magenta');ax.set_title(path.name+' '+split);ax.axis('off')
        fig.tight_layout();fig.savefig(OUT/f'overlay-wide{number:02d}.jpg',dpi=130);plt.close(fig)
        panels.append((path.name,image,graph))
        print(json.dumps(entry),flush=True)
    values=np.array([r['nodes'] for r in samples if r['split']=='train'])
    # Capacity derives from training GT only; validation is checked for overflow.
    queries=max(16,int(np.ceil(values.max()*1.25/8)*8))
    assert max(r['nodes'] for r in samples)<=queries, 'Validation exceeds training-derived capacity'
    write('manifest.json',dict(protocol='native2d-wide-image-disjoint-v1',samples=samples,no_test_evaluation=True,
        provenance='supplied G.A/G.V; dense degree-2 contraction; exact boundary clipping; no crossing reconnection'))
    write('av-wide-audit.json',dict(headers=header,images=audit,passed=True,scope='23 development images fully audited; image07 excluded; 6 test headers only',
        training_node_quantiles=dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(values,[0,.5,.9,.95,.99,1]).tolist())),
        queries=queries,samples={s:sum(r['split']==s for r in samples) for s in ['train','validation']},
        limitations=['patient independence unknown','graph labels are supplied AV-analysis topology, not new independent tracing',
            'image-level metrics are crop macro/micro; full mosaic reconstruction not evaluated','unknown physical calibration: pixels only']))

def inspect_secondary():
    result={}
    av=DATA/'AV Paper Online Files'
    for name,folder in [('AV-DRIVE',av/'AV-DRIVE/training'),('AV-INSPIRE',av/'AV-INSPIRE')]:
        files=sorted(folder.glob('*.mat'));details=[]
        for f in files:
            d=loadmat(f,simplify_cells=True)
            details.append(dict(file=f.name,variables=whosmat(f),graph_fields=list(d['G'])))
        result[name]=dict(files=details,readiness='graphs only; paired original images excluded by redistribution terms; blocked')
    reaver=DATA/'REAVER_Vascular_Networks_Image_Dataset'
    with ZipFile(reaver/'REAVER Dataset Summary.docx') as z:
        description=''.join(ET.fromstring(z.read('word/document.xml')).itertext())
    (OUT/'reaver-description.txt').write_text(description)
    manual=reaver/'ImageJ_Manual'
    if not manual.exists():
        manual=reaver/'Manual'
        assert 'ImageJ_Manual: contains output images from manual analysis' in description
    rows=list(csv.DictReader((reaver/'image_quantification.csv').open()));items=[]
    results=loadmat(manual/'Results.mat',simplify_cells=True)
    for i,path in enumerate(sorted((reaver/'_Original_Images').glob('*.tif'))):
        original=tifffile.imread(path)
        with tifffile.TiffFile(manual/path.name) as tf:
            rgb=tf.asarray();meta=tf.imagej_metadata;tags=tf.pages[0].tags
            spacing=np.array([tags[k].value[1]/tags[k].value[0] for k in ['XResolution','YResolution']])
        raw_parity=np.array_equal(original,rgb[:,:,2]);binary=all(set(np.unique(rgb[:,:,j]))<={0,255} for j in [0,1])
        csv_spacing=[float(r['umppix']) for r in rows if r['Image_Name']==path.name]
        meta=meta or {}
        # Some exported TIFFs have display DPI, not a physical ImageJ unit.
        # In that case use the analysis CSV/MAT only when they agree by name.
        csv_scale=float(csv_spacing[0]);mat_scale=float(np.asarray(results['umppix']).ravel()[i])
        assert np.allclose(csv_spacing,csv_scale,rtol=1e-6) and np.isclose(mat_scale,csv_scale,rtol=1e-6)
        calibrated_tiff=meta.get('unit')=='micron'
        if calibrated_tiff:assert np.allclose(spacing,csv_scale,rtol=1e-6)
        else:spacing=np.array([csv_scale,csv_scale])
        valid=raw_parity and binary
        # Nonbinary manual skeletons are recorded as blocked, never thresholded
        # silently. Red-channel nonbinary exports also need provenance review.
        graph=from_mask(rgb[:,:,1]>0,is_centerline=True,provenance=dict(source=str(manual/path.name))) if valid else None
        item=dict(file=path.name,shape=list(original.shape),manual_folder=str(manual),raw_blue_exact=raw_parity,
            binary_red_green=binary,spacing_micron_xy=spacing.tolist(),csv_and_mat_scale_parity=True,
            tiff_physical_unit_available=calibrated_tiff,scale_source='TIFF + CSV + Results.mat' if calibrated_tiff else 'CSV + Results.mat; TIFF resolution is display DPI',
            strict_pair_gate=valid,raw_blue_difference_fraction=float((image_difference:=original!=rgb[:,:,2]).mean()),
            channel_values_rg=[np.unique(rgb[:,:,j]).tolist() for j in [0,1]],
            skeleton_in_mask=float((rgb[:,:,0][rgb[:,:,1]>0]>0).mean()),
            nodes=len(graph.nodes) if graph else None,edges=len(graph.edges) if graph else None,
            topology=topology(graph.nodes,graph.edges) if graph else None,**(graph.provenance if graph else {}))
        items.append(item);print('REAVER '+path.name,flush=True)
    result['REAVER']=dict(images=items,manual_directory_alias='Description ImageJ_Manual is named Manual in this local copy; raw B and metadata verified',
        readiness='paired manual-analysis data; independent connectivity and specimen identity unavailable; crossing/junction clusters need protocol before training',
        annotation='R manual mask, G supplied manual-analysis skeleton, B original; all *_Auto folders excluded')
    octa=DATA/'octasynth'
    result['OCTA-Synth']={s:{k:sum(p.is_file() for p in (octa/s/k).rglob('*')) for k in ['raw','labels','graphs']} for s in ['train','val','test']}
    result['OCTA-Synth']['graph_directories']={s:sum(p.is_dir() for p in (octa/s/'graphs').iterdir()) for s in ['train','val','test']}
    result['OCTA-Synth']['readiness']='blocked: graph-only local copy; sibling preprocessing requires existing raw PNG and labels, does not generate imaging intensities'
    for name in ['HALVS_DATASET','20cities_2d']:
        p=DATA/name
        result[name]=dict(readmes=[str(f) for f in p.glob('*') if 'readme' in f.name.lower()],
            readiness='botanical image/mask source, graph derived; leaf grouping/checkpoint compatibility not established' if name.startswith('HALVS') else 'road source, not vascular or botanical pretraining')
        if name.startswith('HALVS'):
            result[name]['pairs']={s:{k:len(list((p/'annotation'/s/k).glob('*.png'))) for k in ['img','mask']} for s in ['train','val']}
            ids={s:{f.name for f in (p/'annotation'/s/'img').glob('*.png')} for s in ['train','val']}
            result[name]['split_filename_overlap']=sorted(ids['train']&ids['val'])
            result[name]['filename_pairing']={s:ids[s]=={f.name for f in (p/'annotation'/s/'mask').glob('*.png')} for s in ids}
            result[name]['mask_semantics']='RGB vein orders: red first, green second, blue third; do not treat color mask as unqualified binary'
            result[name]['checkpoint_blocker']='mixed_plants_halvs_2d checkpoint run exists but contains no files; no compatible checkpoint recovered'
    result['RETA']=dict(readiness='archives present; 54 train/27 test; public test vessel masks withheld; extraction/correspondence not yet audited',
        archives={p.name:p.stat().st_size for p in (DATA/'16960855').glob('*.rar')})
    write('secondary-audit.json',result)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    inspect_wide()
    inspect_secondary()
