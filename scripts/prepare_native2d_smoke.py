#!/usr/bin/env python3
"""Read-only source audit and a bounded paired development sample; no test data."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy import sparse
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
# recontract=False: reproduces the prepared data as built before 5 Nov 2026; recontracted targets: scripts/prepare_native2d_recontracted_20261105.py
from native2d.targets import from_dense_graph,from_mask,crop_graph,save_target

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native2d_20261027'
if (OUT/'worktree.json').exists():OUT=Path(json.loads((OUT/'worktree.json').read_text())['output'])
DATA=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def count(path,pattern='*'):return sum(p.is_file() for p in path.glob(pattern)) if path.exists() else 0


def inventory():
    inv={}
    for name in ['octasynth','20cities_2d']:
        inv[name]={split:{sub:count(DATA/name/split/sub) for sub in ['raw','labels','seg','graphs','vtp']} for split in ['train','val','test']}
        if name=='octasynth':
            for split in inv[name]:inv[name][split]['graph_directories']=sum(p.is_dir() for p in (DATA/name/split/'graphs').iterdir())
    for name in ['AV-WIDE','AV-DRIVE/training','AV-DRIVE/test','AV-INSPIRE']:
        inv[name]={'mat_files':count(DATA/'AV Paper Online Files'/name,'*.mat'),
            'paired_image':name=='AV-WIDE','physical_spacing':'unavailable; use pixels',
            'provenance':'supplied AV-labelled dense graph; not an independently hand-traced centerline GT'}
    reaver=DATA/'REAVER_Vascular_Networks_Image_Dataset'
    inv['REAVER']={'original_tiffs':count(reaver/'_Original_Images','*.tif'),'manual_tiffs':count(reaver/'Manual','*.tif'),
        'provenance':'ImageJ manual analysis: R binary segmentation, G skeleton, B original raw image; local dataset summary',
        'subject_ids':'not supplied; image identifiers are not proof of biological subject independence',
        'physical_spacing':'per-image TIFF micron metadata checked against image_quantification.csv'}
    inv['HALVS']={s:{k:count(DATA/'HALVS_DATASET/annotation'/s/k,'*.png') for k in ['img','mask']} for s in ['train','val']}
    inv['HALVS']['provenance']='botanical image/mask pairs; graph derived by skeletonization, not independent graph labels'
    inv['RETA']={'archives':{p.name:p.stat().st_size for p in (DATA/'16960855').glob('*.rar')},
        'readiness':'compressed, not audited/extracted; readme lists 54 train/27 test images with test vessel masks not public'}
    return inv


def overlays(image,graph,out,*,mask=None,skeleton=None):
    fig,ax=plt.subplots(1,2,figsize=(13,6))
    display=image[:,:,1] if image.ndim==3 else image
    low,high=np.percentile(display,[2,98])
    for a in ax:
        a.imshow(display,cmap='gray',vmin=low,vmax=max(low+1,high))
        a.set_xlim(-.5,image.shape[1]-.5);a.set_ylim(image.shape[0]-.5,-.5)
    if mask is not None:ax[0].contour(mask,levels=[.5],colors=['#00e5ff'],linewidths=.5)
    if skeleton is not None:
        y,x=np.where(skeleton);ax[0].scatter(x,y,s=.2,c='#ffdd00',label='provided manual skeleton')
    for i,p in enumerate(graph.paths):
        ax[1].plot(*p.T,color='#00e5ff',lw=2,label='original dense path' if i==0 else None)
        from native2d.targets import fit_controls
        c=fit_controls(p);knots=np.r_[np.zeros(4),np.arange(1,5)/5,np.ones(4)]
        curve=BSpline(knots,c,3)(np.linspace(0,1,257))
        ax[1].plot(*curve.T,color='#ff3daf',lw=1.3,linestyle='--',label='fitted endpoint-constrained K8' if i==0 else None)
    ax[1].scatter(*graph.nodes.T,s=18,c='white',edgecolors='black',label='graph anchors',zorder=5)
    ax[0].set_title('Source image / annotations (display contrast only)');ax[1].set_title('Dense paths and K8 targets — XY pixels')
    ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);prepared=OUT/'prepared';prepared.mkdir(exist_ok=True)
    inv=inventory();samples=[];audit=[]
    for number in [1,2]:
        path=DATA/'AV Paper Online Files/AV-WIDE'/f'wide_image_{number:02d}.mat'
        raw=loadmat(path,simplify_cells=True);image=raw['I2'];g=raw['G']
        a=g['A'];rows,cols=sparse.triu(a,k=1).nonzero() if sparse.issparse(a) else np.where(np.triu(a,1))
        # MATLAB imagesc pixel centers are 1-based. Preserve the documented XY
        # order and convert exactly once to zero-based image pixel centers.
        nodes=g['V'].astype(float)-1.
        graph=from_dense_graph(nodes,np.stack((rows,cols),1),provenance=dict(dataset='AV-WIDE',source=str(path),source_sha256=sha(path),
            coordinate_conversion='G.V XY MATLAB 1-based minus 1',annotation='supplied graph; preserve connectivity; no skeleton rasterization'))
        degree=np.bincount(graph.edges.ravel(),minlength=len(graph.nodes))
        audit.append(dict(dataset='AV-WIDE',subject=f'wide_{number:02d}',dense_nodes=len(nodes),dense_edges=len(rows),
            anchors=len(graph.nodes),branches=len(graph.edges),degree4plus=int((degree>=4).sum()),
            crossing_policy='preserve supplied adjacency; image crossings do not imply a junction'))
        accepted=0
        # Only source images 01/02 are touched for development. Other images and
        # DRIVE's existing test split are never loaded.
        for center in graph.nodes:
            start=np.clip(np.floor(center-32),0,np.array(image.shape[:2])[::-1]-64).astype(int)
            patch=crop_graph(graph,start,[64,64],recontract=False)
            if not (2<=len(patch.nodes)<=12 and 1<=len(patch.edges)<=12):continue
            if any(np.linalg.norm(start-np.array(r['crop_start_xy']))<48 for r in samples if r['subject_id']==f'wide_{number:02d}'):continue
            key=f'wide{number:02d}_{accepted:02d}';target=prepared/f'{key}.npz';img=prepared/f'{key}.png'
            crop=image[start[1]:start[1]+64,start[0]:start[0]+64]
            Image.fromarray(crop).save(img);save_target(target,patch,[64,64])
            (prepared/f'{key}.json').write_text(json.dumps(patch.provenance,indent=2)+'\n')
            overlays(crop,patch,prepared/f'{key}-overlay.png')
            samples.append(dict(sample_id=key,subject_id=f'wide_{number:02d}',subject_identity_scope='source image; patient identity unavailable',
                split='train' if number==1 else 'validation',image=str(img.relative_to(OUT)),target=str(target.relative_to(OUT)),
                crop_start_xy=start.tolist(),source=str(path),source_sha256=sha(path),units='pixel',annotation='supplied dense graph',
                nodes=len(patch.nodes),edges=len(patch.edges)))
            accepted+=1
            if accepted==2:break
        if not accepted:raise RuntimeError(f'No valid bounded development crop for {path}')
    reaver=DATA/'REAVER_Vascular_Networks_Image_Dataset'
    with ZipFile(reaver/'REAVER Dataset Summary.docx') as z:
        text=''.join(ET.fromstring(z.read('word/document.xml')).itertext())
    (OUT/'reaver-label-provenance.txt').write_text(text[text.index('Dataset Summary:'):])
    scale_rows=list(csv.DictReader((reaver/'image_quantification.csv').open()))
    path=reaver/'Manual/diaphragm_60x_1.tif';original=reaver/'_Original_Images/diaphragm_60x_1.tif'
    with tifffile.TiffFile(path) as tf:
        rgb=tf.asarray();tags=tf.pages[0].tags;metadata=tf.imagej_metadata
        spacing=[tags[k].value[1]/tags[k].value[0] for k in ['XResolution','YResolution']]
    image=tifffile.imread(original)
    if not np.array_equal(image,rgb[:,:,2]):raise ValueError('REAVER raw B-channel parity failed')
    if metadata['unit']!='micron' or not all(set(np.unique(rgb[:,:,i]))<={0,255} for i in [0,1]):raise ValueError('REAVER encoding differs from documentation')
    supplied=[float(r['umppix']) for r in scale_rows if r['Image_Name']==path.name]
    if not np.allclose(supplied,spacing[0],rtol=1e-6):raise ValueError('REAVER spacing metadata mismatch')
    graph=from_mask(rgb[:,:,1]>0,is_centerline=True,provenance=dict(dataset='REAVER',source=str(path),source_sha256=sha(path),
        mask_source='manual ImageJ R channel',centerline_source='manual ImageJ G channel',
        graph_annotation='derived by contraction of provided manual skeleton; no independent connectivity labels'))
    # One microscope crop for alignment only, not mixed into the retinal cohort.
    start=np.clip(np.floor(graph.nodes[len(graph.nodes)//2]-64),0,896).astype(int)
    patch=crop_graph(graph,start,[128,128],recontract=False);crop=image[start[1]:start[1]+128,start[0]:start[0]+128]
    save_target(prepared/'reaver-alignment.npz',patch,[128,128],spacing_xy=spacing,units='micron')
    overlays(crop,patch,prepared/'reaver-alignment-overlay.png',mask=rgb[start[1]:start[1]+128,start[0]:start[0]+128,0]>0,
        skeleton=rgb[start[1]:start[1]+128,start[0]:start[0]+128,1]>0)
    audit.append({**graph.provenance,'raw_channel_exact':True,'spacing_micron_xy':spacing,
        'centerline_mask_coverage':float(np.mean(rgb[:,:,0][rgb[:,:,1]>0]>0)),
        'csv_spacing_parity':True})
    manifest=dict(protocol='native2d-development-smoke-v1',no_test_evaluation=True,split_scope='image-disjoint development only; final subject-level protocol pending metadata',samples=samples)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'dataset-audit.json').write_text(json.dumps(dict(inventory=inv,alignment=audit),indent=2)+'\n')
    print(json.dumps(dict(samples=len(samples),train=sum(r['split']=='train' for r in samples),validation=sum(r['split']=='validation' for r in samples),output=str(OUT))))


if __name__=='__main__':main()
