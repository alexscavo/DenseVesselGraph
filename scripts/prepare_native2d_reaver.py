#!/usr/bin/env python3
"""Prepare a separate calibrated diaphragm cohort; no GPU submission or mixing."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import tifffile
# recontract=False: reproduces the prepared data as built before 5 Nov 2026; recontracted targets: scripts/prepare_native2d_recontracted_20261105.py
from native2d.targets import from_mask,crop_graph,save_target
from native2d.evaluation import common_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native2d_20261028'
DATA=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/REAVER_Vascular_Networks_Image_Dataset')

def main():
    audit=json.loads((OUT/'secondary-audit.json').read_text())['REAVER']['images']
    directory=OUT/'reaver-diaphragm';directory.mkdir(exist_ok=True);samples=[];sources=[]
    for number in range(1,7):
        name=f'diaphragm_60x_{number}.tif';row=next(r for r in audit if r['file']==name)
        assert row['strict_pair_gate'] and not row['requires_crossing_review'] and row['skeleton_in_mask']==1
        source=DATA/'_Original_Images'/name;manual=Path(row['manual_folder'])/name
        raw=tifffile.imread(source);rgb=tifffile.imread(manual);spacing=np.array(row['spacing_micron_xy'])
        graph=from_mask(rgb[:,:,1]>0,is_centerline=True,provenance=dict(source=str(manual),
            annotation='contracted supplied ImageJ manual-analysis skeleton; no independently annotated connectivity'))
        split='train' if number<=4 else 'validation'
        for y in range(0,raw.shape[0],128):
            for x in range(0,raw.shape[1],128):
                patch=crop_graph(graph,[x,y],[128,128],recontract=False);key=f'diaphragm60x{number}_x{x}_y{y}'
                image=directory/(key+'.png');target=directory/(key+'.npz')
                Image.fromarray(raw[y:y+128,x:x+128]).save(image)
                # Legacy serializer converts micron metadata to mm. Convert its
                # metric back explicitly; coordinates, paths and controls are
                # dimensionless and therefore unchanged by this unit conversion.
                save_target(target,patch,[128,128],spacing_xy=spacing,units='micron')
                with np.load(target) as stored:values={k:stored[k] for k in stored.files}
                assert values['units'].item()=='mm'
                values['metric']*=1000.;values['units']=np.array('micron')
                np.testing.assert_allclose(values['nodes']@values['metric'].T,patch.nodes*spacing,atol=1e-10)
                np.savez_compressed(target,**values)
                samples.append(dict(sample_id=key,subject_id=f'diaphragm60x{number}',split=split,
                    image=str(image.relative_to(OUT)),target=str(target.relative_to(OUT)),
                    nodes=len(patch.nodes),edges=len(patch.edges),units='micron',source_image=str(source),manual=str(manual)))
        sources.append(dict(image=str(source),manual=str(manual),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            manual_sha256=hashlib.sha256(manual.read_bytes()).hexdigest(),spacing_micron_xy=spacing.tolist(),split=split))
    q=max(r['nodes'] for r in samples if r['split']=='train')
    # The exact dense path on its original graph must have perfect physical F1.
    candidate=next(r for r in samples if r['edges']>0)
    with np.load(OUT/candidate['target']) as z:gt={k:z[k] for k in z.files}
    paths=[gt['full_paths'][a:b] for a,b in zip(gt['path_offsets'][:-1],gt['path_offsets'][1:])]
    check=common_metrics(dict(nodes=gt['nodes'],edges=gt['edges'],edge_polylines=paths),gt)
    assert check['distance_units']=='micron' and check['common_curve_f1']==1 and check['common_branch_f1']==1
    manifest=dict(protocol='reaver-diaphragm60x-manual-development-v1',samples=samples,sources=sources,
        provenance='ImageJ_Manual described by DOCX is Manual locally; exact raw-blue and binary R/G verified on these six images',
        split_scope='4 training / 2 validation source images; specimen independence unknown; no official test supplied',
        annotation='derived manual-analysis skeleton graph; no merging, pruning or automated outputs',
        units='micron',tolerances=[1.,2.],common_v2_anchor_curve_micron=1.,common_v2_sample_spacing_micron=.25,
        training_max_nodes=q,object_queries=max(16,int(np.ceil(q*1.25/8)*8)),
        ready='prepared and physical-unit identity check passed; full-model GPU/memorization gate required before training',unit_identity_check=check)
    (OUT/'reaver-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(samples=len(samples),training_max_nodes=q,queries=manifest['object_queries'],unit_check_passed=True)))

if __name__=='__main__':main()
