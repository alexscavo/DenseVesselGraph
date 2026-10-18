#!/usr/bin/env python3
"""CPU spatial-resolution controls on the exact sampled probe evaluation voxels."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,required=True)
p.add_argument('--dataset',type=Path,default=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/transfer_splines_k8_20261015'))
a=p.parse_args();torch.set_num_threads(1)
s=json.loads((a.results/'summary.json').read_text());rng=np.random.default_rng(s['seed'])
rows=list(csv.DictReader((a.results/'probe-per-patch.csv').open()))
result={}
for split in ('train','val'):
    selected=[r for r in rows if r['probe']=='raw_intensity' and r['split']==split]
    assert set(r['sample_id'] for r in selected)==set(s['ids'][split])
    labels=[];scores={k:[] for k in ('raw','raw_pooled_2','mask_pooled_2_reference','mask_pooled_32_reference')}
    for row in selected:
        name=row['sample_id']
        raw=nib.load(a.dataset/split/'raw'/f'{name}_data.nii.gz').get_fdata(dtype=np.float32)
        mask=nib.load(a.dataset/split/'seg'/f'{name}_seg.nii.gz').get_fdata(dtype=np.float32)>.5
        natural=rng.choice(mask.size,min(2048,mask.size),replace=False)
        if split=='train':
            fg=np.flatnonzero(mask.ravel());bg=np.flatnonzero(~mask.ravel())
            rng.choice(fg,256,replace=len(fg)<256);rng.choice(bg,256,replace=len(bg)<256)
        y=mask.ravel()[natural];assert int(y.sum())==int(row['foreground'])
        labels.extend(y);scores['raw'].extend(raw.ravel()[natural])
        for key,volume,grid in [('raw_pooled_2',raw,2),('mask_pooled_2_reference',mask,2),('mask_pooled_32_reference',mask,32)]:
            x=torch.as_tensor(volume.astype(np.float32))[None,None]
            low=F.adaptive_avg_pool3d(x,(grid,)*3)
            restored=F.interpolate(low,size=raw.shape,mode='trilinear',align_corners=False)[0,0].numpy()
            scores[key].extend(restored.ravel()[natural])
    result[split]={k:float(average_precision_score(labels,v)) for k,v in scores.items()}
    # Scalar readout is monotonic; float sigmoid saturation can merge intensity ranks.
    # Record the difference instead of treating raw and float-probe AP as identical.
    result[split]['raw_ap_difference_from_float_probe']=result[split]['raw']-s['probes']['raw_intensity'][split]['average_precision']
    print(split,result[split],flush=True)
    assert abs(result[split]['raw_ap_difference_from_float_probe'])<.001
out=dict(protocol='average-pool and trilinear reconstruction; same evaluation voxels and AP as frozen probes',
         results=result,caveat='Pooled GT mask is a label-informed resolution reference, NOT a rigorous upper bound on learned features or a deployable prediction. Spatial detail may be encoded in channels and require a richer decoder.')
(a.results/'resolution-controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
