#!/usr/bin/env python3
"""Does the stage-1 offset head actually refine node positions? (3 Nov 2026)

traced_adaptive models, validation patches (every k-th, ~200 per dataset), heatmap peaks at
the train-selected node threshold, matched one-to-one to GT within 2 mm as in evaluation.
Per matched node (errors in mm):
  peak     |peak voxel centre - GT|
  refined  |peak voxel + predicted offset - GT|  (what evaluation uses)
split by whether the peak sits on the GT's rounded voxel, i.e. the only voxel where the offset
is supervised (L1 to GT - round(GT)). Also, at every GT voxel: predicted offset vs its target,
and the size of the targets (GT nodes lying on voxel centres would give zero targets).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from node_heatmap_probe_20261030 import PLAN, make_dataset, SIZE  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402

D = T.D
RUN = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'


def stats(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None, median=float(np.median(x)) if len(x) else None)


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    out = {}
    for d in ['SyntheticMRI', 'IXI', 'BRAVA']:
        res = json.loads((RUN / d / 'traced_adaptive/final.json').read_text())
        nt = res['thresholds']['node']
        model = T.TracedDenseGraph().eval()
        model.load_state_dict(torch.load(RUN / d / 'traced_adaptive/model.pt', map_location='cpu')['model'])
        val = json.loads(PLAN.read_text())['datasets'][d]['val_ids']
        ids = val[::max(1, len(val) // 200)][:200]
        ds = make_dataset(d, ids, augment=False)
        acc = {k: [] for k in ('peak_on', 'ref_on', 'peak_off', 'ref_off', 'target', 'pred_at_gt', 'head_err_at_gt', 'round_err_at_gt', 'pred_norm_at_peaks')}
        for i in range(len(ds)):
            s = ds[i]
            g = s[2][0][:, :3].numpy() * SIZE
            if len(g) == 0:
                continue
            M = s[4][0]['metric'].numpy() / SIZE
            o = model.forward_image(s[0][0].float()[None])[0].float()
            heat = o[0].sigmoid()
            keep = (heat == F.max_pool3d(heat[None, None], 3, 1, 1)[0, 0]) & (heat >= nt)
            idx = keep.nonzero()
            off = o[1:4][:, idx[:, 0], idx[:, 1], idx[:, 2]].T.numpy()
            v = idx.float().numpy()
            ref = np.clip(v + off, -0.5, SIZE - 0.5)
            acc['pred_norm_at_peaks'] += list(np.abs(off).max(1))
            mapping = D.match(torch.as_tensor(ref @ M.T), torch.as_tensor(g @ M.T))
            c = np.clip(np.round(g), 0, SIZE - 1).astype(int)
            for p, q in mapping.items():
                on = bool((v[p] == c[q]).all())
                acc['peak_on' if on else 'peak_off'].append(np.linalg.norm((v[p] - g[q]) @ M.T))
                acc['ref_on' if on else 'ref_off'].append(np.linalg.norm((ref[p] - g[q]) @ M.T))
            pred = o[1:4][:, c[:, 0], c[:, 1], c[:, 2]].T.numpy()
            tgt = g - c
            acc['target'] += list(np.abs(tgt).max(1))
            acc['pred_at_gt'] += list(np.abs(pred).max(1))
            acc['head_err_at_gt'] += list(np.linalg.norm((c + pred - g) @ M.T, axis=1))
            acc['round_err_at_gt'] += list(np.linalg.norm((c - g) @ M.T, axis=1))
        on, off_ = len(acc['peak_on']), len(acc['peak_off'])
        out[d] = dict(node_threshold=nt, patches=len(ds),
                      matched_nodes=dict(peak_on_gt_voxel=on, peak_off_gt_voxel=off_, on_fraction=on / max(1, on + off_)),
                      error_mm={k: stats(acc[k]) for k in ('peak_on', 'ref_on', 'peak_off', 'ref_off')},
                      refinement_helps_fraction=dict(
                          on=float(np.mean(np.array(acc['ref_on']) < np.array(acc['peak_on']))) if on else None,
                          off=float(np.mean(np.array(acc['ref_off']) < np.array(acc['peak_off']))) if off_ else None),
                      at_gt_voxel=dict(target_max_abs_vox=stats(acc['target']), target_zero_fraction=float(np.mean(np.array(acc['target']) < 1e-4)),
                                       predicted_max_abs_vox=stats(acc['pred_at_gt']),
                                       error_rounded_mm=stats(acc['round_err_at_gt']), error_with_offset_mm=stats(acc['head_err_at_gt'])),
                      predicted_max_abs_vox_at_peaks=stats(acc['pred_norm_at_peaks']))
        print('OFFSET', d, json.dumps(out[d]), flush=True)
    (REPO / 'experiments/dense_graph_20261030/offset-refinement-check.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
