#!/usr/bin/env python3
"""How well does the node-role head find intermediate nodes? (12 Nov 2026, GPU, validation, diagnostic.)

For a run trained with --node-roles: every peak >= the run's node threshold on 600 validation patches gets its role
probabilities (before any filtering) and a label from the GT, exactly as in training: matched to a GT node (real),
unmatched within 1.5 mm of a GT centreline (intermediate), unmatched elsewhere (off-vessel).
Reported: class shares, and for P(intermediate) >= tau: recall of the intermediate peaks, precision, and the share of
real nodes removed; plus the same for the off-vessel class.
usage: role_head_quality_20261112.py <plan entry> <run> [--patches 600]
Writes experiments/long_edges_20261111/role-quality-<entry>-<run>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from node_miss_diagnosis_20261110 import dense_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('entry')
ap.add_argument('run')
ap.add_argument('--patches', type=int, default=600)
a = ap.parse_args()
run_dir = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / a.entry / a.run
final = json.loads((run_dir / 'final.json').read_text())
T.apply_saved_mode(final)
assert T.MODE['roles']
node_t = final['thresholds']['node']
model = T.TracedDenseGraph(backbone=final.get('backbone', 'unet')).to('cuda')
model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cuda')['model'])
model.eval()
ids = json.loads(PLAN.read_text())['datasets'][a.entry]['val_ids'][:a.patches]
ds = make_dataset(a.entry, ids, augment=False)
loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=8, collate_fn=D.collate)
probs, labels, scores, offset = [], [], [], 0
for images, nodes, edges, geometry in loader:
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.forward_image(images.to('cuda'))
    feats, heat, prob = out[:, 5:].float(), out[:, :1].float().sigmoid(), out[:, 4].float().sigmoid()
    for b, (p, s) in enumerate(D.peaks(out, node_t)):
        record = ds.records[offset + b]
        if len(p) == 0:
            continue
        with np.load(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz') as z:
            M, g, ed, P, off = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2), z['full_paths'] @ z['metric'].T, z['path_offsets']
        pm = (p / D.SIZE).cpu().numpy() @ M.T
        m = D.match(torch.as_tensor(pm), torch.as_tensor(g)) if len(g) else {}
        dense = np.concatenate([dense_path(P[off[k]:off[k + 1]]) for k in range(len(ed))]) if len(ed) else np.zeros((0, 3))
        d_path = cKDTree(dense).query(pm)[0] if len(dense) else np.full(len(pm), np.inf)
        with torch.no_grad():
            probs.append(model.role_logits(feats[b], heat[b], prob[b], p).softmax(-1).cpu().numpy())
        labels.append(np.array([0 if q in m else 1 if d_path[q] <= 1.5 else 2 for q in range(len(pm))]))
        scores.append(s.cpu().numpy())
    offset += len(images)
probs, labels, scores = np.concatenate(probs), np.concatenate(labels), np.concatenate(scores)
res = dict(entry=a.entry, run=a.run, node_threshold=node_t, peaks=int(len(labels)),
           shares=dict(real=float((labels == 0).mean()), intermediate=float((labels == 1).mean()), off_vessel=float((labels == 2).mean())), by_tau={})
for name, col, cls in (('intermediate', 2, 1), ('off_vessel', 3, 2)):
    for tau in (0.3, 0.5, 0.7):
        drop = probs[:, col] >= tau
        res['by_tau'][f'{name} >= {tau}'] = dict(removed_share_of_peaks=float(drop.mean()), recall=float(drop[labels == cls].mean()),
                                                precision=float((labels[drop] == cls).mean()) if drop.any() else None, real_nodes_removed=float(drop[labels == 0].mean()))
order = lambda col, cls: float(np.mean(probs[labels == cls][:, col][:, None] > probs[labels != cls][:, col][None, :][:, :20000])) if (labels == cls).any() else None
res['auc_intermediate_vs_rest'] = order(2, 1)
res['heat_score_median'] = dict(real=float(np.median(scores[labels == 0])), intermediate=float(np.median(scores[labels == 1])), off_vessel=float(np.median(scores[labels == 2])))
print('ROLEQ', a.entry, json.dumps(res), flush=True)
dst = ROOT / 'experiments/long_edges_20261111'
(dst / f'role-quality-{a.entry}-{a.run}.json').write_text(json.dumps(res, indent=1) + '\n')
