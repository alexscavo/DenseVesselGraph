#!/usr/bin/env python3
"""Smoke test of a simplified plan entry (10 Nov 2026, GPU): patches with an empty GT graph and ordinary ones go
through the loader (hand-written .vtp read by pyvista, cache/graph consistency check), the training targets, one
traced training step's candidate/label code, prediction with an existing model and D.evaluate.
usage: smoke_simplified_entry_20261110.py <simplified entry> <entry:run of an existing traced model>"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
import dense_graph_traced_20261030 as T  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset, targets  # noqa: E402

entry, model_run = sys.argv[1], sys.argv[2]
plan = json.loads(PLAN.read_text())['datasets'][entry]
manifest = json.loads((Path(plan['root']) / 'simplify_manifest.json').read_text())
empty = [i for i in manifest['empty'] if i in set(plan['val_ids'])][:3] + [i for i in manifest['empty'] if i in set(plan['train_ids'])][:3]
ids = empty + plan['val_ids'][:6]
print('empty ids', empty, flush=True)
for augment in (False, True):
    ds = make_dataset(entry, ids, augment=augment)
    batch = D.collate([ds[k] for k in range(len(ids))])
    images, nodes, edges, geometry = batch
    print('augment', augment, 'nodes per patch', [len(n) for n in nodes], 'edges', [len(e) for e in edges], flush=True)
    heat, offset, peak, line = targets(nodes, geometry, 'cuda')
    off2, mask2 = T.neighbourhood_offsets(nodes, 'cuda', 1.75)
    assert heat.shape[0] == len(ids) and float(heat[:len(empty)].sum()) == 0.0 and float(line[:len(empty)].sum()) == 0.0
run_dir = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / model_run.replace(':', '/')
final = json.loads((run_dir / 'final.json').read_text())
T.apply_saved_mode(final)
model = T.TracedDenseGraph(backbone=final.get('backbone', 'unet')).to('cuda')
model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cuda')['model'])
ds = make_dataset(entry, ids, augment=False)
rows = T.predict(model, ds, 'cuda', 0.3, 4)
print('predicted nodes per patch', [len(r['nodes']) for r in rows], flush=True)
print('EVAL all', {k: round(v, 3) if isinstance(v, float) else v for k, v in D.evaluate(rows, 0.3, 0.75).items()}, flush=True)
print('EVAL empty only', {k: round(v, 3) if isinstance(v, float) else v for k, v in D.evaluate(rows[:len(empty)], 0.3, 0.75).items()}, flush=True)
print('length breakdown ok', list(T.length_breakdown(rows, 0.3, 0.75))[:2], flush=True)
# one training step's stage-2 code on the batch (GT source and predicted source)
model.train()
with torch.autocast('cuda', dtype=torch.bfloat16):
    out = model.forward_image(images.to('cuda'))
feats, heat, prob = out[:, 5:].float(), out[:, :1].float().sigmoid().detach(), out[:, 4].float().sigmoid().detach()
for kind in ('gt', 'pred'):
    cands, maps = [], []
    pred = D.peaks(out.detach(), 0.2)
    for b in range(len(ids)):
        metric = geometry[b]['metric'].to('cuda') / D.SIZE
        gt = nodes[b].to('cuda') * D.SIZE
        c = gt + torch.randn_like(gt) * 0.5 if kind == 'gt' else pred[b][0][:D.MAX_PEAKS]
        cands.append(c)
        maps.append({i: i for i in range(len(gt))} if kind == 'gt' else D.match(c @ metric.T, gt @ metric.T))
    contacts, dist, label = T.traced_pairs(prob, cands)
    terms = [T.patch_terms(model, feats[b], heat[b], cands[b], nodes[b].to('cuda') * D.SIZE, edges[b].to('cuda'), geometry[b]['guides'].to('cuda') * D.SIZE,
                           geometry[b]['metric'].to('cuda') / D.SIZE, maps[b], contacts[b], dist[b], label[b], prob[b]) for b in range(len(ids))]
    print(kind, 'patch terms', ['none' if t is None else round(float(t[0]), 3) for t in terms], flush=True)
print('SMOKE OK', flush=True)
