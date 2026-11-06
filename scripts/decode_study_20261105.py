#!/usr/bin/env python3
"""Decoding study (5 Nov 2026), CPU part: better node selection without retraining.

Rows come from decode_sweep_20261105.py (predictions at a grid of node thresholds; each threshold has its own
tracing, because every detected peak seeds the geodesic regions). Variants:
  V0 current    thresholds of final.json (node: max node F1, then edge: max edge F1, on the train monitor)
  V1 joint      (node, edge) thresholds chosen together on the final graph, re-traced at each node threshold
  V2 hysteresis trace among peaks >= t_low, accept edges >= t_edge, keep a connected piece of the accepted graph
                only if it holds a peak >= t_high (weak peaks survive only when edges connect them to a strong one)
  V3 = V2 + drop isolated nodes (degree 0 after decoding; each one is a separate piece in beta_0)
  V4 = V2, but an edge touching a weak peak is accepted only if it joins two unconnected pieces: weak peaks extend
       the graph as a tree and cannot add cycles (V2 added ~0.2 cycles per patch on TopBrain MRA validation)
  V5 = V4 + drop isolated nodes
Selection, fixed in advance and on the TRAIN MONITOR ONLY: maximise edge F1 subject to mean |beta_0| and |beta_1|
errors not above V0's on the same monitor. Validation is used once, to report every selected variant against V0:
node/edge/Branch/Curve F1 (loop-aware protocol) and signed/absolute Betti errors, subject-paired bootstrap.
Writes experiments/decode_study_20261105/<dataset>.json.
"""
import json
import os
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_dense_anchored_20261030 import dense_patch  # noqa: E402
from diagnose_branch_relations import match_nodes  # noqa: E402
from ixi_query_diagnostics import betti, edge_accounting  # noqa: E402
from passthrough_effect_20261104 import paired  # noqa: E402

SWEEP = Path(os.environ['SCRATCH']) / 'experiments/gnbm/decode_sweep_20261105'
OUT = ROOT / 'experiments/decode_study_20261105'
EDGE_GRID = [round(x, 2) for x in np.arange(0.5, 0.91, 0.05)]
GT = {}


def gt(cache):
    if cache not in GT:
        with np.load(cache) as z:
            GT[cache] = (z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2), z['inserted'] if 'inserted' in z.files else None,
                         betti(len(z['nodes']), z['edges'].reshape(-1, 2)), z['metric'].copy())
    return GT[cache]


def decode(r, t_high, t_edge, t_low=None, prune_isolated=False, weak_tree=False):
    """Kept node indices (into r['nodes']) and accepted pair indices (into r['pairs'])."""
    t_low = t_high if t_low is None else t_low
    scores = r['node_scores']
    nodes = np.flatnonzero(scores >= t_low)
    pairs = np.zeros(0, int)
    if r['edge_scores'] is not None and len(r['pairs']):
        ok = (r['edge_scores'] >= t_edge) & np.isin(r['pairs'], nodes).all(1)
        pairs = np.flatnonzero(ok)
    if t_low < t_high:  # hysteresis: keep connected pieces that contain a strong peak
        parent = {int(i): int(i) for i in nodes}

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        if weak_tree:
            # Edges between strong peaks are kept as before; an edge touching a weak peak is accepted (best score
            # first) only if it joins two pieces not yet connected, so weak peaks can never close a cycle.
            strong_node = scores >= t_high
            sure = [k for k in pairs if strong_node[r['pairs'][k]].all()]
            weak = sorted((k for k in pairs if not strong_node[r['pairs'][k]].all()), key=lambda k: -r['edge_scores'][k])
            for k in sure:
                a, b = map(int, r['pairs'][k])
                parent[root(a)] = root(b)
            kept = list(sure)
            for k in weak:
                a, b = map(int, r['pairs'][k])
                if root(a) != root(b):
                    parent[root(a)] = root(b)
                    kept.append(k)
            pairs = np.array(sorted(kept), int)
        for k in pairs:
            a, b = map(int, r['pairs'][k])
            parent[root(a)] = root(b)
        strong = {root(int(i)) for i in nodes if scores[i] >= t_high}
        nodes = np.array([i for i in nodes if root(int(i)) in strong], int)
        pairs = np.array([k for k in pairs if root(int(r['pairs'][k][0])) in strong], int)
    if prune_isolated and len(nodes):
        used = set(r['pairs'][pairs].ravel().tolist()) if len(pairs) else set()
        nodes = np.array([i for i in nodes if int(i) in used], int)
    return nodes, pairs


def as_row(r, nodes, pairs):
    """A row holding exactly the decoded graph (scores 1 = kept), for the standard evaluation at 0.5/0.5."""
    remap = {int(k): i for i, k in enumerate(nodes)}
    out = dict(r, nodes=r['nodes'][nodes], node_scores=np.ones(len(nodes)))
    if r['edge_scores'] is not None and len(pairs):
        out.update(pairs=np.array([[remap[int(a)], remap[int(b)]] for a, b in r['pairs'][pairs]], int),
                   edge_scores=np.ones(len(pairs)), curves=r['curves'][pairs])
    else:
        out.update(pairs=np.zeros((0, 2), int), edge_scores=None, curves=None)
    return out


def fast(rows, **kw):
    """Edge F1 and mean |beta| errors of a decoding over a list of rows (no geometry)."""
    tp = fp = fn = 0
    e0, e1, s0, s1 = [], [], [], []
    for r in rows:
        nodes_mm, gt_edges, inserted, (g0, g1), m = gt(r['cache'])
        nodes, pairs = decode(r, **kw)
        pred_mm = r['nodes'][nodes] @ m.T
        mapping = match_nodes(pred_mm, nodes_mm, 2.0) if len(pred_mm) and len(nodes_mm) else {}
        remap = {int(k): i for i, k in enumerate(nodes)}
        local = np.array([[remap[int(a)], remap[int(b)]] for a, b in r['pairs'][pairs]], int).reshape(-1, 2)
        row, *_ = edge_accounting(np.arange(len(nodes)), local, np.ones(len(local)), np.ones(len(local), bool), mapping,
                                  gt_edges, len(nodes_mm), inserted)
        tp, fp, fn = tp + row['tp'], fp + row['fp'], fn + row['fn']
        p0, p1 = betti(len(nodes), local)
        e0.append(abs(p0 - g0)); e1.append(abs(p1 - g1)); s0.append(p0 - g0); s1.append(p1 - g1)
    return dict(edge_f1=2 * tp / max(1, 2 * tp + fp + fn), mae_beta0=float(np.mean(e0)), mae_beta1=float(np.mean(e1)),
                signed_beta0=float(np.mean(s0)), signed_beta1=float(np.mean(s1)))


def load(dataset, run, split, t):
    return pickle.loads((SWEEP / dataset / run / f'{split}_{t:.2f}.pkl').read_bytes())


def main(dataset, run='traced_adaptive'):
    src = json.loads((SWEEP / dataset / run / 'source.json').read_text())
    grid = src['grid']
    v0 = dict(t_high=src['thresholds']['node'], t_edge=src['thresholds']['edge'])
    if v0['t_high'] not in grid:
        raise ValueError(f'V0 node threshold {v0["t_high"]} not in the sweep grid {grid}')
    mon = {t: load(dataset, run, 'monitor', t) for t in grid}
    base = fast(mon[v0['t_high']], **v0)
    feasible = lambda s: s['mae_beta0'] <= base['mae_beta0'] + 1e-9 and s['mae_beta1'] <= base['mae_beta1'] + 1e-9  # noqa: E731
    candidates = {'V0_current': [(v0, base)]}
    candidates['V1_joint'] = [(dict(t_high=tn, t_edge=te), fast(mon[tn], t_high=tn, t_edge=te)) for tn, te in product(grid, EDGE_GRID)]
    variants = (('V2_hysteresis', False, False), ('V3_hysteresis_pruned', True, False),
                ('V4_hysteresis_weak_tree', False, True), ('V5_hysteresis_weak_tree_pruned', True, True))
    for name, *_ in variants:
        candidates[name] = []
    for tl, th, te in product(grid, grid, EDGE_GRID):
        if th < tl:
            continue
        for name, prune, tree in variants:
            if tl == th and tree:
                continue  # without weak peaks the tree rule is a no-op
            cfg = dict(t_low=tl, t_high=th, t_edge=te, prune_isolated=prune, weak_tree=tree)
            candidates[name].append((cfg, fast(mon[tl], **cfg)))
    chosen = {}
    for name, options in candidates.items():
        ok = [o for o in options if feasible(o[1])]
        cfg, s = max(ok or options, key=lambda o: o[1]['edge_f1'])
        chosen[name] = dict(config=cfg, monitor=s, constraint_met=bool(ok), feasible_options=len(ok), options=len(options))
        print(dataset, name, json.dumps(chosen[name]), flush=True)
    # validation, once, for each chosen configuration
    val_cache, per = {}, {}
    for name, c in chosen.items():
        cfg = c['config']
        t_rows = cfg.get('t_low', cfg['t_high'])
        if t_rows not in val_cache:
            val_cache[t_rows] = load(dataset, run, 'val', t_rows)
        rows = val_cache[t_rows]
        per[name] = {}
        bet = []
        for r in rows:
            nodes, pairs = decode(r, **cfg)
            dr = as_row(r, nodes, pairs)
            per[name][r['sample_id']] = dense_patch(dr, 0.5, 0.5)
            _, _, _, (g0, g1), _ = gt(r['cache'])
            local = dr['pairs']
            p0, p1 = betti(len(dr['nodes']), local)
            bet.append((p0 - g0, p1 - g1))
        b = np.array(bet, float)
        c['validation_betti'] = dict(signed_beta0=float(b[:, 0].mean()), signed_beta1=float(b[:, 1].mean()),
                                     mae_beta0=float(np.abs(b[:, 0]).mean()), mae_beta1=float(np.abs(b[:, 1]).mean()))
    for name in chosen:
        if name != 'V0_current':
            chosen[name]['validation_vs_V0'] = paired(per[name], per['V0_current'])
            d = chosen[name]['validation_vs_V0']['difference']
            print(dataset, name, 'VAL minus V0:', ' '.join(f"{m} {v['point']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for m, v in d.items()),
                  '| betti', json.dumps(chosen[name]['validation_betti']), flush=True)
    chosen['V0_current']['validation'] = paired(per['V0_current'], per['V0_current'])['a']
    print(dataset, 'V0 VAL', json.dumps(chosen['V0_current']['validation']), json.dumps(chosen['V0_current']['validation_betti']), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{dataset}.json').write_text(json.dumps(dict(dataset=dataset, run=run, grid=grid, edge_grid=EDGE_GRID,
                                                         selection='max monitor edge F1 s.t. mean |beta0|,|beta1| errors <= V0 (monitor)',
                                                         chosen=chosen), indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
