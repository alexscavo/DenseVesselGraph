#!/usr/bin/env python3
"""Loop A/B (5 Nov 2026): does training with loop patches change the Betti numbers of the predicted graphs?

Per validation patch: predicted beta_0 (connected pieces) and beta_1 (independent cycles) of the accepted graph,
GT beta_0 / beta_1 (subdivision does not change them), and the signed error predicted - GT. Both arms at their own
train-monitor thresholds and at shared thresholds (higher node threshold; approximate when they differ, see
compare_loop_ab_20261105). Mean signed and absolute errors over all / loop / loop-free patches, and the B - A
difference with a subject-paired bootstrap (2000 resamples, seed 261030). CPU, saved validation rows only.
Writes experiments/loop_subdivision_20261105/loop-ab-betti.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_loop_ab_20261105 import PLAN, load  # noqa: E402
from ixi_query_diagnostics import betti  # noqa: E402

OUT = ROOT / 'experiments/loop_subdivision_20261105/loop-ab-betti.json'


def patch_betti(r, node_t, edge_t):
    with np.load(r['cache']) as z:
        gt_n, gt_e = len(z['nodes']), z['edges'].reshape(-1, 2)
    keep = np.nonzero(r['node_scores'] >= node_t)[0]
    remap = {int(k): i for i, k in enumerate(keep)}
    edges = np.zeros((0, 2), int)
    if r['edge_scores'] is not None and len(r['pairs']):
        ok = np.isin(r['pairs'], keep).all(1) & (r['edge_scores'] >= edge_t)
        edges = np.array([[remap[a], remap[b]] for a, b in r['pairs'][ok]], int).reshape(-1, 2)
    p0, p1 = betti(len(keep), edges)
    g0, g1 = betti(gt_n, gt_e)
    return dict(p0=p0, p1=p1, g0=g0, g1=g1)


def summary(per, ids):
    a = np.array([[per[s][k] for k in ('p0', 'p1', 'g0', 'g1')] for s in ids], float)
    return dict(pred_beta0=a[:, 0].mean(), pred_beta1=a[:, 1].mean(), gt_beta0=a[:, 2].mean(), gt_beta1=a[:, 3].mean(),
                signed_beta0=(a[:, 0] - a[:, 2]).mean(), signed_beta1=(a[:, 1] - a[:, 3]).mean(),
                mae_beta0=np.abs(a[:, 0] - a[:, 2]).mean(), mae_beta1=np.abs(a[:, 1] - a[:, 3]).mean())


def paired(per_b, per_a, ids):
    subjects = {}
    for s in ids:
        subjects.setdefault(s.split('_')[1], []).append(s)
    names = sorted(subjects)
    rng = np.random.default_rng(261030)
    keys = ('signed_beta0', 'signed_beta1', 'mae_beta0', 'mae_beta1', 'pred_beta1')
    point = {k: summary(per_b, ids)[k] - summary(per_a, ids)[k] for k in keys}
    boot = {k: [] for k in keys}
    for _ in range(2000):
        draw = [s for j in rng.integers(len(names), size=len(names)) for s in subjects[names[j]]]
        sb, sa = summary(per_b, draw), summary(per_a, draw)
        for k in keys:
            boot[k].append(sb[k] - sa[k])
    return {k: dict(point=point[k], ci95=[float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]) for k in keys}


def main(datasets):
    plan = json.loads(PLAN.read_text())['datasets']
    out = {}
    for d in datasets:
        root = Path(plan[f'{d}-loops']['root'])
        loops = {r['sample_id'] for r in csv.DictReader((root / 'patch_index.csv').open()) if int(r.get('inserted_count', 0)) > 0}
        (ra, rows_a), (rb, rows_b) = load(f'{d}-noloops', 'traced_adaptive'), load(f'{d}-loops', 'traced_adaptive')
        ta, tb = ra['thresholds'], rb['thresholds']
        shared = dict(node=max(ta['node'], tb['node']), edge=ta['edge'])
        arms = dict(own=({r['sample_id']: patch_betti(r, ta['node'], ta['edge']) for r in rows_a},
                         {r['sample_id']: patch_betti(r, tb['node'], tb['edge']) for r in rows_b}),
                    shared=({r['sample_id']: patch_betti(r, shared['node'], shared['edge']) for r in rows_a},
                            {r['sample_id']: patch_betti(r, shared['node'], shared['edge']) for r in rows_b}))
        res = dict(thresholds=dict(noloops=ta, loops=tb, shared=shared), shared_exact=ta['node'] == tb['node'])
        for mode, (per_a, per_b) in arms.items():
            for subset, keep in (('all', lambda s: True), ('loop_patches', lambda s: s in loops), ('loop_free_patches', lambda s: s not in loops)):
                ids = [s for s in per_a if keep(s)]
                if not ids:
                    continue
                res.setdefault(mode, {})[subset] = dict(patches=len(ids), noloops=summary(per_a, ids), loops=summary(per_b, ids),
                                                        loops_minus_noloops=paired(per_b, per_a, ids))
                sa, sb, df = summary(per_a, ids), summary(per_b, ids), res[mode][subset]['loops_minus_noloops']
                print(f"{d:12s} {mode:6s} {subset:17s} n={len(ids):5d} GT b0 {sa['gt_beta0']:.2f} b1 {sa['gt_beta1']:.2f} | "
                      f"pred b1 A {sa['pred_beta1']:.2f} B {sb['pred_beta1']:.2f} | signed b1 A {sa['signed_beta1']:+.2f} B {sb['signed_beta1']:+.2f} "
                      f"diff {df['signed_beta1']['point']:+.2f} [{df['signed_beta1']['ci95'][0]:+.2f},{df['signed_beta1']['ci95'][1]:+.2f}] | "
                      f"signed b0 A {sa['signed_beta0']:+.2f} B {sb['signed_beta0']:+.2f} diff {df['signed_beta0']['point']:+.2f} "
                      f"[{df['signed_beta0']['ci95'][0]:+.2f},{df['signed_beta0']['ci95'][1]:+.2f}]", flush=True)
        out[d] = res
        OUT.write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or ['IXI', 'SyntheticMRI', 'TopBrainMRA', 'TopBrainCTA'])
