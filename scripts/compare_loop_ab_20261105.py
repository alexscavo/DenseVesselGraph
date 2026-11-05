#!/usr/bin/env python3
"""Loop A/B (5 Nov 2026): training with loop patches (arm B, <D>-loops) minus without (arm A, <D>-noloops).

CPU, saved validation rows. Both arms are scored on the SAME complete validation list (loop patches included)
with the loop-aware protocol (metrics/loop_aware.py via compare_dense_anchored.dense_patch), each at its own
train-monitor thresholds, and also both arms at shared thresholds (the higher node threshold, arm A's edge
threshold; exact when the node thresholds agree, otherwise an approximation, flagged as shared_exact=false). Subject-paired
bootstrap (2000 resamples, seed 261030) over: all validation patches, loop patches only (>=1 inserted anchor),
loop-free patches only. Stage 1 (spline6) and stage 2 (traced_adaptive). No selection on validation.
Writes experiments/loop_subdivision_20261105/loop-ab.json.
"""
import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_dense_anchored_20261030 import SCRATCH, dense_patch  # noqa: E402
from passthrough_effect_20261104 import paired  # noqa: E402

PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'
OUT = ROOT / 'experiments/loop_subdivision_20261105/loop-ab.json'


def load(dataset, run):
    d = SCRATCH / 'dense_graph_20261030' / dataset / run
    stem = 'final' if (d / 'final-validation-rows.pkl').exists() else 'final-v2'
    return json.loads((d / f'{stem}.json').read_text()), pickle.loads((d / f'{stem}-validation-rows.pkl').read_bytes())


def main(datasets):
    plan = json.loads(PLAN.read_text())['datasets']
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for d in datasets:
        root = Path(plan[f'{d}-loops']['root'])
        loops = {r['sample_id'] for r in csv.DictReader((root / 'patch_index.csv').open()) if int(r.get('inserted_count', 0)) > 0}
        for run in ('spline6', 'traced_adaptive'):
            try:
                (ra, rows_a), (rb, rows_b) = load(f'{d}-noloops', run), load(f'{d}-loops', run)
            except FileNotFoundError as error:
                print('MISSING', d, run, error, flush=True)
                continue
            ta, tb = ra['thresholds'], rb['thresholds']
            per_a = {r['sample_id']: dense_patch(r, ta['node'], ta['edge']) for r in rows_a}
            per_b = {r['sample_id']: dense_patch(r, tb['node'], tb['edge']) for r in rows_b}
            # Shared thresholds: saved rows hold only nodes above each arm's own node threshold, so both arms are
            # decoded at the HIGHER node threshold (filtering rows; approximate when it differs from an arm's own,
            # since candidates were traced among that arm's lower-threshold nodes) and at arm A's edge threshold.
            shared = dict(node=max(ta['node'], tb['node']), edge=ta['edge'])
            per_a_sh = {r['sample_id']: dense_patch(r, shared['node'], shared['edge']) for r in rows_a}
            per_b_sh = {r['sample_id']: dense_patch(r, shared['node'], shared['edge']) for r in rows_b}
            res = dict(thresholds=dict(noloops=dict(node=ta['node'], edge=ta['edge']), loops=dict(node=tb['node'], edge=tb['edge']),
                                       shared=shared), shared_exact=ta['node'] == tb['node'])
            for subset, keep in (('all', lambda s: True), ('loop_patches', lambda s: s in loops),
                                 ('loop_free_patches', lambda s: s not in loops)):
                ids = [s for s in per_a if keep(s)]
                if not ids:
                    continue
                res[subset] = dict(patches=len(ids),
                                   own_thresholds=paired({s: per_b[s] for s in ids}, {s: per_a[s] for s in ids}),
                                   shared_thresholds=paired({s: per_b_sh[s] for s in ids}, {s: per_a_sh[s] for s in ids}))
            out.setdefault(d, {})[run] = res
            for subset in ('all', 'loop_patches', 'loop_free_patches'):
                if subset in res:
                    diff = res[subset]['own_thresholds']['difference']
                    print(d, run, subset, res[subset]['patches'], ' '.join(
                        f"{m} {v['point']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for m, v in diff.items()), flush=True)
            OUT.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or ['IXI', 'SyntheticMRI', 'TopBrainMRA', 'TopBrainCTA'])
