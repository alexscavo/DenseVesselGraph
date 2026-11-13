#!/usr/bin/env python3
"""Per-subject check: whole-volume rows against the stored-patch stitching (12 Nov 2026, CPU).

Same validation subjects, same thresholds (the run's own), 'core, hungarian 2 mm', scored raw and cleaned against the
same volume GT. Three arms:
  stored     the stored validation patches with the run's saved rows (final-validation-rows.pkl), grid = the starts of
             the subject's stored patches: exactly what stitch_volume_20261112.py <entry> <run> does;
  rows@stored the new rows (scripts/predict_volume_grid_20261112.py, grid overlap) at the same starts, same grid:
             differs from 'stored' only by inference reproduction (bf16);
  rows@all   the new rows at every position of the dataset grid, grid = all starts (what --rows-dir --grid overlap does).
usage: stitch_rows_vs_stored_20261112.py <entry> <run> --rows-dir DIR [--subjects N]
Writes experiments/stitching_20261112/<entry>-<run>-overlap-vs-stored.json
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from stitch_volume_20261112 import PLAN, RUNS, load_meta, pooled, pred_patch, score, simplify, stitch_core, volume_gt  # noqa: E402

_ONE = None


def _call(x):
    return _ONE(x)


def main():
    global _ONE
    ap = argparse.ArgumentParser()
    ap.add_argument('entry')
    ap.add_argument('run')
    ap.add_argument('--rows-dir', type=Path, required=True)
    ap.add_argument('--subjects', type=int, default=0)
    ap.add_argument('--scale-mm', type=float, default=2.0)
    a = ap.parse_args()
    plan = json.loads(PLAN.read_text())['datasets'][a.entry]
    original = Path(json.loads((Path(plan['root']) / 'simplify_manifest.json').read_text())['source'])
    source_root = Path(json.loads((original / 'generation_config.json').read_text())['source_root'])
    index = {r['sample_id']: r for r in csv.DictReader((original / 'patch_index.csv').open())}
    by = defaultdict(list)
    for i in plan['val_ids']:
        by[index[i]['patient_id']].append(i)
    patients = sorted(by)[:a.subjects] if a.subjects else sorted(by)
    thr = json.loads((RUNS / a.entry / a.run / 'final.json').read_text())['thresholds']
    saved = {r['sample_id']: r for r in pickle.loads((RUNS / a.entry / a.run / 'final-validation-rows.pkl').read_bytes())}
    grid_dir = a.rows_dir / 'overlap'
    meta = load_meta(grid_dir)

    def one(patient):
        gt, A = volume_gt(source_root, patient, a.scale_mm)
        new = {tuple(int(x) for x in r['start']): r for r in pickle.loads((grid_dir / f"{patient}-node{thr['node']:g}.pkl").read_bytes())}
        starts = {i: tuple(int(index[i][k]) for k in ('start_d', 'start_h', 'start_w')) for i in by[patient]}
        stored_grid = [sorted({int(r[k]) for r in index.values() if r['patient_id'] == patient}) for k in ('start_d', 'start_h', 'start_w')]
        arms = {
            'stored': ([(starts[i], *pred_patch(saved[i], thr['node'], thr['edge'])) for i in by[patient]], stored_grid),
            'rows@stored': ([(starts[i], *pred_patch(new[starts[i]], thr['node'], thr['edge'])) for i in by[patient]], stored_grid),
            'rows@all': ([(st, *pred_patch(r, thr['node'], thr['edge'])) for st, r in new.items()], meta[patient]['starts']),
        }
        out = {}
        for arm, (patches, grid) in arms.items():
            g, _ = stitch_core(patches, grid, A, 2.0, 0.0, False)
            out[f'{arm} | raw'] = score(g.graph(), gt)
            simplify(g, a.scale_mm)
            out[f'{arm} | cleaned'] = score(g.graph(), gt)
        out['positions'] = dict(stored=len(by[patient]), all=len(new))
        out['grid_equal'] = [list(x) for x in stored_grid] == [list(x) for x in meta[patient]['starts']]
        return patient, out

    _ONE = one
    res = {}
    with mp.get_context('fork').Pool(min(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')), len(patients))) as pool:
        for p, o in pool.imap_unordered(_call, patients):
            res[p] = o
    f1 = lambda tp, x, y: 2 * tp / max(1, x + y)  # noqa: E731
    keys = [f'{arm} | {stage}' for arm in ('stored', 'rows@stored', 'rows@all') for stage in ('raw', 'cleaned')]
    for p in sorted(res):
        o = res[p]
        parts = [f"{k.replace(' | cleaned', '')}: node F1 {f1(o[k]['node_tp'], o[k]['nodes'], o[k]['gt_nodes']):.3f} edge F1 {f1(o[k]['edge_tp'], o[k]['edges'], o[k]['gt_edges']):.3f} "
                 f"(nodes {o[k]['nodes']}, GT {o[k]['gt_nodes']})" for k in keys if k.endswith('cleaned')]
        print(f"COMPARE {a.entry} {p} [positions stored {o['positions']['stored']} / all {o['positions']['all']}, grid starts equal {o['grid_equal']}] | " + ' | '.join(parts), flush=True)
    pooled_res = {k: pooled([res[p][k] for p in sorted(res)]) for k in keys}
    for k in keys:
        r = pooled_res[k]
        print(f"COMPARE {a.entry} pooled {k}: node P/R/F1 {r['node_precision']:.3f} {r['node_recall']:.3f} {r['node_f1']:.3f} | edge P/R/F1 {r['edge_precision']:.3f} "
              f"{r['edge_recall']:.3f} {r['edge_f1']:.3f} | centreline P/R {r['centreline_precision']:.3f} {r['centreline_recall']:.3f}", flush=True)
    dst = ROOT / 'experiments/stitching_20261112'
    (dst / f'{a.entry}-{a.run}-overlap-vs-stored.json').write_text(json.dumps(dict(entry=a.entry, run=a.run, thresholds=thr, patients=sorted(res),
                                                                                 per_subject={p: res[p] for p in sorted(res)}, pooled=pooled_res), indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
