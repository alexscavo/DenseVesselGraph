#!/usr/bin/env python3
"""Compare trained runs on the validation list of one plan entry, whatever entry each run was trained on (10 Nov 2026).

Used for the GT-simplification arm: the reference model (trained on the original GT) and the arm (trained on the
simplified GT) are both decoded and scored on the SAME validation GT, at the frozen global decoding (node 0.30,
edge 0.75) with node AP @2 mm as the threshold-free measure. Scoring is compare_arms_20261109.evaluate_run; the
difference is the subject-paired bootstrap of protocol_report_20261106.paired. Curves of every run are compressed at
--compress-tol voxels (0 = off) so that runs with different K rules are compared on their output representation.

usage: gt_simplification_compare_20261110.py <target entry> <entry:run reference> <entry:run arm> [<entry:run arm> ...]
Writes experiments/gt_simplification_20261110/<target entry>[-<tag>].json (runs keyed '<entry>:<run>').
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import compare_arms_20261109 as C  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from protocol_report_20261106 import paired  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('target')
    p.add_argument('reference')
    p.add_argument('arms', nargs='+')
    p.add_argument('--compress-tol', type=float, default=0.5)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--tag', default='', help="Results go to <target>-<tag>.json (a comparison with another reference)")
    a = p.parse_args()
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    val_ids = json.loads(PLAN.read_text())['datasets'][a.target]['val_ids']
    if a.smoke:
        val_ids = val_ids[:24]
    ds = make_dataset(a.target, val_ids, augment=False)
    dst = ROOT / 'experiments/gt_simplification_20261110' / ('smoke' if a.smoke else '')
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / (f'{a.target}-{a.tag}.json' if a.tag else f'{a.target}.json')
    out = json.loads(path.read_text()) if path.exists() else dict(target=a.target, decoding=C.V0, runs={}, minus_reference={})
    out.update(reference=a.reference, compress_tol_voxels=a.compress_tol, validation_patches=len(val_ids))
    t0 = time.time()
    ref, ref_counts, ref_ids = C.evaluate_run(*a.reference.split(':'), ds, workers, a.compress_tol)
    out['runs'][a.reference] = ref
    show = lambda e: {k: round(float(e['V0_global'][k]), 4) for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1', 'mae_beta0', 'mae_beta1')}
    print('RUN', a.target, a.reference, json.dumps(show(ref)), 'nodeAP', round(ref['node_ap_2mm']['node_ap'], 4), flush=True)
    for arm in a.arms:
        if not (Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / arm.replace(':', '/') / 'final.json').exists():
            print('MISSING', arm, flush=True)
            continue
        entry, counts, ids = C.evaluate_run(*arm.split(':'), ds, workers, a.compress_tol)
        assert ids == ref_ids
        out['runs'][arm] = entry
        d = out['minus_reference'][arm] = paired(counts, ref_counts, ids)['difference']
        print('RUN', a.target, arm, json.dumps(show(entry)), 'nodeAP', round(entry['node_ap_2mm']['node_ap'], 4), flush=True)
        print('DIFF', a.target, arm, '-', a.reference, json.dumps({k: [round(d[k]['point'], 4)] + [round(x, 4) for x in d[k]['ci95']]
                                                                    for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1', 'mae_beta0', 'mae_beta1')}), flush=True)
    out['seconds'] = time.time() - t0
    path.write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
