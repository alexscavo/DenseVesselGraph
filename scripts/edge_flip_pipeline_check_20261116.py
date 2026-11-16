#!/usr/bin/env python3
"""Verification of --flip-average-edges in scripts/dense_graph_traced_20261030.py (16 Nov 2026, GPU, one entry per job).

Runs the stage-2 CLI (--evaluate-only, run traced_unet_wide_long_r1111) three times:
  edges   --flip-average --flip-average-edges --nms 5, in the run directory -> final-v2-nms5-flipavg-edges.json (new file;
          the CLI refuses to overwrite it). Must reproduce the edge_logit arm of
          experiments/final_try_20261113/edge_flip_average_<entry>.json (thresholds, validation node / edge / Branch F1),
          and its validation rows must equal the study's rows ($SCRATCH/experiments/gnbm/edge_flip_average_20261113/
          <entry>-rows.pkl: same nodes, pairs and curves; edge score = sigmoid(mean of the 8 flip logits)).
  flipavg --flip-average --nms 5, in a shadow run directory (symlinks to model.pt and final.json under
          $SCRATCH/experiments/gnbm/edge_flip_pipeline_check_20261116/) -> must equal the run's final-v2-nms5-flipavg.json.
  plain   no option, same shadow directory -> must equal the run's final.json (and final-v2.json).
Nothing in the run directory is overwritten. With --smoke all three go to a separate shadow directory with the CLI's
--smoke (4 monitor / 4 validation patches) and only the row-level comparison with the study is meaningful.
Writes experiments/final_try_20261113/edge_flip_pipeline_check_<entry>[-smoke].json.

usage: edge_flip_pipeline_check_20261116.py <entry> [--smoke]
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ['SCRATCH'])
RUNS = SCRATCH / 'experiments/gnbm/dense_graph_20261030'
SHADOW = SCRATCH / 'experiments/gnbm/edge_flip_pipeline_check_20261116'
STUDY_ROWS = SCRATCH / 'experiments/gnbm/edge_flip_average_20261113'
OUT = ROOT / 'experiments/final_try_20261113'
RUN = 'traced_unet_wide_long_r1111'
KEYS = ('node_f1', 'edge_f1', 'common_branch_f1', 'common_curve_f1', 'node_precision', 'node_recall', 'edge_precision', 'edge_recall')


def cli(output, entry, extra, smoke):
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/dense_graph_traced_20261030.py'), '--output', str(output), '--workers', '10',
           '--dataset', entry, '--run-name', RUN, '--evaluate-only'] + extra + (['--smoke'] if smoke else [])
    t0 = time.time()
    print('RUN', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    print(f'EXIT {rc} after {time.time() - t0:.0f} s', flush=True)
    if rc:
        raise SystemExit(f'failed: {cmd}')
    return time.time() - t0


def same(f0, f1):
    """Equality of thresholds and of every train_monitor / validation metric of two evaluation files."""
    num = lambda d: {k: v for k, v in d.items() if isinstance(v, (int, float))}
    diff = max([abs(num(f0[s])[k] - num(f1[s])[k]) for s in ('train_monitor', 'validation') for k in num(f0[s]) if k in num(f1[s])] or [0.0])
    return dict(thresholds_equal=f0['thresholds']['node'] == f1['thresholds']['node'] and f0['thresholds']['edge'] == f1['thresholds']['edge'],
                train_monitor_equal=f0['train_monitor'] == f1['train_monitor'], validation_equal=f0['validation'] == f1['validation'],
                max_abs_metric_difference=float(diff),
                validation={k: [f0['validation'].get(k), f1['validation'].get(k)] for k in KEYS})


def rows_vs_study(rows, study):
    study = {r['sample_id']: r for r in study}
    n, same_struct, de, dc, flips, ties = 0, 0, 0.0, 0.0, 0, []
    for r in rows:
        s = study[r['sample_id']]
        n += 1
        if (r['nodes'].shape != s['nodes'].shape or not np.array_equal(r['nodes'], s['nodes']) or r['pairs'].shape != s['pairs'].shape
                or not np.array_equal(r['pairs'], s['pairs'])):
            continue
        same_struct += 1
        if r['edge_scores'] is None:
            continue
        ref = 1 / (1 + np.exp(-s['flip_logits'].astype(np.float64).mean(1)))
        de = max(de, float(np.abs(r['edge_scores'] - ref).max(initial=0)))
        dc = max(dc, float(np.abs(r['curves'] - s['curves']).max(initial=0)))
    return dict(patches=n, identical_nodes_and_pairs=same_struct, max_abs_edge_score_difference=de, max_abs_curve_difference=dc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entry')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run_dir = RUNS / a.entry / RUN
    shadow_root = SHADOW / ('smoke' if a.smoke else 'full')
    shadow = shadow_root / a.entry / RUN
    shadow.mkdir(parents=True, exist_ok=True)
    for name in ('model.pt', 'final.json'):
        if not (shadow / name).exists():
            (shadow / name).symlink_to(run_dir / name)
    edges_root = shadow_root if a.smoke else RUNS
    name = 'final-v2-nms5-flipavg-edges.json'
    seconds = {}
    seconds['edges'] = cli(edges_root, a.entry, ['--flip-average', '--flip-average-edges', '--nms', '5'], a.smoke)
    seconds['flipavg'] = cli(shadow_root, a.entry, ['--flip-average', '--nms', '5'], a.smoke)
    seconds['plain'] = cli(shadow_root, a.entry, [], a.smoke)
    edir = edges_root / a.entry / RUN
    fe = json.loads((edir / name).read_text())
    rows = pickle.loads((edir / name.replace('.json', '-validation-rows.pkl')).read_bytes())
    study_file = OUT / f'edge_flip_average_{a.entry}{"-smoke" if a.smoke else ""}.json'
    study_rows = pickle.loads((STUDY_ROWS / f'{a.entry}{"-smoke" if a.smoke else ""}-rows.pkl').read_bytes())['validation']
    if a.smoke:  # the smoke study used other patches; compare with the full study rows where the sample ids exist
        full = STUDY_ROWS / f'{a.entry}-rows.pkl'
        study_rows = pickle.loads(full.read_bytes())['validation'] if full.exists() else study_rows
        ids = {r['sample_id'] for r in study_rows}
        rows = [r for r in rows if r['sample_id'] in ids]
    out = dict(description=__doc__.split('usage:')[0].strip(), entry=a.entry, smoke=a.smoke, seconds=seconds,
               edges_file=str(edir / name), flip_average_edges_recorded=fe.get('flip_average_edges'),
               rows_vs_study=rows_vs_study(rows, study_rows))
    if not a.smoke:
        study = json.loads(study_file.read_text())['arms']['edge_logit']
        out['edges_vs_study_edge_logit'] = dict(
            node_threshold=[fe['thresholds']['node'], study['node_threshold']], edge_threshold=[fe['thresholds']['edge'], study['edge_threshold']],
            validation={k: [fe['validation'].get(k), study['validation'].get(k)] for k in KEYS},
            train_monitor={k: [fe['train_monitor'].get(k), study['train_monitor'].get(k)] for k in KEYS},
            max_abs_validation_difference=max(abs(fe['validation'][k] - study['validation'][k]) for k in KEYS))
        sh = shadow_root / a.entry / RUN
        out['flipavg_vs_existing'] = same(json.loads((sh / 'final-v2-nms5-flipavg.json').read_text()),
                                          json.loads((run_dir / 'final-v2-nms5-flipavg.json').read_text()))
        out['plain_vs_final_json'] = same(json.loads((sh / 'final-v2.json').read_text()), json.loads((run_dir / 'final.json').read_text()))
        if (run_dir / 'final-v2.json').exists():
            out['plain_vs_existing_final_v2'] = same(json.loads((sh / 'final-v2.json').read_text()), json.loads((run_dir / 'final-v2.json').read_text()))
    path = OUT / f'edge_flip_pipeline_check_{a.entry}{"-smoke" if a.smoke else ""}.json'
    assert a.smoke or not path.exists()
    path.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('CHECK ' + json.dumps(out, default=float), flush=True)


if __name__ == '__main__':
    main()
