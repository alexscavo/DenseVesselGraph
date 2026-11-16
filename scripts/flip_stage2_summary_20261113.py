#!/usr/bin/env python3
"""Stage 2 on the mirrored-augmentation stage-1 fine-tune, and flip-averaged node decoding (13 Nov 2026;
cluster/jean_zay/flip_stage2_20261113.slurm).

Runs per entry (all stage 2, reference recipe: 7.5k updates, warm-up 0, lr 3e-4, seed 364505):
  reference  traced_unet_wide_long_r1111        from spline6_unet_wide_long_r1111 (today's pipeline)
  control    traced_unet_wide_ft_s15_r1113      from spline6_unet_wide_ft_s15_r1113 (6k-update sigma-1.5 fine-tune)
  flip       traced_unet_wide_ft_s15_flip_r1113 from spline6_unet_wide_ft_s15_flip_r1113 (same + axis flips), with --flip
Decodings: default = recorded window (3^3), single pass (final.json of the training run); flipavg5 = --flip-average
--nms 5 (final-v2-nms5-flipavg.json). Node / edge thresholds are re-selected on the training monitor in each case by
dense_graph_traced_20261030.final_evaluation. Verification files of the reference: final-v2.json (evaluate-only, no
option; must equal final.json), final-v2-flipavg.json (TopBrain CTA, flip average at 3^3).

From the saved validation rows: per-patch node / edge / branch / curve counts and Betti errors as
node_decode_study_20261113.stage2_counts (dense_patch + betti), pooled as protocol_report_20261106.pooled (checked
against the json), and subject-paired bootstrap differences (protocol_report_20261106.paired: 2000 resamples of
subjects = the <subject> of sample_<subject>_..., seed 261030). Validation is reported only; test splits are never read.

usage: flip_stage2_summary_20261113.py [--workers N]
"""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
BASE = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
CACHE = Path(os.environ['SCRATCH']) / 'experiments/gnbm/flip_stage2_20261113'
OUT = ROOT / 'experiments/final_try_20261113/flip_stage2.json'
ENTRIES = ['IXI-simp2', 'BRAVA-simp2', 'TopBrainCTA-simp2']
RUNS = dict(reference='traced_unet_wide_long_r1111', control='traced_unet_wide_ft_s15_r1113', flip='traced_unet_wide_ft_s15_flip_r1113')
DECODINGS = dict(default='final.json', flipavg5='final-v2-nms5-flipavg.json')
VERIFY = dict(reevaluated_default='final-v2.json', flipavg3='final-v2-flipavg.json')
STUDY = ROOT / 'experiments/final_try_20261113/node_decode2_stage2_{}.json'
KEYS_JSON = ('node_precision', 'node_recall', 'node_f1', 'edge_precision', 'edge_recall', 'edge_f1', 'common_branch_f1',
             'common_curve_f1', 'beta0_mae', 'beta1_mae')
R = lambda x: None if x is None else round(float(x), 4)
def block(f):
    pick = lambda d: {k: R(d.get(k)) for k in KEYS_JSON}
    return dict(node_threshold=f['thresholds']['node'], edge_threshold=f['thresholds']['edge'], nms=f.get('nms', 3),
                flip_average=f.get('flip_average', False), flip=f.get('flip', False),
                train_monitor=pick(f['train_monitor']), validation=pick(f['validation']))


def counts(run_dir, name, f, workers):
    """Per-patch counts of the validation rows saved with evaluation `name` (cached)."""
    cache = CACHE / f'{run_dir.parent.name}-{run_dir.name}-{name.replace(".json", "")}.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z['counts'], list(z['ids'])
    rows = pickle.loads((run_dir / name.replace('.json', '-validation-rows.pkl')).read_bytes())
    import node_decode_study_20261113 as S
    S.CTX.update(rows=rows, nt=f['thresholds']['node'], et=f['thresholds']['edge'])  # read by forked workers
    with Pool(workers) as pool:
        c = np.asarray(pool.map(S.stage2_counts, range(len(rows)), chunksize=16), float)
    ids = [r['sample_id'] for r in rows]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, counts=c, ids=np.array(ids))
    S.CTX.clear()
    return c, ids


def seconds_per_update(run_dir):
    rows = [json.loads(line) for line in (run_dir / 'training.jsonl').read_text().splitlines()]
    curve = {json.loads(line)['step'] for line in (run_dir / 'learning-curve.jsonl').read_text().splitlines()}
    steps = [(r['step'], r['seconds']) for r in rows]
    pure = [(s1 - s0) / (t1 - t0) for (t0, s0), (t1, s1) in zip(steps, steps[1:]) if not any(t0 < c <= t1 for c in curve)]
    return dict(updates=steps[-1][0], seconds_total_with_curve_points=R(steps[-1][1]),
                seconds_per_update_overall=R(steps[-1][1] / steps[-1][0]), seconds_per_update_median_100=R(np.median(pure)))


def main():
    from protocol_report_20261106 import paired, pooled
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    a = p.parse_args()
    out = dict(description=__doc__.split('usage:')[0].strip(), entries={})
    rnd = lambda d: {k: R(v) for k, v in d.items()}
    for entry in ENTRIES:
        res, C = dict(runs={}, paired_validation={}, verification={}), {}
        for arm, run in RUNS.items():
            run_dir = BASE / entry / run
            rr = dict(run=str(run_dir), decodings={})
            if (run_dir / 'final.json').exists():  # training finished
                rr['training'] = seconds_per_update(run_dir)
                rr['warm_start'] = json.loads((run_dir / 'warm-start.json').read_text())['source']
            for dec, name in DECODINGS.items():
                if not (run_dir / name).exists():
                    rr['decodings'][dec] = dict(file=name, missing=True)
                    continue
                f = json.loads((run_dir / name).read_text())
                c, ids = counts(run_dir, name, f, a.workers)
                pl = pooled(c)
                check = {k: bool(abs(pl[k] - f["validation"][j]) < 6e-5) for k, j in (('node_f1', 'node_f1'), ('edge_f1', 'edge_f1'),
                                                                               ('branch_f1', 'common_branch_f1'))}
                rr['decodings'][dec] = dict(file=name, **block(f), pooled_from_rows=rnd(pl), pooled_reproduces_json=check)
                C[(arm, dec)] = (c, ids)
            res['runs'][arm] = rr
        comparisons = {'flip_minus_control@default': (('flip', 'default'), ('control', 'default')),
                       'flip_minus_control@flipavg5': (('flip', 'flipavg5'), ('control', 'flipavg5')),
                       'flipavg5_minus_default@reference': (('reference', 'flipavg5'), ('reference', 'default')),
                       'flipavg5_minus_default@control': (('control', 'flipavg5'), ('control', 'default')),
                       'flipavg5_minus_default@flip': (('flip', 'flipavg5'), ('flip', 'default')),
                       'flip@flipavg5_minus_reference@default (total change)': (('flip', 'flipavg5'), ('reference', 'default')),
                       'control_minus_reference@default (stage-1 fine-tune, context)': (('control', 'default'), ('reference', 'default'))}
        for name, (x, y) in comparisons.items():
            if x in C and y in C:
                assert C[x][1] == C[y][1], 'validation rows are not aligned'
                d = paired(C[x][0], C[y][0], C[x][1])
                res['paired_validation'][name] = {k: dict(point=R(v['point']), ci95=[R(q) for q in v['ci95']], p=R(v['p']))
                                                  for k, v in d['difference'].items()}
        ids = next(iter(C.values()))[1] if C else []
        res['subjects'] = len({s.split('_')[1] for s in ids})
        ref = BASE / entry / RUNS['reference']
        v = {}
        if (ref / VERIFY['reevaluated_default']).exists():
            f0, f1 = (json.loads((ref / n).read_text()) for n in ('final.json', VERIFY['reevaluated_default']))
            v['evaluate_only_reproduces_final_json'] = dict(
                thresholds_equal=f0['thresholds'] == f1['thresholds'],
                train_monitor_equal=f0['train_monitor'] == f1['train_monitor'], validation_equal=f0['validation'] == f1['validation'],
                max_abs_validation_difference=max(abs(f0['validation'][k] - f1['validation'][k]) for k in f0['validation']
                                                  if isinstance(f0['validation'][k], float)))
        study = json.loads(Path(str(STUDY).format(entry)).read_text())['arms']
        for tag, name in (('flipavg5', DECODINGS['flipavg5']), ('flipavg3', VERIFY['flipavg3'])):
            if (ref / name).exists():
                f = json.loads((ref / name).read_text())
                v[f'this_pipeline_{tag}'] = dict(node_threshold=f['thresholds']['node'], edge_threshold=f['thresholds']['edge'],
                                                 validation_node_f1=R(f['validation']['node_f1']), validation_edge_f1=R(f['validation']['edge_f1']),
                                                 train_monitor_node_f1=R(f['train_monitor']['node_f1']), train_monitor_edge_f1=R(f['train_monitor']['edge_f1']))
        for arm in ('pipeline', 'a_flip_single'):
            s = study[arm]
            v[f'study_{arm}'] = dict(map=(s.get('arm') or {}).get('map'), node_rule=(s.get('arm') or {}).get('rule'),
                                     edge_threshold=s['edge_threshold'], validation_node_f1=R(s['validation']['node_f1']),
                                     validation_edge_f1=R(s['validation']['edge_f1']), train_monitor_node_f1=R(s['train_monitor']['node_f1']),
                                     train_monitor_edge_f1=R(s['train_monitor']['edge_f1']))
        res['verification'] = v
        out['entries'][entry] = res
        OUT.write_text(json.dumps(out, indent=1))
        print('done', entry, flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
