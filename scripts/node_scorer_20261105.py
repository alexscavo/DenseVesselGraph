#!/usr/bin/env python3
"""Learned node scorer, cross-fitted (5 Nov 2026).

Question: is a detected peak a real node? The heatmap peak value is not a probability and does not see the graph.
The scorer rescores every peak (heat >= 0.2, the edge head's training peak threshold) from per-peak features of the
saved predictions: heat and its rank, crowding (distance to the nearest peak, peaks within 3 mm), distance to the patch
boundary, candidate and traced edge counts, the best three incident edge scores and counts above 0.5/0.7/0.85, and
the cosine between the chords to the two best partners (about -1 for a peak in the middle of a vessel).
Overconfidence: the scorer is trained ONLY on out-of-fold predictions (models trained on the other half of the
training subjects, build_crossfit_plan_20261105), never on a model's own training patches; it is a separate model on
detached features, so nothing flows back into the heatmap. Calibration is checked on out-of-fold data (fold swap) and
on the deployed model's train monitor and validation predictions.
Decoding (deployed full model, rows at 0.2): nodes = peaks with scorer probability >= p (fixed 0.5, or chosen on the
train monitor), edges among them >= t_edge, optionally hysteresis on p and pruning of isolated nodes; selection on the
train monitor with the rule of decode_study_20261105 (max edge F1 s.t. mean |beta0|,|beta1| errors <= V0).
Writes experiments/decode_study_20261105/node_scorer_<dataset>.json.
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from decode_study_20261105 import EDGE_GRID, OUT, SWEEP, as_row, decode, fast, gt, paired  # noqa: E402
from compare_dense_anchored_20261030 import dense_patch  # noqa: E402
from diagnose_branch_relations import match_nodes  # noqa: E402
from ixi_query_diagnostics import betti  # noqa: E402

T_LOW = 0.2
NAMES = ['heat', 'heat_rank', 'n_peaks', 'nn_mm', 'peaks_3mm', 'boundary_vox', 'n_cand', 'n_traced',
         'e1', 'e2', 'e3', 'n_e50', 'n_e70', 'n_e85', 'cos_best2', 'partner_heat']


def features(r):
    """(n_peaks, len(NAMES)) features of one row's peaks."""
    m = gt(r['cache'])[4]
    n = len(r['nodes'])
    if not n:
        return np.zeros((0, len(NAMES)))
    s = r['node_scores']
    mm = r['nodes'] @ m.T
    d = np.linalg.norm(mm[:, None] - mm[None], axis=-1) + np.eye(n) * 1e9
    vox = r['nodes'] * 64
    inc = [[] for _ in range(n)]
    if r['edge_scores'] is not None:
        for k, (a, b) in enumerate(r['pairs']):
            inc[a].append(k)
            inc[b].append(k)
    rows = []
    order = np.argsort(np.argsort(-s))
    for i in range(n):
        ks = sorted(inc[i], key=lambda k: -r['edge_scores'][k])
        es = [float(r['edge_scores'][k]) for k in ks]
        top = (es + [0.0, 0.0, 0.0])[:3]
        partners = [int(r['pairs'][k][1] if r['pairs'][k][0] == i else r['pairs'][k][0]) for k in ks[:2]]
        cos = 1.0
        if len(partners) == 2:
            u, v = mm[partners[0]] - mm[i], mm[partners[1]] - mm[i]
            cos = float(u @ v / max(1e-9, np.linalg.norm(u) * np.linalg.norm(v)))
        rows.append([s[i], order[i] / max(1, n - 1), n, min(d[i].min(), 50.0), int((d[i] < 3.0).sum()),
                     float(min(vox[i].min(), (63 - vox[i]).min())), len(ks),
                     int(sum(bool(r['traced'][k]) for k in ks)) if r.get('traced') is not None else 0,
                     *top, sum(e >= 0.5 for e in es), sum(e >= 0.7 for e in es), sum(e >= 0.85 for e in es), cos,
                     float(s[partners[0]]) if partners else 0.0])
    return np.asarray(rows, float)


def labels(r):
    nodes_mm, _, _, _, m = gt(r['cache'])
    y = np.zeros(len(r['nodes']), int)
    if len(r['nodes']) and len(nodes_mm):
        for i in match_nodes(r['nodes'] @ m.T, nodes_mm, 2.0):
            y[i] = 1
    return y


def dataset(rows):
    X = [features(r) for r in rows]
    y = [labels(r) for r in rows]
    return np.concatenate(X), np.concatenate(y)


def fit(X, y):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.15, random_state=364505).fit(X, y)


def calib(model, X, y):
    p = model.predict_proba(X)[:, 1]
    frac, mean = calibration_curve(y, p, n_bins=10, strategy='quantile')
    ece = float(np.mean(np.abs(frac - mean)))
    return dict(peaks=int(len(y)), positives=float(y.mean()), auc=float(roc_auc_score(y, p)), ece_quantile_bins=ece,
                mean_p=float(p.mean()), reliability=dict(mean_predicted=mean.round(3).tolist(), observed=frac.round(3).tolist()),
                heat_auc=float(roc_auc_score(y, X[:, 0])))


def rescored(rows, model):
    out = []
    for r in rows:
        f = features(r)
        out.append(dict(r, node_scores=model.predict_proba(f)[:, 1] if len(f) else r['node_scores'], heat=r['node_scores']))
    return out


def load(dataset_key, run, split, t):
    return pickle.loads((SWEEP / dataset_key / run / f'{split}_{t:.2f}.pkl').read_bytes())


def main(dataset_key, run='traced_adaptive'):
    folds = [dataset_key.replace('-loops', f'-fold{k}') for k in (0, 1)]
    oof = [load(f, run, 'oof', T_LOW) for f in folds]
    data = [dataset(rows) for rows in oof]
    report = dict(dataset=dataset_key, t_low=T_LOW, features=NAMES)
    # calibration on unseen out-of-fold data: train on one fold's OOF predictions, test on the other's
    report['oof_swap'] = [calib(fit(*data[k]), *data[1 - k]) for k in (0, 1)]
    model = fit(np.concatenate([d[0] for d in data]), np.concatenate([d[1] for d in data]))
    src = json.loads((SWEEP / dataset_key / run / 'source.json').read_text())
    v0 = dict(t_high=src['thresholds']['node'], t_edge=src['thresholds']['edge'])
    mon, val = load(dataset_key, run, 'monitor', T_LOW), load(dataset_key, run, 'val', T_LOW)
    report['deployed_monitor_calibration'] = calib(model, *dataset(mon))
    report['deployed_validation_calibration'] = calib(model, *dataset(val))
    print(dataset_key, 'calibration', json.dumps({k: {kk: report[k][kk] for kk in ('auc', 'heat_auc', 'ece_quantile_bins', 'mean_p', 'positives')}
                                                  for k in ('deployed_monitor_calibration', 'deployed_validation_calibration')}),
          'oof swap', json.dumps([{kk: x[kk] for kk in ('auc', 'heat_auc', 'ece_quantile_bins')} for x in report['oof_swap']]), flush=True)
    mon_s, val_s = rescored(mon, model), rescored(val, model)
    base_mon = load(dataset_key, run, 'monitor', v0['t_high'])
    base = fast(base_mon, **v0)
    feasible = lambda s: s['mae_beta0'] <= base['mae_beta0'] + 1e-9 and s['mae_beta1'] <= base['mae_beta1'] + 1e-9  # noqa: E731
    grid_p = [round(x, 2) for x in np.arange(0.2, 0.81, 0.05)]
    options = {
        'S1_fixed_p05_e05': [dict(t_high=0.5, t_edge=0.5)],
        'S2_fixed_p05_edge_selected': [dict(t_high=0.5, t_edge=te) for te in EDGE_GRID],
        'S3_selected': [dict(t_high=tp, t_edge=te) for tp, te in product(grid_p, EDGE_GRID)],
        'S4_selected_hysteresis_pruned': [dict(t_low=tl, t_high=th, t_edge=te, prune_isolated=True)
                                          for tl, th, te in product(grid_p, grid_p, EDGE_GRID) if th >= tl],
        'S5_selected_hysteresis_weak_tree': [dict(t_low=tl, t_high=th, t_edge=te, weak_tree=True)
                                             for tl, th, te in product(grid_p, grid_p, EDGE_GRID) if th > tl],
    }
    chosen = {}
    for name, cfgs in options.items():
        scored = [(c, fast(mon_s, **c)) for c in cfgs]
        ok = [o for o in scored if feasible(o[1])]
        cfg, s = max(ok or scored, key=lambda o: o[1]['edge_f1'])
        chosen[name] = dict(config=cfg, monitor=s, constraint_met=bool(ok) or len(cfgs) == 1 and feasible(s), feasible_options=len(ok), options=len(cfgs))
        print(dataset_key, name, json.dumps(chosen[name]), flush=True)
    base_val = load(dataset_key, run, 'val', v0['t_high'])
    per0 = {r['sample_id']: dense_patch(as_row(r, *decode(r, **v0)), 0.5, 0.5) for r in base_val}
    bet0 = []
    for r in base_val:
        nodes, pairs = decode(r, **v0)
        dr = as_row(r, nodes, pairs)
        _, _, _, (g0, g1), _ = gt(r['cache'])
        bet0.append(np.subtract(betti(len(dr['nodes']), dr['pairs']), (g0, g1)))
    b0 = np.array(bet0, float)
    report['V0'] = dict(config=v0, monitor=base, validation=paired(per0, per0)['a'],
                        validation_betti=dict(signed_beta0=b0[:, 0].mean(), signed_beta1=b0[:, 1].mean(),
                                              mae_beta0=np.abs(b0[:, 0]).mean(), mae_beta1=np.abs(b0[:, 1]).mean()))
    for name, c in chosen.items():
        per, bet = {}, []
        for r in val_s:
            nodes, pairs = decode(r, **c['config'])
            dr = as_row(r, nodes, pairs)
            per[r['sample_id']] = dense_patch(dr, 0.5, 0.5)
            _, _, _, (g0, g1), _ = gt(r['cache'])
            bet.append(np.subtract(betti(len(dr['nodes']), dr['pairs']), (g0, g1)))
        b = np.array(bet, float)
        c['validation_betti'] = dict(signed_beta0=b[:, 0].mean(), signed_beta1=b[:, 1].mean(), mae_beta0=np.abs(b[:, 0]).mean(), mae_beta1=np.abs(b[:, 1]).mean())
        c['validation_vs_V0'] = paired(per, per0)
        d = c['validation_vs_V0']['difference']
        print(dataset_key, name, 'VAL minus V0:', ' '.join(f"{m} {v['point']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for m, v in d.items()),
              '| betti', json.dumps(c['validation_betti']), flush=True)
    report['chosen'] = chosen
    print(dataset_key, 'V0 VAL', json.dumps(report['V0']['validation']), json.dumps(report['V0']['validation_betti']), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'node_scorer_{dataset_key}.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    pickle.dump(model, (OUT / f'node_scorer_{dataset_key}.pkl').open('wb'))


if __name__ == '__main__':
    main(*sys.argv[1:])
