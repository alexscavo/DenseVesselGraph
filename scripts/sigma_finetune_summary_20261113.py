#!/usr/bin/env python3
"""Summary of the wider node-heatmap target fine-tunes (13 Nov 2026; cluster/jean_zay/submit_sigma_finetune_20261113.sh).

Per entry and arm (sigma 1.5 = control, 2.0, 2.5, 3.5; stage 1 fine-tuned 6000 updates from spline6_unet_wide_long_r1111)
and per decoding window (3: final.json, 5 / 7: final-nms<K>.json; node / edge thresholds re-selected on the training
monitor for each window by dense_graph_20261030.final_evaluation): selected thresholds, node P / R / F1 and stage-1 edge
F1 on the training monitor and on validation, and node recall at the fixed node threshold 0.5 on validation.

Choice (training monitor only): per entry, the (arm, window) with the highest training-monitor node F1. For that pick and
the control (sigma 1.5 at window 3, and at the control's own best window), from the saved validation rows:
  - per-patch node / edge counts reproduced exactly as D.evaluate (edge_accounting; loop caches: inserted anchors are
    don't-care), checked against the totals of the final json;
  - node recall per GT node class (boundary / junction / terminal / other: node_heatmap_probe_20261030.node_types) at
    the selected node threshold, over real (non-inserted) GT nodes;
  - subject-paired bootstrap (2000 resamples of volumes, sample_<volume>_<patch>) of the validation node F1 and edge F1
    differences pick - control.
Validation is reported only; nothing here is chosen on it. Test splits are never read.

usage: sigma_finetune_summary_20261113.py [--out experiments/final_try_20261113/sigma_finetune.json]
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from node_heatmap_probe_20261030 import node_types  # noqa: E402

ENTRIES = ['IXI-simp2', 'BRAVA-simp2', 'TopBrainCTA-simp2']
SIGMAS = ['1.5', '2.0', '2.5', '3.5']
ARMS = SIGMAS + ['1.5+flip']  # 1.5+flip: control recipe + random axis flips in the augmentation (--flip)
WINDOWS = [3, 5, 7]
BASE = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
CHECKPOINT = 'spline6_unet_wide_long_r1111'
R = lambda x: None if x is None else round(float(x), 4)


def run_name(arm):
    sigma, _, extra = arm.partition('+')
    return f"spline6_unet_wide_ft_s{sigma.replace('.', '')}{'_' + extra if extra else ''}_r1113"


def final_name(window):
    return 'final.json' if window == 3 else f'final-nms{window}.json'


def block(f):
    t, v, fx = f['train_monitor'], f['validation'], f['validation_fixed_0p5']
    return dict(sigma=f.get('sigma'), nms=f.get('nms'), node_threshold=f['thresholds']['node'], edge_threshold=f['thresholds']['edge'],
                train_monitor={k: R(t[k]) for k in ('node_precision', 'node_recall', 'node_f1', 'edge_f1')},
                validation={k: R(v[k]) for k in ('node_precision', 'node_recall', 'node_f1', 'edge_f1')},
                validation_node_recall_at_0p5=R(fx['node_recall']), validation_node_precision_at_0p5=R(fx['node_precision']))


def curve(run):
    rows = [json.loads(line) for line in (run / 'learning-curve.jsonl').read_text().splitlines()]
    pts = [dict(step=r['step'], node_threshold=r['node_threshold'], train_monitor_node_f1=R(r['train_monitor']['node_f1']),
                validation_monitor_node_f1=R(r['validation_monitor']['node_f1'])) for r in rows]
    trained = [p for p in pts if p['step'] > 0]
    return dict(step0=next((p for p in pts if p['step'] == 0), None), first=trained[0], middle=trained[len(trained) // 2 - 1],
                last=trained[-1], all=pts)


def patch_counts(rows_path, node_t, edge_t):
    """Per-patch node / edge counts as D.evaluate (geometry off) + per-class GT recall counts."""
    from diagnose_branch_relations import match_nodes
    from ixi_query_diagnostics import edge_accounting
    rows = pickle.loads(Path(rows_path).read_bytes())
    out, classes = [], {}
    for r in rows:
        with np.load(r['cache']) as z:
            gt_nodes, gt_edges, metric = z['nodes'], z['edges'].reshape(-1, 2), z['metric']
            inserted = z['inserted'] if 'inserted' in z.files else None
        keep = np.nonzero(r['node_scores'] >= node_t)[0]
        remap = {int(k): i for i, k in enumerate(keep)}
        nodes = r['nodes'][keep]
        pairs = r['pairs'] if r['edge_scores'] is not None else np.zeros((0, 2), int)
        valid = np.isin(pairs, keep).all(1) if len(pairs) else np.zeros(0, bool)
        sub = pairs[valid]
        scores = r['edge_scores'][valid] if len(sub) else np.zeros(0)
        local = np.array([[remap[a], remap[b]] for a, b in sub], int).reshape(-1, 2)
        mapping = match_nodes(nodes @ metric.T, gt_nodes @ metric.T, 2.0)
        row, *_ = edge_accounting(np.arange(len(nodes)), local, scores, scores >= edge_t, mapping, gt_edges, len(gt_nodes), inserted)
        out.append((r['sample_id'], row['node_tp'], row['node_fp'], row['node_fn'], row['tp'], row['fp'], row['fn']))
        types = node_types(torch.as_tensor(gt_nodes), torch.as_tensor(gt_edges))
        real = ~np.asarray(inserted, bool) if inserted is not None else np.ones(len(gt_nodes), bool)
        hit = np.zeros(len(gt_nodes), bool)
        hit[list(mapping.values())] = True
        for t in ('boundary', 'junction', 'terminal', 'other'):
            s = (types == t) & real
            a, b = classes.get(t, (0, 0))
            classes[t] = (a + int(hit[s].sum()), b + int(s.sum()))
    del rows
    return out, classes


def f1(c):
    tp, fp, fn = c
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def totals(counts):
    a = np.array([c[1:] for c in counts], float).sum(0)
    return dict(node_f1=f1(a[0:3]), node_precision=a[0] / max(1, a[0] + a[1]), node_recall=a[0] / max(1, a[0] + a[2]), edge_f1=f1(a[3:6]))


def paired_bootstrap(pick, control, n=2000, seed=0):
    ids = [c[0] for c in pick]
    assert ids == [c[0] for c in control], 'validation rows are not aligned'
    volume = np.array([i.rsplit('_', 1)[0] for i in ids])
    groups = {v: np.flatnonzero(volume == v) for v in np.unique(volume)}
    names = sorted(groups)
    P = np.array([c[1:] for c in pick], float)
    C = np.array([c[1:] for c in control], float)
    Pg = np.stack([P[groups[v]].sum(0) for v in names])
    Cg = np.stack([C[groups[v]].sum(0) for v in names])
    rng = np.random.default_rng(seed)
    d_node, d_edge = [], []
    for _ in range(n):
        s = rng.integers(0, len(names), len(names))
        p, c = Pg[s].sum(0), Cg[s].sum(0)
        d_node.append(f1(p[0:3]) - f1(c[0:3]))
        d_edge.append(f1(p[3:6]) - f1(c[3:6]))
    p, c = Pg.sum(0), Cg.sum(0)
    ci = lambda d: [R(np.percentile(d, 2.5)), R(np.percentile(d, 97.5))]
    return dict(volumes=len(names), resamples=n, node_f1_difference=R(f1(p[0:3]) - f1(c[0:3])), node_f1_difference_ci95=ci(d_node),
                edge_f1_difference=R(f1(p[3:6]) - f1(c[3:6])), edge_f1_difference_ci95=ci(d_edge),
                node_recall_difference=R(p[0] / (p[0] + p[2]) - c[0] / (c[0] + c[2])))


def rows_file(run, window):
    return run / final_name(window).replace('.json', '-validation-rows.pkl')


def class_block(classes):
    return {t: dict(recall=R(a / b) if b else None, gt_nodes=b) for t, (a, b) in classes.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'experiments/final_try_20261113/sigma_finetune.json')
    p.add_argument('--entries', nargs='*', default=ENTRIES)
    p.add_argument('--no-rows', action='store_true', help='Skip the per-class / paired analysis (json files only)')
    a = p.parse_args()
    summary = json.loads(a.out.read_text()) if a.out.exists() else {}
    summary.setdefault('entries', {})
    for entry in a.entries:
        ck = json.loads((BASE / entry / CHECKPOINT / 'final.json').read_text())
        res = dict(checkpoint=dict(run=str(BASE / entry / CHECKPOINT), **block(ck)), arms={})
        for sigma in ARMS:
            run = BASE / entry / run_name(sigma)
            if not (run / 'final.json').exists():
                res['arms'][sigma] = dict(run=str(run), missing=True)
                continue
            windows = {str(w): block(json.loads((run / final_name(w)).read_text())) for w in WINDOWS if (run / final_name(w)).exists()}
            best_w = max(windows, key=lambda w: windows[w]['train_monitor']['node_f1'])
            res['arms'][sigma] = dict(run=str(run), windows=windows, best_window_on_train_monitor=int(best_w), curve=curve(run))
        done = {s: v for s, v in res['arms'].items() if not v.get('missing')}
        if '1.5' in done:
            cand = [(s, int(w), v['windows'][w]['train_monitor']['node_f1']) for s, v in done.items() if s in SIGMAS for w in v['windows']]
            s, w, score = max(cand, key=lambda c: c[2])
            ctl_w = done['1.5']['best_window_on_train_monitor']
            res['pick_on_train_monitor'] = dict(sigma=s, window=w, train_monitor_node_f1=score,
                                                rule='highest training-monitor node F1 over the sigma arms x windows (flip arm reported separately)')
            res['control'] = dict(sigma='1.5', window=3, control_best_window_on_train_monitor=ctl_w)
            if not a.no_rows:
                analyses = {}
                todo = dict(pick=(s, w), control_w3=('1.5', 3), control_best=('1.5', ctl_w))
                if '1.5+flip' in done:
                    todo['flip_best'] = ('1.5+flip', done['1.5+flip']['best_window_on_train_monitor'])
                for tag, (sg, win) in todo.items():
                    key = (sg, win)
                    if key in [v['key'] for v in analyses.values()]:
                        analyses[tag] = next(v for v in analyses.values() if v['key'] == key)
                        continue
                    b = done[sg]['windows'][str(win)]
                    counts, classes = patch_counts(rows_file(BASE / entry / run_name(sg), win), b['node_threshold'], b['edge_threshold'])
                    t = totals(counts)
                    check = all(abs(t[k] - b['validation'][k]) < 6e-5 for k in ('node_f1', 'node_precision', 'node_recall', 'edge_f1'))
                    analyses[tag] = dict(key=key, counts=counts, classes=class_block(classes), check=check)
                    print(entry, tag, sg, win, 'reproduces final json:', check, flush=True)
                res['validation_recall_by_class'] = {tag: dict(sigma=v['key'][0], window=v['key'][1], node_threshold=done[v['key'][0]]['windows'][str(v['key'][1])]['node_threshold'],
                                                               reproduces_final_json=v['check'], **v['classes']) for tag, v in analyses.items()}
                res['paired_validation_pick_minus_control'] = {
                    'vs_control_w3': paired_bootstrap(analyses['pick']['counts'], analyses['control_w3']['counts']),
                    'vs_control_best_window': paired_bootstrap(analyses['pick']['counts'], analyses['control_best']['counts'])}
                if 'flip_best' in analyses:
                    res['paired_validation_flip_minus_control'] = {
                        'vs_control_w3': paired_bootstrap(analyses['flip_best']['counts'], analyses['control_w3']['counts']),
                        'vs_control_best_window': paired_bootstrap(analyses['flip_best']['counts'], analyses['control_best']['counts'])}
        summary['entries'][entry] = res
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=1))
    summary['description'] = __doc__.split('usage:')[0].strip()
    a.out.write_text(json.dumps(summary, indent=1))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
