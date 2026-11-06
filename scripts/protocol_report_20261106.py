#!/usr/bin/env python3
"""Evaluation protocol, step 2 (6 Nov 2026): operating points, validation, statistics, figures. CPU.

Reads the per-setting tables of protocol_tables_20261106 for the five datasets and reports, per dataset and per
decoding family (V0 = single thresholds; V4 = tree-rule hysteresis), four operating points:
  global_lodo    chosen on the TRAIN MONITORS OF THE OTHER FOUR DATASETS only (macro average over datasets): what a new,
                 unseen dataset would get. The review-safe headline.
  global_pooled  chosen on all five train monitors (macro): the single frozen setting for future datasets (its own
                 monitor is part of the selection data, so it is not a held-out estimate).
  per_dataset    chosen on the dataset's own train monitor (V0: the thresholds of final.json; V4: rule A).
  oracle         best validation edge F1 within the family: an UPPER BOUND, never a result.
Rules (fixed in advance, identical for every dataset and for LODO/pooled):
  V0  two-step: node threshold by node F1, then edge threshold by edge F1 (the training script's rule).
  V4  rule A: max edge F1 s.t. mean |beta0| and |beta1| errors <= V0 (same rule, same data) and Curve F1 >= V0.
Validation: full loop-aware metrics and Betti errors at every chosen setting; V4 vs V0 at each operating point and
LODO vs per-dataset, subject-paired bootstrap (2000, seed 261030), two-sided bootstrap p, Holm over the five datasets
for the primary endpoints; non-inferiority margins fixed in advance (F1 -0.01; Betti error +0.10 per patch).
Robustness: monitor-size variance (rules re-run on 200 subsamples of 100 and bootstrap resamples of 256 patches) and
sensitivity curves on validation (descriptive only). Threshold-free metrics are read from ap_<dataset>.json and box
mAP/mAR is recomputed at every chosen setting.
Writes experiments/evaluation_protocol_20261106/{report.json, report.md} and figures under
docs/research/figures/evaluation_protocol_20261106/.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from compare_dense_anchored_20261030 import KEYS, dense_patch, metrics  # noqa: E402
from decode_study_20261105 import as_row, decode, gt, load  # noqa: E402
from ixi_query_diagnostics import betti  # noqa: E402

DATASETS = os.environ.get('PROTOCOL_DATASETS', 'SyntheticMRI-loops+IXI-loops+BRAVA+TopBrainMRA-loops+TopBrainCTA-loops').split('+')  # '+': sbatch --export splits on commas
HERE = ROOT / 'experiments/evaluation_protocol_20261106'
OUTDIR = HERE / os.environ.get('PROTOCOL_OUT', '')  # smoke runs write to a subfolder
FIG = ROOT / 'docs/research/figures/evaluation_protocol_20261106'
MARGIN_F1, MARGIN_BETTI = 0.01, 0.10
PRIMARY = ('edge_f1', 'curve_f1', 'mae_beta0', 'mae_beta1')
F1S = ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1')
LOWER_BETTER = ('mae_beta0', 'mae_beta1')
B, SEED, SUBSAMPLES = 2000, 261030, 200
TOL = 1e-9


# ------------------------------------------------------------------ tables and summaries
def load_tables():
    T = {}
    for d in DATASETS:
        meta = json.loads((HERE / f'tables/{d}.json').read_text())
        z = np.load(HERE / f'tables/{d}.npz')
        T[d] = dict(meta=meta, monitor=z['monitor'], val=z['val'],
                    mcol={c: i for i, c in enumerate(meta['monitor_columns'])}, vcol={c: i for i, c in enumerate(meta['val_columns'])})
    keys = [json.dumps(s, sort_keys=True) for s in T[DATASETS[0]]['meta']['settings']]
    assert all([json.dumps(s, sort_keys=True) for s in T[d]['meta']['settings']] == keys for d in DATASETS), 'settings differ'
    return T


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 1.0)


def summarize_monitor(t, idx=None):
    """Per-setting monitor summary (arrays over settings), pooled over patches idx (all if None)."""
    a = t['monitor'] if idx is None else t['monitor'][:, idx]
    c = {k: a[:, :, t['mcol'][k]].sum(1) for k in KEYS}
    p = np.where(c['curve_pred'] > 0, c['curve_pred_ok'] / np.maximum(c['curve_pred'], 1e-12), 1.0)
    r = np.where(c['curve_gt'] > 0, c['curve_gt_ok'] / np.maximum(c['curve_gt'], 1e-12), 1.0)
    m = t['mcol']
    return dict(node_f1=f1(c['node_tp'], c['node_fp'], c['node_fn']), edge_f1=f1(c['tp'], c['fp'], c['fn']),
                branch_f1=f1(c['branch_tp'], c['branch_fp'], c['branch_fn']),
                curve_f1=np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0),
                mae_beta0=np.abs(a[:, :, m['p0']] - a[:, :, m['g0']]).mean(1), mae_beta1=np.abs(a[:, :, m['p1']] - a[:, :, m['g1']]).mean(1))


def summarize_val_fast(t):
    a, v = t['val'], t['vcol']
    s = lambda k: a[:, :, v[k]].sum(1)  # noqa: E731
    return dict(node_f1=f1(s('node_tp'), s('node_fp'), s('node_fn')), edge_f1=f1(s('tp'), s('fp'), s('fn')),
                mae_beta0=np.abs(a[:, :, v['p0']] - a[:, :, v['g0']]).mean(1), mae_beta1=np.abs(a[:, :, v['p1']] - a[:, :, v['g1']]).mean(1))


def macro(summaries):
    return {k: np.mean([s[k] for s in summaries], 0) for k in summaries[0]}


# ------------------------------------------------------------------ rules
def rule_v0(summ, settings):
    """Two-step: node threshold by node F1 (independent of the edge threshold), then edge threshold by edge F1."""
    v0 = [i for i, s in enumerate(settings) if s['family'] == 'V0']
    nodes = sorted({settings[i]['t_high'] for i in v0})
    node_f1 = {tn: summ['node_f1'][[i for i in v0 if settings[i]['t_high'] == tn]] for tn in nodes}
    assert all(np.allclose(v, v[0]) for v in node_f1.values()), 'V0 node F1 must not depend on the edge threshold'
    tn = max(nodes, key=lambda t: node_f1[t][0])  # first maximum in grid order
    cand = [i for i in v0 if settings[i]['t_high'] == tn]
    return max(cand, key=lambda i: summ['edge_f1'][i])


def rule_a(summ, settings, ref):
    v4 = [i for i, s in enumerate(settings) if s['family'] == 'V4']
    ok = [i for i in v4 if summ['mae_beta0'][i] <= summ['mae_beta0'][ref] + TOL and summ['mae_beta1'][i] <= summ['mae_beta1'][ref] + TOL
          and summ['curve_f1'][i] >= summ['curve_f1'][ref] - TOL]
    return (max(ok, key=lambda i: summ['edge_f1'][i]), len(ok)) if ok else (ref, 0)


def best_edge_v4(summ, settings):
    """Unconstrained V4: max edge F1 only (reported to show the rule-A constraint does not create the gains)."""
    return max((i for i, s in enumerate(settings) if s['family'] == 'V4'), key=lambda i: summ['edge_f1'][i])


def index_of(settings, cfg):
    key = json.dumps(cfg, sort_keys=True)
    found = [i for i, s in enumerate(settings) if json.dumps(s, sort_keys=True) == key]
    assert len(found) == 1, cfg
    return found[0]


# ------------------------------------------------------------------ validation (full) and statistics
CTX = {}


def _eval_patch(args):
    i, s = args
    r = CTX['rows'][i]
    dr = as_row(r, *decode(r, **{k: v for k, v in s.items() if k != 'family'}))
    c = dense_patch(dr, 0.5, 0.5)
    _, _, _, (g0, g1), _ = gt(r['cache'])
    p0, p1 = betti(len(dr['nodes']), dr['pairs'])
    return [c[k] for k in KEYS] + [abs(p0 - g0), abs(p1 - g1)]


def full_val(dataset, run, s, workers):
    CTX['rows'] = load(dataset, run, 'val', s.get('t_low', s['t_high']))
    with Pool(workers) as pool:
        out = np.asarray(pool.map(_eval_patch, [(i, s) for i in range(len(CTX['rows']))], chunksize=16), float)
    return out, [r['sample_id'] for r in CTX['rows']]


def pooled(a):
    c = dict(zip(KEYS, a[:, :len(KEYS)].sum(0)))
    m = metrics(c)
    m.update(mae_beta0=a[:, len(KEYS)].mean(), mae_beta1=a[:, len(KEYS) + 1].mean())
    return m


def paired(a, b, ids):
    """a - b, bootstrap over subjects. Returns point, ci95 and two-sided bootstrap p per metric."""
    subj = np.array([s.split('_')[1] for s in ids])
    names = np.unique(subj)
    groups = [np.flatnonzero(subj == n) for n in names]
    rng = np.random.default_rng(SEED)
    pa, pb = pooled(a), pooled(b)
    point = {k: pa[k] - pb[k] for k in pa}
    boot = {k: [] for k in pa}
    for _ in range(B):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        xa, xb = pooled(a[idx]), pooled(b[idx])
        for k in boot:
            boot[k].append(xa[k] - xb[k])
    out = {}
    for k in boot:
        v = np.asarray(boot[k])
        p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
        out[k] = dict(point=float(point[k]), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], p=float(p))
    return dict(a=pa, b=pb, difference=out)


def verdict(k, d):
    lo, hi = d['ci95']
    if k in LOWER_BETTER:
        return 'better' if hi < 0 else ('non-inferior' if hi < MARGIN_BETTI else 'not shown non-inferior')
    return 'better' if lo > 0 else ('non-inferior' if lo > -MARGIN_F1 else 'not shown non-inferior')


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


# ------------------------------------------------------------------ main
def main():
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    T = load_tables()
    S = T[DATASETS[0]]['meta']['settings']
    mon = {d: summarize_monitor(T[d]) for d in DATASETS}
    valf = {d: summarize_val_fast(T[d]) for d in DATASETS}
    report = dict(protocol=__doc__, margins=dict(f1=MARGIN_F1, betti=MARGIN_BETTI), primary=PRIMARY, datasets={}, checks={})
    chosen = {d: {} for d in DATASETS}
    # global, pooled over all five
    g_summ = macro([mon[d] for d in DATASETS])
    g_v0 = rule_v0(g_summ, S)
    g_v4, g_n = rule_a(g_summ, S, g_v0)
    g_v4u = best_edge_v4(g_summ, S)
    report['global_pooled'] = dict(V0=S[g_v0], V4=S[g_v4], V4_unconstrained=S[g_v4u], V4_feasible=g_n)
    for d in DATASETS:
        others = [x for x in DATASETS if x != d]
        l_summ = macro([mon[x] for x in others])
        l_v0 = rule_v0(l_summ, S)
        l_v4, l_n = rule_a(l_summ, S, l_v0)
        fin = T[d]['meta']['final_thresholds']
        p_v0 = index_of(S, dict(family='V0', t_high=fin['node'], t_edge=fin['edge']))
        p_v4, p_n = rule_a(mon[d], S, p_v0)
        rule_v0_own = rule_v0(mon[d], S)
        fam = lambda f: [i for i, s in enumerate(S) if s['family'] == f]  # noqa: E731
        o_v0 = max(fam('V0'), key=lambda i: valf[d]['edge_f1'][i])
        o_v4 = max(fam('V4'), key=lambda i: valf[d]['edge_f1'][i])
        chosen[d] = dict(global_lodo=dict(V0=l_v0, V4=l_v4, V4_unconstrained=best_edge_v4(l_summ, S)),
                         global_pooled=dict(V0=g_v0, V4=g_v4, V4_unconstrained=g_v4u),
                         per_dataset=dict(V0=p_v0, V4=p_v4, V4_unconstrained=best_edge_v4(mon[d], S)), oracle=dict(V0=o_v0, V4=o_v4))
        report['checks'][d] = dict(final_thresholds=fin, table_rule_v0_on_own_monitor=S[rule_v0_own],
                                   final_thresholds_reproduced_by_table_rule=rule_v0_own == p_v0,
                                   per_dataset_V4_feasible=p_n, lodo_V4_feasible=l_n)
    # full validation at every chosen setting (cached per dataset and setting)
    results = {}
    for d in DATASETS:
        run = T[d]['meta']['run']
        cache, ids_ref = {}, None
        for op, fams in chosen[d].items():
            for fam_name, i in fams.items():
                if i not in cache:
                    cache[i], ids = full_val(d, run, S[i], workers)
                    ids_ref = ids_ref or ids
                    assert ids == ids_ref
                    # consistency: full and fast validation counts agree on node/edge counts
                    fast = T[d]['val'][i]
                    for k in ('node_tp', 'node_fp', 'node_fn', 'tp', 'fp', 'fn'):
                        assert np.allclose(cache[i][:, KEYS.index(k)], fast[:, T[d]['vcol'][k]]), (d, S[i], k)
        entry = dict(settings={op: {f: S[i] for f, i in fams.items()} for op, fams in chosen[d].items()}, validation={}, comparisons={})
        for op, fams in chosen[d].items():
            entry['validation'][op] = {f: pooled(cache[i]) for f, i in fams.items()}
        for op in ('global_lodo', 'global_pooled', 'per_dataset'):
            entry['comparisons'][f'{op}: V4 - V0'] = paired(cache[chosen[d][op]['V4']], cache[chosen[d][op]['V0']], ids_ref)
            entry['comparisons'][f'{op}: V4_unconstrained - V0'] = paired(cache[chosen[d][op]['V4_unconstrained']], cache[chosen[d][op]['V0']], ids_ref)
        for f in ('V0', 'V4'):
            entry['comparisons'][f'{f}: global_lodo - per_dataset'] = paired(cache[chosen[d]['global_lodo'][f]], cache[chosen[d]['per_dataset'][f]], ids_ref)
        results[d] = (entry, cache, ids_ref)
        report['datasets'][d] = entry
        print('DONE validation', d, flush=True)
    # Holm over datasets and verdicts, per comparison and primary endpoint
    report['holm'] = {}
    for comp in report['datasets'][DATASETS[0]]['comparisons']:
        for k in PRIMARY:
            ps = [report['datasets'][d]['comparisons'][comp]['difference'][k]['p'] for d in DATASETS]
            adj = holm(ps)
            for d, a in zip(DATASETS, adj):
                diff = report['datasets'][d]['comparisons'][comp]['difference'][k]
                diff['p_holm'] = a
                diff['verdict'] = verdict(k, diff)
            report['holm'][f'{comp} | {k}'] = dict(zip(DATASETS, adj))
    # monitor-size variance of the per-dataset rules
    rng = np.random.default_rng(SEED)
    report['monitor_size_variance'] = {}
    for d in DATASETS:
        n_mon = T[d]['monitor'].shape[1]
        out = {}
        for label, size, replace in (('100_without_replacement', min(100, n_mon), False), (f'{n_mon}_bootstrap', n_mon, True)):
            picks = {'V0': [], 'V4': []}
            for _ in range(SUBSAMPLES):
                idx = rng.choice(n_mon, size=size, replace=replace)
                summ = summarize_monitor(T[d], idx)
                v0 = rule_v0(summ, S)
                v4, _ = rule_a(summ, S, v0)
                picks['V0'].append(v0)
                picks['V4'].append(v4)
            out[label] = {f: dict(
                distinct_settings=len(set(p)), most_common_share=float(max(np.bincount(p)) / len(p)),
                val_edge_f1=dict(mean=float(valf[d]['edge_f1'][p].mean()), p05=float(np.percentile(valf[d]['edge_f1'][p], 5)), p95=float(np.percentile(valf[d]['edge_f1'][p], 95))),
                val_mae_beta1=dict(mean=float(valf[d]['mae_beta1'][p].mean()), p05=float(np.percentile(valf[d]['mae_beta1'][p], 5)), p95=float(np.percentile(valf[d]['mae_beta1'][p], 95))))
                for f, p in picks.items()}
        report['monitor_size_variance'][d] = out
    # threshold-free metrics and box mAP/mAR at the chosen settings
    from traced_ap_20261105 import decoded_rows
    from dense_ap_20261030 import box_map
    for d in DATASETS:
        ap = ROOT / f'experiments/decode_study_20261105/ap_{d}.json'
        entry = report['datasets'][d]
        if ap.exists():
            a = json.loads(ap.read_text())
            entry['threshold_free'] = dict(node_ap_2mm=a['node_ap_2mm'], edge_ranking_ap_at_V0_node_threshold=
                                           {k: v for k, v in a['V0'].items() if k.startswith('edge_ap')})
        entry['box_mAP_mAR'] = {}
        for op, fams in chosen[d].items():
            for f, i in fams.items():
                s = {k: v for k, v in S[i].items() if k != 'family'}
                rows = load(d, T[d]['meta']['run'], 'val', s.get('t_low', s['t_high']))
                entry['box_mAP_mAR'][f'{op}/{f}'] = box_map(decoded_rows(rows, s), 0.0, 0.0, strict=False)
        print('DONE box', d, flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / 'report.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    write_markdown(report)
    figures(T, S, valf, chosen)


def fmt(d):
    return f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def write_markdown(report):
    L = ['# Evaluation protocol results (6 Nov 2026, validation; generated by scripts/protocol_report_20261106.py)', '',
         f"Margins: F1 {-MARGIN_F1}, Betti error +{MARGIN_BETTI}. Holm over the five datasets per comparison and primary endpoint.", '']
    L += ['## Absolute validation metrics', '', '| Dataset | Operating point | Family | Node F1 | Edge F1 | Branch F1 | Curve F1 | |β0| err | |β1| err |', '|---|---|---|---|---|---|---|---|---|']
    for d, e in report['datasets'].items():
        for op, fams in e['validation'].items():
            for f, m in fams.items():
                L.append(f"| {d} | {op} | {f} | {m['node_f1']:.3f} | {m['edge_f1']:.3f} | {m['branch_f1']:.3f} | {m['curve_f1']:.3f} | {m['mae_beta0']:.2f} | {m['mae_beta1']:.2f} |")
    for comp in report['datasets'][DATASETS[0]]['comparisons']:
        L += ['', f'## {comp}', '', '| Dataset | Node F1 | Edge F1 | Branch F1 | Curve F1 | |β0| err | |β1| err |', '|---|---|---|---|---|---|---|']
        for d, e in report['datasets'].items():
            x = e['comparisons'][comp]['difference']
            cells = []
            for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1', 'mae_beta0', 'mae_beta1'):
                c = fmt(x[k])
                if k in PRIMARY:
                    c += f" (p_Holm {x[k]['p_holm']:.3f}, {x[k]['verdict']})"
                cells.append(c)
            L.append(f"| {d} | " + ' | '.join(cells) + ' |')
    L += ['', '## Settings', '']
    L.append(f"- global_pooled (frozen for future datasets): V0 {report['global_pooled']['V0']}, V4 {report['global_pooled']['V4']}, "
             f"V4 unconstrained {report['global_pooled']['V4_unconstrained']}")
    for d, e in report['datasets'].items():
        L.append(f"- {d}: " + '; '.join(f"{op}: " + ' / '.join(f"{f} {v}" for f, v in fams.items()) for op, fams in e['settings'].items()))
    L += ['', '## Checks', ''] + [f"- {d}: {json.dumps(c)}" for d, c in report['checks'].items()]
    L += ['', '## Monitor-size variance (per-dataset rules re-run on monitor subsamples)', '']
    for d, v in report['monitor_size_variance'].items():
        for label, fams in v.items():
            for f, x in fams.items():
                L.append(f"- {d} {label} {f}: {x['distinct_settings']} distinct settings (top {x['most_common_share']:.0%}); val edge F1 "
                         f"{x['val_edge_f1']['mean']:.3f} [{x['val_edge_f1']['p05']:.3f}, {x['val_edge_f1']['p95']:.3f}]; val |β1| err "
                         f"{x['val_mae_beta1']['mean']:.2f} [{x['val_mae_beta1']['p05']:.2f}, {x['val_mae_beta1']['p95']:.2f}]")
    L += ['', '## Threshold-free and box metrics', '']
    for d, e in report['datasets'].items():
        tf = e.get('threshold_free', {})
        if tf:
            L.append(f"- {d}: node AP@2mm {tf['node_ap_2mm']['node_ap']:.3f} (max recall {tf['node_ap_2mm']['max_recall']:.3f}); "
                     + ', '.join(f"{k} {v['ap']:.3f}" for k, v in tf['edge_ranking_ap_at_V0_node_threshold'].items()))
        L.append(f"  - box mAP/mAR: " + '; '.join(f"{k}: node {v['box_node_mAP']:.3f}/{v['box_node_mAR']:.3f}, edge {v['box_edge_mAP']:.3f}/{v['box_edge_mAR']:.3f}"
                                                   for k, v in e['box_mAP_mAR'].items()))
    (OUTDIR / 'report.md').write_text('\n'.join(L) + '\n')


def figures(T, S, valf, chosen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(15, 3.2 * len(DATASETS)))
    for r, d in enumerate(DATASETS):
        v = valf[d]
        p0, p4, l0, l4 = (chosen[d]['per_dataset']['V0'], chosen[d]['per_dataset']['V4'], chosen[d]['global_lodo']['V0'], chosen[d]['global_lodo']['V4'])
        # (a) V0: edge F1 vs node threshold at the per-dataset edge threshold
        te = S[p0]['t_edge']
        xs = [i for i, s in enumerate(S) if s['family'] == 'V0' and s['t_edge'] == te]
        ax = axes[r, 0]
        ax.plot([S[i]['t_high'] for i in xs], v['edge_f1'][xs], 'o-', label='edge F1')
        ax.plot([S[i]['t_high'] for i in xs], v['node_f1'][xs], 's-', label='node F1')
        ax2 = ax.twinx(); ax2.plot([S[i]['t_high'] for i in xs], v['mae_beta1'][xs], 'r^--', label='|β1| err'); ax2.set_ylabel('|β1| error', color='r')
        ax.axvline(S[p0]['t_high'], color='k', ls=':', label='per-dataset'); ax.axvline(S[l0]['t_high'], color='g', ls='--', label='LODO')
        ax.set_title(f'{d} V0: node threshold (edge {te})', fontsize=9); ax.set_xlabel('node threshold'); ax.legend(fontsize=7, loc='lower left')
        # (b) V0: edge F1 vs edge threshold at the per-dataset node threshold
        tn = S[p0]['t_high']
        xs = [i for i, s in enumerate(S) if s['family'] == 'V0' and s['t_high'] == tn]
        ax = axes[r, 1]
        ax.plot([S[i]['t_edge'] for i in xs], v['edge_f1'][xs], 'o-'); ax2 = ax.twinx(); ax2.plot([S[i]['t_edge'] for i in xs], v['mae_beta1'][xs], 'r^--')
        ax.axvline(S[p0]['t_edge'], color='k', ls=':'); ax.axvline(S[l0]['t_edge'], color='g', ls='--')
        ax.set_title(f'{d} V0: edge threshold (node {tn})', fontsize=9); ax.set_xlabel('edge threshold'); ax.set_ylabel('edge F1')
        # (c) V4: edge F1 vs high threshold at the per-dataset low and edge thresholds
        tl, te4 = S[p4].get('t_low'), S[p4]['t_edge']
        xs = [i for i, s in enumerate(S) if s['family'] == 'V4' and s.get('t_low') == tl and s['t_edge'] == te4]
        ax = axes[r, 2]
        if xs:
            ax.plot([S[i]['t_high'] for i in xs], v['edge_f1'][xs], 'o-'); ax2 = ax.twinx(); ax2.plot([S[i]['t_high'] for i in xs], v['mae_beta1'][xs], 'r^--')
            ax.axvline(S[p4]['t_high'], color='k', ls=':')
            if S[l4].get('t_low') == tl and S[l4]['t_edge'] == te4:
                ax.axvline(S[l4]['t_high'], color='g', ls='--')
        ax.set_title(f'{d} V4: high threshold (low {tl}, edge {te4})', fontsize=9); ax.set_xlabel('high threshold'); ax.set_ylabel('edge F1')
    fig.suptitle('Validation sensitivity (descriptive only; settings are chosen on training monitors). Dotted: per-dataset choice; dashed: leave-one-dataset-out',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig((FIG if not os.environ.get('PROTOCOL_OUT') else OUTDIR) / 'sensitivity.png', dpi=110)


if __name__ == '__main__':
    main()
