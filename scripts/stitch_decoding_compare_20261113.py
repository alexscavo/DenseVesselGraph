#!/usr/bin/env python3
"""Whole-volume stitching under three decodings of traced_unet_wide_long_r1111 (16 Nov 2026, CPU, reads json only).

Decodings (validation subjects, grid 'overlap', stitching 'core, hungarian 2 mm', cleaned graphs, each at the run's
thresholds under that decoding, chosen on monitor_train patches):
  original       window 3, single pass                                  <entry>-<run>-overlap.json
  flipavg        --flip-average --nms 5 (node heat over 8 flips)        <entry>-<run>-overlap-nms5-flipavg.json
  flipavg_edges  + --flip-average-edges (edge logits over 8 flips)      <entry>-<run>-overlap-nms5-flipavg-edges.json
(experiments/stitching_20261112/). Per decoding: pooled cleaned node / edge P / R / F1, centreline P / R (1.5 mm),
components and loops per volume (and GT), share of cut points matched (2 x merged pairs / cut nodes) — recomputed from the
per-subject counts and checked against the pooled values in the files. Subject-paired differences of pooled node and
edge F1 (and the other pooled ratios): 2000 bootstrap resamples of subjects, seed 261030, 95% percentile interval,
two-sided bootstrap p as protocol_report_20261106.paired. Validation only; few subjects per entry (see 'subjects').
Writes experiments/final_try_20261113/stitch_overlap_flipavg-edges_vs_others.json (refuses to overwrite).

usage: stitch_decoding_compare_20261113.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'experiments/stitching_20261112'
OUT = ROOT / 'experiments/final_try_20261113/stitch_overlap_flipavg-edges_vs_others.json'
RUN = 'traced_unet_wide_long_r1111'
ENTRIES = ['IXI-simp2', 'BRAVA-simp2', 'TopBrainCTA-simp2']
DECODINGS = dict(original='overlap', flipavg='overlap-nms5-flipavg', flipavg_edges='overlap-nms5-flipavg-edges')
KEY = 'pred | core, hungarian 2 mm | cleaned'
PAIRS = {'flipavg_edges_minus_flipavg': ('flipavg_edges', 'flipavg'), 'flipavg_edges_minus_original': ('flipavg_edges', 'original'),
         'flipavg_minus_original (context)': ('flipavg', 'original')}
B, SEED = 2000, 261030


def pooled(rows):
    t = {k: sum(r[k] for r in rows) for k in rows[0]}
    f = lambda tp, a, b: 2 * tp / max(1, a + b)  # noqa: E731  (as stitch_volume_20261112.pooled)
    n = len(rows)
    return dict(node_precision=t['node_tp'] / max(1, t['nodes']), node_recall=t['node_tp'] / max(1, t['gt_nodes']), node_f1=f(t['node_tp'], t['nodes'], t['gt_nodes']),
                edge_precision=t['edge_tp'] / max(1, t['edges']), edge_recall=t['edge_tp'] / max(1, t['gt_edges']), edge_f1=f(t['edge_tp'], t['edges'], t['gt_edges']),
                centreline_precision=t['cl_pred_ok'] / max(1, t['cl_pred']), centreline_recall=t['cl_gt_ok'] / max(1, t['cl_gt']),
                beta0_per_volume=t['beta0'] / n, gt_beta0_per_volume=t['gt_beta0'] / n, beta1_per_volume=t['beta1'] / n, gt_beta1_per_volume=t['gt_beta1'] / n,
                nodes_per_volume=t['nodes'] / n, edges_per_volume=t['edges'] / n)


def main():
    if OUT.exists():
        sys.exit(f'{OUT} exists; not overwriting')
    rng_keys = ('node_f1', 'edge_f1', 'node_precision', 'node_recall', 'edge_precision', 'edge_recall', 'centreline_precision',
                'centreline_recall', 'beta0_per_volume', 'beta1_per_volume')
    out = dict(description=__doc__.split('usage:')[0].strip(), stitching=KEY, entries={})
    for entry in ENTRIES:
        files = {d: json.loads((SRC / f'{entry}-{RUN}-{s}.json').read_text()) for d, s in DECODINGS.items()}
        subjects = sorted(set.intersection(*[set(f['per_subject']) for f in files.values()]))
        res = dict(subjects=subjects, decodings={}, paired={})
        per = {}
        for d, f in files.items():
            assert sorted(f['per_subject']) == subjects, (entry, d, 'subject sets differ')
            rows = [f['per_subject'][p][KEY] for p in subjects]
            per[d] = rows
            pl = pooled(rows)
            r = f['results'][KEY]
            check = max(abs(pl[k] - r[k]) for k in pl)
            assert check < 1e-9, (entry, d, check)
            res['decodings'][d] = dict(file=f'{entry}-{RUN}-{DECODINGS[d]}.json', thresholds=f['thresholds'], decoding=f.get('decoding'),
                                       pooled=pl, cut_nodes_per_volume=r['cut_nodes'], merged_pairs_per_volume=r['merged_pairs'],
                                       unmatched_cuts_per_volume=r['unmatched_cuts'], share_of_cut_points_matched=2 * r['merged_pairs'] / max(1e-9, r['cut_nodes']),
                                       pooled_matches_file=True,
                                       per_subject={p: dict(node_f1=pooled([row])['node_f1'], edge_f1=pooled([row])['edge_f1']) for p, row in zip(subjects, rows)})
        rng = np.random.default_rng(SEED)
        idx_all = [rng.integers(len(subjects), size=len(subjects)) for _ in range(B)]
        for name, (x, y) in PAIRS.items():
            d = {}
            for k in rng_keys:
                point = pooled(per[x])[k] - pooled(per[y])[k]
                boot = np.array([pooled([per[x][i] for i in idx])[k] - pooled([per[y][i] for i in idx])[k] for idx in idx_all])
                p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
                d[k] = dict(point=float(point), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], p=float(p))
            d['subjects_better_node_f1'] = sum(pooled([a])['node_f1'] > pooled([b])['node_f1'] for a, b in zip(per[x], per[y]))
            d['subjects_better_edge_f1'] = sum(pooled([a])['edge_f1'] > pooled([b])['edge_f1'] for a, b in zip(per[x], per[y]))
            res['paired'][name] = d
        out['entries'][entry] = res
        for dname, v in res['decodings'].items():
            p = v['pooled']
            print(f"{entry:18s} {dname:14s} thr {v['thresholds']['node']}/{v['thresholds']['edge']} node P/R/F1 {p['node_precision']:.3f} {p['node_recall']:.3f} {p['node_f1']:.3f} "
                  f"edge P/R/F1 {p['edge_precision']:.3f} {p['edge_recall']:.3f} {p['edge_f1']:.3f} cl P/R {p['centreline_precision']:.3f} {p['centreline_recall']:.3f} "
                  f"comp {p['beta0_per_volume']:.1f} (GT {p['gt_beta0_per_volume']:.1f}) loops {p['beta1_per_volume']:.1f} (GT {p['gt_beta1_per_volume']:.1f}) "
                  f"cuts matched {v['share_of_cut_points_matched']:.3f}", flush=True)
        for name, d in res['paired'].items():
            print(f"{entry:18s} {name}: node F1 {d['node_f1']['point']:+.4f} [{d['node_f1']['ci95'][0]:+.4f}, {d['node_f1']['ci95'][1]:+.4f}] "
                  f"edge F1 {d['edge_f1']['point']:+.4f} [{d['edge_f1']['ci95'][0]:+.4f}, {d['edge_f1']['ci95'][1]:+.4f}] "
                  f"(better node / edge in {d['subjects_better_node_f1']} / {d['subjects_better_edge_f1']} of {len(subjects)})", flush=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
