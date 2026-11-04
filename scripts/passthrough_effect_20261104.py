#!/usr/bin/env python3
"""What the pass-through decision changes, free of the threshold shifts between arms (4 Nov 2026).

CPU, saved validation rows, subject-paired bootstrap (2000 resamples, seed 261030). Writes
experiments/dense_graph_20261030/passthrough-effect.json with, per dataset:
  on_vs_off   traced_passthrough with its selected tau vs the same model with no contraction, at the same
              node and edge thresholds. This isolates the decision from training and threshold selection.
  blocked     GT edges blocked by a spurious in-between detection, with the blocked set defined BEFORE
              contraction and recovery counted in the final graph. analyse() in twohop_analysis_20261103
              defines it after contraction, so a successful contraction leaves its own denominator.
  shared      arm_a evaluated at arm_b's thresholds vs arm_b (only when they differ). An edge threshold shift
              is exact. Raising a node threshold only filters saved rows: candidates were traced among the lower
              threshold's nodes, so this is an approximation of re-running the arm at that threshold.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
from compare_dense_anchored_20261030 import KEYS, SCRATCH, dense_patch, metrics  # noqa: E402
from dense_graph_traced_20261030 import contract  # noqa: E402
from twohop_analysis_20261103 import END_MM, NEAR_MM  # noqa: E402


def load(run, arm):
    result = json.loads((run / arm / 'final.json').read_text())
    return result, pickle.loads((run / arm / 'final-validation-rows.pkl').read_bytes())


def decode(rows, node_t, edge_t, tau):
    return {r['sample_id']: dense_patch(contract(r, node_t, edge_t, tau), node_t, edge_t) for r in rows}


def paired(a, b):
    """a - b, pooled over patches and bootstrapped over subjects."""
    assert sorted(a) == sorted(b), 'validation populations differ'
    subjects = {}
    for i in a:
        subjects.setdefault(i.split('_')[1], []).append(i)
    names = sorted(subjects)

    def pooled(per, chosen):
        total = dict.fromkeys(KEYS, 0)
        for s in chosen:
            for i in subjects[s]:
                for k in KEYS:
                    total[k] += per[i][k]
        return total

    ta, tb = pooled(a, names), pooled(b, names)
    ma, mb = metrics(ta), metrics(tb)
    rng = np.random.default_rng(261030)
    boot = {m: [] for m in ma}
    for _ in range(2000):
        draw = [names[j] for j in rng.integers(len(names), size=len(names))]
        x, y = metrics(pooled(a, draw)), metrics(pooled(b, draw))
        for m in boot:
            boot[m].append(x[m] - y[m])
    return dict(a=ma, b=mb, counts_a={k: int(ta[k]) for k in ('node_tp', 'node_fp', 'node_fn', 'tp', 'fp', 'fn')},
                counts_b={k: int(tb[k]) for k in ('node_tp', 'node_fp', 'node_fn', 'tp', 'fp', 'fn')},
                difference={m: dict(point=ma[m] - mb[m], ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]) for m, v in boot.items()})


def blocked_recovery(rows, node_t, edge_t, tau):
    """Pooled and per-subject counts, so that arms with different blocked sets are reported as fractions."""
    per = {}
    for r in rows:
        tally = per.setdefault(r['sample_id'].split('_')[1], dict(edges=0, recovered=0))
        if r['edge_scores'] is None:
            continue
        with np.load(r['cache']) as z:
            M, gn, ge = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2)
            P, o = z['full_paths'] @ z['metric'].T, z['path_offsets']
        keep = np.nonzero(r['node_scores'] >= node_t)[0]
        pred = r['nodes'][keep] @ M.T
        m = D.match(torch.as_tensor(pred), torch.as_tensor(gn))
        inv = {g: i for i, g in m.items()}
        spurious = np.array([i for i in range(len(keep)) if i not in m], int)
        final = contract(r, node_t, edge_t, tau)
        accepted = {tuple(sorted(map(int, p))) for p, s in zip(r['pairs'], final['edge_scores'])
                    if s >= edge_t and min(final['node_scores'][p[0]], final['node_scores'][p[1]]) >= node_t}
        for e, (a, b) in enumerate(ge):
            if a not in inv or b not in inv or not len(spurious):
                continue
            path = P[o[e]:o[e + 1]]
            along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            inner = path[(along > END_MM) & (along < along[-1] - END_MM)]
            if not len(inner) or not (np.linalg.norm(pred[spurious][:, None] - inner[None], axis=-1).min(1) < NEAR_MM).any():
                continue
            tally['edges'] += 1
            tally['recovered'] += int(tuple(sorted((int(keep[inv[a]]), int(keep[inv[b]])))) in accepted)
    edges, recovered = sum(v['edges'] for v in per.values()), sum(v['recovered'] for v in per.values())
    return dict(edges=edges, recovered=recovered, fraction=recovered / max(1, edges), per_subject=per)


def blocked_difference(a, b):
    """Recovered fraction of a minus b, bootstrapped over subjects. The arms' blocked sets differ, so this compares rates."""
    names = sorted(a['per_subject'])
    rate = lambda t, chosen: sum(t['per_subject'][s]['recovered'] for s in chosen) / max(1, sum(t['per_subject'][s]['edges'] for s in chosen))  # noqa: E731
    rng = np.random.default_rng(261030)
    draws = [[names[j] for j in rng.integers(len(names), size=len(names))] for _ in range(2000)]
    boot = [rate(a, d) - rate(b, d) for d in draws]
    return dict(point=rate(a, names) - rate(b, names), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])


def main():
    path = ROOT / 'experiments/dense_graph_20261030/passthrough-effect.json'
    out = {}
    for dataset in ('SyntheticMRI', 'IXI', 'BRAVA'):
        run = SCRATCH / f'dense_graph_20261030/{dataset}'
        arms = {a: load(run, a) for a in ('traced_adaptive', 'traced_offset', 'traced_passthrough')}
        t = {a: arms[a][0]['thresholds'] for a in arms}
        result, rows = arms['traced_passthrough']
        nt, et, tau = t['traced_passthrough']['node'], t['traced_passthrough']['edge'], result['passthrough']['threshold']
        own = {a: decode(arms[a][1], t[a]['node'], t[a]['edge'], tau if a == 'traced_passthrough' else None) for a in arms}
        off = decode(rows, nt, et, None)
        entry = dict(thresholds={a: dict(node=t[a]['node'], edge=t[a]['edge']) for a in arms}, tau=tau,
                     on_vs_off=paired(own['traced_passthrough'], off))
        off_reported = result['passthrough']['validation_without_contraction']
        entry['parity_off_vs_reported'] = abs(entry['on_vs_off']['b']['edge_f1'] - off_reported['edge_f1'])
        entry['blocked'] = {a: blocked_recovery(arms[a][1], t[a]['node'], t[a]['edge'], tau if a == 'traced_passthrough' else None) for a in arms}
        entry['blocked']['traced_passthrough_no_contraction'] = blocked_recovery(rows, nt, et, None)
        b = entry['blocked']
        entry['blocked_difference'] = dict(on_minus_off=blocked_difference(b['traced_passthrough'], b['traced_passthrough_no_contraction']),
                                           passthrough_minus_offset=blocked_difference(b['traced_passthrough'], b['traced_offset']),
                                           passthrough_minus_adaptive=blocked_difference(b['traced_passthrough'], b['traced_adaptive']))
        entry['shared'] = {}
        for a, b in (('traced_offset', 'traced_adaptive'), ('traced_offset', 'traced_passthrough')):
            if t[a]['node'] > t[b]['node']:
                entry['shared'][f'{a}_at_{b}_thresholds'] = 'skipped: saved rows do not reach the lower node threshold'
            elif (t[a]['node'], t[a]['edge']) != (t[b]['node'], t[b]['edge']):
                at = decode(arms[a][1], t[b]['node'], t[b]['edge'], None)
                entry['shared'][f'{a}_at_{b}_thresholds'] = dict(paired(at, own[b]), exact=t[a]['node'] == t[b]['node'])
        out[dataset] = entry
        print('EFFECT', dataset, json.dumps({k: v for k, v in entry.items() if k != 'blocked'})[:3000], flush=True)
        path.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
