#!/usr/bin/env python3
"""Why do traced_adaptive regions fail to touch for GT edges whose endpoints are both found? (3 Nov 2026)

CPU, from saved validation rows. Geodesic regions block skip edges by design, so a third
retained detection lying on the GT vessel between the two endpoints takes the contact. For each
GT edge with both endpoints matched (2 mm), flag "intermediate detection" if any other retained
predicted node is within 1.5 mm of the dense GT path at least 2 mm (along the path) away from
either end, and cross-tabulate with the edge outcome, split by whether that detection is matched to a GT
node (2 mm) or spurious. Also: fraction of ALL GT edges whose
predicted centreline is broken (from centreline-oracle-<ds>-traced_adaptive-gaps.pkl).
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_20261030 as D  # noqa: E402
from compare_dense_anchored_20261030 import SCRATCH  # noqa: E402

NEAR_MM, END_MM = 1.5, 2.0


def main(arm='traced_adaptive'):
    out = {}
    for ds in ['SyntheticMRI', 'IXI', 'BRAVA']:
        d = SCRATCH / f'dense_graph_20261030/{ds}/{arm}'
        th = json.loads((d / 'final.json').read_text())['thresholds']
        node_t, edge_t = th['node'], th['edge']
        rows = pickle.loads((d / 'final-validation-rows.pkl').read_bytes())
        gaps = pickle.loads((ROOT / f'experiments/dense_graph_20261030/centreline-oracle-{ds}-{arm}-gaps.pkl').read_bytes())
        table = {k: Counter() for k in ('all', '>=20mm')}
        broken = Counter()
        for r in rows:
            with np.load(r['cache']) as z:
                M, gn, ge = z['metric'], z['nodes'] @ z['metric'].T, z['edges'].reshape(-1, 2)
                P, o = z['full_paths'] @ z['metric'].T, z['path_offsets']
            keep = np.nonzero(r['node_scores'] >= node_t)[0]
            pred = r['nodes'][keep] @ M.T
            m = D.match(torch.as_tensor(pred), torch.as_tensor(gn))
            inv = {g: i for i, g in m.items()}
            score = {tuple(sorted(map(int, p))): s for p, s in zip(r['pairs'], r['edge_scores'])} if r['edge_scores'] is not None else {}
            traced = {tuple(sorted(map(int, p))): bool(t) for p, t in zip(r['pairs'], r['traced'])} if r['traced'] is not None else {}
            for e, (a, b) in enumerate(ge):
                path = P[o[e]:o[e + 1]]
                seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
                along = np.concatenate([[0.0], np.cumsum(seg)])
                arc = along[-1]
                g = gaps[r['sample_id']][e]
                found = a in inv and b in inv
                broken[('both_found' if found else 'endpoint_missing', 'broken' if g >= 0 else 'intact')] += 1
                if not found:
                    continue
                ia, ib = inv[a], inv[b]
                inner = path[(along > END_MM) & (along < arc - END_MM)]
                rest = np.setdiff1d(np.arange(len(pred)), [ia, ib])
                near = rest[np.linalg.norm(pred[rest][:, None] - inner[None], axis=-1).min(1) < NEAR_MM] if len(inner) and len(rest) else rest[:0]
                matched = set(inv.values())
                # C matched to a GT node (e.g. a junction of a branch touching this vessel) vs spurious (no GT node within 2 mm)
                block = 'clear' if not len(near) else ('intermediate_matched' if any(int(c) in matched for c in near) else 'intermediate_spurious')
                k = tuple(sorted((int(keep[ia]), int(keep[ib]))))
                if k not in score:
                    outcome = 'not_candidate'
                else:
                    outcome = f"{'traced' if traced.get(k, False) else 'fallback'}_{'recovered' if score[k] >= edge_t else 'rejected'}"
                key = (block, outcome)
                table['all'][key] += 1
                if arc >= 20:
                    table['>=20mm'][key] += 1
        res = {}
        for name, t in table.items():
            res[name] = {}
            for side in ('intermediate_spurious', 'intermediate_matched', 'clear'):
                n = sum(v for (s, _), v in t.items() if s == side)
                nt = sum(v for (s, o), v in t.items() if s == side and not o.startswith('traced'))
                rec = sum(v for (s, o), v in t.items() if s == side and o.endswith('recovered'))
                res[name][side] = dict(edges=n, not_traced_fraction=nt / max(1, n), recovered_fraction=rec / max(1, n),
                                       outcomes={o: v for (s, o), v in t.items() if s == side})
        tot = sum(broken.values())
        out[ds] = dict(thresholds=th, by_intermediate_detection=res,
                       centreline_broken_all_gt_edges=sum(v for (f, b), v in broken.items() if b == 'broken') / max(1, tot),
                       centreline_broken={f'{f}/{b}': v for (f, b), v in broken.items()})
        print('BLOCK', ds, json.dumps(out[ds]), flush=True)
        del rows
    (ROOT / 'experiments/dense_graph_20261030/tracing-blocking-check.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
