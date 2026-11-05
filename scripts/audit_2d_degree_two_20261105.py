#!/usr/bin/env python3
"""Audit (5 Nov 2026): in native 2D targets, is every degree-2 node an inserted loop breaking point?

metrics/loop_aware.inserted_from_degree treats degree-2 nodes of 2D targets as inserted anchors. A breaking point
always lies on a cycle (it splits a parallel pair, a self-loop or a ring), so a degree-2 node that is NOT on a cycle
would be something else (e.g. a junction whose third branch was cropped away) and the rule would mislabel it.
Per dataset and split: crops, crops with degree-2 nodes, degree-2 nodes, and how many lie on a cycle. Targets with an
explicit `inserted` mask (recontracted, native2d/targets.crop_graph) also report how many degree-2 nodes are flagged and
how many flagged nodes are not degree 2 / not on a cycle.

Usage: audit_2d_degree_two_20261105.py [--root DATA_ROOT] [--out JSON] [dataset ...]
(default root: data_root of experiments/native2d_mixed_20261028/plan.json; default out: the 5 Nov audit JSON)."""
import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(json.loads((ROOT / 'experiments/native2d_mixed_20261028/plan.json').read_text())['data_root'])


def on_cycle(v, edges, n):
    """v (degree 2) lies on a cycle iff its two neighbours stay connected without v."""
    nb = [q if p == v else p for p, q in edges if v in (p, q)]
    a, b = nb[0], nb[1]
    adj = defaultdict(list)
    for p, q in edges:
        if v in (p, q):
            continue
        adj[p].append(q)
        adj[q].append(p)
    seen, stack = {a}, [a]
    while stack:
        x = stack.pop()
        if x == b:
            return True
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return a == b  # both edges to the same neighbour: a 2-cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DATA)
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments/loop_subdivision_20261105/audit_2d_degree_two.json')
    parser.add_argument('datasets', nargs='*', default=['avwide', 'reaver', 'octasynth', 'plants'])
    args = parser.parse_args()
    out = {}
    for dataset in args.datasets:
        manifest = args.root / dataset / 'manifest.json'
        if not manifest.exists():
            print('missing', manifest)
            continue
        m = json.loads(manifest.read_text())
        rows = m['samples']
        archive = None
        stats = defaultdict(lambda: dict(crops=0, crops_with_degree2=0, degree2=0, on_cycle=0))
        flags = defaultdict(lambda: dict(crops_with_mask=0, inserted=0, degree2_flagged=0, inserted_not_degree2=0,
                                         inserted_off_cycle=0, inserted_on_boundary=0))
        for row in rows:
            split = row.get('split', '?')
            if row.get('target_archive'):
                archive = archive or ZipFile(manifest.parent / row['target_archive'])
                z = np.load(io.BytesIO(archive.read(row['target'])))
            else:
                z = np.load(manifest.parent / row['target'])
            edges = [tuple(map(int, e)) for e in z['edges'].reshape(-1, 2)]
            n = len(z['nodes'])
            degree = np.bincount(np.asarray(edges, int).ravel(), minlength=n) if edges else np.zeros(n, int)
            d2 = np.flatnonzero(degree == 2)
            s = stats[split]
            s['crops'] += 1
            s['crops_with_degree2'] += int(len(d2) > 0)
            s['degree2'] += len(d2)
            cyc = {int(v) for v in d2 if on_cycle(int(v), edges, n)}
            s['on_cycle'] += len(cyc)
            if 'inserted' in z.files:
                f, ins = flags[split], np.asarray(z['inserted'], bool)
                f['crops_with_mask'] += 1
                f['inserted'] += int(ins.sum())
                f['degree2_flagged'] += int(ins[d2].sum())
                f['inserted_not_degree2'] += int((ins & (degree != 2)).sum())
                f['inserted_off_cycle'] += sum(int(v) not in cyc for v in np.flatnonzero(ins & (degree == 2)))
                f['inserted_on_boundary'] += int((ins & np.asarray(z['boundary'], bool)).sum())
        out[dataset] = {k: {**v, **flags.get(k, {})} for k, v in stats.items()}
        print(dataset, json.dumps(out[dataset]), flush=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
