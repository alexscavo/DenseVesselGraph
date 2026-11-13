#!/usr/bin/env python3
"""Long-edge outcomes under the decodings chosen by decode_study_20261105 (12 Nov 2026, CPU, validation).

For the run studied in experiments/decode_study_20261105/<entry>.json: each chosen decoding (V0 current thresholds,
V1 joint thresholds, V2 hysteresis, V4 hysteresis with the tree rule; all chosen on the training monitor) is applied
to the validation rows of the sweep, and every GT edge >= 10 mm is classified as in long_edge_breaks_20261111.
usage: long_edge_decoding_20261112.py <plan entry>
Writes experiments/long_edges_20261111/decoding-<entry>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import decode_study_20261105 as S  # noqa: E402
import long_edge_breaks_20261111 as L  # noqa: E402


def decoded(r, nodes, pairs):
    remap = {int(k): i for i, k in enumerate(nodes)}
    out = dict(r, nodes=r['nodes'][nodes], node_scores=np.ones(len(nodes)))
    if r['edge_scores'] is not None and len(pairs):
        out.update(pairs=np.array([[remap[int(a)], remap[int(b)]] for a, b in r['pairs'][pairs]], int), edge_scores=np.ones(len(pairs)),
                   curves=r['curves'][pairs], traced=np.asarray(r['traced'])[pairs])
    else:
        out.update(pairs=np.zeros((0, 2), int), edge_scores=None, curves=None, traced=None)
    return out


def main(entry):
    study = json.loads((S.OUT / f'{entry}.json').read_text())
    out = dict(entry=entry, run=study['run'], decodings={})
    cache = {}
    for name in ('V0_current', 'V1_joint', 'V2_hysteresis', 'V4_hysteresis_weak_tree'):
        c = study['chosen'][name]
        cfg = c['config']
        t = cfg.get('t_low', cfg['t_high'])
        if t not in cache:
            cache[t] = S.load(entry, study['run'], 'val', t)
        rows = [decoded(r, *S.decode(r, **cfg)) for r in cache[t]]
        edges = [e for r in rows for e in L.classify(r, 0.5, 0.5)]
        summary = L.summarise(edges)
        out['decodings'][name] = dict(config=cfg, validation_vs_V0=c.get('validation_vs_V0', {}).get('difference'), validation=c.get('validation'),
                                      validation_betti=c.get('validation_betti'), long_edges=summary)
        print('DECODING', entry, name, json.dumps(cfg), '|', ' | '.join(
            f"{b}: recovered {v.get('recovered', {}).get('share', 0):.3f}, endpoint missing {v.get('endpoint missing', {}).get('share', 0):.3f}, "
            f"broken {v.get('broken, all pieces accepted', {}).get('share', 0) + v.get('broken, a piece missing', {}).get('share', 0):.3f}" for b, v in summary.items()), flush=True)
    (ROOT / 'experiments/long_edges_20261111' / f'decoding-{entry}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
