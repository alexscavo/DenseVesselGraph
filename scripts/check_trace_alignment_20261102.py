#!/usr/bin/env python3
"""Do the GT centreline traces lie on bright vessel voxels? (2 Nov 2026)

For ~120 validation patches per dataset: image contrast on the rasterised GT centreline vs a
ring 3-6 voxels around it, (mean_on - mean_ring) / std(patch). Repeated with the traces flipped
along each axis or with H/W swapped. Correct alignment shows clearly higher contrast as-is than
under any transform; a transform that beats 'as-is' indicates a registration/orientation error.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from node_heatmap_probe_20261030 import make_dataset, PLAN, SIZE  # noqa: E402


def ball(r):
    g = np.arange(-r, r + 1)
    return (g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2) <= r * r


TRANSFORMS = {
    'as-is': lambda p: p,
    'flip D': lambda p: p * [-1, 1, 1] + [SIZE - 1, 0, 0],
    'flip H': lambda p: p * [1, -1, 1] + [0, SIZE - 1, 0],
    'flip W': lambda p: p * [1, 1, -1] + [0, 0, SIZE - 1],
    'swap H/W': lambda p: p[:, [0, 2, 1]],
}


def main():
    plan = json.loads(PLAN.read_text())['datasets']
    out = {}
    for ds in ['SyntheticMRI', 'IXI', 'BRAVA']:
        ids = plan[ds]['val_ids'][::max(1, len(plan[ds]['val_ids']) // 120)][:120]
        data = make_dataset(ds, ids, augment=False)
        res = {k: [] for k in TRANSFORMS}
        per_subject = {}
        for i in range(len(data)):
            s = data[i]
            img = np.asarray(s[0][0][0], float)
            g = s[4][0]['guides'].numpy() * SIZE
            if len(g) == 0:
                continue
            pts = g.reshape(-1, 3)
            for name, t in TRANSFORMS.items():
                v = np.round(np.clip(t(pts), 0, SIZE - 1)).astype(int)
                on = np.zeros(img.shape, bool)
                on[v[:, 0], v[:, 1], v[:, 2]] = True
                ring = binary_dilation(on, ball(6)) & ~binary_dilation(on, ball(3))
                if ring.sum() == 0:
                    continue
                c = (img[on].mean() - img[ring].mean()) / (img.std() + 1e-6)
                res[name].append(c)
                if name == 'as-is':
                    per_subject.setdefault(ids[i].split('_')[1], []).append(c)
        out[ds] = dict(patches=len(res['as-is']),
                       transforms={k: dict(mean_contrast=float(np.mean(v)), brighter_than_ring=float(np.mean(np.array(v) > 0)))
                                   for k, v in res.items()},
                       as_is_by_subject={k: dict(patches=len(v), mean_contrast=float(np.mean(v))) for k, v in per_subject.items()})
        print('DS', ds, json.dumps(out[ds]['transforms']), flush=True)
        print('SUBJECTS', ds, json.dumps(out[ds]['as_is_by_subject']), flush=True)
    (ROOT / 'experiments/dense_graph_20261030/trace-alignment-check.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
