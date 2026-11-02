"""Read current serialized K8 graph anchors; no images, inference or test patches."""
import csv
import hashlib
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
MAIN = WORKTREE.parent / 'Graph-Native-Betti-Matching-Representation'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    values = np.array([r['nodes'] for r in rows])
    return dict(patches=len(rows), subjects=len({r['subject'] for r in rows}),
                min=int(values.min()), p50=float(np.median(values)),
                p95=float(np.quantile(values, .95)), p99=float(np.quantile(values, .99)),
                max=int(values.max()), mean=float(values.mean()),
                overflow={str(q): int((values > q).sum()) for q in [120, 192, 256]},
                **{k: sum(r[k] for r in rows) for k in
                   ['edges', 'isolated_nodes', 'degree1_nodes', 'degree2_nodes',
                    'junction_nodes', 'boundary_nodes']})


def run():
    started = time.monotonic()
    original = json.loads((WORKTREE / 'experiments/ixi_local_components_20261026/plan.json').read_text())
    multi = json.loads((MAIN / 'experiments/multidataset_continuous/plan.json').read_text())
    result = dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  scope='Serialized train/val K8 nodes, edges and boundary flags only; no test arrays or images',
                  datasets={}, n07_plan_sha256=digest(WORKTREE / 'experiments/ixi_local_components_20261026/plan.json'))
    rows, anomalies = [], []
    for dataset in ['IXI', 'BRAVA']:
        spec = multi['datasets'][dataset]
        root = Path(spec['root'])
        manifest = json.loads((root / 'manifest.json').read_text())
        info = dict(root=str(root), manifest_sha256=digest(root / 'manifest.json'),
                    manifest_matches_existing_plan=digest(root / 'manifest.json') == spec['manifest_sha256'],
                    max_anchors=manifest['max_anchors'], controls=manifest['controls'],
                    exclusions=manifest['exclusion_counts'],
                    boundary_policy=manifest['boundary_policy'], splits={})
        result['datasets'][dataset] = info
        subject_sets = {}
        for split, idkey in [('train', 'train_ids'), ('val', 'validation_ids')]:
            paths = sorted((root / split / 'spline').glob('*.npz'))
            ids = [p.stem for p in paths]
            assert set(ids) == set(manifest[idkey]), (dataset, split, 'membership mismatch')
            assert len(ids) == len(manifest[idkey]) == manifest['counts'][split]
            identity = hashlib.sha256()
            subset = set(original['dataset']['train_ids' if split == 'train' else 'val_ids']) if dataset == 'IXI' else set()
            current = []
            for index, path in enumerate(paths):
                with zipfile.ZipFile(path) as archive:
                    def array(key):
                        with archive.open(key + '.npy') as stream:
                            return np.lib.format.read_array(stream, allow_pickle=False)
                    nodes, edges, boundary = array('nodes'), array('edges'), array('boundary')
                    with archive.open('controls.npy') as stream:
                        version = np.lib.format.read_magic(stream)
                        shape, _, _ = np.lib.format._read_array_header(stream, version)
                    assert nodes.ndim == 2 and nodes.shape[1] == 3 and np.isfinite(nodes).all()
                    assert edges.shape == (len(edges), 2)
                    assert len(boundary) == len(nodes) and shape == (len(edges), 8, 3)
                    assert not len(edges) or (edges.min() >= 0 and edges.max() < len(nodes))
                    degree = np.bincount(edges.reshape(-1), minlength=len(nodes))
                    identity.update(path.name.encode())
                    for entry in archive.infolist():
                        identity.update(f'{entry.filename}:{entry.CRC}:{entry.file_size}'.encode())
                subject = spec['subjects'][path.stem.split('_')[1]]
                row = dict(dataset=dataset, split=split, subject=subject, sample_id=path.stem,
                           nodes=len(nodes), edges=len(edges),
                           isolated_nodes=int((degree == 0).sum()), degree1_nodes=int((degree == 1).sum()),
                           degree2_nodes=int((degree == 2).sum()), junction_nodes=int((degree >= 3).sum()),
                           boundary_nodes=int(boundary.sum()), n07_subset=path.stem in subset)
                rows.append(row)
                current.append(row)
                if len(nodes) > 120:
                    anomalies.append(row)
                if (index + 1) % 2000 == 0:
                    print(dataset, split, index + 1, flush=True)
            info['splits'][split] = summarize(current)
            info['splits'][split]['cache_crc_manifest_sha256'] = identity.hexdigest()
            subject_sets[split] = {r['subject'] for r in current}
            if dataset == 'IXI':
                assert sum(r['n07_subset'] for r in current) == len(subset)
                info.setdefault('n07_subset', {})[split] = summarize([r for r in current if r['n07_subset']])
        assert not subject_sets['train'] & subject_sets['val'], 'subject leakage'
        info['subject_disjoint'] = True
        info['all_retained_development_targets_fit_120'] = not any(info['splits'][s]['overflow']['120'] for s in ['train', 'val'])
    for filename, data in [('per-patch.csv', rows), ('overflow-above120.csv', anomalies)]:
        with (HERE / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(data)
    groups = defaultdict(list)
    for r in rows:
        groups[r['dataset'], r['split'], r['subject']].append(r)
    per_subject = [dict(dataset=d, split=s, subject=u, **summarize(r))
                   for (d, s, u), r in sorted(groups.items())]
    result['seconds'] = time.monotonic() - started
    (HERE / 'per-subject.json').write_text(json.dumps(per_subject, indent=2) + '\n')
    (HERE / 'target-capacity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
