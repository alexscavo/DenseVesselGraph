#!/usr/bin/env python3
"""Read the existing train/val targets; never regenerate targets or open test data."""
import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / 'Graph-Native-Betti-Matching-Representation'
OUT = ROOT / 'experiments/q0_ixi_audit_20261028'
IXI = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/transfer_splines_k8_20261015')
MODERN = ROOT.parent / 'gnbm-ixi-local-components-20261026/experiments/ixi_local_components_20261026/plan.json'
BRAVA = ROOT.parent / 'gnbm-brava-long-20261027/experiments/brava_long_20261027/plan.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def distribution(values):
    values = np.asarray(values)
    return dict(patches=len(values), minimum=int(values.min()), maximum=int(values.max()),
                mean=float(values.mean()), p95=float(np.quantile(values, .95)),
                p99=float(np.quantile(values, .99)),
                exceeding={str(n): int((values > n).sum()) for n in (120, 192, 256)})


def load_graph(path):
    # Exact same PyVista point/line parser and float32 conversion as the loader,
    # without importing torch or constructing/loading a neural network.
    mesh = pv.read(path)
    nodes = np.asarray(mesh.points, dtype=np.float32)
    flat = np.asarray(mesh.lines, dtype=np.int64)
    assert flat.size % 3 == 0, path
    cells = flat.reshape(-1, 3)
    assert np.all(cells[:, 0] == 2), path
    return nodes, cells[:, 1:].copy()


def inspect(item):
    root, dataset, split, name, subject, selected, old_count, expected_count = item
    graph_path = root / split / 'vtp' / f'{name}_graph.vtp'
    cache_path = root / split / 'spline' / f'{name}.npz'
    raw = cache_path.read_bytes()
    nodes, edges = load_graph(graph_path)
    with np.load(io.BytesIO(raw), allow_pickle=False) as cache:
        assert nodes.shape == cache['nodes'].shape, name
        assert np.allclose(nodes, cache['nodes'], atol=1e-6, rtol=0), name
        assert np.array_equal(edges, cache['edges']), name
        boundary = cache['boundary'].astype(bool)
        assert boundary.shape == (len(nodes),), name
        assert cache['controls'].shape == (len(edges), 8, 3), name
        assert cache['guides'].shape == (len(edges), 64, 3), name
        assert np.isfinite(cache['metric']).all(), name
        path_points = len(cache['full_paths'])
    assert np.isfinite(nodes).all(), name
    assert not len(edges) or (edges.min() >= 0 and edges.max() < len(nodes)), name
    if expected_count is not None:
        assert len(nodes) == expected_count, name
    degree = np.bincount(edges.ravel(), minlength=len(nodes))
    return dict(dataset=dataset, split=split, sample=name, subject=subject,
                modern_selected=selected, nodes=len(nodes), edges=len(edges),
                older_index_nodes=old_count, isolated=int((degree == 0).sum()),
                degree1=int((degree == 1).sum()), degree2=int((degree == 2).sum()),
                junctions=int((degree >= 3).sum()), boundary=int(boundary.sum()),
                boundary_degree2=int(((degree == 2) & boundary).sum()),
                interior_degree2=int(((degree == 2) & ~boundary).sum()),
                interior_terminals=int(((degree == 1) & ~boundary).sum()),
                original_path_points=path_points,
                spline_sha256=hashlib.sha256(raw).hexdigest(), vtp_sha256=sha(graph_path))


def items():
    modern, brava = read(MODERN), read(BRAVA)
    manifest = read(IXI / 'manifest.json')
    index = {r['sample_id']: r for r in csv.DictReader((IXI / 'patch_index.csv').open())
             if r['split'] in ('train', 'val')}
    result = []
    for split, key in [('train', 'train_ids'), ('val', 'validation_ids')]:
        selected = set(modern['dataset'][f'{split}_ids'])
        assert selected <= set(manifest[key])
        for name in manifest[key]:
            row = index[name]
            assert row['split'] == split
            result.append((IXI, 'IXI', split, name, row['patient_id'], name in selected,
                           int(row['node_count']), int(row['anchor_count'])))
    brava_root = Path(brava['dataset'])
    for split in ('train', 'val'):
        for name in brava['ids'][split]:
            result.append((brava_root, 'BRAVA', split, name, brava['subjects'][name],
                           False, None, None))
    return result


def historical():
    report = MAIN / 'docs/research/evidence/ixi_capacity_192_256'
    spec = read(report / 'selection.json')
    summary = read(report / 'summary.json')
    name = summary['overflow_smoke']['sample_id']
    source = Path(spec['source'])
    path = source / 'train/vtp' / f'{name}_graph.vtp'
    if not path.exists():
        return dict(classification='PRE-CONTRACTION historical population, per archived benchmark/source provenance; original VTP unavailable for independent recount',
                    source=str(source), target=str(path), sample=name, target_available=False,
                    nodes=summary['overflow_smoke']['target_nodes'],
                    edges=summary['overflow_smoke']['target_edges'], degree2=None,
                    benchmark_job='22067', metric_report=str(report / 'summary.json'),
                    metric_report_sha256=sha(report / 'summary.json'),
                    selection_sha256=sha(report / 'selection.json'),
                    limitation='239 is an archived measurement, not a freshly verified current K8 target count')
    nodes, edges = load_graph(path)
    degree = np.bincount(edges.ravel(), minlength=len(nodes))
    assert len(nodes) == summary['overflow_smoke']['target_nodes'] == 239
    return dict(classification='PRE-CONTRACTION: includes intermediate degree-2 object nodes',
                source=str(source), target=str(path), sample=name, sha256=sha(path),
                nodes=len(nodes), edges=len(edges), degree2=int((degree == 2).sum()),
                isolated=int((degree == 0).sum()),
                source_generation=read(source / 'generation_config.json'),
                benchmark_job='22067', metric_report=str(report / 'summary.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    tasks = items()
    if args.smoke:
        groups = {(t[1], t[2]) for t in tasks}
        selected = [next(t for t in tasks if (t[1], t[2]) == key) for key in sorted(groups)]
        selected += [max((t for t in tasks if t[1] == 'IXI' and t[2] == split), key=lambda t: t[-1])
                     for split in ('train', 'val')]
        tasks = list(dict.fromkeys(selected))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(inspect, tasks), 1):
            rows.append(row)
            if i % 1000 == 0:
                print('audited', i, '/', len(tasks), flush=True)
    summaries = {}
    for dataset in ('IXI', 'BRAVA'):
        groups = {}
        for split in ('train', 'val'):
            rr = [r for r in rows if r['dataset'] == dataset and r['split'] == split]
            summary = distribution([r['nodes'] for r in rr])
            summary['subjects'] = len({r['subject'] for r in rr})
            summary['node_roles'] = {k: sum(r[k] for r in rr) for k in
                ('isolated', 'degree1', 'degree2', 'junctions', 'boundary',
                 'boundary_degree2', 'interior_degree2', 'interior_terminals')}
            summary['overflow_120'] = [dict(sample=r['sample'], subject=r['subject'], nodes=r['nodes'],
                modern_selected=r['modern_selected']) for r in rr if r['nodes'] > 120]
            summary['per_subject'] = {s: distribution([r['nodes'] for r in rr if r['subject'] == s])
                                      for s in sorted({r['subject'] for r in rr})}
            selected = [r for r in rr if r['modern_selected']]
            if selected:
                summary['modern_selected'] = distribution([r['nodes'] for r in selected])
                summary['modern_selected']['subjects'] = len({r['subject'] for r in selected})
            groups[split] = summary
        overlap = ({r['subject'] for r in rows if r['dataset'] == dataset and r['split'] == 'train'} &
                   {r['subject'] for r in rows if r['dataset'] == dataset and r['split'] == 'val'})
        assert not overlap, (dataset, overlap)
        summaries[dataset] = groups
    result = dict(completed=True, smoke=args.smoke, rows=len(rows), seconds=time.monotonic()-start,
                  datasets=summaries, exact_loader_vtp_cache_parity=True, test_targets_opened=0,
                  percentiles='NumPy linear interpolated quantiles on retained patches',
                  input_manifests={str(p): sha(p) for p in
                    (IXI / 'manifest.json', IXI / 'patch_index.csv', MODERN, BRAVA)},
                  historical_239=historical())
    prefix = 'smoke' if args.smoke else 'targets'
    write(OUT / f'{prefix}.json', result)
    with (OUT / f'{prefix}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(completed=True, smoke=args.smoke, patches=len(rows),
          seconds=result['seconds'], distributions={d: {s: {k: v for k, v in z.items()
          if k in ('patches', 'subjects', 'maximum', 'p95', 'p99', 'exceeding', 'modern_selected')}
          for s, z in groups.items()} for d, groups in summaries.items()}), indent=2), flush=True)


if __name__ == '__main__':
    main()
