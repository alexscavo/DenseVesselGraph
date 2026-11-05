#!/usr/bin/env python3
"""Rebuild the native 2D crop targets with crop re-contraction (5 Nov 2026), into a NEW folder.

The prepared targets (experiments/native2d_long_20261028/<dataset>/manifest.json, read by
scripts/dense_graph_2d_20261030.make_dataset) contracted the whole-image graph once and then cut crops, so a crop could
keep ordinary degree-2 nodes (a junction whose third branch lies outside the crop, a breaking point of a loop the crop
cut open). Here every crop is rebuilt from the SAME sources at the SAME position, with the same splits and images,
via native2d/targets.crop_graph(recontract=True): degree-2 nodes are contracted except boundary anchors, and breaking
points are inserted only for parallel branches, self-loops and anchor-free rings (3D rule). Targets store explicit
`inserted` and `boundary` masks.

Each crop is also rebuilt with recontract=False and compared array-for-array with the stored target: this checks that
the sources and the encoding (including the uniform-K8 refits of native2d_20261028_targetfix) are reproduced exactly.

Usage: prepare_native2d_recontracted_20261105.py <avwide|reaver|octasynth|plants> [--limit N] [--out DIR]
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from native2d.targets import Graph2D, crop_graph, fit_controls, from_dense_graph, from_mask, save_target
from metrics.loop_aware import inserted_from_degree
from training.evaluation.metrics import graph_betti_numbers

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'experiments/native2d_long_20261028'
DEFAULT_OUT = ROOT / 'experiments/native2d_recontracted_20261105'
DATA = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets')
STORAGE = dict(avwide='files', reaver='files', octasynth='zip', plants='zip')


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def npz_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def load(data):
    with np.load(io.BytesIO(data) if isinstance(data, bytes) else data) as z:
        return {k: z[k].copy() for k in z.files}


def saved(graph, **kwargs):
    stream = io.BytesIO()
    save_target(stream, graph, [128, 128], **kwargs)
    return load(stream.getvalue())


def refit(target):
    """native2d_20261028_targetfix uniform-K8 refit (prepare_native2d_fit_revision / audit_native2d_fit_revision)."""
    m = target['metric']
    inverse = np.linalg.inv(m)
    paths = [target['full_paths'][a:b] @ m.T for a, b in zip(target['path_offsets'][:-1], target['path_offsets'][1:])]
    target['controls'] = np.stack([fit_controls(p) @ inverse.T for p in paths]) if paths else np.empty((0, 8, 2))
    return target


# ---------------------------------------------------------------- per-dataset sources and encodings (as built)

def avwide_graph(source):
    """prepare_native2d_study.inspect_wide: whole-image contracted graph and its bounding boxes."""
    from scipy import sparse
    from scipy.io import loadmat
    raw = loadmat(source, simplify_cells=True)
    g = raw['G']
    nodes = np.asarray(g['V'], float) - 1
    a = sparse.csr_matrix(g['A'])
    rows, cols = sparse.triu(a, k=1).nonzero()
    edges = np.stack((rows, cols), 1)
    assert (a != a.T).nnz == 0 and not a.diagonal().any()
    graph = from_dense_graph(nodes, edges, provenance=dict(dataset='AV-WIDE', source=str(source)))
    return graph, np.array([[p.min(0), p.max(0)] for p in graph.paths])


def avwide_crop(state, row, recontract):
    graph, bounds = state
    x, y = row['crop_start_xy']
    keep = np.where(np.all(bounds[:, 1] >= [x, y], axis=1) & np.all(bounds[:, 0] <= [x + 127, y + 127], axis=1))[0]
    local = Graph2D(graph.nodes, graph.edges[keep], [graph.paths[i] for i in keep], graph.provenance)
    if len(keep) != len(graph.edges):
        used = np.unique(graph.edges[keep])
        isolates = np.where(np.bincount(graph.edges.ravel(), minlength=len(graph.nodes)) == 0)[0]
        used = np.union1d(used, isolates)
        remap = np.full(len(graph.nodes), -1)
        remap[used] = np.arange(len(used))
        local = Graph2D(graph.nodes[used], remap[graph.edges[keep]], [graph.paths[i] for i in keep], graph.provenance)
    return crop_graph(local, [x, y], [128, 128], recontract=recontract)


def avwide_encode(patch, row):
    return refit(saved(patch))


REAVER_AUDIT = ROOT / 'experiments/native2d_20261028/secondary-audit.json'


def reaver_graph(manual):
    import tifffile
    rgb = tifffile.imread(manual)
    return from_mask(rgb[:, :, 1] > 0, is_centerline=True, provenance=dict(source=str(manual)))


def reaver_crop(graph, row, recontract):
    x, y = map(int, re.search(r'_x(\d+)_y(\d+)$', row['sample_id']).groups())
    return crop_graph(graph, [x, y], [128, 128], recontract=recontract)


@lru_cache(maxsize=None)
def reaver_spacing(file):
    audit = json.loads(REAVER_AUDIT.read_text())['REAVER']['images']
    return np.array(next(r for r in audit if r['file'] == file)['spacing_micron_xy'])


def reaver_encode(patch, row):
    """prepare_native2d_reaver (micron metric) followed by the targetfix uniform-K8 refit."""
    spacing = reaver_spacing(Path(row['manual']).name)
    values = saved(patch, spacing_xy=spacing, units='micron')
    assert values['units'].item() == 'mm'
    values['metric'] *= 1000.
    values['units'] = np.array('micron')
    np.testing.assert_allclose(values['nodes'] @ values['metric'].T, patch.nodes * spacing, atol=1e-10)
    return refit(values)


def octa_graph(source):
    from PIL import Image
    from scripts.prepare_native2d_long import csv_graph
    parent = Path(source).name.removesuffix('data.png')
    with Image.open(source) as im:
        width, height = im.size
    xyz, edges = csv_graph(DATA / 'octa-synth-packed/graphs' / parent)
    return from_dense_graph(xyz[:, [1, 0]] * [width, height], edges, provenance=dict(dataset='OCTA-Synth'))


def octa_crop(graph, row, recontract):
    return crop_graph(graph, row['crop_start_xy'], [128, 128], recontract=recontract)


def zip_encode(patch, row):
    from scripts.prepare_native2d_long import encode
    return load(encode(patch))


def plant_graph(source):
    import pyvista as pv
    from scripts.prepare_native2d_long import plant_edges
    mesh = pv.read(source)
    points = np.asarray(mesh.points, float)
    edges, _ = plant_edges(mesh.lines)
    assert not any(a == b for a, b in edges.tolist())
    return from_dense_graph(points[:, :2] * 128, edges, provenance=dict(source=str(source)))


def plant_crop(graph, row, recontract):
    return crop_graph(graph, [0, 0], [128, 128], recontract=recontract)


DATASETS = dict(
    avwide=(lambda row: row['source'], avwide_graph, avwide_crop, avwide_encode),
    reaver=(lambda row: row['manual'], reaver_graph, reaver_crop, reaver_encode),
    octasynth=(lambda row: row['source'], octa_graph, octa_crop, zip_encode),
    plants=(lambda row: row['source_graph'], plant_graph, plant_crop, zip_encode),
)


# ---------------------------------------------------------------- checks

def describe(target):
    n = len(target['nodes'])
    edges = np.asarray(target['edges'], int).reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=n) if len(edges) else np.zeros(n, int)
    cycle = inserted_from_degree(n, edges)
    off = target['path_offsets']
    paths = [target['full_paths'][a:b] @ target['metric'].T for a, b in zip(off[:-1], off[1:])]
    return dict(nodes=n, edges=len(edges), degree2=int((degree == 2).sum()), degree2_on_cycle=int(((degree == 2) & cycle).sum()),
                betti=list(graph_betti_numbers(n, edges)), length=float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in paths)),
                degree=degree, cycle=cycle)


def check_new(old, new):
    """Invariants of a recontracted target against the stored one; returns a list of violations."""
    a, b = describe(old), describe(new)
    bad = []
    pairs = [tuple(sorted(e)) for e in new['edges'].reshape(-1, 2).tolist()]
    if any(u == v for u, v in pairs) or len(set(pairs)) != len(pairs):
        bad.append('not_simple')
    inserted, boundary = new['inserted'], new['boundary']
    if inserted.shape != (b['nodes'],) or boundary.shape != (b['nodes'],):
        bad.append('mask_shape')
    elif not np.array_equal(inserted, b['degree'] == 2):
        bad.append('degree2_not_inserted')
    elif not np.all(b['cycle'][inserted]):
        bad.append('inserted_off_cycle')
    if np.any(inserted & boundary):
        bad.append('inserted_boundary')
    if a['betti'] != b['betti']:
        bad.append('betti')
    if not np.isclose(a['length'], b['length'], rtol=1e-9, atol=1e-7):
        bad.append('length')
    if b['nodes'] > a['nodes']:
        bad.append('more_nodes')
    if len(new['controls']):
        if not np.allclose(new['controls'][:, [0, -1]], new['nodes'][new['edges']], atol=1e-7):
            bad.append('control_endpoints')
        if not np.isfinite(new['controls']).all() or np.abs(new['controls'] @ new['metric'].T).max() >= 1024:
            bad.append('controls')
    if len(old['full_paths']):  # every stored path point is still a path point (geometry unchanged)
        from scipy.spatial import cKDTree
        if len(new['full_paths']) == 0:
            bad.append('geometry')
        else:
            distance, _ = cKDTree(new['full_paths'] @ new['metric'].T).query(old['full_paths'] @ old['metric'].T)
            if distance.max() > 1e-6:
                bad.append('geometry')
    return a, b, bad


def same(x, y, keys):
    return all(k in x and k in y and x[k].shape == y[k].shape and x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k]) for k in keys)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=sorted(DATASETS))
    parser.add_argument('--limit', type=int, default=None, help='smoke test: first N rows only, written under <out>/smoke')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    name = args.dataset
    source_of, build, crop, encode = DATASETS[name]
    old_manifest_path = OLD / name / 'manifest.json'
    manifest = json.loads(old_manifest_path.read_text())
    rows = manifest['samples'][:args.limit] if args.limit else manifest['samples']
    out = (args.out / 'smoke' if args.limit else args.out) / name
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out}/manifest.json exists: refusing to overwrite')
    out.mkdir(parents=True, exist_ok=True)
    missing = sorted({source_of(r) for r in rows if not Path(source_of(r)).exists()})
    if missing:
        write(out / 'missing-sources.json', dict(missing=missing))
        raise SystemExit(f'{len(missing)} sources missing, see {out}/missing-sources.json')
    storage = STORAGE[name]
    archive = ZipFile(OLD / name / 'targets.zip') if storage == 'zip' else None
    packed = ZipFile(out / 'targets.zip.tmp', 'w', ZIP_STORED) if storage == 'zip' else None
    if storage == 'files':
        (out / 'prepared').mkdir(exist_ok=True)
    started = time.time()
    cache_key, state = None, None
    new_rows, records, failures = [], [], []
    for i, row in enumerate(rows):
        source = source_of(row)
        if source != cache_key:
            state, cache_key = build(source), source
        old = load(archive.read(row['target'])) if archive else load(OLD / name / row['target'])
        legacy = encode(crop(state, row, False), row)
        keys = sorted(old)
        reproduced = sorted(legacy) == keys and same(legacy, old, keys)
        try:
            new = encode(crop(state, row, True), row)
        except ValueError as error:
            failures.append(dict(sample_id=row['sample_id'], split=row['split'], error=str(error)))
            continue
        a, b, bad = check_new(old, new)
        changed = not same(new, old, keys)
        records.append(dict(sample_id=row['sample_id'], split=row['split'], legacy_reproduced=reproduced, changed=changed,
                            violations=bad, nodes=[a['nodes'], b['nodes']], edges=[a['edges'], b['edges']],
                            degree2=[a['degree2'], b['degree2']], degree2_on_cycle=[a['degree2_on_cycle'], b['degree2_on_cycle']],
                            inserted=int(new['inserted'].sum()), boundary=int(new['boundary'].sum()), betti=b['betti'],
                            length=[a['length'], b['length']]))
        data = npz_bytes(new)
        r = copy.deepcopy(row)
        if storage == 'zip':
            packed.writestr(row['target'], data)
        else:
            path = out / 'prepared' / Path(row['target']).name
            path.write_bytes(data)
            r['target'] = str(path.relative_to(out))
        r.update(nodes=b['nodes'], edges=b['edges'], inserted_nodes=int(new['inserted'].sum()))
        new_rows.append(r)
        if (i + 1) % 1000 == 0:
            print(f'{name} {i + 1}/{len(rows)} {time.time() - started:.0f}s', flush=True)
    if packed:
        packed.close()
        (out / 'targets.zip.tmp').rename(out / 'targets.zip')
    splits = sorted({r['split'] for r in records})

    def summary(sel):
        return dict(crops=len(sel), legacy_reproduced=sum(r['legacy_reproduced'] for r in sel), changed=sum(r['changed'] for r in sel),
                    with_violations=sum(bool(r['violations']) for r in sel),
                    nodes_before=sum(r['nodes'][0] for r in sel), nodes_after=sum(r['nodes'][1] for r in sel),
                    edges_before=sum(r['edges'][0] for r in sel), edges_after=sum(r['edges'][1] for r in sel),
                    degree2_before=sum(r['degree2'][0] for r in sel), degree2_before_on_cycle=sum(r['degree2_on_cycle'][0] for r in sel),
                    degree2_after=sum(r['degree2'][1] for r in sel), degree2_after_on_cycle=sum(r['degree2_on_cycle'][1] for r in sel),
                    inserted_after=sum(r['inserted'] for r in sel), crops_with_inserted=sum(r['inserted'] > 0 for r in sel),
                    max_nodes_before=max((r['nodes'][0] for r in sel), default=0), max_nodes_after=max((r['nodes'][1] for r in sel), default=0))
    report = dict(dataset=name, source_manifest=str(old_manifest_path), rows=len(rows), written=len(new_rows), failures=failures,
                  per_split={s: summary([r for r in records if r['split'] == s]) for s in splits}, total=summary(records),
                  violations=[r for r in records if r['violations']], not_reproduced=[r['sample_id'] for r in records if not r['legacy_reproduced']],
                  changed=[dict(sample_id=r['sample_id'], split=r['split'], nodes=r['nodes'], degree2=r['degree2'], inserted=r['inserted'])
                           for r in records if r['changed']],
                  seconds=time.time() - started)
    write(out / 'rebuild-report.json', report)
    top = {k: v for k, v in manifest.items() if k != 'samples'}
    write(out / 'manifest.json', dict(**top, samples=new_rows, rebuilt_from_manifest=str(old_manifest_path),
                                      rebuilt_from_manifest_sha256=hashlib.sha256(old_manifest_path.read_bytes()).hexdigest(),
                                      target_revision='recontracted_20261105: crop re-contraction (native2d/targets.crop_graph recontract=True); '
                                      'same sources, crops, splits and images; explicit inserted/boundary node masks'))
    train = [r['nodes'] for r in new_rows if r['split'] == 'train']
    if train:
        write(out / 'audit.json', dict(passed=not report['violations'] and not failures, units=manifest.get('units'),
                                       samples={s: sum(r['split'] == s for r in new_rows) for s in ['train', 'validation']},
                                       training_node_quantiles=dict(zip(['min', 'p50', 'p90', 'p95', 'p99', 'max'], np.quantile(train, [0, .5, .9, .95, .99, 1]).tolist())),
                                       maximum_nodes=int(max(r['nodes'] for r in new_rows)), training_maximum_nodes=int(max(train)),
                                       rebuild_report='rebuild-report.json'))
    print(json.dumps(dict(dataset=name, total=report['total'], failures=len(failures), violations=len(report['violations']),
                          not_reproduced=len(report['not_reproduced']), seconds=round(report['seconds'])), indent=1), flush=True)


if __name__ == '__main__':
    main()
