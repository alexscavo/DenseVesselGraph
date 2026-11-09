#!/usr/bin/env python3
"""TubeTK (Skwerer trees) as a 3D target dataset, prepared like BRAVA (5 Nov 2026).

Ground truth = the Skwerer centreline trees (sub-X_space-MRA_desc-skwerer_vesseltree.pkl: parent-linked nodes in
RAS mm of the MRA world frame, radius in mm), imported directly as a graph like BRAVA's SWC import: no
skeletonization, no re-extraction from masks, the CaravelMetrics graph/ folder is not used. Input image = the
skull-stripped MRA (desc-brain_mask, outside 0). The graph is clipped to the same mask (edges with an endpoint
outside are cut; inside parts kept). The only other graph edits are (1) contraction of zero-length edges
(coincident consecutive samples; geometry unchanged) and (2) on Normal-008 and Normal-014 removal of the repeated subtree
(exact duplicates only: identical positions and identical edges, verified). The seg slot holds the provided
rasterized Skwerer tree mask x brain mask (a rasterization of the GT, not an independent segmentation).
Bullitt trees (forest of unlinked tubes) are used only for a coverage QC (Bullitt length not covered by
Skwerer within 1 mm).

Modes: inventory | subjects [--only S ...] | smoke | stage | patches | collect
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from scripts.ixi_vessel_graph import VesselGraph, write_source_graph
from scripts.prepare_brava import compact_graph, digest, save

SOURCE = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TubeTK')
PRE = SOURCE / 'TubeTK_preprocessed'
DEFAULT_OUT = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TubeTK_skwerer_20261105')
# Normal-008: README says its Skwerer tree repeats a subtree (31,096 duplicated nodes). Normal-014 has the same defect
# (26,091 position duplicates, not in the README), found by the candidate count recorded for every subject on 6 Nov;
# the identical exact-duplicate rule is applied to it (flagged for the user's review).
DEDUPE_SUBJECTS = {'sub-Normal008', 'sub-Normal014'}
SMOKE_SUBJECTS = ['sub-Normal008', 'sub-Normal099']  # duplicate subtree + both trees; tall grid, mask-containment case
SMOKE_PATCH_SUBJECTS = ['sub-Normal002', 'sub-Normal008', 'sub-Normal099']  # the patch generator needs >= 3 patients
COVER_MM = 1.0                               # Bullitt sample covered when a Skwerer centreline is within 1 mm
LARGE_RADIUS_MM = (1.5, 2.0)                 # calibre proxies for missing ICA / siphon (no anatomical labels exist)
SHIFT_MM = 2.0                               # alignment control: same graph shifted +-2 mm along each axis
SPLIT_SEED = 42


def paths(subject):
    m, v = PRE / subject / 'mra', PRE / subject / 'vessel'
    return dict(preproc=m / f'{subject}_desc-preproc_MRA.nii.gz', brain=m / f'{subject}_desc-brain_MRA.nii.gz',
                mask=m / f'{subject}_space-MRA_desc-brain_mask.nii.gz',
                skwerer=v / f'{subject}_space-MRA_desc-skwerer_vesseltree.pkl',
                bullitt=v / f'{subject}_space-MRA_desc-bullitt_vesseltree.pkl',
                skwerer_mask=v / f'{subject}_space-MRA_label-vessel_desc-skwerer_mask.nii.gz')


def read_tsv(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def inventory():
    manifest = read_tsv(SOURCE / 'manifest.tsv')
    participants = {r['participant_id']: r for r in read_tsv(SOURCE / 'participants.tsv')}
    rows = []
    for r in manifest:
        if r['vessel_tree_skwerer'] != 'true':
            continue
        s = r['participant_id']; p = paths(s)
        for key in ('preproc', 'brain', 'mask', 'skwerer', 'skwerer_mask'):
            assert p[key].is_file(), (s, key)
        assert (r['vessel_tree_bullitt'] == 'true') == p['bullitt'].is_file(), s
        part = participants[s]
        rows.append(dict(subject=s, has_bullitt=p['bullitt'].is_file(), family_id=part['family_id'] if part['family_id'] != 'n/a' else '',
                         exclude_suggested=part['exclude_suggested'] == 'true', t1_sequence=r['t1_sequence']))
    rows.sort(key=lambda r: r['subject'])
    for i, r in enumerate(rows):
        r['patient_id'] = f'{i:06d}'
    return rows


# ---------------------------------------------------------------- graph import

def load_tree(path):
    tree = pickle.load(path.open('rb'))
    nodes = tree['nodes']
    assert tree['space'] == 'RAS_mm', tree['space']
    ids = np.array([n['id'] for n in nodes]); assert np.array_equal(ids, np.arange(len(nodes))), path
    parent = np.array([n['parent'] for n in nodes], dtype=np.int64)
    xyz = np.array([n['xyz'] for n in nodes], dtype=np.float64)
    radius = np.array([n['radius'] for n in nodes], dtype=np.float64)
    assert np.isfinite(xyz).all() and (radius >= 0).all() and parent.max() < len(nodes)
    child = np.flatnonzero(parent >= 0)
    edges = np.column_stack([parent[child], child])
    extra = {k: np.array([n[k] for n in nodes]) for k in ('tube_id', 'artery') if k in nodes[0]}
    return xyz, radius, edges, extra


def _components(n, edges):
    parent = np.arange(n)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([root(i) for i in range(n)])


def _relabel(xyz, radius, edges, keep):
    index = -np.ones(len(xyz), dtype=np.int64); index[keep] = np.arange(int(keep.sum()))
    e = index[edges]; assert (e >= 0).all()
    return xyz[keep], radius[keep], e


def contract_zero_length(xyz, radius, edges):
    """Merge coincident consecutive samples (zero-length edges). Positions are identical, so no geometry changes."""
    zero = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1) < 1e-9
    if not zero.any():
        return xyz, radius, edges, 0
    rep = _components(len(xyz), edges[zero])
    e = rep[edges]; e = e[e[:, 0] != e[:, 1]]
    e = np.unique(np.sort(e, axis=1), axis=0)
    keep = np.zeros(len(xyz), bool); keep[rep] = True
    xyz2, radius2, e2 = _relabel(xyz, radius, e, keep)
    return xyz2, radius2, e2, int(len(xyz) - keep.sum())


def exact_duplicates(xyz, edges):
    """Exact duplicate edges: an edge whose unordered pair of endpoint positions equals that of an earlier edge
    sharing no node with it (edges sharing a node are a back-and-forth of one chain, not a copy). Returns the
    duplicate-edge mask and the canonical node of each position (first occurrence)."""
    _, first_index, inverse = np.unique(xyz, axis=0, return_index=True, return_inverse=True)
    pos = inverse.reshape(-1)
    seen = {}; duplicate = np.zeros(len(edges), bool)
    for k, (a, b) in enumerate(edges.tolist()):
        key = (min(pos[a], pos[b]), max(pos[a], pos[b]))
        earlier = seen.setdefault(key, [])
        if any(not ({a, b} & set(edges[j].tolist())) for j in earlier):
            duplicate[k] = True
        else:
            earlier.append(k)
    return duplicate, first_index[pos]


def dedupe(xyz, radius, edges):
    """Remove exact duplicates only: duplicate edges (identical endpoint positions, see exact_duplicates) and the
    nodes left without edges whose position repeats a kept node. Edges of the copy that are NOT identical to an
    original edge are kept; their copy endpoints coincide exactly with an original node and are merged into it."""
    duplicate, canonical = exact_duplicates(xyz, edges)
    kept = edges[~duplicate]
    used = np.zeros(len(xyz), bool); used[kept.ravel()] = True
    repeated = canonical != np.arange(len(xyz))
    removed = repeated & ~used                     # pure copy nodes
    # copy nodes still carrying a non-identical edge: merge into the original node at the same position
    merge = repeated & used & np.isin(np.arange(len(xyz)), edges[duplicate].ravel())
    target = np.arange(len(xyz)); target[merge] = canonical[merge]
    extra = kept[merge[kept].any(1)]
    reattached = [dict(edge=[int(a), int(b)], to=[int(target[a]), int(target[b])],
                       length_mm=round(float(np.linalg.norm(xyz[a] - xyz[b])), 4)) for a, b in extra.tolist()]
    e = target[kept]; assert (e[:, 0] != e[:, 1]).all()
    keep = ~(removed | merge)
    xyz2, radius2, e2 = _relabel(xyz, radius, e, keep)
    before = VesselGraph(xyz, np.zeros(len(xyz)), list(map(tuple, edges.tolist())), [], radius).betti()
    after = VesselGraph(xyz2, np.zeros(len(xyz2)), list(map(tuple, e2.tolist())), [], radius2).betti()
    return xyz2, radius2, e2, dict(removed_nodes=int((removed | merge).sum()), pure_copy_nodes=int(removed.sum()),
                                   merged_copy_endpoints=int(merge.sum()), removed_duplicate_edges=int(duplicate.sum()),
                                   kept_nonidentical_copy_edges=reattached, betti_before=list(before), betti_after=list(after),
                                   rule='edge removed iff an earlier node-disjoint edge has the identical unordered pair of endpoint positions')


def world_to_voxel(xyz, affine):
    return (np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(affine).T)[:, :3]


def inside_mask(vox, mask):
    idx = np.rint(vox).astype(np.int64)
    inb = ((idx >= 0) & (idx < np.asarray(mask.shape))).all(1)
    out = np.zeros(len(vox), bool)
    out[inb] = mask[idx[inb, 0], idx[inb, 1], idx[inb, 2]]
    return out


def densify(xyz, edges, step=0.25):
    """Node positions plus interpolated samples on edges longer than `step` mm (Skwerer link edges reach ~5 mm)."""
    a, b = xyz[edges[:, 0]], xyz[edges[:, 1]]
    length = np.linalg.norm(b - a, axis=1)
    extra = [a[i] + (b[i] - a[i]) * t[:, None] for i in np.flatnonzero(length > step)
             for t in [np.linspace(0, 1, int(np.ceil(length[i] / step)) + 1)[1:-1]]]
    return np.concatenate([xyz] + extra) if extra else xyz


def edge_lengths(xyz, edges):
    return np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)


# ---------------------------------------------------------------- per subject

def process(subject, out):
    directory = out / 'volumes' / subject
    if (directory / 'complete.json').exists():
        marker = json.loads((directory / 'complete.json').read_text())
        if marker.get('completed'):
            return marker
    directory.mkdir(parents=True, exist_ok=True)
    p = paths(subject)
    hashes = {k: digest(v) for k, v in p.items() if v.is_file()}
    pre_img, brain_img, mask_img = nib.load(p['preproc']), nib.load(p['brain']), nib.load(p['mask'])
    affine, shape = brain_img.affine, brain_img.shape
    for img in (pre_img, mask_img):
        assert img.shape == shape and np.allclose(img.affine, affine, atol=1e-5)
        assert img.header.get_xyzt_units()[0] == 'mm'
    spacing = np.asarray(brain_img.header.get_zooms()[:3], dtype=float)
    assert np.allclose(np.abs(affine[:3, :3]), np.diag(spacing), atol=1e-5), 'non-diagonal affine'
    mask = np.asanyarray(mask_img.dataobj) > 0
    pre = np.asanyarray(pre_img.dataobj); brain = np.asanyarray(brain_img.dataobj)
    stripped = np.where(mask, pre, 0).astype(pre.dtype)
    brain_equals = bool(np.array_equal(brain, stripped) and brain.dtype == pre.dtype)
    image_path = directory / 'brain_skullstripped.nii.gz'
    for link in (image_path, directory / 'brain_mask.nii.gz'):
        if link.is_symlink() or link.exists():
            link.unlink()
    if brain_equals:  # use the provided desc-brain_MRA unchanged
        image_path.symlink_to(p['brain'].resolve())
    else:
        header = pre_img.header.copy(); header.set_xyzt_units('mm')
        nib.save(nib.Nifti1Image(stripped, affine, header), image_path)
    (directory / 'brain_mask.nii.gz').symlink_to(p['mask'].resolve())
    del pre, brain

    # seg slot: rasterized Skwerer tree (provided) restricted to the brain mask
    vmask_img = nib.load(p['skwerer_mask']); assert vmask_img.shape == shape and np.allclose(vmask_img.affine, affine, atol=1e-5)
    vmask = np.asanyarray(vmask_img.dataobj) > 0
    seg = (vmask & mask).astype(np.uint8)
    header = mask_img.header.copy(); header.set_data_dtype(np.uint8); header.set_xyzt_units('mm')
    nib.save(nib.Nifti1Image(seg, affine, header), directory / 'vessel_skwerer_inmask.nii.gz')

    # ---- Skwerer graph import
    xyz, radius, edges, _ = load_tree(p['skwerer'])
    raw_counts = dict(nodes=len(xyz), edges=len(edges), roots=int(len(xyz) - len(edges)),
                      position_duplicate_extra=int(len(xyz) - len(np.unique(xyz, axis=0))))
    xyz, radius, edges, zero_merged = contract_zero_length(xyz, radius, edges)
    dup_report = None
    if subject in DEDUPE_SUBJECTS:
        xyz, radius, edges, dup_report = dedupe(xyz, radius, edges)
    else:
        duplicate, canonical = exact_duplicates(xyz, edges)
        dup_report = dict(removed_nodes=0, candidate_duplicate_edges_not_removed=int(duplicate.sum()),
                          repeated_positions_not_removed=int((canonical != np.arange(len(xyz))).sum()))
    vox = world_to_voxel(xyz, affine)
    degree = np.bincount(edges.ravel(), minlength=len(xyz))
    full = VesselGraph(vox, degree, list(map(tuple, edges.tolist())), [vox[list(e)] for e in edges], radius / spacing[0])
    full_betti = full.betti()
    total_mm = float(edge_lengths(xyz, edges).sum())
    lengths = edge_lengths(xyz, edges)

    # ---- clip to the brain mask
    inside = inside_mask(vox, mask)
    kept_edges = edges[inside[edges].all(1)]
    kept_mm = float(edge_lengths(xyz, kept_edges).sum())
    used = np.zeros(len(xyz), bool); used[kept_edges.ravel()] = True
    cxyz, cradius, cedges = _relabel(xyz, radius, kept_edges, used)
    cvox = world_to_voxel(cxyz, affine)
    cdeg = np.bincount(cedges.ravel(), minlength=len(cxyz))
    dense = VesselGraph(cvox, cdeg, list(map(tuple, cedges.tolist())), [cvox[list(e)] for e in cedges],
                        cradius / spacing[0], skeleton_voxels=len(cxyz))
    dense_betti = dense.betti()
    assert dense_betti[1] == 0, ('cycle in clipped Skwerer graph', dense_betti)
    comp = _components(len(cxyz), cedges.tolist())
    comp_len = defaultdict(float)
    for (a, b), L in zip(cedges.tolist(), edge_lengths(cxyz, cedges)):
        comp_len[comp[a]] += L
    comp_len = np.array(sorted(comp_len.values()))
    reps = dict(junction_only=compact_graph(dense, spacing, cradius, False),
                adaptive=compact_graph(dense, spacing, cradius, True))
    for name, g in reps.items():
        write_source_graph(directory / 'graphs' / name, g, affine, shape)
    np.savez_compressed(directory / 'dense_graph.npz', world_mm=cxyz, voxel=cvox, radius_mm=cradius, edges=cedges)

    # ---- alignment: centreline samples on bright voxels vs controls
    img = np.asanyarray(nib.load(image_path).dataobj).astype(np.float32)
    in_mask_values = img[mask]
    bright = float(np.percentile(in_mask_values, 95))
    samples = densify(cxyz, cedges)
    def bright_fraction(points_mm):
        v = np.rint(world_to_voxel(points_mm, affine)).astype(np.int64)
        ok = ((v >= 0) & (v < np.asarray(shape))).all(1)
        vals = np.zeros(len(v), np.float32); vals[ok] = img[v[ok, 0], v[ok, 1], v[ok, 2]]
        return float((vals >= bright).mean()), float(np.median(vals))
    frac, median_value = bright_fraction(samples)
    shifts = {}
    for axis in range(3):
        for sign in (-1, 1):
            d = np.zeros(3); d[axis] = sign * SHIFT_MM
            shifts[f'{"xyz"[axis]}{"+" if sign > 0 else "-"}{SHIFT_MM:g}mm'] = bright_fraction(samples + d)[0]
    rng = np.random.default_rng(0)
    random_fraction = float((in_mask_values[rng.choice(len(in_mask_values), 20000, replace=False)] >= bright).mean())
    seg_support = float(seg[tuple(np.rint(cvox).astype(int).T)].mean()) if len(cvox) else float('nan')
    alignment = dict(bright_threshold_p95_in_mask=bright, centreline_bright_fraction=frac, centreline_median_intensity=median_value,
                     random_in_mask_bright_fraction=random_fraction, shifted_bright_fraction=shifts,
                     best_shift_fraction=max(shifts.values()), samples=int(len(samples)),
                     centreline_in_seg_fraction=seg_support)

    # ---- Bullitt coverage QC
    coverage = None; missing_vox = np.zeros((0, 3), np.float32)
    if p['bullitt'].is_file():
        bxyz, bradius, bedges, extra = load_tree(p['bullitt'])
        bvox = world_to_voxel(bxyz, affine); binside = inside_mask(bvox, mask)
        be = bedges[binside[bedges].all(1)]
        blen = edge_lengths(bxyz, be); mid = 0.5 * (bxyz[be[:, 0]] + bxyz[be[:, 1]])
        dist, _ = cKDTree(samples).query(mid, distance_upper_bound=COVER_MM + 1e-9)
        miss = dist > COVER_MM
        er = 0.5 * (bradius[be[:, 0]] + bradius[be[:, 1]])
        tube = extra['tube_id'][be[:, 0]]; artery = extra['artery'][be[:, 0]]
        per_tube = []
        for t in np.unique(tube):
            sel = tube == t; L = float(blen[sel].sum()); Lm = float(blen[sel & miss].sum())
            if Lm > 0:
                per_tube.append(dict(tube_id=int(t), length_mm=round(L, 2), missing_mm=round(Lm, 2),
                                     mean_radius_mm=round(float(np.average(er[sel], weights=blen[sel] + 1e-12)), 3),
                                     mean_z_mm=round(float(mid[sel, 2].mean()), 1)))
        per_tube.sort(key=lambda r: -r['missing_mm'])
        per_artery = {str(bool(a)): dict(length_mm=float(blen[artery == a].sum()), missing_mm=float(blen[(artery == a) & miss].sum()))
                      for a in np.unique(artery)}
        coverage = dict(bullitt_nodes=int(len(bxyz)), bullitt_tubes=int(len(np.unique(extra['tube_id']))),
                        bullitt_length_in_mask_mm=float(blen.sum()), bullitt_length_total_mm=float(edge_lengths(bxyz, bedges).sum()),
                        missing_mm=float(blen[miss].sum()), missing_bullitt_fraction=float(blen[miss].sum() / max(blen.sum(), 1e-9)),
                        **{f'missing_radius_ge_{r:g}mm_mm': float(blen[miss & (er >= r)].sum()) for r in LARGE_RADIUS_MM},
                        **{f'bullitt_radius_ge_{r:g}mm_mm': float(blen[er >= r].sum()) for r in LARGE_RADIUS_MM},
                        tubes_with_missing=len(per_tube), tubes_fully_missing=sum(r['missing_mm'] >= 0.95 * r['length_mm'] for r in per_tube),
                        worst_tubes=per_tube[:8], per_artery_flag=per_artery,
                        artery_flag_values=sorted(map(str, np.unique(extra['artery']).tolist())))
        missing_vox = bvox[be[miss]].mean(1).astype(np.float32)
    removed = ~inside
    np.savez_compressed(directory / 'qc_points.npz', skwerer_kept_vox=cvox.astype(np.float32),
                        skwerer_removed_vox=vox[removed].astype(np.float32), bullitt_missing_vox=missing_vox,
                        skwerer_edges=cedges.astype(np.int32))
    marker = dict(completed=True, subject=subject, shape=list(shape), spacing=spacing.tolist(), affine=affine.tolist(),
                  source_sha256={k: hashes[k] for k in hashes}, source_paths={k: str(v) for k, v in p.items() if v.is_file()},
                  mask_used='desc-brain_mask (carotid-inclusive widened HD-BET mask)',
                  brain_mra_equals_preproc_times_mask=brain_equals,
                  image=str(image_path), image_source=('desc-brain_MRA (verified == desc-preproc_MRA * desc-brain_mask)' if brain_equals
                                                       else 'desc-preproc_MRA * desc-brain_mask written here (desc-brain_MRA differs)'),
                  seg='vessel_skwerer_inmask.nii.gz = provided desc-skwerer vessel mask (rasterized tree) * desc-brain_mask',
                  seg_voxels=int(seg.sum()), vessel_mask_voxels=int(vmask.sum()), vessel_mask_inside_fraction=float(seg.sum() / max(vmask.sum(), 1)),
                  skwerer_raw=raw_counts, zero_length_edges_contracted=zero_merged, duplicates=dup_report,
                  skwerer_full=dict(nodes=len(xyz), edges=len(edges), betti=list(full_betti), length_mm=total_mm,
                                    edge_length_p99_mm=float(np.percentile(lengths, 99)), edge_length_max_mm=float(lengths.max()),
                                    edges_longer_than_0p5mm=int((lengths > .5).sum())),
                  clipping=dict(node_inside_fraction=float(inside.mean()), kept_length_mm=kept_mm, kept_length_fraction=kept_mm / max(total_mm, 1e-9),
                                cut_edges=int(len(edges) - len(kept_edges)), components=int(dense_betti[0]),
                                components_shorter_than_1mm=int((comp_len < 1).sum()), smallest_component_mm=float(comp_len.min()) if len(comp_len) else 0.,
                                largest_component_mm=float(comp_len.max()) if len(comp_len) else 0.),
                  representations={k: dict(nodes=g.node_count, edges=g.edge_count, betti=list(g.betti())) for k, g in reps.items()},
                  dense=dict(nodes=dense.node_count, edges=dense.edge_count, betti=list(dense_betti)),
                  alignment=alignment, bullitt_coverage=coverage,
                  graph_policy='Skwerer parent links preserved; zero-length edges contracted; exact duplicate subtree removed on Normal-008 and Normal-014 only; '
                               'clipped to desc-brain_mask (edge kept iff both endpoints in mask, nearest voxel); adaptive = optimal_keep_mask '
                               'with max(0.1 mm, 0.75 x radius_mm) as BRAVA; no skeletonization, pruning, smoothing or vessel-mask containment')
    save(directory / 'complete.json', marker)
    return marker


def run_subjects(out, subjects, workers):
    results, failures = {}, {}
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(process, s, out): s for s in subjects}
        for f in as_completed(futures):
            s = futures[f]
            try:
                m = f.result(); results[s] = m
                print('done', s, 'kept', round(m['clipping']['kept_length_fraction'], 4), 'adaptive', m['representations']['adaptive'],
                      'bright', round(m['alignment']['centreline_bright_fraction'], 3), flush=True)
            except Exception as error:  # record every failure, do not hide later ones
                failures[s] = repr(error); print('FAILED', s, repr(error), flush=True)
    save(out / 'logs' / f'subjects_failures_{len(subjects)}.json', failures)
    if failures:
        raise SystemExit(f'{len(failures)} subjects failed: {sorted(failures)}')
    return results


# ---------------------------------------------------------------- split / staging

def family_safe_split(records, seed=SPLIT_SEED):
    """assign_splits (70/15/15, seed 42, as IXI/BRAVA), then keep each family on one side: a family split across
    sets goes wholly to train, and the vacated val/test places are refilled from family-free train subjects in the
    helper's own hash order, so the 67/14/16 totals are unchanged."""
    from scripts.prepare_real_graph_patch_sources import assign_splits
    base = assign_splits(records, seed); assignment = dict(base)
    key = lambda r: hashlib.sha256(f"{seed}:{r['dataset']}:{r['modality']}:{r['subject']}".encode()).hexdigest()
    families = defaultdict(list)
    for r in records:
        if r['family_id']:
            families[r['family_id']].append(r)
    moves = []
    for fam, members in sorted(families.items()):
        if len({assignment[m['patient_id']] for m in members}) > 1:
            for m in members:
                if assignment[m['patient_id']] != 'train':
                    moves.append(dict(subject=m['subject'], family_id=fam, from_split=assignment[m['patient_id']], to_split='train'))
                    assignment[m['patient_id']] = 'train'
    deficit = Counter(base.values()); deficit.subtract(Counter(assignment.values()))
    donors = sorted([r for r in records if not r['family_id'] and assignment[r['patient_id']] == 'train'], key=key)
    for split in ('val', 'test'):
        for _ in range(deficit[split]):
            r = donors.pop(0); assignment[r['patient_id']] = split
            moves.append(dict(subject=r['subject'], family_id='', from_split='train', to_split=split))
    assert Counter(assignment.values()) == Counter(base.values())
    for fam, members in families.items():
        assert len({assignment[m['patient_id']] for m in members}) == 1, fam
    return assignment, base, moves


def stage(out, subjects=None, base=None, smoke_assignment=None):
    rows = inventory()
    if subjects is not None:
        rows = [r for r in rows if r['subject'] in subjects]
    base = base or out
    source = base / 'sources'; source.mkdir(parents=True, exist_ok=False)
    records = []
    for r in rows:
        folder = out / 'volumes' / r['subject']; marker = json.loads((folder / 'complete.json').read_text()); assert marker['completed']
        records.append(dict(patient_id=r['patient_id'], dataset='tubetk', modality='mra', source_split='train', subject=r['subject'],
                            family_id=r['family_id'], image=str(folder / 'brain_skullstripped.nii.gz'), seg=str(folder / 'vessel_skwerer_inmask.nii.gz'),
                            brain_mask=str(folder / 'brain_mask.nii.gz'), graph=str(folder / 'graphs' / 'adaptive')))
    if smoke_assignment is None:
        assignment, helper, moves = family_safe_split(records)
    else:
        assignment = {r['patient_id']: smoke_assignment[i] for i, r in enumerate(records)}; helper, moves = dict(assignment), []
    for r in records:
        for name, origin, suffix in [('raw', r['image'], '.nii.gz'), ('seg', r['seg'], '.nii.gz'), ('brain_mask', r['brain_mask'], '.nii.gz'),
                                     ('graphs/adaptive', r['graph'], '')]:
            d = source / name; d.mkdir(parents=True, exist_ok=True)
            (d / (r['patient_id'] + suffix)).symlink_to(Path(origin).resolve())
        r['patch_split'] = assignment[r['patient_id']]; r['helper_split'] = helper[r['patient_id']]
    with (source / 'source_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    save(source / 'source_manifest.json', dict(subjects=records, family_moves=moves,
         segmentation_role='rasterized_skwerer_tree_in_brain_mask_not_independent_segmentation',
         graph_policy='Skwerer tree, clipped to desc-brain_mask; radius-adaptive simplification max(0.1mm, 0.75*radius_mm) as BRAVA'))
    with (source / 'patient_split.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['patient_id', 'split']); w.writeheader()
        w.writerows(dict(patient_id=k, split=v) for k, v in sorted(assignment.items()))
    return source


SEMANTICS = dict(
    segmentation_role='rasterized_skwerer_tree_in_brain_mask_not_independent_segmentation',
    vessel_segmentations_created=False,
    seg_source='TubeTK_preprocessed/sub-X/vessel/sub-X_space-MRA_label-vessel_desc-skwerer_mask.nii.gz x desc-brain_mask: the GT tree '
               'rasterized (every node/segment a ball of its radius; sub-voxel vessels are 1-voxel lines). It is a rendering of the '
               'ground-truth graph, not an independent segmentation; it exists only because the legacy loader needs a seg slot.',
    note='Do not use seg/ as independent vessel evidence (vessel Dice against it measures agreement with the rasterized GT tree, '
         'mask-input oracles are graph-derived). Image input = skull-stripped MRA.',
    normalization='Same existing IXI/BRAVA normalization (legacy MAD threshold, float P99.5 fallback when MAD is zero)')


def patches(source, output, workers):
    subprocess.run([sys.executable, 'scripts/generate_synthetic_mri_dataset.py', '--root', str(source), '--output-dir', str(output),
                    '--split-output', str(source / 'patient_split.csv'), '--patch-size', '64', '64', '64', '--pad', '0', '0', '0',
                    '--maximum-stride', '40', '40', '40', '--workers', str(workers), '--resume', '--allow-existing-split-proportions'],
                   check=True, cwd=REPO)
    from scripts.curate_vessel_training_patches import curate
    summary = curate(output, apply=True, all_splits=True)
    save(output / 'tubetk_semantics.json', SEMANTICS)
    print(json.dumps(summary, indent=2), flush=True)


def smoke(out, workers):
    run_subjects(out, SMOKE_PATCH_SUBJECTS, min(workers, 3))  # completed subjects are reused
    base = out / 'smoke'
    source = stage(out, SMOKE_PATCH_SUBJECTS, base, smoke_assignment=['train', 'val', 'test'])
    patches(source, base / 'patches', workers)
    print('smoke staged+patched', flush=True)


# ---------------------------------------------------------------- collect

def _stats(values):
    v = np.asarray(values, float)
    return dict(n=int(len(v)), median=float(np.median(v)), min=float(v.min()), max=float(v.max()), p5=float(np.percentile(v, 5)),
                p95=float(np.percentile(v, 95)), mean=float(v.mean())) if len(v) else dict(n=0)


def collect(out, cache_name):
    rows = inventory(); markers = {r['subject']: json.loads((out / 'volumes' / r['subject'] / 'complete.json').read_text()) for r in rows}
    src = json.loads((out / 'sources' / 'source_manifest.json').read_text())
    split = {r['subject']: r['patch_split'] for r in src['subjects']}
    cache = out / cache_name; manifest = json.loads((cache / 'manifest.json').read_text()); assert (cache / 'READY').exists()
    index = list(csv.DictReader((cache / 'patch_index.csv').open()))
    patches_root = out / 'patches'
    view = json.loads((patches_root / 'training_view.json').read_text())
    grid = list(csv.DictReader((patches_root / 'patch_index.csv').open()))
    pid_split = {r['patient_id']: r['patch_split'] for r in src['subjects']}
    assert all(pid_split[r['patient_id']] == r['split'] for r in grid), 'patch split differs from subject split'
    assert all(pid_split[r['patient_id']] == r['split'] for r in index), 'cache split differs from subject split'
    per_patch = {s: dict(nodes=_stats([int(r['anchor_count']) for r in index if r['split'] == s]),
                         branches=_stats([int(r['branch_count']) for r in index if r['split'] == s]),
                         inserted_loop_anchors=int(sum(int(r['inserted_count']) for r in index if r['split'] == s)))
                 for s in ('train', 'val', 'test')}
    # alignment outliers: robust z on bright fraction, or a +-2 mm shifted copy at least as bright
    frac = np.array([markers[s]['alignment']['centreline_bright_fraction'] for s in markers])
    med = float(np.median(frac)); mad = float(np.median(np.abs(frac - med))) * 1.4826
    subjects = []
    for r in rows:
        m = markers[r['subject']]; a = m['alignment']; c = m['bullitt_coverage']
        rec = dict(subject=r['subject'], patient_id=r['patient_id'], split=split[r['subject']], family_id=r['family_id'],
                   has_bullitt=r['has_bullitt'], brain_mra_equals_preproc_times_mask=m['brain_mra_equals_preproc_times_mask'],
                   kept_length_fraction=m['clipping']['kept_length_fraction'], kept_length_mm=m['clipping']['kept_length_mm'],
                   skwerer_length_mm=m['skwerer_full']['length_mm'], components_after_clip=m['clipping']['components'],
                   zero_length_edges_contracted=m['zero_length_edges_contracted'], duplicate_nodes_removed=m['duplicates']['removed_nodes'],
                   adaptive_nodes=m['representations']['adaptive']['nodes'], adaptive_edges=m['representations']['adaptive']['edges'],
                   bright_fraction=a['centreline_bright_fraction'], random_bright_fraction=a['random_in_mask_bright_fraction'],
                   best_shift_bright_fraction=a['best_shift_fraction'], centreline_in_seg_fraction=a['centreline_in_seg_fraction'],
                   alignment_flag=bool(a['centreline_bright_fraction'] < med - 3 * mad or a['best_shift_fraction'] >= a['centreline_bright_fraction']),
                   vessel_mask_inside_fraction=m['vessel_mask_inside_fraction'])
        if c:
            rec.update(missing_bullitt_fraction=c['missing_bullitt_fraction'], missing_bullitt_mm=c['missing_mm'],
                       bullitt_length_in_mask_mm=c['bullitt_length_in_mask_mm'],
                       missing_radius_ge_1p5mm_mm=c['missing_radius_ge_1.5mm_mm'], missing_radius_ge_2mm_mm=c['missing_radius_ge_2mm_mm'],
                       tubes_fully_missing=c['tubes_fully_missing'],
                       siphon_flag=bool(c['missing_radius_ge_2mm_mm'] >= 5 or c['missing_bullitt_fraction'] >= 0.10))
        subjects.append(rec)
    with (out / 'manifest.csv').open('w', newline='') as f:
        fields = list(dict.fromkeys(['subject', 'patient_id', 'split'] + [k for s in subjects for k in s]))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(subjects)
    (out / 'patient_split.csv').write_text((out / 'sources' / 'patient_split.csv').read_text())
    siphon = sorted([s for s in subjects if s['has_bullitt']], key=lambda s: -s['missing_bullitt_fraction'])
    kept = [s['kept_length_fraction'] for s in subjects]
    dedupe = {s: markers[s]['duplicates'] for s in DEDUPE_SUBJECTS}
    # Compute nodes may lack Git; submitters can supply the revision explicitly.
    commit = os.environ.get('TUBETK_GIT_HEAD')
    git = shutil.which('git')
    if not commit and git:
        commit = subprocess.run([git, 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    prep = dict(
        completed=True, created='2026-11-05', output=str(out), script='scripts/prepare_tubetk_skwerer_20261105.py',
        script_sha256=digest(Path(__file__)), git_head=commit,
        git_note=('working tree may contain uncommitted changes (reviewed and committed by the user)' if commit
                  else 'Git unavailable; git_head is unknown; script_sha256 identifies the preparation script'),
        source_dataset=str(SOURCE), source_readme_sha256=digest(SOURCE / 'README.md'), manifest_tsv_sha256=digest(SOURCE / 'manifest.tsv'),
        participants_tsv_sha256=digest(SOURCE / 'participants.tsv'),
        ground_truth='Skwerer trees (desc-skwerer_vesseltree.pkl) imported directly as parent-linked graphs; graph/ (CaravelMetrics) not used',
        image='skull-stripped MRA = desc-brain_MRA, verified equal to desc-preproc_MRA * desc-brain_mask on every subject' if all(s['brain_mra_equals_preproc_times_mask'] for s in subjects)
              else 'see per-subject brain_mra_equals_preproc_times_mask',
        mask_used='desc-brain_mask (carotid-inclusive widened HD-BET mask), image and graph clipped to the same mask',
        semantics=SEMANTICS, subjects_total=len(subjects),
        split=dict(helper='scripts.prepare_real_graph_patch_sources.assign_splits seed 42 + family repair (family_safe_split)',
                   counts=dict(Counter(split.values())), family_moves=src['family_moves']),
        kept_length_fraction=_stats(kept), duplicate_removal=dedupe,
        raw_position_duplicates={s: markers[s]['skwerer_raw']['position_duplicate_extra'] for s in DEDUPE_SUBJECTS},
        duplicate_candidates_other_subjects={s: m['duplicates'] for s, m in markers.items() if s not in DEDUPE_SUBJECTS
                                             and m['duplicates'].get('candidate_duplicate_edges_not_removed', 0)},
        zero_length_edges_contracted=_stats([s['zero_length_edges_contracted'] for s in subjects]),
        alignment=dict(method='fraction of Skwerer centreline samples (0.25 mm) whose nearest voxel is >= P95 of in-mask intensity; '
                              'controls: random in-mask voxels and the same samples shifted +-2 mm per axis',
                       bright_fraction=_stats(frac), random_fraction=_stats([s['random_bright_fraction'] for s in subjects]),
                       best_shift_fraction=_stats([s['best_shift_bright_fraction'] for s in subjects]),
                       outlier_rule='bright_fraction < median - 3*MAD*1.4826, or a shifted copy at least as bright',
                       flagged=[s['subject'] for s in subjects if s['alignment_flag']]),
        siphon_qc=dict(method=f'Bullitt edges with both endpoints in desc-brain_mask; missing if the edge midpoint has no clipped Skwerer '
                              f'centreline sample (0.25 mm densified) within {COVER_MM} mm. Bullitt artery flag: values '
                              f'{sorted({v for s in markers.values() if s["bullitt_coverage"] for v in s["bullitt_coverage"]["artery_flag_values"]})} '
                              '(no per-artery labels exist); calibre proxies: missing length at mean edge radius >= 1.5 / >= 2.0 mm',
                       flag_rule='siphon_flag = missing_radius_ge_2mm_mm >= 5 or missing_bullitt_fraction >= 0.10 (recorded only; no exclusion)',
                       table=[{k: s[k] for k in ('subject', 'split', 'missing_bullitt_fraction', 'missing_bullitt_mm', 'bullitt_length_in_mask_mm',
                                                 'missing_radius_ge_1p5mm_mm', 'missing_radius_ge_2mm_mm', 'tubes_fully_missing', 'siphon_flag')}
                              for s in siphon],
                       worst_tubes={s['subject']: markers[s['subject']]['bullitt_coverage']['worst_tubes'][:5] for s in siphon[:10]}),
        patches=dict(grid_patches_by_split=view['grid_patches_by_split'], active_patches_by_split=view['active_patches_by_split'],
                     excluded_by_curation=view['excluded_reasons'], cache=str(cache), cache_counts=manifest['counts'],
                     cache_exclusions=manifest['exclusion_counts'], cache_excluded=manifest['excluded'][:50],
                     loop_policy=manifest['loop_policy'], per_patch=per_patch),
        test_usage='Test subjects/patches were only constructed (images, graphs, patches, K8 targets). No threshold, checkpoint, '
                   'hyperparameter or QC choice used them; the 3D viewer uses a validation patch and the figures use train/val subjects.',
        per_subject=subjects)
    save(out / 'preparation.json', prep)
    print(json.dumps({k: prep[k] for k in ('subjects_total', 'split', 'kept_length_fraction', 'duplicate_removal')}, indent=2))
    print(json.dumps(prep['patches'], indent=2)[:3000])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['inventory', 'subjects', 'smoke', 'stage', 'patches', 'collect'])
    ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cache', default='transfer_splines_k8_loops_20261105')
    a = ap.parse_args()
    if a.mode == 'inventory':
        rows = inventory(); print(json.dumps(rows[:3], indent=1), len(rows), sum(r['has_bullitt'] for r in rows))
    elif a.mode == 'subjects':
        run_subjects(a.output, a.only or [r['subject'] for r in inventory()], a.workers)
    elif a.mode == 'smoke':
        smoke(a.output, a.workers)
    elif a.mode == 'stage':
        stage(a.output)
    elif a.mode == 'patches':
        patches(a.output / 'sources', a.output / 'patches', a.workers)
    else:
        collect(a.output, a.cache)
