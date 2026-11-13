#!/usr/bin/env python3
"""Decoder ceiling by overfitting 10 training patches (12 Nov 2026).

Hypothesis: on the ORIGINAL GT the two-stage traced model cannot reproduce even the patches it was trained on (GT nodes
closer than 2 mm give one heatmap peak), while on the GT SIMPLIFIED at 2 mm it gets close to perfect.

select (login node, CPU): per dataset, the 10 patches are taken from the TRAINING ids of the simplified entry: training
  patches sorted by GT node count in the simplified cache (ties by id), patches whose simplified graph has no node or no
  edge dropped, then the patch at the midpoint of each of the 10 deciles. The same ids are used for the original-GT
  entry. Writes six plan entries '<entry>-overfit10' (copies of the source entry with train_ids, val_ids, monitor_train
  and monitor_val all set to the 10 ids; no existing entry is changed; plan.json is backed up first) and
  experiments/overfit_ceiling_20261112/selection.json.
evaluate (GPU): one trained traced run of an overfit entry, on its 10 patches against its own GT:
  - node / edge P, R, F1 at the run's thresholds (selected on the same 10 patches = its training data; final.json) and
    at the fixed decoding node 0.30 / edge 0.75 (compare_arms_20261109.evaluate_run, same counts as D.evaluate);
  - what is still wrong, at both decodings: missed GT nodes (another GT node within 2 mm / isolated), false nodes (by
    where they sit), missed GT edges (endpoint missed / both endpoints found but no traced candidate / traced candidate
    rejected by the edge head), false edges (MM / MU / UU). Counts follow the loop-aware protocol of the evaluation
    (metrics/loop_aware.py): inserted anchors are don't-care nodes and GT chains through them are one edge.
  Writes experiments/overfit_ceiling_20261112/<entry>.json.

usage: overfit_ceiling_20261112.py select
       overfit_ceiling_20261112.py evaluate <overfit entry> <run> [--suffix=-smoke]
       overfit_ceiling_20261112.py summary
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'
OUT = ROOT / 'experiments/overfit_ceiling_20261112'
PAIRS = [('IXI-loops', 'IXI-simp2'), ('BRAVA', 'BRAVA-simp2'), ('TopBrainCTA-loops', 'TopBrainCTA-simp2')]
SUFFIX = '-overfit10'
N = 10
TOL = 2.0
FIXED = dict(node=0.30, edge=0.75)


def cache_paths(root, ids):
    """sample id -> spline cache path (train or val split folder of the cache root)."""
    from data.loaders.discovery import discover_synthetic_mri
    out = {}
    for split in ('train', 'val'):
        for r in discover_synthetic_mri(Path(root), split):
            out[r.sample_id] = r.graph.parent.parent / 'spline' / f'{r.sample_id}.npz'
    return {i: out[i] for i in ids}


def counts(path):
    with np.load(path) as z:
        ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(z['nodes']), bool)
        return dict(nodes=int(len(z['nodes'])), real_nodes=int((~ins).sum()), edges=int(len(z['edges'].reshape(-1, 2))))


def select(a):
    plan = json.loads(PLAN.read_text())
    ds = plan['datasets']
    backup = Path(os.environ['SCRATCH']) / 'tmp/plan-backup-20261112-overfit.json'
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(PLAN, backup)
    record = dict(rule='simplified-entry training ids sorted by (GT node count in the simplified cache, id); patches with '
                       'no node or no edge dropped; patch at floor((k + 0.5) * n / 10), k = 0..9', backup=str(backup), datasets={})
    new = {}
    for orig, simp in PAIRS:
        manifest = json.loads((Path(ds[simp]['root']) / 'simplify_manifest.json').read_text())
        assert Path(manifest['source']) == Path(ds[orig]['root']), (orig, simp, manifest['source'])
        assert ds[orig]['train_ids'] == ds[simp]['train_ids'], (orig, simp)
        train = ds[simp]['train_ids']
        sc = {i: counts(p) for i, p in cache_paths(ds[simp]['root'], train).items()}
        kept = sorted((i for i in train if sc[i]['nodes'] > 0 and sc[i]['edges'] > 0), key=lambda i: (sc[i]['nodes'], i))
        n = len(kept)
        ids = [kept[int(np.floor((k + 0.5) * n / N))] for k in range(N)]
        oc = {i: counts(p) for i, p in cache_paths(ds[orig]['root'], ids).items()}
        record['datasets'][simp.split('-')[0]] = dict(
            original_entry=orig, simplified_entry=simp, training_patches=len(train), dropped_empty=len(train) - n,
            ids=ids, simplified=[sc[i] for i in ids], original=[oc[i] for i in ids],
            simplified_node_count_deciles=[sc[kept[int(np.floor(q * (n - 1)))]]['nodes'] for q in np.linspace(0, 1, 11)])
        for src in (orig, simp):
            e = json.loads(json.dumps(ds[src]))
            e.update(train_ids=list(ids), val_ids=list(ids), monitor_train=list(ids), monitor_val=list(ids),
                     max_target_nodes=max((oc if src == orig else sc)[i]['nodes'] for i in ids),
                     full_train_count=N, full_val_count=N, parent=src,
                     design=f'overfit ceiling 12 Nov 2026: {src} restricted to {N} training patches (train = monitor = '
                            f'validation lists); ids chosen on {simp} (scripts/overfit_ceiling_20261112.py select)')
            for k in ('oof_ids', 'loop_training_patches', 'loop_validation_patches'):
                e.pop(k, None)
            new[src + SUFFIX] = e
    plan = json.loads(PLAN.read_text())  # re-read just before writing
    for k, e in new.items():
        if k in plan['datasets'] and plan['datasets'][k] != e:
            raise SystemExit(f'{k} already exists with other content; not overwritten')
        plan['datasets'][k] = e
    tmp = PLAN.with_suffix('.json.tmp-overfit')
    tmp.write_text(json.dumps(plan, indent=1) + '\n')
    tmp.replace(PLAN)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'selection.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


def prf(tp, fp, fn):
    return dict(precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0, tp=int(tp), fp=int(fp), fn=int(fn))


def gt_hops(n, edges, u, v, limit=4):
    """Hop distance between GT nodes u and v (None if more than `limit` or disconnected)."""
    adj = {}
    for a, b in edges:
        adj.setdefault(int(a), set()).add(int(b)), adj.setdefault(int(b), set()).add(int(a))
    seen, frontier = {u}, {u}
    for h in range(1, limit + 1):
        frontier = {y for x in frontier for y in adj.get(x, ()) if y not in seen}
        if v in frontier:
            return h
        seen |= frontier
    return None


def breakdown_patch(r, node_t, edge_t):
    """Per patch: official counts (edge_accounting) and the classification of every miss / false detection."""
    from scipy.spatial import cKDTree
    from diagnose_branch_relations import match_nodes
    from ixi_query_diagnostics import edge_accounting
    from metrics.loop_aware import _merge_chains
    from node_miss_diagnosis_20261110 import dense_path
    with np.load(r['cache']) as z:
        M, gn, ge = z['metric'], z['nodes'], z['edges'].reshape(-1, 2)
        ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(gn), bool)
        P, off = z['full_paths'] @ M.T, z['path_offsets']
    g = gn @ M.T
    n_gt = len(gn)
    keep = np.nonzero(r['node_scores'] >= node_t)[0]
    remap = {int(k): i for i, k in enumerate(keep)}
    p = r['nodes'][keep] @ M.T
    pairs = r['pairs'] if r['edge_scores'] is not None else np.zeros((0, 2), int)
    valid = np.isin(pairs, keep).all(1) if len(pairs) else np.zeros(0, bool)
    sub = pairs[valid]
    scores = r['edge_scores'][valid] if len(sub) else np.zeros(0)
    traced = r['traced'][valid] if len(sub) else np.zeros(0, bool)
    local = np.array([[remap[int(a)], remap[int(b)]] for a, b in sub], int).reshape(-1, 2)
    mapping = match_nodes(p, g, TOL) if len(p) and n_gt else {}
    row, chosen, *_ = edge_accounting(np.arange(len(p)), local, scores, scores >= edge_t, mapping, ge, n_gt,
                                      ins if ins.any() else None)
    real_map = {q: gi for q, gi in mapping.items() if not ins[gi]}
    inv = {gi: q for q, gi in real_map.items()}
    paths = [dense_path(P[off[k]:off[k + 1]]) for k in range(len(ge))]
    path_tree = cKDTree(np.concatenate(paths)) if paths else None
    # ---- nodes
    missed = []
    for i in np.flatnonzero(~ins):
        if i in inv:
            continue
        d = np.linalg.norm(g - g[i], axis=1)
        d[i] = np.inf
        near = np.flatnonzero(d <= TOL)
        dp = np.linalg.norm(p - g[i], axis=1) if len(p) else np.zeros(0)
        missed.append(dict(kind='another GT node within 2 mm' if len(near) else 'isolated',
                           near_real=int((~ins[near]).sum()), near_inserted=int(ins[near].sum()),
                           nearest_gt_mm=float(d.min()) if n_gt > 1 else None,
                           kept_detection_within_2mm=bool((dp <= TOL).any()),
                           nearest_detection_mm=float(dp.min()) if len(dp) else None))
    false_nodes = []
    for q in range(len(p)):
        if q in mapping:
            continue
        d_node = float(np.linalg.norm(g - p[q], axis=1).min()) if n_gt else np.inf
        d_path = float(path_tree.query(p[q])[0]) if path_tree is not None else np.inf
        false_nodes.append(dict(where='within 2 mm of a GT node' if d_node <= TOL else 'on a GT centreline, between nodes'
                                if d_path <= 1.5 else 'off the GT graph', score=float(r['node_scores'][keep[q]]),
                                d_node=d_node, d_path=d_path))
    # ---- edges, as loop_aware_counts (plain protocol = no inserted node: every chain is one edge)
    gt_merged, _ = _merge_chains([tuple(map(int, e)) for e in ge], set(np.flatnonzero(ins).tolist()), n_gt)
    acc = local[chosen]
    edges = [tuple(map(int, e)) for e in acc]
    degree = np.bincount(acc.ravel(), minlength=len(p)) if len(acc) else np.zeros(len(p), int)
    through = {q for q, gi in mapping.items() if ins[gi] and degree[q] == 2}
    pred_merged, _ = _merge_chains(edges, through, len(p))
    pool = Counter((u, v) for u, v, _ in gt_merged)
    false_edges = []
    for u, v, members in pred_merged:
        key = tuple(sorted((real_map[u], real_map[v]))) if u in real_map and v in real_map else None
        if key is not None and pool[key] > 0:
            pool[key] -= 1
            continue
        kind = 'MM' if key is not None else 'MU' if (u in real_map or v in real_map) else 'UU'
        rec = dict(kind=kind, chain=len(members))
        if kind == 'MM':
            h = gt_hops(n_gt, ge, key[0], key[1]) if key[0] != key[1] else 0
            rec['gt_hops'] = h
        false_edges.append(rec)
    cand = {tuple(sorted(map(int, e))): k for k, e in enumerate(local)}
    missed_edges = []
    members_of = {}
    for u, v, mem in gt_merged:
        members_of.setdefault((u, v), []).append(mem)
    for (u, v), left in pool.items():
        for j in range(left):
            mem = members_of[(u, v)][j]
            arc = float(sum(np.linalg.norm(np.diff(paths[k], axis=0), axis=1).sum() for k in mem))
            rec = dict(arc_mm=arc, through_inserted=len(mem) - 1)
            if u not in inv or v not in inv:
                rec.update(kind='endpoint missed', both_missed=u not in inv and v not in inv)
            else:
                A, B = inv[u], inv[v]
                pts = np.concatenate([paths[k] for k in mem])
                tree = cKDTree(pts)
                dd = tree.query(p)[0] if len(p) else np.zeros(0)
                interior = [q for q in range(len(p)) if q not in (A, B) and dd[q] <= 1.5
                            and np.linalg.norm(p[q] - g[u]) > TOL and np.linalg.norm(p[q] - g[v]) > TOL]
                rec.update(detection_on_path=len(interior), detection_on_path_unmatched=sum(q not in mapping for q in interior))
                k = cand.get((min(A, B), max(A, B)))
                if A == B:
                    rec['kind'] = 'both ends on one detection (loop)'
                elif k is None:
                    rec['kind'] = 'both endpoints found, not a candidate'
                elif not traced[k]:
                    rec.update(kind='both endpoints found, straight-fallback candidate only', score=float(scores[k]))
                elif scores[k] < edge_t:
                    rec.update(kind='traced candidate, rejected by the edge head', score=float(scores[k]))
                else:
                    rec.update(kind='accepted but not credited (multigraph / chain)', score=float(scores[k]))
            missed_edges.append(rec)
    assert len(missed) == row['node_fn'], (len(missed), row['node_fn'])
    assert len(false_nodes) == row['node_fp'], (len(false_nodes), row['node_fp'])
    assert len(missed_edges) == row['fn'], (len(missed_edges), row['fn'])
    assert len(false_edges) == row['fp'], (len(false_edges), row['fp'])
    return dict(sample_id=r['sample_id'], gt_real_nodes=int((~ins).sum()), gt_inserted=int(ins.sum()), gt_edges=row['gt_edges'],
                pred_nodes=int(len(p)), node_tp=row['node_tp'], node_fp=row['node_fp'], node_fn=row['node_fn'],
                edge_tp=row['tp'], edge_fp=row['fp'], edge_fn=row['fn'],
                missed_nodes=missed, false_nodes=false_nodes, missed_edges=missed_edges, false_edges=false_edges)


def tally(patches):
    c = lambda items, key: dict(Counter(x[key] for x in items))
    mn = [x for q in patches for x in q['missed_nodes']]
    fnn = [x for q in patches for x in q['false_nodes']]
    me = [x for q in patches for x in q['missed_edges']]
    fe = [x for q in patches for x in q['false_edges']]
    tot = {k: sum(q[k] for q in patches) for k in ('node_tp', 'node_fp', 'node_fn', 'edge_tp', 'edge_fp', 'edge_fn', 'gt_real_nodes', 'gt_edges', 'pred_nodes')}
    return dict(
        totals=tot, node=prf(tot['node_tp'], tot['node_fp'], tot['node_fn']), edge=prf(tot['edge_tp'], tot['edge_fp'], tot['edge_fn']),
        missed_gt_nodes=dict(total=len(mn), by_kind=c(mn, 'kind'),
                             with_kept_detection_within_2mm=sum(x['kept_detection_within_2mm'] for x in mn),
                             crowded_with_kept_detection_within_2mm=sum(x['kept_detection_within_2mm'] for x in mn if x['kind'] != 'isolated'),
                             nearest_gt_node_mm={f'{lo}-{hi}': sum(x['nearest_gt_mm'] is not None and lo <= x['nearest_gt_mm'] < hi for x in mn)
                                                 for lo, hi in ((0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 1e9))}),
        false_nodes=dict(total=len(fnn), by_where=c(fnn, 'where')),
        missed_gt_edges=dict(total=len(me), by_kind=c(me, 'kind'),
                             endpoint_missed_both=sum(x.get('both_missed', False) for x in me),
                             not_candidate_with_detection_on_path=sum(x.get('detection_on_path', 0) > 0 for x in me
                                                                      if x['kind'] == 'both endpoints found, not a candidate'),
                             through_inserted_anchor=sum(x['through_inserted'] > 0 for x in me)),
        false_edges=dict(total=len(fe), by_kind=c(fe, 'kind'),
                         MM_by_gt_hops={str(k): v for k, v in Counter(x.get('gt_hops') for x in fe if x['kind'] == 'MM').items()},
                         chains_through_anchor_detections=sum(x['chain'] > 1 for x in fe)))


def curve_summary(run_dir):
    """First / middle / last learning-curve points and logged training losses of one run."""
    out = {}
    lc = run_dir / 'learning-curve.jsonl'
    if lc.exists():
        pts = [json.loads(x) for x in lc.read_text().splitlines() if x.strip()]
        out['curve'] = [dict(step=q['step'], node_threshold=q['node_threshold'], edge_threshold=q['edge_threshold'],
                             **{k: round(v, 4) for k, v in q['train_monitor'].items()}) for q in pts]
    tr = run_dir / 'training.jsonl'
    if tr.exists():
        rows = [json.loads(x) for x in tr.read_text().splitlines() if x.strip()]
        terms = [k for k in ('node', 'offset', 'centerline', 'edge', 'geometry_mm', 'passthrough', 'role') if k in rows[0]]
        for q in rows:
            q['logged_sum'] = sum(q[k] for k in terms)
        w = max(1, len(rows) // 20)  # windows of 5% of the logged rows (one row per 100 updates, single-batch values)
        pick = lambda s: {k: float(np.mean([q[k] for q in s])) for k in terms + ['logged_sum']} | dict(steps=[s[0]['step'], s[-1]['step']])
        mid = len(rows) // 2
        out['loss'] = dict(terms=terms, window_rows=w, first=pick(rows[:w]), middle=pick(rows[mid - w // 2: mid - w // 2 + w]),
                           last=pick(rows[-w:]), seconds_per_update=rows[-1]['seconds'] / rows[-1]['step'])
    return out


def low_peaks(model, ds, node_t, workers, D, floor=0.01):
    """Missed GT nodes at node threshold node_t: is there a heatmap peak below the threshold? All 3^3-NMS peaks with
    heat >= floor are matched one-to-one to the GT nodes (2 mm, as the evaluation); a GT node missed at node_t is then
    'peak below threshold' (its low-floor match, with the score), 'only nearby peak belongs to another GT node', or
    'no peak within 2 mm (>= floor)'. Also the maximum heat within 2 mm of the node (voxel ball)."""
    import torch
    from diagnose_branch_relations import match_nodes
    from node_heatmap_probe_20261030 import collate
    model.eval()
    out_rows, offset = [], 0
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=workers, collate_fn=collate)
    with torch.no_grad():
        for images, nodes, edges, geometry in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_image(images.to('cuda'))
            heat = out[:, 0].float().sigmoid()
            for b, (pk, sc) in enumerate(D.peaks(out, floor)):
                rec = ds.records[offset + b]
                with np.load(rec.graph.parent.parent / 'spline' / f'{rec.sample_id}.npz') as z:
                    M, gn = z['metric'], z['nodes']
                    ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(gn), bool)
                g = gn @ M.T
                p = (pk / 64).cpu().numpy() @ M.T
                s = sc.cpu().numpy()
                kept = np.flatnonzero(s >= node_t)
                hi = match_nodes(p[kept], g, TOL) if len(kept) and len(g) else {}
                lo = match_nodes(p, g, TOL) if len(p) and len(g) else {}
                hit = set(hi.values())
                lo_gt = {gi: q for q, gi in lo.items()}
                vm = M / 64  # voxel -> mm
                r = int(np.ceil(TOL / np.linalg.norm(vm, axis=0).min()))
                o = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing='ij'), -1).reshape(-1, 3)
                o = o[np.linalg.norm(o @ vm.T, axis=1) <= TOL]
                h = heat[b].cpu().numpy()
                for i in np.flatnonzero(~ins):
                    if i in hit:
                        continue
                    vox = np.clip(np.round(gn[i] * 64).astype(int)[None] + o, 0, 63)
                    near = np.linalg.norm(p - g[i], axis=1) <= TOL if len(p) else np.zeros(0, bool)
                    if i in lo_gt:
                        kind, score = 'peak below threshold', float(s[lo_gt[i]])
                    elif near.any():
                        kind, score = 'only nearby peak belongs to another GT node', float(s[near].max())
                    else:
                        kind, score = f'no peak within 2 mm (>= {floor})', None
                    out_rows.append(dict(sample_id=rec.sample_id, kind=kind, peak_score=score,
                                         heat_max_2mm=float(h[vox[:, 0], vox[:, 1], vox[:, 2]].max()),
                                         heat_at_node=float(h[tuple(np.clip(np.round(gn[i] * 64).astype(int), 0, 63))])))
            offset += len(images)
    hist = lambda v: {f'{lo}-{hi}': int(sum(lo <= x < hi for x in v)) for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 1.01))}
    return dict(floor=floor, node_threshold=node_t, missed=len(out_rows), by_kind=dict(Counter(x['kind'] for x in out_rows)),
                below_threshold_peak_scores=hist([x['peak_score'] for x in out_rows if x['kind'] == 'peak below threshold']),
                heat_max_2mm=hist([x['heat_max_2mm'] for x in out_rows]), nodes=out_rows)


def evaluate(a):
    import torch
    import compare_arms_20261109 as C
    import dense_graph_20261030 as D
    import dense_graph_traced_20261030 as T
    from node_heatmap_probe_20261030 import make_dataset
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    runs = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030'
    run_dir = runs / a.entry / a.run
    final = json.loads((run_dir / 'final.json').read_text())
    ids = json.loads(PLAN.read_text())['datasets'][a.entry]['val_ids']
    ds = make_dataset(a.entry, ids, augment=False)
    tol = final['mode'].get('compress_tol', 0.0)
    # fixed decoding node 0.30 / edge 0.75, compare_arms_20261109 (also node AP @2 mm)
    entry, cnt, cids = C.evaluate_run(a.entry, a.run, ds, workers, tol)
    c = dict(zip(C.KEYS, cnt[:, :len(C.KEYS)].sum(0)))
    fixed = dict(node=prf(c['node_tp'], c['node_fp'], c['node_fn']), edge=prf(c['tp'], c['fp'], c['fn']),
                 V0_global=entry['V0_global'], node_ap_2mm=entry['node_ap_2mm'])
    # the run's own thresholds (selected on these 10 patches = its training data)
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], two_hop=T.MODE['two_hop'], passthrough=T.MODE['passthrough'],
                               backbone=final.get('backbone', 'unet'), feature_channels=final.get('feature_channels', 32)).to('cuda')
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cuda')['model'])
    nt, et = final['thresholds']['node'], final['thresholds']['edge']
    rows_sel = T.predict(model, ds, 'cuda', nt, workers)
    rows_fix = T.predict(model, ds, 'cuda', FIXED['node'], workers)
    check = D.evaluate(rows_sel, nt, et, geometry=False)
    result = dict(entry=a.entry, run=a.run, patches=ids, compress_tol=tol, augment=final.get('augment'),
                  selected_thresholds=dict(node=nt, edge=et, policy=final['thresholds']['policy']),
                  selected=dict(final_json_train_monitor={k: final['train_monitor'][k] for k in D.CURVE_KEYS},
                                recomputed={k: check[k] for k in D.CURVE_KEYS}),
                  fixed_0p30_0p75=fixed,
                  breakdown_selected=tally([breakdown_patch(r, nt, et) for r in rows_sel]),
                  breakdown_fixed=tally([breakdown_patch(r, FIXED['node'], FIXED['edge']) for r in rows_fix]))
    result['per_patch_selected'] = [breakdown_patch(r, nt, et) for r in rows_sel]  # with the per-item lists
    result['missed_node_peaks_selected'] = low_peaks(model, ds, nt, workers, D)
    result['missed_node_peaks_fixed'] = low_peaks(model, ds, FIXED['node'], workers, D)
    ws = run_dir / 'warm-start.json'
    s1 = Path(json.loads(ws.read_text())['source']).parent if ws.exists() else None
    result['stage1_run'] = str(s1) if s1 else None
    result['curves'] = dict(stage1=curve_summary(s1) if s1 else None, stage2=curve_summary(run_dir))
    if s1 and (s1 / 'final.json').exists():
        f1 = json.loads((s1 / 'final.json').read_text())
        result['stage1_final_train_monitor'] = dict(thresholds=f1['thresholds'], **{k: f1['train_monitor'][k] for k in D.CURVE_KEYS})
    # the fixed-decoding breakdown should reproduce compare_arms' counts (a second predict pass; recorded, not asserted)
    result['check_breakdown_fixed_equals_compare_arms'] = all(
        result['breakdown_fixed'][k][x] == fixed[k][x] for k in ('node', 'edge') for x in ('tp', 'fp', 'fn'))
    result['check_selected_equals_final_json'] = all(
        abs(result['selected']['recomputed'][k] - result['selected']['final_json_train_monitor'][k]) < 1e-9 for k in D.CURVE_KEYS)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{a.entry}{a.suffix}.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    show = lambda d: {k: round(d[k], 4) if isinstance(d[k], float) else d[k] for k in ('precision', 'recall', 'f1')}
    print('SELECTED', a.entry, nt, et, json.dumps(dict(node=show(result['breakdown_selected']['node']), edge=show(result['breakdown_selected']['edge']))))
    print('FIXED', a.entry, json.dumps(dict(node=show(fixed['node']), edge=show(fixed['edge']))))
    print('DONE', flush=True)


def summary(a):
    """Table of the six entries (CPU): results, thresholds, curve points, miss breakdown -> summary.json + stdout."""
    out = {}
    f = lambda x: '  -  ' if x is None else f'{x:.3f}'
    for orig, simp in PAIRS:
        for src in (orig, simp):
            e = src + SUFFIX
            path = OUT / f'{e}.json'
            if not path.exists():
                print('MISSING', e)
                continue
            r = json.loads(path.read_text())
            out[e] = dict(selected_thresholds=r['selected_thresholds'],
                          selected={k: r['breakdown_selected'][k] for k in ('node', 'edge')},
                          fixed_0p30_0p75={k: r['fixed_0p30_0p75'][k] for k in ('node', 'edge')},
                          node_ap_2mm=r['fixed_0p30_0p75']['node_ap_2mm'],
                          breakdown_selected={k: r['breakdown_selected'][k] for k in ('missed_gt_nodes', 'false_nodes', 'missed_gt_edges', 'false_edges')},
                          breakdown_fixed={k: r['breakdown_fixed'][k] for k in ('missed_gt_nodes', 'false_nodes', 'missed_gt_edges', 'false_edges')},
                          curves=r['curves'], checks=[r['check_breakdown_fixed_equals_compare_arms'], r['check_selected_equals_final_json']])
            s, x = r['breakdown_selected'], r['fixed_0p30_0p75']
            print(f"{e:30s} sel n{r['selected_thresholds']['node']:.2f}/e{r['selected_thresholds']['edge']:.2f} "
                  f"node P {f(s['node']['precision'])} R {f(s['node']['recall'])} F1 {f(s['node']['f1'])} | "
                  f"edge P {f(s['edge']['precision'])} R {f(s['edge']['recall'])} F1 {f(s['edge']['f1'])} || 0.30/0.75 "
                  f"node P {f(x['node']['precision'])} R {f(x['node']['recall'])} F1 {f(x['node']['f1'])} | "
                  f"edge P {f(x['edge']['precision'])} R {f(x['edge']['recall'])} F1 {f(x['edge']['f1'])}")
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['select', 'evaluate', 'summary'])
    p.add_argument('entry', nargs='?')
    p.add_argument('run', nargs='?')
    p.add_argument('--suffix', default='', help="evaluate: results go to <entry><suffix>.json (e.g. '-smoke')")
    a = p.parse_args()
    {'select': select, 'evaluate': evaluate, 'summary': summary}[a.command](a)
