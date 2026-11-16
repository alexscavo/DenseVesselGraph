#!/usr/bin/env python3
"""Node-decoding study, part 3 (13 Nov 2026, third request): does the local shape of the predicted centreline around a
candidate node separate real nodes from false ones? No retraining. Reuses scripts/node_decode_study_20261113.py
(parts 1-2, whose outputs are not touched).

Candidates: stage-1 run (spline6_unet_wide_long_r1111) on monitor_train and val_ids, sigmoid heat, offsets and
  centreline probability averaged over the 8 axis flips of the input (flip8), local maxima in a 5^3 window,
  score >= FLOOR3 = 0.05, position = voxel + predicted offset (part-1 decode('raw5')).
Arm count (CPU): centreline mask = flip8 centreline probability >= tau; around the candidate voxel c (rounded refined
  position), the mask inside the ball |x - c| <= r + 2 (voxel units); the 26-connected component that contains c, or the
  mask voxel nearest to c within 2 voxels (none: 'off-vessel', -1, grouped with 0 arms); arms = number of 26-connected
  components of that component restricted to the shell r < |x - c| <= r + 2; capped at 3. Outside the patch = no mask.
  tau in TAUS, r in RADII. Group: 'face' if the refined position is < 3 voxels from a patch face (as part 2's band:
  min over axes of min(x + 0.5, 63.5 - x) < 3), else 0 / 1 / 2 / 3+ arms.
Candidate types (matching of all candidates >= 0.05 to the GT, one-to-one, 2 mm): matched to a real GT node of class
  boundary / junction / terminal / other (part-2 classes: cache boundary flag, then degree), matched to an inserted
  anchor; unmatched: duplicate (within 2 mm of a matched GT node), midvessel (within 1.5 mm of a GT centreline),
  elsewhere.
Rules (thresholds chosen on monitor_train by node F1): ref = one threshold on the 0.02..0.95 / 0.01 grid (= part 2's
  flip8:raw5|single); single05 = one threshold on the 0.05..0.70 / 0.05 grid; full = one threshold per group, coordinate
  ascent from single05 until no change, per (tau, r), best (tau, r); a = 2-arm group own threshold, all others one shared
  threshold (exhaustive 2-D grid); b = 1-arm group own threshold, others shared. Validation: node P/R/F1, recall per
  class, false detections by type (part-2 detail2_patch), subject-paired node F1 difference vs ref.
Stage 2: the node sets of ref, full (and a if within 0.005 monitor F1 of full) replace D.peaks inside the unchanged
  T.predict of traced_unet_wide_long_r1111 (as part 2's stage2v2), edge threshold re-selected on monitor_train.

usage: node_topology_study_20261113.py {sanity3,collect3,analyse3,stage2v3,figures3,all3} <entry>[+<entry>] [--smoke]
Outputs: $SCRATCH/experiments/gnbm/node_decode_20261113/<entry>{-smoke}-p3-{monitor,val}.pkl,
         experiments/final_try_20261113/node_topology_<entry>{-smoke}.json, viz_tmp/v3_*.png
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import node_decode_study_20261113 as S  # noqa: E402

SIZE, STORE, OUT, VIZ, CTX = S.SIZE, S.STORE, S.OUT, S.VIZ, S.CTX
TAUS = (0.3, 0.5, 0.7)
RADII = (4, 6, 8)
COMBOS = [(t, r) for t in TAUS for r in RADII]
FLOOR3 = 0.05
FACE = 3.0
WIN = 5
GROUPS = ('face', '0', '1', '2', '3+')
TYPES = ('boundary', 'junction', 'terminal', 'other', 'anchor', 'duplicate', 'midvessel', 'elsewhere', 'near_unmatched_gt')
HEAT_EDGES = (0.05, 0.15, 0.25, 0.35, 0.5)
HEAT_BINS = ('0.05-0.15', '0.15-0.25', '0.25-0.35', '0.35-0.5', '>=0.5')
G5 = [round(0.05 * k, 2) for k in range(1, 15)]
RMAX = max(RADII) + 2
REF_WINDOW = 'flip8:raw5|single'


def out_json(a):
    return OUT / f'node_topology_{S.tag(a)}.json'


# ------------------------------------------------------------------------------------------------------ arm count
_G = {}


def geom():
    if not _G:
        ax = np.arange(-RMAX, RMAX + 1)
        d2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
        g = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
        near = g[(g ** 2).sum(1) <= 4]
        near = near[np.argsort((near ** 2).sum(1), kind='stable')]
        _G.update(near=near + RMAX, s26=np.ones((3, 3, 3), bool))
        for r in range(1, RMAX - 1):
            lo, hi = RMAX - r - 2, RMAX + r + 3
            sub = d2[lo:hi, lo:hi, lo:hi]
            _G[r] = (lo, sub <= (r + 2) ** 2, sub > r * r)
    return _G


def arm_counts(prob, pos, taus=TAUS, radii=RADII):
    """(N, len(taus), len(radii)) int8 arm counts of the thresholded centreline around each candidate (capped at 3);
    -1 = off-vessel (no mask voxel within 2 voxels of the candidate)."""
    from scipy.ndimage import label
    G = geom()
    n = len(pos)
    out = np.zeros((n, len(taus), len(radii)), np.int8)
    if n == 0:
        return out
    P = np.pad(np.asarray(prob, np.float32), RMAX)
    c = np.clip(np.rint(np.asarray(pos, np.float64)).astype(int), 0, SIZE - 1) + RMAX
    for i in range(n):
        x, y, z = c[i]
        cube = P[x - RMAX:x + RMAX + 1, y - RMAX:y + RMAX + 1, z - RMAX:z + RMAX + 1]
        for a, tau in enumerate(taus):
            m = cube >= tau
            hit = m[G['near'][:, 0], G['near'][:, 1], G['near'][:, 2]]
            if not hit.any():
                out[i, a] = -1
                continue
            seed = G['near'][int(np.argmax(hit))]
            for b, r in enumerate(radii):
                lo, ball, shell = G[r]
                sub = m[lo:-lo or None, lo:-lo or None, lo:-lo or None] & ball
                lab, _ = label(sub, G['s26'])
                comp = lab == lab[seed[0] - lo, seed[1] - lo, seed[2] - lo]
                out[i, a, b] = min(label(comp & shell, G['s26'])[1], 3)
    return out


def groups_of(pos, arms):
    """Group index (0 face, 1 = 0 arms / off-vessel, 2 = 1 arm, 3 = 2 arms, 4 = 3+ arms) for one (tau, r)."""
    g = np.maximum(arms.astype(int), 0) + 1
    g[S.face_dist(pos) < FACE] = 0
    return g


def sanity3(a=None):
    """Synthetic masks: straight tube 2, tube ending at the candidate 1, Y 3, cross 3 (capped), blob 0, empty -1,
    oblique tube 2, tube passing 2 voxels from the candidate 2, tube with probability 0.6 at tau 0.7 -> -1."""
    ax = np.arange(SIZE, dtype=np.float64)
    X = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1)
    c = np.array([32.0, 32.0, 32.0])

    def seg(p0, p1, rad=1.5):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        t = np.clip(((X - p0) @ d) / (d @ d), 0, 1)
        return np.linalg.norm(X - (p0 + t[..., None] * d), axis=-1) <= rad

    def rays(dirs, L=25):
        m = np.zeros((SIZE,) * 3, bool)
        for v in dirs:
            v = np.asarray(v, float) / np.linalg.norm(v)
            m |= seg(c, c + L * v)
        return m
    s3 = np.sqrt(3) / 2
    cases = dict(straight=(rays([(1, 0, 0), (-1, 0, 0)]), 2), end=(rays([(1, 0, 0)]), 1),
                 Y=(rays([(1, 0, 0), (-0.5, s3, 0), (-0.5, -s3, 0)]), 3),
                 cross4=(rays([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]), 3),
                 blob=(np.linalg.norm(X - c, axis=-1) <= 2.0, 0), empty=(np.zeros((SIZE,) * 3, bool), -1),
                 oblique=(rays([(1, 1, 1), (-1, -1, -1)]), 2),
                 offset2=(seg((12, 34, 32), (52, 34, 32), 0.6), 2))
    res, ok = {}, True
    for name, (m, want) in cases.items():
        got = arm_counts(m.astype(np.float32), c[None])[0]
        res[name] = dict(expected=want, got=got.tolist())
        ok &= bool((got == want).all())
    m = rays([(1, 0, 0), (-1, 0, 0)]).astype(np.float32) * 0.6
    got = arm_counts(m, c[None])[0]
    res['prob0.6'] = dict(expected='[2,2,2] at tau 0.3/0.5, -1 at tau 0.7', got=got.tolist())
    ok &= bool((got[:2] == 2).all() and (got[2] == -1).all())
    pts = np.c_[np.random.default_rng(0).uniform(10, 54, 200), np.full(200, 32.0), np.full(200, 33.0)]
    t0 = time.time()
    arm_counts(rays([(1, 0, 0), (-1, 0, 0)]).astype(np.float32), pts)
    res['seconds_per_candidate_9_combos'] = (time.time() - t0) / 200
    res['all_ok'] = ok
    print('SANITY3', json.dumps(res), flush=True)
    assert ok, res
    return res


# ------------------------------------------------------------------------------------------------------ collect (GPU)
def figure_ids_of(a, records):
    if a.smoke:
        return S.figure_ids(records)
    return json.loads((OUT / f'node_decode_{a.entry}.json').read_text())['figure_ids']


def collect3(a):
    import multiprocessing as mp
    import torch
    import dense_graph_20261030 as D
    from node_heatmap_probe_20261030 import make_dataset
    device = 'cuda'
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    model, _ = S.load_stage1(a.entry, device)
    STORE.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context('fork').Pool(max(1, workers - 4))
    for split, ids in S.splits_of(a).items():
        t0 = time.time()
        ds = make_dataset(a.entry, ids, augment=False)
        fig = set(figure_ids_of(a, ds.records)) if split == 'val' else set()
        loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=3, collate_fn=D.collate)
        recs, pending, offset = [], [], 0
        with torch.no_grad():
            for images, _, _, _ in loader:
                images = images.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model.forward_image(images)[:, :5]
                heat, off, prob = S.flip_maps(model, images, out)
                dec = S.decode(heat, off, prob, f'raw{WIN}', FLOOR3)
                pr = prob.float().cpu().numpy()
                for b in range(len(images)):
                    record = ds.records[offset + b]
                    pos = dec[b]['pos'].cpu().numpy().astype(np.float32)
                    rec = dict(sample_id=record.sample_id, cache=S.cache_of(record), pos=pos,
                               score=dec[b]['score'].cpu().numpy().astype(np.float32))
                    pending.append(pool.apply_async(arm_counts, (pr[b], pos)))
                    if record.sample_id in fig:
                        rec['mip'] = {name: np.stack([vol.max(axis=k) for k in range(3)]).astype(np.float16) for name, vol in
                                      (('image', images[b, 0].float().cpu().numpy()), ('heat', heat[b, 0].cpu().numpy()),
                                       ('centreline', pr[b]))}
                    recs.append(rec)
                offset += len(images)
        t_gpu = time.time() - t0
        for rec, p in zip(recs, pending):
            rec['arms'] = p.get()
        (STORE / f'{S.tag(a)}-p3-{split}.pkl').write_bytes(pickle.dumps(dict(figure_ids=sorted(fig), records=recs), protocol=4))
        n = np.mean([len(r['score']) for r in recs])
        print(f'COLLECTED3 {a.entry} {split} {len(recs)} patches, {n:.1f} candidates/patch >= {FLOOR3}, gpu loop {t_gpu:.0f}s, '
              f'total {time.time() - t0:.0f}s', flush=True)
    pool.close()
    del model
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------ analyse (CPU)
def candidate_types(p_mm, gt):
    """Type index (TYPES) of every candidate from the matching of all of them."""
    mp = S.match(p_mm, gt['g'])
    matched = np.zeros(len(gt['g']), bool)
    matched[list(mp.values())] = True
    t = np.zeros(len(p_mm), np.int8)
    for q in range(len(p_mm)):
        if q in mp:
            j = mp[q]
            t[q] = TYPES.index('anchor') if gt['ins'][j] else TYPES.index(str(gt['cls'][j]))
            continue
        d = np.linalg.norm(gt['g'] - p_mm[q], axis=1) if len(gt['g']) else np.zeros(0)
        if (d <= S.TOL).any():
            t[q] = TYPES.index('duplicate' if matched[d <= S.TOL].any() else 'near_unmatched_gt')
        elif gt['tree'] is not None and gt['tree'].query(p_mm[q])[0] <= 1.5:
            t[q] = TYPES.index('midvessel')
        else:
            t[q] = TYPES.index('elsewhere')
    return t


def prep_patch(i):
    rec = CTX['recs'][i]
    gt = S.gt_of2(rec['cache'])
    p = (rec['pos'] / SIZE) @ gt['M'].T
    return p, candidate_types(p, gt)


_MEMO = {}


def eval_task(args):
    """(len(rules), 3) node tp / fp / fn of one patch; rule = (combo index, 5 group thresholds)."""
    i, rules = args
    rec = CTX['recs'][i]
    gt = S.gt_of2(rec['cache'])
    out = np.zeros((len(rules), 3), np.int32)
    for k, (ci, thr) in enumerate(rules):
        keep = rec['score'] >= np.asarray(thr)[rec['groups'][ci]]
        key = (i, np.packbits(keep).tobytes(), len(keep))
        if key not in _MEMO:
            q = np.flatnonzero(keep)
            _MEMO[key] = S.counts(S.match(rec['p_mm'][q], gt['g']), len(q), gt['ins'])
        out[k] = _MEMO[key]
    return out


def f1_of(c):
    return 2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 2], 1)


def evaluate_rules(pool, n, rules):
    return f1_of(np.sum(pool.map(eval_task, [(i, rules) for i in range(n)], chunksize=4), 0))


def kept(rec, ci, thr):
    return rec['score'] >= np.asarray(thr)[rec['groups'][ci]]


def diag_tables(recs):
    """(combos, TYPES, GROUPS, HEAT_BINS) candidate counts."""
    tab = np.zeros((len(COMBOS), len(TYPES), len(GROUPS), len(HEAT_BINS)), np.int64)
    for r in recs:
        hb = np.searchsorted(np.asarray(HEAT_EDGES), r['score'], side='right') - 1
        for ci in range(len(COMBOS)):
            np.add.at(tab[ci], (r['types'], r['groups'][ci], hb), 1)
    return tab


def table_json(tab):
    return {t: {g: tab[k, j].tolist() for j, g in enumerate(GROUPS)} for k, t in enumerate(TYPES)}


def separation(tab):
    """Compact view of one (combo) table, interior candidates only: arm-group shares per type, all heat bins and
    heat < 0.35."""
    out = {}
    for name, hb in (('all_heat', slice(None)), ('heat_below_0.35', slice(0, 3))):
        d = {}
        for k, t in enumerate(TYPES):
            row = tab[k, 1:, hb].sum(-1)
            if row.sum():
                d[t] = dict(n=int(row.sum()), share_0_1_2_3=[round(float(x), 3) for x in row / row.sum()])
        out[name] = d
    return out


def analyse3(a):
    import multiprocessing as mp
    import torch  # noqa: F401  (node_types)
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    t0 = time.time()
    data = {s: pickle.loads((STORE / f'{S.tag(a)}-p3-{s}.pkl').read_bytes()) for s in ('monitor', 'val')}
    for split in ('monitor', 'val'):
        recs = data[split]['records']
        CTX.update(recs=recs)
        for r, (p, t) in zip(recs, S.run_pool(prep_patch, len(recs), workers)):
            r['p_mm'], r['types'] = p, t
            r['groups'] = np.stack([groups_of(r['pos'], r['arms'][:, TAUS.index(tau), RADII.index(rad)]) for tau, rad in COMBOS]) \
                if len(r['pos']) else np.zeros((len(COMBOS), 0), int)
    print(f'PREP3 {time.time() - t0:.0f}s', flush=True)
    mon = data['monitor']['records']
    CTX.update(recs=mon)
    n = len(mon)
    with mp.get_context('fork').Pool(workers) as pool:
        f_ref = evaluate_rules(pool, n, [(0, (t,) * 5) for t in S.GRID])
        t_ref = S.GRID[int(np.argmax(f_ref))]
        f_s5 = evaluate_rules(pool, n, [(0, (t,) * 5) for t in G5])
        t_s5 = G5[int(np.argmax(f_s5))]
        # coordinate ascent per combo, in lockstep
        thr = {ci: [t_s5] * 5 for ci in range(len(COMBOS))}
        best = {ci: float(f_s5.max()) for ci in thr}
        trace = {ci: [] for ci in thr}
        active, it = set(thr), 0
        while active:
            it += 1
            changed = set()
            for g in range(5):
                rules, idx = [], []
                for ci in sorted(active):
                    for t in G5:
                        th = list(thr[ci])
                        th[g] = t
                        rules.append((ci, tuple(th)))
                        idx.append(ci)
                f = evaluate_rules(pool, n, rules)
                for ci in sorted(active):
                    sel = [k for k, c in enumerate(idx) if c == ci]
                    fs = f[sel]
                    cur = G5.index(thr[ci][g])
                    k = cur if fs[cur] >= fs.max() - 1e-12 else int(np.argmax(fs))
                    if k != cur:
                        thr[ci][g] = G5[k]
                        changed.add(ci)
                    best[ci] = float(fs[k])
                    trace[ci].append([it, GROUPS[g], G5[k], round(best[ci], 5)])
            active = changed
            if it > 30:
                break
        print(f'ASCENT3 {it} iterations {time.time() - t0:.0f}s', flush=True)
        restricted = {}
        for name, gi in (('a_mid_own', 3), ('b_tip_own', 2)):
            rules = [(ci, tuple(m if g == gi else s for g in range(5))) for ci in range(len(COMBOS)) for s in G5 for m in G5]
            f = evaluate_rules(pool, n, rules).reshape(len(COMBOS), len(G5), len(G5))
            restricted[name] = f
    full_ci = max(range(len(COMBOS)), key=lambda ci: (best[ci], -ci))
    rules = dict(ref=dict(combo=None, thresholds=[t_ref] * 5, monitor_f1_search=float(f_ref.max())),
                 single05=dict(combo=None, thresholds=[t_s5] * 5, monitor_f1_search=float(f_s5.max())),
                 full=dict(combo=full_ci, thresholds=list(thr[full_ci]), monitor_f1_search=best[full_ci]))
    per_combo = {f'tau{t}_r{r}': dict(full=dict(thresholds=dict(zip(GROUPS, thr[ci])), monitor_f1=best[ci], trace=trace[ci]))
                 for ci, (t, r) in enumerate(COMBOS)}
    for name, gi in (('a_mid_own', 3), ('b_tip_own', 2)):
        f = restricted[name]
        ci, s, m = np.unravel_index(int(np.argmax(f)), f.shape)
        th = [G5[m] if g == gi else G5[s] for g in range(5)]
        rules[name] = dict(combo=int(ci), thresholds=th, monitor_f1_search=float(f.max()))
        for cj, (t, r) in enumerate(COMBOS):
            sj, mj = np.unravel_index(int(np.argmax(f[cj])), f[cj].shape)
            per_combo[f'tau{t}_r{r}'][name] = dict(shared=G5[sj], own=G5[mj], monitor_f1=float(f[cj].max()))
    print(f'RULES3 {json.dumps(rules)} {time.time() - t0:.0f}s', flush=True)
    # details (part-2 detail2_patch on the kept candidates of each rule)
    details = {}
    for split in ('monitor', 'val'):
        recs = []
        for r in data[split]['records']:
            c = {}
            for vid, ru in rules.items():
                k = kept(r, ru['combo'] or 0, ru['thresholds'])
                c[vid] = dict(pos=r['pos'][k], score=r['score'][k])
            recs.append(dict(sample_id=r['sample_id'], cache=r['cache'], cands=c))
        CTX.update(recs=recs, rules={v: (v, ('single', -1.0)) for v in rules}, band={v: FACE for v in rules}, ap=False)
        details[split] = S.run_pool(S.detail2_patch, len(recs), workers)
    for vid, ru in rules.items():
        ru['tau_r'] = None if ru['combo'] is None else list(COMBOS[ru['combo']])
        ru['thresholds_by_group'] = dict(zip(GROUPS, ru['thresholds']))
        ru['train_monitor'] = S.summarise2(details['monitor'], vid)
        ru['validation'] = S.summarise2(details['val'], vid)
        ru['validation_node_f1_minus_ref'] = S.paired_node(details['val'], details['val'], vid, 'ref') if vid != 'ref' else None
    diag = {}
    for split in ('monitor', 'val'):
        tab = diag_tables(data[split]['records'])
        diag[split] = {f'tau{t}_r{r}': dict(table=table_json(tab[ci]), interior_arm_shares=separation(tab[ci]))
                       for ci, (t, r) in enumerate(COMBOS)}
    cls = {split: dict(zip(S.CLASSES, map(int, sum(d['cls_tot'] for d in det)))) for split, det in details.items()}
    p2 = json.loads((OUT / f'node_decode2_{a.entry}.json').read_text())['variants'][REF_WINDOW] if not a.smoke else None
    result = dict(
        entry=a.entry, stage1_run=S.STAGE1, smoke=a.smoke, floor=FLOOR3, window=WIN, maps='flip8 (heat, offsets, centreline)',
        taus=list(TAUS), radii=list(RADII), face_band_voxels=FACE, groups=list(GROUPS), types=list(TYPES),
        heat_bins=list(HEAT_BINS), threshold_grid_rules=G5, threshold_grid_ref=[S.GRID[0], S.GRID[-1], 0.01],
        patches=dict(monitor=len(data['monitor']['records']), val=len(data['val']['records'])),
        candidates=dict(monitor=int(sum(len(r['score']) for r in data['monitor']['records'])),
                        val=int(sum(len(r['score']) for r in data['val']['records']))),
        gt_nodes_by_class=cls, sanity=sanity3(), rules=rules, per_combo_monitor=per_combo,
        parity_ref_vs_part2=None if p2 is None else dict(part2_threshold=p2['rule'][1], part2_val_f1=p2['validation']['f1'],
                                                         part2_monitor_f1=p2['train_monitor']['f1']),
        diagnostic=diag, figure_ids=data['val']['figure_ids'],
        notes=['all choices on monitor_train only; validation for reporting; no test split',
               'arm count: ball r+2 around the rounded refined position, 26-connectivity, component of the candidate (or '
               'nearest mask voxel within 2 voxels), components in the shell r < d <= r+2, capped at 3; off-vessel = 0 group',
               'candidate types use the one-to-one 2 mm matching of ALL candidates >= 0.05 (diagnostic); false detections '
               'of a rule use the matching of its kept candidates (part-2 detail2_patch)',
               'rules a/b: exhaustive 2-D grid (shared, own) per (tau, r), best (tau, r) chosen separately per rule'])
    out_json(a).write_text(json.dumps(result, indent=1, default=float) + '\n')
    print_report(result)
    print(f'ANALYSED3 {time.time() - t0:.0f}s', flush=True)


def print_report(r):
    f = lambda x: '  -  ' if x is None else f'{x:.3f}'  # noqa: E731
    print(f"\n== {r['entry']} candidates {r['candidates']} classes {json.dumps(r['gt_nodes_by_class'])}")
    print('parity ref vs part 2', r['parity_ref_vs_part2'])
    for vid, v in r['rules'].items():
        m, va = v['train_monitor'], v['validation']
        d = v['validation_node_f1_minus_ref']
        rc, fd = va['recall_by_class'], va['false_detections']
        print(f"{vid:9s} tau,r {v['tau_r']} thr {v['thresholds']} | mon P {f(m['precision'])} R {f(m['recall'])} F1 {f(m['f1'])} | "
              f"val P {f(va['precision'])} R {f(va['recall'])} F1 {f(va['f1'])} | val rec b/j/t/o {f(rc['boundary'])}/"
              f"{f(rc['junction'])}/{f(rc['terminal'])}/{f(rc['other'])} | valFP dup/line/else {fd['duplicate_within_2mm_of_matched_gt']}/"
              f"{fd['within_1p5mm_of_gt_centreline']}/{fd['elsewhere']} face/int {fd['in_face_band']}/{fd['interior']} | dF1 "
              + ('ref' if d is None else f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"))
    for k, v in r['per_combo_monitor'].items():
        print(k, 'full', round(v['full']['monitor_f1'], 4), v['full']['thresholds'], '| a', round(v['a_mid_own']['monitor_f1'], 4),
              '| b', round(v['b_tip_own']['monitor_f1'], 4))
    ci = r['rules']['full']['tau_r']
    key = f'tau{ci[0]}_r{ci[1]}'
    for split in ('monitor', 'val'):
        print(f'-- diagnostic {split} {key} interior arm shares', json.dumps(r['diagnostic'][split][key]['interior_arm_shares']))


# ------------------------------------------------------------------------------------------------------ stage 2 (GPU)
S3 = {}


def stage2v3(a):
    import multiprocessing as mp
    import torch
    import dense_graph_20261030 as D
    import dense_graph_traced_20261030 as T
    from node_heatmap_probe_20261030 import make_dataset
    from protocol_report_20261106 import paired, pooled
    device = 'cuda'
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    res3 = json.loads(out_json(a).read_text())
    run_dir = S.RUNS / a.entry / S.STAGE2
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    assert not T.MODE['passthrough'] and not T.MODE['roles'] and not T.MODE['two_hop'] and not T.MODE['gap_features']
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], two_hop=T.MODE['two_hop'], passthrough=T.MODE['passthrough'],
                               backbone=final.get('backbone', 'unet'), feature_channels=final.get('feature_channels', 32)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    model.eval()
    s1, _ = S.load_stage1(a.entry, device)
    sp = S.splits_of(a)
    mon_ds, val_ds = make_dataset(a.entry, sp['monitor'], augment=False), make_dataset(a.entry, sp['val'], augment=False)
    R = res3['rules']
    arms = ['ref', 'full'] + (['a_mid_own'] if R['a_mid_own']['train_monitor']['f1'] >= R['full']['train_monitor']['f1'] - 0.005 else [])
    print('STAGE2V3 arms', arms, flush=True)
    views = [t for t in S.transforms() if 'flip8' in S.sets_of(t)]
    assert len(views) == 8
    original_peaks, original_forward = D.peaks, model.forward_image
    pool = mp.get_context('fork').Pool(max(1, workers - 4))

    def fwd(images):
        out = original_forward(images)
        heat = off = prob = 0
        for t in views:
            o = s1.forward_image(S.t_forward(images, t))[:, :5].float()
            heat = heat + S.t_back(o[:, :1].sigmoid(), t)
            off = off + S.t_back_vec(o[:, 1:4], t)
            prob = prob + S.t_back(o[:, 4:5].sigmoid(), t)
        S3['maps'] = (heat / 8, off / 8, prob[:, 0] / 8)
        return out

    def make_peaks(rule):
        thr = np.asarray(rule['thresholds'], np.float64)
        tau_r = rule['tau_r']

        def peaks(out, threshold, nms=None):
            heat, off, prob = S3['maps']
            dec = S.decode(heat, off, prob, f'raw{WIN}', FLOOR3)
            pos = [d['pos'].cpu().numpy().astype(np.float32) for d in dec]
            if tau_r is not None:
                pr = prob.float().cpu().numpy()
                arm = pool.starmap(arm_counts, [(pr[b], pos[b], (tau_r[0],), (tau_r[1],)) for b in range(len(dec))])
            res = []
            for b, d in enumerate(dec):
                sc = d['score'].cpu().numpy().astype(np.float32)
                g = groups_of(pos[b], arm[b][:, 0, 0]) if tau_r is not None else np.zeros(len(sc), int)
                keep = torch.as_tensor(sc >= thr[g], device=d['score'].device)
                res.append((d['pos'][keep].contiguous(), d['score'][keep].contiguous()))
            return res
        return peaks

    out = dict(arms={}, minus_ref={}, change='Stage-1 flip8 5^3 candidates kept by the rule (thresholds chosen in analyse3) replace '
               'D.peaks inside the unchanged T.predict of the stage-2 run (as part 2 stage2v2); stage 2 keeps its own features '
               'and centreline map; edge threshold re-selected on monitor_train by edge F1 (D.select_edge_threshold); '
               'evaluation D.evaluate; per-patch counts compare_dense_anchored.dense_patch + Betti; paired = '
               'protocol_report_20261106.paired (subject bootstrap).')
    cnts = {}
    for name in arms:
        t1 = time.time()
        try:
            D.peaks = make_peaks(R[name])
            model.forward_image = fwd
            mon = T.predict(model, mon_ds, device, 0.0, workers)
            et = D.select_edge_threshold(mon, 0.0)
            val = T.predict(model, val_ds, device, 0.0, workers)
        finally:
            D.peaks = original_peaks
            model.forward_image = original_forward
        res = dict(rule=dict(tau_r=R[name]['tau_r'], thresholds_by_group=R[name]['thresholds_by_group']), edge_threshold=et,
                   train_monitor=D.evaluate(mon, 0.0, et), validation=D.evaluate(val, 0.0, et))
        CTX.update(rows=val, nt=0.0, et=et)
        cnt = np.asarray(S.run_pool(S.stage2_counts, len(val), workers), float)
        res['validation_pooled_counts'] = pooled(cnt)
        res['node_check_vs_analyse3'] = dict(stage2_validation_node_f1=res['validation']['node_f1'],
                                             analyse3_validation_node_f1=R[name]['validation']['f1'],
                                             stage2_monitor_node_f1=res['train_monitor']['node_f1'],
                                             analyse3_monitor_node_f1=R[name]['train_monitor']['f1'])
        ids = [r['sample_id'] for r in val]
        cnts[name] = (cnt, ids)
        out['arms'][name] = res
        v = res['validation']
        print(f"STAGE2V3 {a.entry} {name}: edge thr {et} val node F1 {v['node_f1']:.4f} edge F1 {v['edge_f1']:.4f} "
              f"branch F1 {v['common_branch_f1']:.4f} curve F1 {v['common_curve_f1']:.4f} ({time.time() - t1:.0f}s)", flush=True)
    pool.close()
    for name in arms[1:]:
        assert cnts[name][1] == cnts['ref'][1]
        d = paired(cnts[name][0], cnts['ref'][0], cnts['ref'][1])['difference']
        out['minus_ref'][name] = {k: d[k] for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1', 'mae_beta0', 'mae_beta1')}
        print('PAIRED3', a.entry, name, json.dumps({k: [round(v['point'], 4), *[round(x, 4) for x in v['ci95']]]
                                                    for k, v in out['minus_ref'][name].items()}), flush=True)
    res3 = json.loads(out_json(a).read_text())
    res3['stage2'] = out
    out_json(a).write_text(json.dumps(res3, indent=1, default=float) + '\n')
    print('DONE stage2v3', flush=True)


# ------------------------------------------------------------------------------------------------------ figures (CPU)
GCOL = {'face': '#8a8a8a', '0': '#6250d6', '1': '#1baf7a', '2': '#eb6834', '3+': '#e3343c'}


def figures3(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch  # noqa: F401
    res = json.loads(out_json(a).read_text())
    rule = res['rules']['full']
    ci = COMBOS.index(tuple(rule['tau_r']))
    data = pickle.loads((STORE / f'{S.tag(a)}-p3-val.pkl').read_bytes())
    VIZ.mkdir(parents=True, exist_ok=True)
    short = a.entry.replace('-simp2', '') + ('-smoke' if a.smoke else '')
    th = rule['thresholds']
    for rec in data['records']:
        if 'mip' not in rec:
            continue
        gt = S.gt_of2(rec['cache'])
        with np.load(rec['cache']) as z:
            gv = z['nodes'] * SIZE
            P, off = z['full_paths'] * SIZE, z['path_offsets']
        tau, rad = COMBOS[ci]
        arms = rec['arms'][:, TAUS.index(tau), RADII.index(rad)]
        grp = groups_of(rec['pos'], arms)
        keep = rec['score'] >= np.asarray(th)[grp]
        q = np.flatnonzero(keep)
        p = rec['pos'][q]
        mp = S.match((p / SIZE) @ gt['M'].T, gt['g'])
        found = np.zeros(len(gv), bool)
        found[list(mp.values())] = True
        real = ~gt['ins']
        unmatched = np.array([k for k in range(len(q)) if k not in mp], int)
        drop = np.flatnonzero(~keep)
        fig, ax = plt.subplots(3, 3, figsize=(11.5, 12.4), dpi=72)
        axes_of = [(1, 2), (0, 2), (0, 1)]
        for row, name in enumerate(('image', 'heat', 'centreline')):
            for k_ax in range(3):
                h = ax[row, k_ax]
                h.imshow(rec['mip'][name][k_ax].astype(np.float32), cmap='gray' if row == 0 else 'magma',
                         vmin=None if row == 0 else 0, vmax=None if row == 0 else 1, origin='lower', interpolation='nearest')
                r_, c_ = axes_of[k_ax]
                if row == 0:
                    for e in range(len(off) - 1):
                        seg = P[off[e]:off[e + 1]]
                        h.plot(seg[:, c_], seg[:, r_], lw=0.6, color='#86b6ef', alpha=0.9)
                for cname in S.CLASSES:
                    s = (gt['cls'] == cname) & real
                    for flag in (True, False):
                        m = s & (found == flag)
                        if m.any():
                            h.scatter(gv[m, c_], gv[m, r_], s=60, marker=S.MARKERS[cname], facecolors='none',
                                      edgecolors='white' if flag else '#ffd400', linewidths=1.0 if flag else 1.6)
                for gi, gname in enumerate(GROUPS):
                    m = grp[q] == gi
                    if m.any():
                        h.scatter(p[m, c_], p[m, r_], s=16, color=GCOL[gname], zorder=3)
                    m = grp[drop] == gi
                    if m.any():
                        h.scatter(rec['pos'][drop][m, c_], rec['pos'][drop][m, r_], s=14, marker='+', color=GCOL[gname],
                                  alpha=0.7, linewidths=0.8, zorder=3)
                if len(unmatched):
                    h.scatter(p[unmatched, c_], p[unmatched, r_], s=46, marker='x', color='#e34948', linewidths=1.2, zorder=4)
                h.set_xlim(-0.5, SIZE - 0.5)
                h.set_ylim(-0.5, SIZE - 0.5)
                h.set_xticks([])
                h.set_yticks([])
                h.set_title(f'{name} (flip8) MIP, axis {k_ax}', fontsize=8.5)
        miss = {c: int(((gt['cls'] == c) & real & ~found).sum()) for c in S.CLASSES}
        fig.suptitle(f"{a.entry} {rec['sample_id']}  v3 rule 'full': tau {tau}, r {rad}, thresholds face/0/1/2/3+ = {th}\n"
                     f"GT nodes {int(real.sum())} (found {int((found & real).sum())}), kept {len(q)} of {len(rec['score'])} candidates, "
                     f"unmatched kept {len(unmatched)}; missed boundary/junction/terminal/other "
                     f"{miss['boundary']}/{miss['junction']}/{miss['terminal']}/{miss['other']}\n"
                     f"GT: square boundary, triangle junction, circle terminal, diamond other; white = found, yellow = missed\n"
                     f"candidates by arm group: grey face, purple 0, green 1, orange 2, red 3+; dot = kept, + = dropped; red x = kept, unmatched",
                     fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(VIZ / f"v3_{short}_{rec['sample_id']}.png")
        plt.close(fig)
    print('FIGURES3', sorted(x.name for x in VIZ.glob(f'v3_{short}_*.png')), flush=True)


def run_command(a):
    if a.command == 'sanity3':
        sanity3()
    if a.command in ('collect3', 'all3'):
        collect3(a)
    if a.command in ('analyse3', 'all3'):
        analyse3(a)
    if a.command in ('figures3', 'all3'):
        figures3(a)
    if a.command in ('stage2v3', 'all3'):
        stage2v3(a)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['sanity3', 'collect3', 'analyse3', 'stage2v3', 'figures3', 'all3'])
    p.add_argument('entry')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    for entry in args.entry.split('+'):
        run_command(argparse.Namespace(**{**vars(args), 'entry': entry}))
