#!/usr/bin/env python3
"""Node-decoding study on trained models, no retraining (13 Nov 2026).

Question: does another way of turning the stage-1 node heatmap into detections improve node detection? Hypothesis:
nodes whose labelled position is uncertain by 1-2 mm (terminals, traced junctions) get a low, smeared peak because the
CenterNet focal loss (Gaussian sigma 1.5 voxel target, one positive voxel) penalises heat a few voxels away from the
labelled voxel; then the heat summed / averaged around a peak ("mass") should separate real from false nodes better
than the peak height, and such nodes should have a low peak but a normal mass.

collect (GPU): stage-1 run (spline6_unet_wide_long_r1111) on monitor_train and every val_id, no augmentation. Per
  patch, candidates (score >= FLOOR = 0.02) of every decoding:
    raw<w>        local maxima of the sigmoid heat in a w^3 window (w = 3: the current rule D.peaks), score = heat,
                  position = voxel + predicted offset;
    mass_<f>_w<w> the heat smoothed by filter f (box 3/5/7, Gaussian sigma 1.0/1.5/2.0 voxel; replicate padding),
                  divided by the filter's response at the centre of an ideal target blob (Gaussian sigma 1.5, peak 1),
                  so a perfect blob scores 1.0 (a monotone rescale: AP and F1 are unchanged by it); local maxima of the
                  smoothed map in a w^3 window (w >= filter size); score = normalised smoothed value; position either
                  '@argmax' (raw-heat argmax inside the window, + its offset) or '@centroid' (heat-weighted centroid
                  of the window);
    flip:<...>    the same on the heat (and offsets) averaged over the 8 axis-flip combinations of the input;
  plus, per candidate, the maximum centreline probability within 2 voxels ('+cl' variants keep >= 0.5);
  per GT node, the maximum raw heat and the maximum box-5 mass within 2 mm (mm ball through the patch metric), the
  same around every V0 candidate, and maximum-intensity projections of 8 validation patches for figures.
analyse (CPU): per variant, node AP @2 mm (the project's definition, dense_ap_20261030.node_ap_state, all GT nodes;
  and a loop-aware version where inserted anchors are don't-care), the threshold maximising node F1 on monitor_train
  (grid 0.02..0.95 step 0.01), node P/R/F1 there on monitor_train and validation (matching = match_nodes, 2 mm,
  one-to-one, loop-aware as metrics/loop_aware.py: inserted anchors take part in the matching, are never missed, and a
  prediction matched to one is not a false positive), recall per node class (node_heatmap_probe.node_types), false
  detections split into duplicate / on a GT centreline (1.5 mm) / elsewhere, and the subject-paired validation node F1
  difference against V0 (protocol_report_20261106.paired). Amplitude diagnostic per class. Choices on monitor_train only.
figures (CPU): PNGs under experiments/final_try_20261113/viz_tmp/.
stage2 (GPU): the chosen variant(s) replace D.peaks inside the unchanged T.predict of the stage-2 run
  (traced_unet_wide_long_r1111), decoding the stage-2 model's OWN heatmap (its U-Net is fine-tuned from stage 1), node
  threshold re-selected on monitor_train by node F1 on that heatmap, edge threshold re-selected on monitor_train by
  edge F1 (D.select_edge_threshold); validation scored with D.evaluate and per patch with compare_dense_anchored.
  dense_patch (the counts of compare_arms_20261109), subject-paired against the unchanged pipeline.

usage: node_decode_study_20261113.py {collect,analyse,figures,stage2,all} <entry> [--smoke] [--variants a,b]
Outputs: $SCRATCH/experiments/gnbm/node_decode_20261113/<entry>{-smoke}-{monitor,val}.pkl (per-patch candidates),
         experiments/final_try_20261113/node_decode_<entry>{-smoke}.json, viz_tmp/*.png,
         experiments/final_try_20261113/node_decode_stage2_<entry>{-smoke}.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
SCRATCH = Path(os.environ.get('SCRATCH', '/tmp'))
RUNS = SCRATCH / 'experiments/gnbm/dense_graph_20261030'
STORE = SCRATCH / 'experiments/gnbm/node_decode_20261113'
OUT = ROOT / 'experiments/final_try_20261113'
VIZ = OUT / 'viz_tmp'
STAGE1, STAGE2 = 'spline6_unet_wide_long_r1111', 'traced_unet_wide_long_r1111'
SIZE = 64
FLOOR = 0.02
TOL = 2.0
CL_MIN, CL_RADIUS = 0.5, 2
GRID = [round(float(t), 2) for t in np.arange(0.02, 0.9501, 0.01)]
SIGMA_TARGET = 1.5
FILTERS = {'box3': ('box', 3), 'box5': ('box', 5), 'box7': ('box', 7),
           'g1.0': ('gauss', 1.0), 'g1.5': ('gauss', 1.5), 'g2.0': ('gauss', 2.0)}
MASS = [('box3', 3), ('box3', 5), ('box3', 7), ('box5', 5), ('box5', 7), ('box7', 7),
        ('g1.0', 3), ('g1.0', 5), ('g1.0', 7), ('g1.5', 5), ('g1.5', 7), ('g2.0', 5), ('g2.0', 7)]
RAW = [3, 5, 7]
BASES = [f'raw{w}' for w in RAW] + [f'mass_{f}_w{w}' for f, w in MASS]
CLASSES = ('boundary', 'junction', 'terminal', 'other')
QUANTILES = (10, 25, 50, 75, 90)
N_FIG = 8


def tag(a):
    return f"{a.entry}{'-smoke' if a.smoke else ''}"


def splits_of(a):
    from node_heatmap_probe_20261030 import PLAN
    plan = json.loads(PLAN.read_text())['datasets'][a.entry]
    s = dict(monitor=list(plan['monitor_train']), val=list(plan['val_ids']))
    if a.smoke:
        s = dict(monitor=s['monitor'][:16], val=s['val'][:40])
    return s


# ------------------------------------------------------------------------------------------------------ decoding (GPU)
_KERNELS, _NORM = {}, {}


def kernel1d(name, device):
    key = (name, str(device))
    if key not in _KERNELS:
        import torch
        kind, v = FILTERS[name]
        if kind == 'box':
            k = torch.ones(v, device=device) / v
        else:
            r = int(np.ceil(3 * v))
            x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
            k = torch.exp(-x ** 2 / (2 * v ** 2))
            k = k / k.sum()
        _KERNELS[key] = k
    return _KERNELS[key]


def smooth(heat, name):
    """Separable filter of heat (B,1,X,Y,Z), replicate padding."""
    import torch.nn.functional as F
    k = kernel1d(name, heat.device)
    r = len(k) // 2
    x = F.pad(heat, (r,) * 6, mode='replicate')
    x = F.conv3d(x, k.view(1, 1, -1, 1, 1))
    x = F.conv3d(x, k.view(1, 1, 1, -1, 1))
    return F.conv3d(x, k.view(1, 1, 1, 1, -1))


def norm_of(name, device):
    """Filter response at the centre of the ideal training target (Gaussian sigma 1.5 voxel, peak 1)."""
    key = (name, str(device))
    if key not in _NORM:
        import torch
        ax = torch.arange(SIZE, device=device, dtype=torch.float32) - 32
        d2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
        blob = torch.exp(-d2 / (2 * SIGMA_TARGET ** 2))[None, None]
        _NORM[key] = float(smooth(blob, name)[0, 0, 32, 32, 32])
    return _NORM[key]


def cube(w, device):
    import torch
    r = torch.arange(-(w // 2), w // 2 + 1, device=device)
    return torch.stack(torch.meshgrid(r, r, r, indexing='ij'), -1).view(-1, 3)


def ball_vox(radius, device):
    o = cube(2 * radius + 1, device)
    return o[(o.float() ** 2).sum(-1) <= radius ** 2]


def gather(vol, b, pts):
    """vol (B,X,Y,Z); b (N,) batch index; pts (N,K,3) long -> values (N,K) and in-bounds mask."""
    ok = ((pts >= 0) & (pts < SIZE)).all(-1)
    c = pts.clamp(0, SIZE - 1)
    return vol[b[:, None].expand_as(c[..., 0]), c[..., 0], c[..., 1], c[..., 2]], ok


def decode(heat, off, prob, base, floor):
    """All candidates of one base decoding for a batch.
    heat (B,1,X,Y,Z) sigmoid heat, off (B,3,X,Y,Z) offsets, prob (B,X,Y,Z) centreline probability.
    Returns per patch a dict pos (N,3) voxel [, pos_c centroid], score (N,), cl (N,) sorted by score (desc)."""
    import torch
    import torch.nn.functional as F
    if base.startswith('raw'):
        w = int(base[3:])
        m = heat
    else:
        _, f, w = base.split('_')
        w = int(w[1:])
        m = smooth(heat, f) / norm_of(f, heat.device)
    keep = (m == F.max_pool3d(m, w, 1, w // 2)) & (m >= floor)
    idx = keep[:, 0].nonzero()
    bi = idx[:, 0]
    score = m[:, 0][keep[:, 0]]
    offv = off.permute(0, 2, 3, 4, 1)
    h = heat[:, 0]
    extra = {}
    if base.startswith('raw'):
        pos = idx[:, 1:].float() + offv[bi, idx[:, 1], idx[:, 2], idx[:, 3]]
    else:
        nb = idx[:, None, 1:] + cube(w, heat.device)[None]
        hv, ok = gather(h, bi, nb)
        a = torch.where(ok, hv, torch.full_like(hv, -1.0)).argmax(1)
        av = nb.clamp(0, SIZE - 1)[torch.arange(len(nb), device=nb.device), a]
        pos = av.float() + offv[bi, av[:, 0], av[:, 1], av[:, 2]]
        wts = torch.where(ok, hv, torch.zeros_like(hv))
        extra['pos_c'] = ((wts[..., None] * nb.float()).sum(1) / wts.sum(1).clamp_min(1e-6)[:, None]).clamp(-0.5, SIZE - 0.5)
    pos = pos.clamp(-0.5, SIZE - 0.5)
    pr = pos.round().long()
    cv, ok = gather(prob, bi, pr[:, None] + ball_vox(CL_RADIUS, heat.device)[None])
    cl = torch.where(ok, cv, torch.zeros_like(cv)).amax(1)
    out = []
    for b in range(len(heat)):
        s = bi == b
        order = score[s].argsort(descending=True)
        d = dict(pos=pos[s][order], score=score[s][order], cl=cl[s][order])
        for k, v in extra.items():
            d[k] = v[s][order]
        out.append(d)
    return out


def maps(out):
    """sigmoid heat (B,1,..), offsets (B,3,..), centreline probability (B,..) from a model output."""
    out = out.float()
    return out[:, :1].sigmoid(), out[:, 1:4], out[:, 4].sigmoid()


def flip_maps(model, images, out0):
    """Heat, offsets and centreline probability averaged over the 8 axis flips (outputs flipped back; the offset
    component of a flipped axis changes sign). out0 = the output of the unflipped input."""
    import torch
    h0, f0, p0 = maps(out0)
    heat, off, prob = h0.clone(), f0.clone(), p0.clone()
    for n in (1, 2, 3):
        for dims in combinations((2, 3, 4), n):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model.forward_image(images.flip(dims))[:, :5]
            h, f, p = maps(o.flip(dims))
            for d in dims:
                f[:, d - 2] = -f[:, d - 2]
            heat += h
            off += f
            prob += p
    return heat / 8, off / 8, prob / 8


def ball_mm(metric_vox, device):
    """Integer voxel offsets within TOL mm (metric_vox: voxel -> mm, 3x3)."""
    import torch
    vox = metric_vox.norm(dim=0)
    reach = [int(np.ceil(TOL / float(v))) for v in vox]
    g = torch.stack(torch.meshgrid(*[torch.arange(-k, k + 1, device=device) for k in reach], indexing='ij'), -1).view(-1, 3)
    return g[(g.float() @ metric_vox.T).norm(dim=-1) <= TOL]


def ball_max(vol, b, pts_vox, offsets):
    import torch
    if len(pts_vox) == 0:
        return np.zeros(0, np.float32)
    p = pts_vox.round().long().clamp(0, SIZE - 1)
    v, ok = gather(vol, torch.full((len(p),), b, device=p.device, dtype=torch.long), p[:, None] + offsets[None])
    return torch.where(ok, v, torch.zeros_like(v)).amax(1).cpu().numpy()


def cache_of(record):
    return str(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz')


def figure_ids(ds_records):
    """Validation patches for figures: sorted ids with >= 8 real GT nodes, N_FIG evenly spaced."""
    ok = []
    for r in sorted(ds_records, key=lambda r: r.sample_id):
        with np.load(cache_of(r)) as z:
            ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(z['nodes']), bool)
            if (~ins).sum() >= 8:
                ok.append(r.sample_id)
    n = len(ok)
    return [ok[int(np.floor((k + 0.5) * n / N_FIG))] for k in range(min(N_FIG, n))]


def collect_model(model, ds, device, workers, figure=(), bases=BASES, flips=True, amplitude=True):
    """Run a model over a dataset and return per-patch candidate records."""
    import torch
    import dense_graph_20261030 as D
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=workers, collate_fn=D.collate)
    model.eval()
    recs, offset = [], 0
    with torch.no_grad():
        for images, nodes, edges, geometry in loader:
            images = images.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_image(images)[:, :5]
            plain = maps(out)
            sets = {'': plain}
            if flips:
                sets['flip:'] = flip_maps(model, images, out)
            dec = {}
            for prefix, (heat, off, prob) in sets.items():
                for base in bases:
                    dec[prefix + base] = decode(heat, off, prob, base, FLOOR)
            heat, off, prob = plain
            m5 = (smooth(heat, 'box5') / norm_of('box5', device))[:, 0] if amplitude else None
            for b in range(len(images)):
                record = ds.records[offset + b]
                cache = cache_of(record)
                with np.load(cache) as z:
                    M, gn = z['metric'], z['nodes']
                mv = torch.as_tensor(M / SIZE, dtype=torch.float32, device=device)
                rec = dict(sample_id=record.sample_id, cache=cache, voxel_mm=np.linalg.norm(M / SIZE, axis=0),
                           cands={k: {f: t.cpu().numpy().astype(np.float32) for f, t in v[b].items()} for k, v in dec.items()})
                if amplitude:
                    o = ball_mm(mv, device)
                    g = torch.as_tensor(gn * SIZE, dtype=torch.float32, device=device)
                    v0 = torch.as_tensor(rec['cands']['raw3']['pos'], device=device)
                    rec.update(gt_peak2mm=ball_max(heat[:, 0], b, g, o), gt_mass2mm=ball_max(m5, b, g, o),
                               v0_peak2mm=ball_max(heat[:, 0], b, v0, o), v0_mass2mm=ball_max(m5, b, v0, o))
                if record.sample_id in figure:
                    rec['mip'] = {name: np.stack([vol.max(axis=k) for k in range(3)]).astype(np.float16) for name, vol in
                                  (('image', images[b, 0].float().cpu().numpy()), ('heat', heat[b, 0].cpu().numpy()),
                                   ('centreline', prob[b].cpu().numpy()))}
                recs.append(rec)
            offset += len(images)
    return recs


def load_stage1(entry, device):
    import torch
    import dense_graph_20261030 as D
    run_dir = RUNS / entry / STAGE1
    final = json.loads((run_dir / 'final.json').read_text())
    model = D.DenseGraph(final['mode'], final['interior'], backbone=final.get('backbone', 'unet'),
                         feature_channels=final.get('feature_channels', 32)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    return model.eval(), final


def collect(a):
    import torch
    from node_heatmap_probe_20261030 import make_dataset
    device = 'cuda'
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8')) - 2
    model, final = load_stage1(a.entry, device)
    STORE.mkdir(parents=True, exist_ok=True)
    for split, ids in splits_of(a).items():
        t0 = time.time()
        ds = make_dataset(a.entry, ids, augment=False)
        fig = figure_ids(ds.records) if split == 'val' else []
        recs = collect_model(model, ds, device, workers, figure=fig)
        (STORE / f'{tag(a)}-{split}.pkl').write_bytes(pickle.dumps(dict(figure_ids=fig, records=recs), protocol=4))
        n = np.mean([len(r['cands']['raw3']['score']) for r in recs])
        print(f'COLLECTED {a.entry} {split} {len(recs)} patches, {n:.1f} V0 candidates/patch >= {FLOOR}, {time.time() - t0:.0f}s',
              flush=True)
    del model
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------ analysis (CPU)
def variant_ids():
    out = []
    for prefix in ('', 'flip:'):
        for base in BASES:
            for pm in (['off'] if base.startswith('raw') else ['argmax', 'centroid']):
                for cl in ('', '+cl'):
                    out.append(f'{prefix}{base}@{pm}{cl}')
    return out


def parse(vid):
    cl = vid.endswith('+cl')
    name, pm = vid.replace('+cl', '').split('@')
    return name, pm, cl


def cands(rec, vid):
    name, pm, cl = parse(vid)
    c = rec['cands'][name]
    pos, score = (c['pos_c'] if pm == 'centroid' else c['pos']), c['score']
    if cl:
        k = c['cl'] >= CL_MIN
        pos, score = pos[k], score[k]
    return pos, score


GT_CACHE = {}


def gt_of(cache):
    """GT nodes (mm), inserted mask, class per node, dense GT path tree (mm)."""
    if cache not in GT_CACHE:
        import torch
        from scipy.spatial import cKDTree
        from node_heatmap_probe_20261030 import node_types
        from node_miss_diagnosis_20261110 import dense_path
        with np.load(cache) as z:
            M, gn, ge = z['metric'], z['nodes'], z['edges'].reshape(-1, 2)
            ins = z['inserted'].astype(bool) if 'inserted' in z.files else np.zeros(len(gn), bool)
            P, off = z['full_paths'] @ M.T, z['path_offsets']
        cls = node_types(torch.as_tensor(gn), torch.as_tensor(ge)) if len(gn) else np.zeros(0, '<U8')
        paths = [dense_path(P[off[k]:off[k + 1]]) for k in range(len(ge))]
        GT_CACHE[cache] = dict(M=M, nodes=gn, g=gn @ M.T, ins=ins, cls=np.asarray(cls),
                               tree=cKDTree(np.concatenate(paths)) if paths else None)
    return GT_CACHE[cache]


def match(p_mm, g_mm):
    from diagnose_branch_relations import match_nodes
    return match_nodes(p_mm, g_mm, TOL) if len(p_mm) and len(g_mm) else {}


def counts(mapping, n_pred, ins):
    tp = sum(not ins[j] for j in mapping.values())
    return tp, n_pred - len(mapping), int((~ins).sum()) - tp


CTX = {}


def sweep_patch(i):
    """(V, G, 3) node tp / fp / fn of one monitor patch for every variant and grid threshold."""
    rec = CTX['recs'][i]
    gt = gt_of(rec['cache'])
    out = np.zeros((len(CTX['vids']), len(GRID), 3), np.int32)
    neg = -np.asarray(GRID)
    for v, vid in enumerate(CTX['vids']):
        pos, score = cands(rec, vid)
        p = (pos / SIZE) @ gt['M'].T
        ks = np.searchsorted(-score, neg, side='right')
        memo = {}
        for t, k in enumerate(ks):
            if k not in memo:
                memo[k] = counts(match(p[:k], gt['g']), k, gt['ins'])
            out[v, t] = memo[k]
    return out


def ap_state_loop_aware(p_mm, score, g_mm, ins):
    """node_ap_state with inserted anchors as don't-care: greedy score-order match to unmatched GT (inserted included);
    a prediction matched to an inserted anchor leaves the ranking; recall over real GT nodes."""
    order = np.argsort(-score, kind='stable')
    d = np.linalg.norm(p_mm[order][:, None] - g_mm[None], axis=-1) if len(g_mm) else np.zeros((len(order), 0))
    matched = np.zeros(len(g_mm), bool)
    tp, keep = np.zeros(len(order), bool), np.ones(len(order), bool)
    for i in range(len(order)):
        c = np.flatnonzero(~matched & (d[i] <= TOL))
        if len(c):
            j = c[np.argmin(d[i, c])]
            matched[j] = True
            if ins[j]:
                keep[i] = False
            else:
                tp[i] = True
    return score[order][keep], tp[keep], int((~ins).sum())


def detail_patch(i):
    """Per variant at its chosen threshold: counts, class hits, false-detection places, matched distances; AP states."""
    from dense_ap_20261030 import node_ap_state
    rec = CTX['recs'][i]
    gt = gt_of(rec['cache'])
    real = ~gt['ins']
    cls_tot = np.array([int(((gt['cls'] == c) & real).sum()) for c in CLASSES])
    res = {}
    for vid in CTX['vids']:
        pos, score = cands(rec, vid)
        p_all = (pos / SIZE) @ gt['M'].T
        k = int((score >= CTX['thr'][vid]).sum())
        p = p_all[:k]
        mp = match(p, gt['g'])
        tp, fp, fn = counts(mp, k, gt['ins'])
        hit = np.zeros(len(gt['g']), bool)
        hit[list(mp.values())] = True
        hit &= real
        cls_hit = np.array([int((hit & (gt['cls'] == c)).sum()) for c in CLASSES])
        place = np.zeros(4, np.int32)  # duplicate, centreline, elsewhere, near an unmatched GT node (expected 0)
        matched_gt = np.zeros(len(gt['g']), bool)
        matched_gt[list(mp.values())] = True
        fp_idx = [q for q in range(k) if q not in mp]
        for q in fp_idx:
            d = np.linalg.norm(gt['g'] - p[q], axis=1) if len(gt['g']) else np.zeros(0)
            if (d <= TOL).any():
                place[0 if matched_gt[d <= TOL].any() else 3] += 1
            elif gt['tree'] is not None and gt['tree'].query(p[q])[0] <= 1.5:
                place[1] += 1
            else:
                place[2] += 1
        dist = [np.linalg.norm(p[q] - gt['g'][j]) for q, j in mp.items() if not gt['ins'][j]]
        entry = dict(c=np.array([tp, fp, fn], np.int32), cls_hit=cls_hit, place=place, dist=(float(np.sum(dist)), len(dist)),
                     ap=node_ap_state(pos / SIZE, score, gt['nodes'], gt['M']) if CTX['ap'] else None,
                     ap_la=ap_state_loop_aware(p_all, score, gt['g'], gt['ins']) if CTX['ap'] else None)
        if vid == 'raw3@off':
            entry['hit'] = hit
            entry['fp_idx'] = np.asarray(fp_idx, int)
        res[vid] = entry
    return dict(sample_id=rec['sample_id'], cls_tot=cls_tot, res=res)


def prf(c):
    tp, fp, fn = (int(x) for x in c)
    return dict(precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0, tp=tp, fp=fp, fn=fn)


def run_pool(fn, n, workers):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers) as pool:
        return pool.map(fn, range(n), chunksize=8)


def summarise(details, vid):
    from dense_ap_20261030 import node_ap
    c = sum(d['res'][vid]['c'] for d in details)
    hits = sum(d['res'][vid]['cls_hit'] for d in details)
    tot = sum(d['cls_tot'] for d in details)
    place = sum(d['res'][vid]['place'] for d in details)
    ds_, dn = sum(d['res'][vid]['dist'][0] for d in details), sum(d['res'][vid]['dist'][1] for d in details)
    out = dict(**prf(c), recall_by_class={k: (float(h / t) if t else None) for k, h, t in zip(CLASSES, hits, tot)},
               gt_by_class={k: int(t) for k, t in zip(CLASSES, tot)},
               false_detections=dict(duplicate_within_2mm_of_matched_gt=int(place[0]), within_1p5mm_of_gt_centreline=int(place[1]),
                                     elsewhere=int(place[2]), near_unmatched_gt_node_expected_0=int(place[3])),
               matched_distance_mm_mean=ds_ / max(1, dn), patches=len(details))
    if details[0]['res'][vid]['ap'] is not None:
        st = [d['res'][vid]['ap'] for d in details]
        out['node_ap_2mm'] = node_ap(st) if sum(len(s[0]) for s in st) else dict(node_ap=0.0)
        st = [d['res'][vid]['ap_la'] for d in details]
        out['node_ap_2mm_loop_aware'] = node_ap(st) if sum(len(s[0]) for s in st) else dict(node_ap=0.0)
    return out


def quant(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), **{f'q{q}': (float(np.percentile(v, q)) if len(v) else None) for q in QUANTILES})


def amplitude(recs, details):
    """Peak heat and box-5 mass within 2 mm of every real GT node, by class and V0 outcome; V0 false detections."""
    out = {}
    rows = {c: dict(peak=[], mass=[], hit=[]) for c in CLASSES}
    fp = dict(peak=[], mass=[], score=[])
    for r, d in zip(recs, details):
        gt = gt_of(r['cache'])
        hit = d['res']['raw3@off']['hit']
        for c in CLASSES:
            s = (gt['cls'] == c) & ~gt['ins']
            rows[c]['peak'] += r['gt_peak2mm'][s].tolist()
            rows[c]['mass'] += r['gt_mass2mm'][s].tolist()
            rows[c]['hit'] += hit[s].tolist()
        q = d['res']['raw3@off']['fp_idx']
        fp['peak'] += r['v0_peak2mm'][q].tolist()
        fp['mass'] += r['v0_mass2mm'][q].tolist()
        fp['score'] += r['cands']['raw3']['score'][q].tolist()
    for c, v in rows.items():
        pk, ms, h = np.asarray(v['peak']), np.asarray(v['mass']), np.asarray(v['hit'], bool)
        ratio = ms / np.maximum(pk, 1e-6)
        out[c] = dict(all=dict(peak=quant(pk), mass=quant(ms), mass_over_peak=quant(ratio)),
                      found_by_V0=dict(peak=quant(pk[h]), mass=quant(ms[h]), mass_over_peak=quant(ratio[h])),
                      missed_by_V0=dict(peak=quant(pk[~h]), mass=quant(ms[~h]), mass_over_peak=quant(ratio[~h])))
    pk, ms = np.asarray(fp['peak']), np.asarray(fp['mass'])
    out['V0_false_detections'] = dict(peak=quant(pk), mass=quant(ms), mass_over_peak=quant(ms / np.maximum(pk, 1e-6)),
                                      own_score=quant(fp['score']))
    return out, rows, fp


def paired_node(det_a, det_b, vid_a, vid_b):
    """Subject-paired bootstrap of node F1 / P / R (a - b) with protocol_report_20261106.paired."""
    from compare_dense_anchored_20261030 import KEYS
    from protocol_report_20261106 import paired
    ids = [d['sample_id'] for d in det_a]
    assert ids == [d['sample_id'] for d in det_b]

    def arr(det, vid):
        x = np.zeros((len(det), len(KEYS) + 2))
        x[:, :3] = [d['res'][vid]['c'] for d in det]
        return x
    out = paired(arr(det_a, vid_a), arr(det_b, vid_b), ids)['difference']['node_f1']
    return dict(point=out['point'], ci95=out['ci95'], p=out['p'])


def analyse(a):
    import torch  # noqa: F401  (node_types)
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    data = {s: pickle.loads((STORE / f'{tag(a)}-{s}.pkl').read_bytes()) for s in ('monitor', 'val')}
    vids = variant_ids()
    t0 = time.time()
    # thresholds on monitor_train
    CTX.update(recs=data['monitor']['records'], vids=vids)
    sweep = np.sum(run_pool(sweep_patch, len(CTX['recs']), workers), 0)  # (V, G, 3)
    f1 = 2 * sweep[..., 0] / np.maximum(2 * sweep[..., 0] + sweep[..., 1] + sweep[..., 2], 1)
    thr = {vid: GRID[int(np.argmax(f1[v]))] for v, vid in enumerate(vids)}
    print(f'SWEEP done {time.time() - t0:.0f}s', flush=True)
    s1_final = json.loads((RUNS / a.entry / STAGE1 / 'final.json').read_text())
    details = {}
    for split in ('monitor', 'val'):
        CTX.update(recs=data[split]['records'], vids=vids, thr=thr, ap=True)
        details[split] = run_pool(detail_patch, len(CTX['recs']), workers)
        print(f'DETAIL {split} done {time.time() - t0:.0f}s', flush=True)
    # parity: V0 at the stage-1 run's own node threshold
    CTX.update(recs=data['monitor']['records'], vids=['raw3@off'], thr={'raw3@off': s1_final['thresholds']['node']}, ap=False)
    par_m = run_pool(detail_patch, len(CTX['recs']), workers)
    CTX.update(recs=data['val']['records'])
    par_v = run_pool(detail_patch, len(CTX['recs']), workers)
    parity = dict(node_threshold=s1_final['thresholds']['node'])
    for name, det, ref in (('train_monitor', par_m, s1_final['train_monitor']), ('validation', par_v, s1_final['validation'])):
        mine = prf(sum(d['res']['raw3@off']['c'] for d in det))
        parity[name] = dict(this_script={k: mine[k] for k in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')},
                            final_json={k: ref[f'node_{k}'] for k in ('precision', 'recall', 'f1')})
    variants = {}
    for v, vid in enumerate(vids):
        variants[vid] = dict(threshold=thr[vid], monitor_f1_curve=[round(float(x), 4) for x in f1[v]],
                             train_monitor=summarise(details['monitor'], vid), validation=summarise(details['val'], vid))
    # position mode per mass configuration (closer to GT on monitor_train), best mass configuration by monitor F1
    chosen_pm = {}
    for prefix in ('', 'flip:'):
        for f, w in MASS:
            base = f'{prefix}mass_{f}_w{w}'
            dd = {pm: variants[f'{base}@{pm}']['train_monitor']['matched_distance_mm_mean'] for pm in ('argmax', 'centroid')}
            chosen_pm[base] = min(dd, key=dd.get)
    mass_plain = [f'mass_{f}_w{w}@{chosen_pm[f"mass_{f}_w{w}"]}' for f, w in MASS]
    best_mass = max(mass_plain, key=lambda v: variants[v]['train_monitor']['f1'])
    v0 = 'raw3@off'
    table = ([v0, 'raw5@off', 'raw7@off'] + mass_plain + ['flip:raw3@off', 'flip:' + best_mass, 'raw3@off+cl', best_mass + '+cl'])
    best_any = max(vids, key=lambda v: variants[v]['train_monitor']['f1'])
    if best_any not in table:
        table.append(best_any)
    for vid in vids:
        variants[vid]['validation_node_f1_minus_V0'] = paired_node(details['val'], details['val'], vid, v0) if vid != v0 else None
    beat = sorted([v for v in vids if variants[v]['train_monitor']['f1'] > variants[v0]['train_monitor']['f1'] + 0.01],
                  key=lambda v: -variants[v]['train_monitor']['f1'])
    amp = {}
    for split in ('monitor', 'val'):
        amp[split] = amplitude(data[split]['records'], details[split])[0]
    vox = np.array([r['voxel_mm'] for s in ('monitor', 'val') for r in data[s]['records']])
    uniq, cnt = np.unique(np.round(vox, 4), axis=0, return_counts=True)
    result = dict(
        entry=a.entry, stage1_run=STAGE1, smoke=a.smoke, floor=FLOOR, tolerance_mm=TOL, threshold_grid=[GRID[0], GRID[-1], 0.01],
        patches=dict(monitor=len(data['monitor']['records']), val=len(data['val']['records'])),
        voxel_mm=dict(values=[dict(voxel_mm=u.tolist(), patches=int(c), voxels_per_2mm=(2.0 / u).round(2).tolist()) for u, c in zip(uniq, cnt)]),
        mass_normalisation={f: norm_of_cpu(f) for f in FILTERS},
        parity_V0_at_stage1_threshold=parity, chosen_position_mode=chosen_pm, best_mass_variant=best_mass,
        best_variant_monitor_f1=best_any, variants_beating_V0_by_0p01_on_monitor=beat, table_rows=table,
        variants=variants, amplitude=amp, figure_ids=data['val']['figure_ids'],
        notes=['thresholds, position modes and the best variant are chosen on monitor_train only',
               'node_ap_2mm = dense_ap_20261030.node_ap (greedy, all GT nodes incl. inserted anchors, as compare_arms); '
               'node_ap_2mm_loop_aware = inserted anchors are don\'t-care',
               'mass scores are divided by the filter response to an ideal sigma-1.5 blob (mass_normalisation)'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'node_decode_{tag(a)}.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    print_table(result)
    print(f'ANALYSED {time.time() - t0:.0f}s', flush=True)
    return result


def norm_of_cpu(f):
    return norm_of(f, 'cpu')


def print_table(r):
    print(f"\n== {r['entry']} voxel sizes {r['voxel_mm']['values']}")
    print('parity', json.dumps(r['parity_V0_at_stage1_threshold']))
    f = lambda x: '  -  ' if x is None else f'{x:.3f}'  # noqa: E731
    for vid in r['table_rows']:
        v = r['variants'][vid]
        m, va = v['train_monitor'], v['validation']
        d = v['validation_node_f1_minus_V0']
        print(f"{vid:28s} t={v['threshold']:.2f} AP m {f(m['node_ap_2mm']['node_ap'])} v {f(va['node_ap_2mm']['node_ap'])} | "
              f"mon P {f(m['precision'])} R {f(m['recall'])} F1 {f(m['f1'])} | val P {f(va['precision'])} R {f(va['recall'])} "
              f"F1 {f(va['f1'])} | val rec b/j/t {f(va['recall_by_class']['boundary'])}/{f(va['recall_by_class']['junction'])}/"
              f"{f(va['recall_by_class']['terminal'])} | valFP dup/line/else {va['false_detections']['duplicate_within_2mm_of_matched_gt']}/"
              f"{va['false_detections']['within_1p5mm_of_gt_centreline']}/{va['false_detections']['elsewhere']} | dF1 "
              + ('ref' if d is None else f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"))
    print('beat V0 by >0.01 on monitor:', r['variants_beating_V0_by_0p01_on_monitor'][:10])


# ------------------------------------------------------------------------------------------------------ figures (CPU)
COLORS = dict(boundary='#2a78d6', junction='#eb6834', terminal='#1baf7a', other='#6250d6')
MARKERS = dict(boundary='s', junction='^', terminal='o', other='D')


def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = json.loads((OUT / f'node_decode_{tag(a)}.json').read_text())
    t0 = res['variants']['raw3@off']['threshold']
    data = {s: pickle.loads((STORE / f'{tag(a)}-{s}.pkl').read_bytes()) for s in ('monitor', 'val')}
    VIZ.mkdir(parents=True, exist_ok=True)
    short = a.entry.replace('-simp2', '') + ('-smoke' if a.smoke else '')
    for rec in data['val']['records']:
        if 'mip' not in rec:
            continue
        gt = gt_of(rec['cache'])
        with np.load(rec['cache']) as z:
            gv = z['nodes'] * SIZE
            P, off = z['full_paths'] * SIZE, z['path_offsets']
            n_edges = len(z['edges'].reshape(-1, 2))
        pos, score = cands(rec, 'raw3@off')
        k = int((score >= t0).sum())
        p = pos[:k]
        mp = match((p / SIZE) @ gt['M'].T, gt['g'])
        found = np.zeros(len(gv), bool)
        found[list(mp.values())] = True
        fp = np.array([q for q in range(k) if q not in mp], int)
        real = ~gt['ins']
        fig, ax = plt.subplots(3, 3, figsize=(11.5, 12), dpi=72)
        axes_of = [(1, 2), (0, 2), (0, 1)]  # MIP along axis k shows (row, col) = these voxel axes
        for row, name in enumerate(('image', 'heat', 'centreline')):
            for k_ax in range(3):
                h = ax[row, k_ax]
                img = rec['mip'][name][k_ax].astype(np.float32)
                h.imshow(img, cmap='gray' if name == 'image' else 'magma', vmin=None if name == 'image' else 0, vmax=None if name == 'image' else 1,
                         origin='lower', interpolation='nearest')
                r_, c_ = axes_of[k_ax]
                if row == 0:
                    for e in range(len(off) - 1):
                        seg = P[off[e]:off[e + 1]]
                        h.plot(seg[:, c_], seg[:, r_], lw=0.6, color='#86b6ef', alpha=0.9)
                for cname in CLASSES:
                    s = (gt['cls'] == cname) & real
                    for flag in (True, False):
                        q = s & (found == flag)
                        if q.any():
                            h.scatter(gv[q, c_], gv[q, r_], s=36 if row == 0 else 22, marker=MARKERS[cname],
                                      facecolors=COLORS[cname] if flag else 'none', edgecolors=COLORS[cname], linewidths=1.2)
                if gt['ins'].any():
                    q = gt['ins']
                    h.scatter(gv[q, c_], gv[q, r_], s=14, marker='.', color='#c3c2b7')
                if len(fp):
                    h.scatter(p[fp, c_], p[fp, r_], s=40, marker='x', color='#e34948', linewidths=1.4)
                h.set_xlim(-0.5, SIZE - 0.5)
                h.set_ylim(-0.5, SIZE - 0.5)
                h.set_xticks([])
                h.set_yticks([])
                h.set_title(f'{name} MIP along voxel axis {k_ax}', fontsize=9)
        n_real = int(real.sum())
        nf = int((found & real).sum())
        miss = {c: int(((gt['cls'] == c) & real & ~found).sum()) for c in CLASSES}
        fig.suptitle(f"{a.entry}  {rec['sample_id']}  V0 (3^3 peaks, heat >= {t0:.2f}): GT nodes {n_real} (found {nf}), "
                     f"GT edges {n_edges}, detections {k}, unmatched detections {len(fp)}\n"
                     f"missed boundary/junction/terminal/other {miss['boundary']}/{miss['junction']}/{miss['terminal']}/{miss['other']}\n"
                     f"marker: square boundary, triangle junction, circle terminal, diamond other; filled = found, hollow = missed; "
                     f"red x = unmatched V0 detection; grey dot = inserted anchor", fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(VIZ / f"{short}_{rec['sample_id']}.png")
        plt.close(fig)
    # amplitude histograms
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), dpi=80)
    for row, split in enumerate(('monitor', 'val')):
        CTX.update(recs=data[split]['records'], vids=['raw3@off'], thr={'raw3@off': t0}, ap=False)
        det = [detail_patch(i) for i in range(len(CTX['recs']))]
        _, rows, fp = amplitude(data[split]['records'], det)
        for col, key in enumerate(('peak', 'mass')):
            h = ax[row, col]
            bins = np.linspace(0, 1.0 if key == 'peak' else max(1.2, float(np.percentile(np.concatenate([rows[c]['mass'] for c in CLASSES] + [fp['mass']]), 99.5))), 41)
            for c in CLASSES:
                v = np.clip(rows[c][key], 0, bins[-1])
                if len(v):
                    h.hist(v, bins=bins, histtype='step', density=True, lw=1.8, color=COLORS[c], label=f'GT {c} (n={len(v)})')
            v = np.clip(fp[key], 0, bins[-1])
            if len(v):
                h.hist(v, bins=bins, histtype='step', density=True, lw=1.8, ls='--', color='#e34948', label=f'V0 false detections (n={len(v)})')
            if key == 'peak':
                h.axvline(t0, color='#52514e', lw=1, ls=':')
            h.set_xlabel('max raw heat within 2 mm' if key == 'peak' else 'max box-5 mass within 2 mm (1 = ideal target blob)')
            h.set_ylabel('density')
            h.set_title(f"{'monitor_train' if split == 'monitor' else 'validation'}: {key}", fontsize=10)
            h.legend(fontsize=7, frameon=False)
    fig.suptitle(f'{a.entry}: heat amplitude at GT nodes by class, and at V0 false detections (dotted: V0 threshold {t0:.2f})', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(VIZ / f'{short}_amplitude_histograms.png')
    plt.close(fig)
    print('FIGURES', sorted(x.name for x in VIZ.glob(f'{short}_*.png')), flush=True)


# ------------------------------------------------------------------------------------------------------ stage 2 (GPU)
FLIP = {}


def make_decoder(vid):
    """A replacement of D.peaks(out, threshold, nms=None) for variant vid; flip variants read the flip-averaged maps
    that the wrapped forward_image stored for the current batch."""
    import torch  # noqa: F401
    name, pm, cl = parse(vid)
    flip = name.startswith('flip:')
    base = name.replace('flip:', '')

    def peaks(out, threshold, nms=None):
        heat, off, prob = FLIP['maps'] if flip else maps(out[:, :5])
        res = []
        for d in decode(heat, off, prob, base, max(threshold, 1e-9)):
            pos = d['pos_c'] if pm == 'centroid' else d['pos']
            keep = d['cl'] >= CL_MIN if cl else torch.ones_like(d['score'], dtype=torch.bool)
            res.append((pos[keep].contiguous(), d['score'][keep].contiguous()))
        return res
    return peaks


def stage2_counts(i):
    from compare_dense_anchored_20261030 import KEYS, dense_patch
    from decode_study_20261105 import gt
    from ixi_query_diagnostics import betti
    r, nt, et = CTX['rows'][i], CTX['nt'], CTX['et']
    c = dense_patch(r, nt, et)
    keep = np.flatnonzero(r['node_scores'] >= nt)
    remap = {int(k): j for j, k in enumerate(keep)}
    edges = np.zeros((0, 2), int)
    if r['edge_scores'] is not None and len(r['pairs']):
        ok = (r['edge_scores'] >= et) & np.isin(r['pairs'], keep).all(1)
        edges = np.array([[remap[int(x)], remap[int(y)]] for x, y in r['pairs'][ok]], int).reshape(-1, 2)
    p0, p1 = betti(len(keep), edges)
    _, _, _, (g0, g1), _ = gt(r['cache'])
    return [c[k] for k in KEYS] + [abs(p0 - g0), abs(p1 - g1)]


def stage2(a):
    import torch
    import dense_graph_20261030 as D
    import dense_graph_traced_20261030 as T
    from node_heatmap_probe_20261030 import make_dataset
    from protocol_report_20261106 import paired, pooled
    device = 'cuda'
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    run_dir = RUNS / a.entry / STAGE2
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    assert not T.MODE['passthrough'] and not T.MODE['roles'] and not T.MODE['two_hop']
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], two_hop=T.MODE['two_hop'], passthrough=T.MODE['passthrough'],
                               backbone=final.get('backbone', 'unet'), feature_channels=final.get('feature_channels', 32)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    model.eval()
    sp = splits_of(a)
    mon_ds, val_ds = make_dataset(a.entry, sp['monitor'], augment=False), make_dataset(a.entry, sp['val'], augment=False)
    study = json.loads((OUT / f'node_decode_{tag(a)}.json').read_text())
    vids = a.variants.split(',') if a.variants else study['variants_beating_V0_by_0p01_on_monitor'][:2]
    original_peaks, original_forward = D.peaks, model.forward_image

    def flip_forward(images):
        out = original_forward(images)
        FLIP['maps'] = flip_maps_raw(original_forward, images, out[:, :5])
        return out

    def run(vid, nt):
        """T.predict on monitor and validation with D.peaks replaced by the decoder of vid (None = unchanged)."""
        try:
            if vid is not None:
                D.peaks = make_decoder(vid)
                if vid.startswith('flip:'):
                    model.forward_image = flip_forward
            mon = T.predict(model, mon_ds, device, nt, workers)
            et = D.select_edge_threshold(mon, nt)
            val = T.predict(model, val_ds, device, nt, workers)
        finally:
            D.peaks = original_peaks
            model.forward_image = original_forward
        return mon, val, et

    def score(name, vid, nt, et_fixed=None):
        t1 = time.time()
        mon, val, et = run(vid, nt)
        reselected = et
        et = et if et_fixed is None else et_fixed
        res = dict(variant=vid, node_threshold=nt, edge_threshold=et, edge_threshold_reselected_on_monitor=reselected, train_monitor=D.evaluate(mon, nt, et),
                   validation=D.evaluate(val, nt, et))
        CTX.update(rows=val, nt=nt, et=et)
        cnt = np.asarray(run_pool(stage2_counts, len(val), workers), float)
        res['validation_pooled_counts'] = pooled(cnt)
        print(f'STAGE2 {name} {vid} node {nt} edge {et}: val node F1 {res["validation"]["node_f1"]:.4f} edge F1 '
              f'{res["validation"]["edge_f1"]:.4f} branch F1 {res["validation"]["common_branch_f1"]:.4f} ({time.time() - t1:.0f}s)', flush=True)
        return res, cnt, [r['sample_id'] for r in val]

    out = dict(entry=a.entry, stage2_run=STAGE2, mode=dict(T.MODE), variants=vids, arms={}, minus_pipeline={},
               change='D.peaks (inside the unchanged T.predict) replaced by the variant decoder on the stage-2 model\'s own '
                      'heatmap; node threshold re-selected on monitor_train by node F1 (grid 0.02..0.95), edge threshold '
                      're-selected on monitor_train by edge F1 (D.EDGE_THRESHOLDS). Nothing else changed.')
    ref, ref_cnt, ids = score('pipeline', None, final['thresholds']['node'], final['thresholds']['edge'])
    out['arms']['pipeline'] = ref
    out['parity_with_final_json'] = {k: dict(this_script=ref['validation'][k], final_json=final['validation'][k])
                                     for k in ('node_f1', 'edge_f1', 'common_branch_f1')}
    out['parity_with_final_json']['edge_threshold'] = dict(this_script_reselected=ref['edge_threshold_reselected_on_monitor'], final_json=final['thresholds']['edge'])
    # node thresholds of the variants (and of V0 on the fine grid) on the stage-2 heatmap, monitor_train only
    need = ['raw3@off'] + [v for v in vids if v != 'raw3@off']
    bases = sorted({parse(v)[0].replace('flip:', '') for v in need})
    recs = collect_model(model, mon_ds, device, workers - 2, bases=bases, flips=any(v.startswith('flip:') for v in need), amplitude=False)
    CTX.update(recs=recs, vids=need)
    sw = np.sum(run_pool(sweep_patch, len(recs), workers), 0)
    f1 = 2 * sw[..., 0] / np.maximum(2 * sw[..., 0] + sw[..., 1] + sw[..., 2], 1)
    nts = {v: GRID[int(np.argmax(f1[i]))] for i, v in enumerate(need)}
    out['stage2_heat_monitor_node_f1'] = {v: dict(threshold=nts[v], f1=float(f1[i].max())) for i, v in enumerate(need)}
    arms = [('V0_fine_grid', 'raw3@off')] if abs(nts['raw3@off'] - final['thresholds']['node']) > 1e-9 else []
    arms += [(v, v) for v in vids if v != 'raw3@off']
    for name, vid in arms:
        res, cnt, ids_v = score(name, vid, nts[vid])
        assert ids_v == ids
        out['arms'][name] = res
        d = paired(cnt, ref_cnt, ids)['difference']
        out['minus_pipeline'][name] = {k: d[k] for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1', 'mae_beta0', 'mae_beta1')}
        print('PAIRED', name, json.dumps({k: [round(v['point'], 4), *[round(x, 4) for x in v['ci95']]] for k, v in out['minus_pipeline'][name].items()}), flush=True)
    (OUT / f'node_decode_stage2_{tag(a)}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print('DONE stage2', flush=True)


def flip_maps_raw(forward, images, out0):
    """flip_maps with an explicit forward function (stage-2 wrapper)."""
    class _M:
        forward_image = staticmethod(forward)
    return flip_maps(_M, images, out0)


# ====================================================================================================== PART 2
# (13 Nov 2026, second request) Fixed node classes (cache 'boundary' flag), two-threshold decoding (face band vs
# interior), test-time averaging over flips / rotations / both, stage 2 fed with the new stage-1 node sets.
# Results: experiments/final_try_20261113/node_decode2_<entry>.json and node_decode2_stage2_<entry>.json;
# figures viz_tmp/v2_*.png. Part-1 outputs are not touched.
SETS = ('id', 'flip8', 'rot24', 'all48')
WINDOWS2 = (3, 5)
MAPS2 = [f'{s}:raw{w}' for s in SETS for w in WINDOWS2]
G2 = [round(float(t), 2) for t in np.arange(0.10, 0.6001, 0.01)]
BANDS = (1, 2, 3, 4)


def transforms():
    """The 48 axis-aligned transforms (axis permutation, then flips); the first is the identity."""
    from itertools import permutations, product
    return [(perm, fl) for perm in permutations(range(3)) for fl in product((False, True), repeat=3)]


def perm_sign(perm):
    return -1 if sum(perm[i] > perm[j] for i in range(3) for j in range(i + 1, 3)) % 2 else 1


def sets_of(t):
    perm, fl = t
    s = {'all48'}
    if perm == (0, 1, 2):
        s.add('flip8')
        if not any(fl):
            s.add('id')
    if perm_sign(perm) * (-1) ** sum(fl) == 1:
        s.add('rot24')
    return s


def t_forward(x, t):
    """Transformed volume (B,C,X,Y,Z): its axis k is original axis perm[k], reversed if fl[k]."""
    perm, fl = t
    y = x.permute(0, 1, *[2 + q for q in perm])
    dims = [2 + k for k in range(3) if fl[k]]
    return (y.flip(dims) if dims else y).contiguous()


def t_back(y, t):
    """Scalar channels of a transformed-frame volume back to the original frame."""
    perm, fl = t
    dims = [2 + k for k in range(3) if fl[k]]
    y = y.flip(dims) if dims else y
    return y.permute(0, 1, *[2 + perm.index(j) for j in range(3)])


def t_back_vec(v, t):
    """Vector field (B,3,...) whose component k lives on transformed axis k, back to the original frame."""
    import torch
    perm, fl = t
    vb = t_back(v, t)
    out = torch.empty_like(vb)
    for k in range(3):
        out[:, perm[k]] = -vb[:, k] if fl[k] else vb[:, k]
    return out


def check_transforms(device):
    """Synthetic checks before use: (1) a delta voxel lands where the analytic map says and comes back to the same
    voxel; (2) an offset vector written at the transformed voxel comes back as the original offset; (3) the 24 'rot24'
    elements are exactly the transforms produced by the training augmentation (data.augmentations.rotate_volume over
    all quarter-turn triples)."""
    import torch
    from data.augmentations import rotate_volume
    g = torch.Generator().manual_seed(0)
    TT = transforms()
    for t in TT:
        perm, fl = t
        for _ in range(4):
            c = torch.randint(0, SIZE, (3,), generator=g)
            delta = torch.rand(3, generator=g) - 0.5
            V = torch.zeros(1, 1, SIZE, SIZE, SIZE, device=device)
            V[0, 0, c[0], c[1], c[2]] = 1
            Vp = t_forward(V, t)
            cp = torch.stack(torch.unravel_index(Vp.flatten().argmax().cpu(), (SIZE,) * 3))
            expect = torch.tensor([SIZE - 1 - int(c[perm[k]]) if fl[k] else int(c[perm[k]]) for k in range(3)])
            assert torch.equal(cp, expect), (t, c, cp, expect)
            assert torch.equal(t_back(Vp, t), V), t
            O = torch.zeros(1, 3, SIZE, SIZE, SIZE, device=device)
            O[0, :, cp[0], cp[1], cp[2]] = torch.tensor([(-1 if fl[k] else 1) * float(delta[perm[k]]) for k in range(3)], device=device)
            Ob = t_back_vec(O, t)
            assert torch.allclose(Ob[0, :, c[0], c[1], c[2]].cpu(), delta), t
            assert int((Ob.abs().sum(1) > 0).sum()) == 1, t
    x = torch.rand(1, 1, 9, 9, 9, generator=g)
    found = set()
    for turns in [(i, j, k) for i in range(4) for j in range(4) for k in range(4)]:
        r = rotate_volume(x[0], turns)[None]
        hits = [n for n, t in enumerate(TT) if torch.equal(t_forward(x, t), r)]
        assert len(hits) == 1, turns
        found.add(hits[0])
    rot = {n for n, t in enumerate(TT) if 'rot24' in sets_of(t)}
    assert found == rot, (sorted(found), sorted(rot))
    return dict(transforms=len(TT), delta_roundtrip='ok', offset_roundtrip='ok',
                rot24_equals_training_quarter_turn_group=True,
                set_sizes={s: sum(s in sets_of(t) for t in TT) for s in SETS})


def averaged_maps(model, images, TT, member):
    """Per set: sigmoid heat and offsets averaged over the set's views (mapped back to the original frame)."""
    import torch
    sums = {s: [0, 0, 0] for s in SETS}
    seconds = 0.0
    for t, mem in zip(TT, member):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model.forward_image(t_forward(images, t))[:, :5].float()
        h, f = t_back(o[:, :1].sigmoid(), t), t_back_vec(o[:, 1:4], t)
        torch.cuda.synchronize()
        seconds += time.time() - t0
        for s in mem:
            sums[s][0] = sums[s][0] + h
            sums[s][1] = sums[s][1] + f
            sums[s][2] += 1
    return {s: (v[0] / v[2], v[1] / v[2]) for s, v in sums.items()}, seconds


def collect2(a):
    import torch
    from node_heatmap_probe_20261030 import make_dataset
    import dense_graph_20261030 as D
    device = 'cuda'
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8')) - 2
    check = check_transforms(device)
    print('TRANSFORM CHECK', json.dumps(check), flush=True)
    model, _ = load_stage1(a.entry, device)
    TT = transforms()
    member = [sets_of(t) for t in TT]
    STORE.mkdir(parents=True, exist_ok=True)
    timing = {}
    for split, ids in splits_of(a).items():
        t_split = time.time()
        ds = make_dataset(a.entry, ids, augment=False)
        fig = figure_ids(ds.records) if split == 'val' else []
        loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=workers, collate_fn=D.collate)
        recs, offset, fwd_seconds, agree = [], 0, 0.0, None
        with torch.no_grad():
            for images, _, _, _ in loader:
                images = images.to(device)
                avg, sec = averaged_maps(model, images, TT, member)
                fwd_seconds += sec
                if agree is None:  # sanity: views agree with the identity view after mapping back
                    agree = {s: float(torch.corrcoef(torch.stack([avg['id'][0].flatten(), avg[s][0].flatten()]))[0, 1]) for s in SETS}
                zeros = torch.zeros_like(avg['id'][0][:, 0])
                dec = {f'{s}:raw{w}': decode(avg[s][0], avg[s][1], zeros, f'raw{w}', FLOOR) for s in SETS for w in WINDOWS2}
                for b in range(len(images)):
                    record = ds.records[offset + b]
                    rec = dict(sample_id=record.sample_id, cache=cache_of(record),
                               cands={k: {f: v[b][f].cpu().numpy().astype(np.float32) for f in ('pos', 'score')} for k, v in dec.items()})
                    if record.sample_id in fig:
                        rec['heat_mip'] = {s: np.stack([avg[s][0][b, 0].max(dim=k)[0].cpu().numpy() for k in range(3)]).astype(np.float16) for s in SETS}
                    recs.append(rec)
                offset += len(images)
        (STORE / f'{tag(a)}-p2-{split}.pkl').write_bytes(pickle.dumps(dict(figure_ids=fig, records=recs), protocol=4))
        timing[split] = dict(patches=len(recs), seconds_forward_all_48_views=fwd_seconds,
                             seconds_per_view_per_patch=fwd_seconds / 48 / max(1, len(recs)),
                             seconds_total=time.time() - t_split, heat_correlation_with_identity_first_batch=agree)
        print(f'COLLECTED2 {a.entry} {split} {json.dumps(timing[split])}', flush=True)
    (STORE / f'{tag(a)}-p2-meta.json').write_text(json.dumps(dict(transform_check=check, timing=timing), indent=1))


GT2 = {}


def gt_of2(cache):
    """As gt_of, with the classes of the request of 13 Nov (2): the cache's explicit 'boundary' flag (patch exits, as
    node_miss_diagnosis_20261110), then degree >= 3 junction, degree 1 terminal, other (degree 2 / isolated)."""
    if cache not in GT2:
        base = dict(gt_of(cache))
        with np.load(cache) as z:
            bnd = z['boundary'].astype(bool)
            deg = np.bincount(z['edges'].reshape(-1).astype(int), minlength=len(z['nodes']))
        base['cls_geometric'] = base['cls']
        base['cls'] = np.where(bnd, 'boundary', np.where(deg >= 3, 'junction', np.where(deg == 1, 'terminal', 'other')))
        GT2[cache] = base
    return GT2[cache]


def face_dist(pos):
    return np.minimum(pos + 0.5, SIZE - 0.5 - pos).min(1) if len(pos) else np.zeros(0)


def select2(rec, m, rule):
    """Candidates of map m kept by rule ('single', t) or ('two', b, t_face, t_interior), score order kept."""
    c = rec['cands'][m]
    pos, score = c['pos'], c['score']
    if rule[0] == 'single':
        k = score >= rule[1]
    else:
        _, b, tf, ti = rule
        k = np.where(face_dist(pos) < b, score >= tf, score >= ti)
    return pos[k], score[k]


def sweep2_patch(i):
    rec = CTX['recs'][i]
    gt = gt_of2(rec['cache'])
    one = np.zeros((len(MAPS2), len(GRID), 3), np.int32)
    two = np.zeros((len(MAPS2), len(BANDS), len(G2), len(G2), 3), np.int32)
    for mi, m in enumerate(MAPS2):
        pos, score = rec['cands'][m]['pos'], rec['cands'][m]['score']
        p = (pos / SIZE) @ gt['M'].T
        memo = {}
        for t, k in enumerate(np.searchsorted(-score, -np.asarray(GRID), side='right')):
            if k not in memo:
                memo[k] = counts(match(p[:k], gt['g']), k, gt['ins'])
            one[mi, t] = memo[k]
        fd = face_dist(pos)
        for bi, b in enumerate(BANDS):
            jf, ji = np.flatnonzero(fd < b), np.flatnonzero(fd >= b)
            kf = np.searchsorted(-score[jf], -np.asarray(G2), side='right')
            ki = np.searchsorted(-score[ji], -np.asarray(G2), side='right')
            uf, inv_f = np.unique(kf, return_inverse=True)
            ui, inv_i = np.unique(ki, return_inverse=True)
            tab = np.zeros((len(uf), len(ui), 3), np.int32)
            for x, nf in enumerate(uf):
                for y, ni in enumerate(ui):
                    q = np.concatenate([jf[:nf], ji[:ni]])
                    tab[x, y] = counts(match(p[q], gt['g']), len(q), gt['ins'])
            two[mi, bi] = tab[inv_f][:, inv_i]
    return one, two


def detail2_patch(i):
    from dense_ap_20261030 import node_ap_state
    rec = CTX['recs'][i]
    gt = gt_of2(rec['cache'])
    real = ~gt['ins']
    cls_tot = np.array([int(((gt['cls'] == c) & real).sum()) for c in CLASSES])
    cls_tot_geo = np.array([int(((gt['cls_geometric'] == c) & real).sum()) for c in CLASSES])
    res = {}
    for vid, (m, rule) in CTX['rules'].items():
        pos, score = select2(rec, m, rule)
        p = (pos / SIZE) @ gt['M'].T
        mp = match(p, gt['g'])
        tp, fp, fn = counts(mp, len(p), gt['ins'])
        hit = np.zeros(len(gt['g']), bool)
        hit[list(mp.values())] = True
        matched_gt = hit.copy()
        hit &= real
        place = np.zeros(4, np.int32)
        band = np.zeros(2, np.int32)  # false detections in the face band (b of the map's two-threshold rule) / interior
        fd = face_dist(pos)
        for q in range(len(p)):
            if q in mp:
                continue
            band[0 if fd[q] < CTX['band'][m] else 1] += 1
            d = np.linalg.norm(gt['g'] - p[q], axis=1) if len(gt['g']) else np.zeros(0)
            if (d <= TOL).any():
                place[0 if matched_gt[d <= TOL].any() else 3] += 1
            elif gt['tree'] is not None and gt['tree'].query(p[q])[0] <= 1.5:
                place[1] += 1
            else:
                place[2] += 1
        res[vid] = dict(c=np.array([tp, fp, fn], np.int32), cls_hit=np.array([int((hit & (gt['cls'] == c)).sum()) for c in CLASSES]),
                        place=place, band=band, dist=(0.0, 0), ap=None, ap_la=None, hit=hit)
        if rule[0] == 'single' and CTX['ap']:
            c = rec['cands'][m]
            p_all = (c['pos'] / SIZE) @ gt['M'].T
            res[vid]['ap'] = node_ap_state(c['pos'] / SIZE, c['score'], gt['nodes'], gt['M'])
            res[vid]['ap_la'] = ap_state_loop_aware(p_all, c['score'], gt['g'], gt['ins'])
    return dict(sample_id=rec['sample_id'], cls_tot=cls_tot, cls_tot_geometric=cls_tot_geo, res=res)


def summarise2(details, vid):
    out = summarise(details, vid)
    band = sum(d['res'][vid]['band'] for d in details)
    out['false_detections'].update(in_face_band=int(band[0]), interior=int(band[1]))
    out.pop('matched_distance_mm_mean', None)
    return out


def analyse2(a):
    import torch  # noqa: F401
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    data = {s: pickle.loads((STORE / f'{tag(a)}-p2-{s}.pkl').read_bytes()) for s in ('monitor', 'val')}
    meta = json.loads((STORE / f'{tag(a)}-p2-meta.json').read_text())
    t0 = time.time()
    CTX.update(recs=data['monitor']['records'])
    res = run_pool(sweep2_patch, len(CTX['recs']), workers)
    one = np.sum([r[0] for r in res], 0)
    two = np.sum([r[1] for r in res], 0)
    f1 = lambda c: 2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 2], 1)  # noqa: E731
    f_one, f_two = f1(one), f1(two)
    rules, band = {}, {}
    for mi, m in enumerate(MAPS2):
        rules[f'{m}|single'] = (m, ('single', GRID[int(np.argmax(f_one[mi]))]))
        bi, x, y = np.unravel_index(int(np.argmax(f_two[mi])), f_two[mi].shape)
        rules[f'{m}|two'] = (m, ('two', BANDS[bi], G2[x], G2[y]))
        band[m] = BANDS[bi]
    print(f'SWEEP2 done {time.time() - t0:.0f}s', flush=True)
    details = {}
    for split in ('monitor', 'val'):
        CTX.update(recs=data[split]['records'], rules=rules, band=band, ap=True)
        details[split] = run_pool(detail2_patch, len(CTX['recs']), workers)
        print(f'DETAIL2 {split} done {time.time() - t0:.0f}s', flush=True)
    v0 = 'id:raw3|single'
    variants = {}
    for vid, (m, rule) in rules.items():
        mi = MAPS2.index(m)
        e = dict(map=m, rule=list(rule), train_monitor=summarise2(details['monitor'], vid), validation=summarise2(details['val'], vid))
        if rule[0] == 'two':
            bi = BANDS.index(rule[1])
            e['monitor_f1_best_per_band'] = {str(b): float(f_two[mi, k].max()) for k, b in enumerate(BANDS)}
            e['monitor_f1_single_threshold_best'] = float(f_one[mi].max())
            e['monitor_f1_two_threshold_best'] = float(f_two[mi, bi].max())
        e['validation_node_f1_minus_V0'] = paired_node(details['val'], details['val'], vid, v0) if vid != v0 else None
        e['validation_node_f1_minus_same_map_single'] = (paired_node(details['val'], details['val'], vid, f'{m}|single')
                                                         if rule[0] == 'two' else None)
        variants[vid] = e
    mon_f1 = lambda v: variants[v]['train_monitor']['f1']  # noqa: E731
    pick_a = max([f'flip8:raw{w}|single' for w in WINDOWS2], key=mon_f1)
    pick_b = max([f'flip8:raw{w}|two' for w in WINDOWS2], key=mon_f1)
    pick_c = max([f'{s}:raw{w}|two' for s in SETS if s != 'id' for w in WINDOWS2], key=mon_f1)
    cls = {split: dict(fixed=dict(zip(CLASSES, map(int, sum(d['cls_tot'] for d in det)))),
                       geometric_part1=dict(zip(CLASSES, map(int, sum(d['cls_tot_geometric'] for d in det)))))
           for split, det in details.items()}
    result = dict(entry=a.entry, stage1_run=STAGE1, smoke=a.smoke, floor=FLOOR, tolerance_mm=TOL,
                  single_threshold_grid=[GRID[0], GRID[-1], 0.01], two_threshold_grid=dict(bands_voxels=list(BANDS), thresholds=[G2[0], G2[-1], 0.01]),
                  face_band_rule='candidate in the face band iff min over axes of min(x + 0.5, 63.5 - x) < b (refined position, voxels)',
                  class_rule='boundary = cache boundary flag; else junction (degree >= 3), terminal (degree 1), other; inserted anchors excluded',
                  gt_nodes_by_class=cls, transform_check=meta['transform_check'], timing=meta['timing'],
                  patches=dict(monitor=len(data['monitor']['records']), val=len(data['val']['records'])),
                  V0=v0, item4_choices=dict(a_flip_single=pick_a, b_flip_two=pick_b, c_best_set_two=pick_c), variants=variants,
                  notes=['all choices (thresholds, band, windows, item-4 picks) on monitor_train only',
                         'V0 = identity heat, 3^3 peaks, single threshold chosen on the 0.02..0.95 grid (as part 1)',
                         'node AP is given for single-score maps only (the two-threshold rule has no single ranking)'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'node_decode2_{tag(a)}.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    print_table2(result)
    print(f'ANALYSED2 {time.time() - t0:.0f}s', flush=True)


def print_table2(r):
    f = lambda x: '  -  ' if x is None else f'{x:.3f}'  # noqa: E731
    print(f"\n== {r['entry']} classes {json.dumps(r['gt_nodes_by_class'])}")
    print('timing', json.dumps(r['timing']))
    for vid, v in r['variants'].items():
        m, va = v['train_monitor'], v['validation']
        d0, d1 = v['validation_node_f1_minus_V0'], v['validation_node_f1_minus_same_map_single']
        g = lambda d: 'ref' if d is None else f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"  # noqa: E731
        ap = f" AP v {f(va['node_ap_2mm']['node_ap'])}" if 'node_ap_2mm' in va else ''
        rc = va['recall_by_class']
        fd = va['false_detections']
        print(f"{vid:20s} {str(v['rule']):28s} mon P {f(m['precision'])} R {f(m['recall'])} F1 {f(m['f1'])} | val P {f(va['precision'])} "
              f"R {f(va['recall'])} F1 {f(va['f1'])}{ap} | val rec b/j/t/o {f(rc['boundary'])}/{f(rc['junction'])}/{f(rc['terminal'])}/{f(rc['other'])} "
              f"| valFP dup/line/else {fd['duplicate_within_2mm_of_matched_gt']}/{fd['within_1p5mm_of_gt_centreline']}/{fd['elsewhere']} "
              f"face/int {fd['in_face_band']}/{fd['interior']} | dV0 {g(d0)} | dSingle {g(d1) if v['rule'][0] == 'two' else '-'}")
    print('item4', r['item4_choices'])


# ------------------------------------------------------------------------------------------------------ stage 2, part 2
S2 = {}


def stage2_arms(study):
    rules = {vid: v['rule'] for vid, v in study['variants'].items()}
    ch = study['item4_choices']
    arms = {'stage1_V0': dict(variant=study['V0'], cl=None)}
    arms['a_flip_single'] = dict(variant=ch['a_flip_single'], cl=None)
    arms['b_flip_two'] = dict(variant=ch['b_flip_two'], cl=None)
    arms['b_flip_two_clavg'] = dict(variant=ch['b_flip_two'], cl='flip8')
    if ch['c_best_set_two'] != ch['b_flip_two']:
        arms['c_best_set_two'] = dict(variant=ch['c_best_set_two'], cl=None)
    for v in arms.values():
        v['map'], v['rule'] = v['variant'].split('|')[0], rules[v['variant']]
    return arms


def stage2v2(a):
    import torch
    import dense_graph_20261030 as D
    import dense_graph_traced_20261030 as T
    from node_heatmap_probe_20261030 import make_dataset
    from protocol_report_20261106 import pooled
    device = 'cuda'
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    check_transforms(device)
    run_dir = RUNS / a.entry / STAGE2
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(final)
    assert not T.MODE['passthrough'] and not T.MODE['roles'] and not T.MODE['two_hop'] and not T.MODE['gap_features']
    model = T.TracedDenseGraph(gap_features=T.MODE['gap_features'], two_hop=T.MODE['two_hop'], passthrough=T.MODE['passthrough'],
                               backbone=final.get('backbone', 'unet'), feature_channels=final.get('feature_channels', 32)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location=device)['model'])
    model.eval()
    s1, _ = load_stage1(a.entry, device)
    sp = splits_of(a)
    mon_ds, val_ds = make_dataset(a.entry, sp['monitor'], augment=False), make_dataset(a.entry, sp['val'], augment=False)
    study = json.loads((OUT / f'node_decode2_{tag(a)}.json').read_text())
    arms = stage2_arms(study)
    todo = ['pipeline'] + list(arms) if not a.arms else a.arms.split(',')
    skipped = [n for n in todo if n != 'pipeline' and n not in arms]  # e.g. arm c when it equals arm b
    todo = [n for n in todo if n not in skipped]
    print('STAGE2V2 arms', todo, 'skipped (not defined for this entry)', skipped, flush=True)
    original_peaks, original_forward = D.peaks, model.forward_image
    TT = transforms()

    def make_forward(arm):
        views = [t for t in TT if arm['map'].split(':')[0] in sets_of(t)]
        cl_views = [t for t in TT if arm['cl'] in sets_of(t)] if arm['cl'] else []

        def fwd(images):
            out = original_forward(images)
            heat = off = 0
            for t in views:  # stage-1 maps averaged over the arm's views (called inside T.predict's autocast)
                o = s1.forward_image(t_forward(images, t))[:, :5].float()
                heat = heat + t_back(o[:, :1].sigmoid(), t)
                off = off + t_back_vec(o[:, 1:4], t)
            S2['maps'] = (heat / len(views), off / len(views))
            if cl_views:  # stage-2 centreline probability averaged over the same views: only the tracing input changes
                prob = 0
                for t in cl_views:
                    prob = prob + t_back(original_forward(t_forward(images, t))[:, 4:5].float().sigmoid(), t)
                prob = (prob / len(cl_views)).clamp(1e-6, 1 - 1e-6)
                out = out.clone()
                out[:, 4] = torch.logit(prob[:, 0]).to(out.dtype)
            return out
        return fwd

    def make_peaks(arm):
        w, rule = int(arm['map'].split('raw')[1]), arm['rule']

        def peaks(out, threshold, nms=None):
            heat, off = S2['maps']
            res = []
            for d in decode(heat, off, torch.zeros_like(heat[:, 0]), f'raw{w}', FLOOR):
                pos, score = d['pos'], d['score']
                if rule[0] == 'single':
                    keep = score >= rule[1]
                else:
                    fd = torch.minimum(pos + 0.5, SIZE - 0.5 - pos).amin(1)
                    keep = torch.where(fd < rule[1], score >= rule[2], score >= rule[3])
                res.append((pos[keep].contiguous(), score[keep].contiguous()))
            return res
        return peaks

    for name in todo:
        t1 = time.time()
        if name == 'pipeline':
            nt, et_fixed, arm = final['thresholds']['node'], final['thresholds']['edge'], None
        else:
            nt, et_fixed, arm = 0.0, None, arms[name]
        try:
            if arm is not None:
                D.peaks = make_peaks(arm)
                model.forward_image = make_forward(arm)
            mon = T.predict(model, mon_ds, device, nt, workers)
            et_sel = D.select_edge_threshold(mon, nt)
            val = T.predict(model, val_ds, device, nt, workers)
        finally:
            D.peaks = original_peaks
            model.forward_image = original_forward
        et = et_sel if et_fixed is None else et_fixed
        res = dict(arm=arm, node_threshold_in_evaluate=nt, edge_threshold=et, edge_threshold_reselected_on_monitor=et_sel,
                   train_monitor=D.evaluate(mon, nt, et), validation=D.evaluate(val, nt, et))
        CTX.update(rows=val, nt=nt, et=et)
        cnt = np.asarray(run_pool(stage2_counts, len(val), workers), float)
        res['validation_pooled_counts'] = pooled(cnt)
        if name == 'pipeline':
            res['parity_with_final_json'] = {k: dict(this_script=res['validation'][k], final_json=final['validation'][k])
                                             for k in ('node_f1', 'edge_f1', 'common_branch_f1', 'common_curve_f1')}
            res['parity_with_final_json']['train_monitor_edge_f1'] = dict(this_script=res['train_monitor']['edge_f1'], final_json=final['train_monitor']['edge_f1'])
        else:
            nd = study['variants'][arm['variant']]['validation']
            res['node_check_vs_analyse2'] = dict(stage2_validation_node_f1=res['validation']['node_f1'], analyse2_validation_node_f1=nd['f1'])
        (STORE / f'{tag(a)}-s2v2-{name}.pkl').write_bytes(pickle.dumps(dict(result=res, counts=cnt, ids=[r['sample_id'] for r in val])))
        v = res['validation']
        print(f"STAGE2V2 {a.entry} {name}: edge thr {et} val node F1 {v['node_f1']:.4f} edge F1 {v['edge_f1']:.4f} "
              f"branch F1 {v['common_branch_f1']:.4f} curve F1 {v['common_curve_f1']:.4f} ({time.time() - t1:.0f}s)", flush=True)
    if all((STORE / f'{tag(a)}-s2v2-{n}.pkl').exists() for n in ['pipeline'] + list(arms)):
        merge2(a)


def merge2(a):
    from protocol_report_20261106 import paired
    study = json.loads((OUT / f'node_decode2_{tag(a)}.json').read_text())
    arms = stage2_arms(study)
    load = lambda n: pickle.loads((STORE / f'{tag(a)}-s2v2-{n}.pkl').read_bytes())  # noqa: E731
    ref = load('pipeline')
    out = dict(entry=a.entry, stage1_run=STAGE1, stage2_run=STAGE2,
               change='Stage-1 nodes (the decoding of items 2-3, at the thresholds chosen there on the stage-1 model) replace '
                      'D.peaks inside the unchanged T.predict of the stage-2 run; stage 2 keeps its own features, heat and '
                      'centreline map (the clavg arm: its centreline probability averaged over the same views); edge threshold '
                      're-selected on monitor_train by edge F1 (D.select_edge_threshold); evaluation D.evaluate, per-patch counts '
                      'compare_dense_anchored.dense_patch + Betti, paired = protocol_report_20261106.paired (2000 subject resamples).',
               arms={'pipeline': ref['result']}, minus_pipeline={})
    for name in arms:
        r = load(name)
        assert r['ids'] == ref['ids']
        out['arms'][name] = r['result']
        d = paired(r['counts'], ref['counts'], r['ids'])['difference']
        out['minus_pipeline'][name] = {k: d[k] for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1', 'mae_beta0', 'mae_beta1')}
        print('PAIRED2', a.entry, name, json.dumps({k: [round(v['point'], 4), *[round(x, 4) for x in v['ci95']]] for k, v in out['minus_pipeline'][name].items()}), flush=True)
    best = max(arms, key=lambda n: out['arms'][n]['train_monitor']['edge_f1'])
    out['best_arm_by_monitor_edge_f1'] = best
    (OUT / f'node_decode2_stage2_{tag(a)}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    figures2(a, arms[best], best)


def figures2(a, arm, name):
    """v2 figures: the 8 validation patches of part 1, nodes found / missed / false under the given stage-2 arm."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p1 = pickle.loads((STORE / f'{tag(a)}-val.pkl').read_bytes())
    p2 = pickle.loads((STORE / f'{tag(a)}-p2-val.pkl').read_bytes())
    mips = {r['sample_id']: r['mip'] for r in p1['records'] if 'mip' in r}
    VIZ.mkdir(parents=True, exist_ok=True)
    short = a.entry.replace('-simp2', '') + ('-smoke' if a.smoke else '')
    sset = arm['map'].split(':')[0]
    for rec in p2['records']:
        if rec['sample_id'] not in mips:
            continue
        gt = gt_of2(rec['cache'])
        with np.load(rec['cache']) as z:
            gv = z['nodes'] * SIZE
            P, off = z['full_paths'] * SIZE, z['path_offsets']
            n_edges = len(z['edges'].reshape(-1, 2))
        p, _ = select2(rec, arm['map'], arm['rule'])
        mp = match((p / SIZE) @ gt['M'].T, gt['g'])
        found = np.zeros(len(gv), bool)
        found[list(mp.values())] = True
        fp = np.array([q for q in range(len(p)) if q not in mp], int)
        real = ~gt['ins']
        fig, ax = plt.subplots(3, 3, figsize=(11.5, 12), dpi=72)
        axes_of = [(1, 2), (0, 2), (0, 1)]
        panels = (('image', mips[rec['sample_id']]['image']), (f'heat ({sset} average)', rec['heat_mip'][sset]),
                  ('centreline (stage 1, identity view)', mips[rec['sample_id']]['centreline']))
        for row, (title, vol) in enumerate(panels):
            for k_ax in range(3):
                h = ax[row, k_ax]
                img = vol[k_ax].astype(np.float32)
                h.imshow(img, cmap='gray' if row == 0 else 'magma', vmin=None if row == 0 else 0, vmax=None if row == 0 else 1,
                         origin='lower', interpolation='nearest')
                r_, c_ = axes_of[k_ax]
                if row == 0:
                    for e in range(len(off) - 1):
                        seg = P[off[e]:off[e + 1]]
                        h.plot(seg[:, c_], seg[:, r_], lw=0.6, color='#86b6ef', alpha=0.9)
                for cname in CLASSES:
                    s = (gt['cls'] == cname) & real
                    for flag in (True, False):
                        q = s & (found == flag)
                        if q.any():
                            h.scatter(gv[q, c_], gv[q, r_], s=36 if row == 0 else 22, marker=MARKERS[cname],
                                      facecolors=COLORS[cname] if flag else 'none', edgecolors=COLORS[cname], linewidths=1.2)
                if gt['ins'].any():
                    h.scatter(gv[gt['ins'], c_], gv[gt['ins'], r_], s=14, marker='.', color='#c3c2b7')
                if len(fp):
                    h.scatter(p[fp, c_], p[fp, r_], s=40, marker='x', color='#e34948', linewidths=1.4)
                h.set_xlim(-0.5, SIZE - 0.5)
                h.set_ylim(-0.5, SIZE - 0.5)
                h.set_xticks([])
                h.set_yticks([])
                h.set_title(f'{title}, MIP along voxel axis {k_ax}', fontsize=8.5)
        miss = {c: int(((gt['cls'] == c) & real & ~found).sum()) for c in CLASSES}
        fig.suptitle(f"{a.entry}  {rec['sample_id']}  v2 nodes = stage-2 arm '{name}': {arm['variant']} rule {arm['rule']}\n"
                     f"GT nodes {int(real.sum())} (found {int((found & real).sum())}), GT edges {n_edges}, detections {len(p)}, "
                     f"unmatched {len(fp)}; missed boundary/junction/terminal/other {miss['boundary']}/{miss['junction']}/{miss['terminal']}/{miss['other']}\n"
                     f"classes: cache boundary flag, then degree; square boundary, triangle junction, circle terminal, diamond other\n"
                     f"filled = found, hollow = missed; red x = unmatched detection; grey dot = inserted anchor", fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(VIZ / f"v2_{short}_{rec['sample_id']}.png")
        plt.close(fig)
    print('FIGURES2', sorted(x.name for x in VIZ.glob(f'v2_{short}_*.png')), flush=True)


def run_command(a):
    if a.command in ('collect', 'all'):
        collect(a)
    if a.command in ('analyse', 'all'):
        analyse(a)
    if a.command in ('figures', 'all'):
        figures(a)
    if a.command == 'stage2':
        stage2(a)
    if a.command in ('collect2', 'all2'):
        collect2(a)
    if a.command in ('analyse2', 'all2'):
        analyse2(a)
    if a.command == 'stage2v2':
        stage2v2(a)
    if a.command == 'merge2':
        merge2(a)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['collect', 'analyse', 'figures', 'stage2', 'all',
                                       'collect2', 'analyse2', 'all2', 'stage2v2', 'merge2'])
    p.add_argument('entry')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--variants', default=None, help='stage2: comma-separated variant ids (default: the top two of '
                                                    'variants_beating_V0_by_0p01_on_monitor)')
    p.add_argument('--arms', default=None, help='stage2v2: comma-separated arm names (default: pipeline and every arm)')
    args = p.parse_args()
    for entry in args.entry.split('+'):  # '+'-separated entries run one after the other in one job
        a = argparse.Namespace(**{**vars(args), 'entry': entry})
        run_command(a)

