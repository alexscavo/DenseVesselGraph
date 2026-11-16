#!/usr/bin/env python3
"""Test-time flip averaging of the stage-2 EDGE scores, on top of the adopted flip-averaged node decoding (13 Nov 2026;
no retraining). Run: traced_unet_wide_long_r1111 of each entry, decoding --flip-average --nms 5.

Arms (nodes and candidate paths are identical in every arm: nodes = D.peaks on the flip-averaged heat
(D.decoding_output), candidates traced once on the unflipped centreline map, exactly as T.predict):
  reference   T.predict as is (single-pass edge scores and curves); thresholds of final-v2-nms5-flipavg.json; the edge
              threshold is also re-selected on monitor_train as a check.
  edge_prob   for each of the 8 axis flips k of the input patch: network pass on the flipped patch, every candidate
              path (and its end nodes) mirrored into the flipped frame (continuous voxel coordinate x -> 63 - x on the
              flipped axes), features sampled and scored with the edge head there; edge score = mean of the 8
              probabilities. Curves = reference (unflipped) curves.
  edge_logit  same with sigmoid(mean of the 8 logits).
  The better of edge_prob / edge_logit on monitor_train edge F1 (each at its own monitor-selected edge threshold) is the
  'edge_avg' arm.
  curve_avg   (variant B, only if edge_avg gains >= 0.005 monitor_train edge F1 over the reference) edge_avg scores and
              curves averaged over the 8 flips (each flip's uncompressed curve mirrored back to the original frame, same
              interior control count as the reference, then compress_curves as T.predict).
Node threshold = that of final-v2-nms5-flipavg.json (nodes unchanged); edge threshold re-selected on monitor_train
(D.select_edge_threshold) per arm. Validation is reported only; no test split is read.

Checks: (1) reference rows equal the saved final-v2-nms5-flipavg-validation-rows.pkl and its validation node / edge F1;
(2) the mirrored scorer on the identity flip (a second network pass, no mirroring) equals the T.predict scores;
(3) node anatomy: heat of the flipped pass sampled at mirrored node coordinates vs the unflipped heat at the nodes
(controls: same flipped heat at the unmirrored coordinates; flipped heat flipped back as a volume, sampled at the nodes,
must equal the mirrored sample).

Per-patch counts: node_decode_study_20261113.stage2_counts; subject-paired bootstrap: protocol_report_20261106.paired
(2000 resamples of subjects, seed 261030). Output: experiments/final_try_20261113/edge_flip_average_<entry>.json
(never overwritten; -smoke variant for the smoke test); rows under $SCRATCH/experiments/gnbm/edge_flip_average_20261113/.
T and D are imported unchanged (T.predict is replicated here for this run's mode, not modified).

usage: edge_flip_average_study_20261113.py --entry IXI-simp2 [--smoke] [--batch 8]
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
STORE = SCRATCH / 'experiments/gnbm/edge_flip_average_20261113'
OUT = ROOT / 'experiments/final_try_20261113'
RUN = 'traced_unet_wide_long_r1111'
REF = 'final-v2-nms5-flipavg.json'
FLIPS = [()] + [d for n in (1, 2, 3) for d in combinations((2, 3, 4), n)]  # image dims; point axis = dim - 2
KEYS = ('node_precision', 'node_recall', 'node_f1', 'edge_precision', 'edge_recall', 'edge_f1', 'common_branch_f1',
        'common_curve_f1', 'common_curve_precision', 'common_curve_recall', 'beta0_mae', 'beta1_mae')
GAIN_FOR_B = 0.005
ANATOMY_BATCHES = 3


def mirror(pts, dims, size):
    """Continuous voxel coordinates (..., 3) into / out of the frame of the input flipped on image `dims`."""
    if not dims:
        return pts
    q = pts.clone()
    for d in dims:
        q[..., d - 2] = size - 1 - q[..., d - 2]
    return q


def sample(vol, pts, size):
    """Trilinear sample of vol (C,D,H,W) at voxel points (n,3), the convention of T.path_features -> (n,C)."""
    import torch.nn.functional as F
    g = (pts / (size - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
    return F.grid_sample(vol[None], g, align_corners=True).view(vol.shape[0], -1).T


def predict_flip(model, ds, device, nt, workers, batch, anatomy):
    """T.predict (mode without roles / two-hop / passthrough / prob hook) + per-flip edge logits and mirrored-back
    uncompressed curves of every candidate. Returns rows (reference fields as T.predict) and timing / check stats."""
    import torch
    import dense_graph_20261030 as D
    import dense_graph_traced_20261030 as T
    from node_heatmap_probe_20261030 import collate
    from oracle_geometry_20261030 import to_mm
    size = T.SIZE
    sync = torch.cuda.synchronize
    model.eval()
    rows = []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate)
    offset, nb = 0, 0
    tm = dict(reference=0.0, flip_forward=0.0, flip_scoring=0.0)
    ident = []  # max |p_identity - p_reference| per patch
    anat = {str(d): dict(mirrored=[], unmirrored=[], ref=[], flipback_minus_mirrored=[]) for d in FLIPS[1:]}
    with torch.no_grad():
        for images, nodes, edges, geometry in loader:
            x_img = images.to(device)
            sync()
            t0 = time.time()
            # ---- reference: verbatim T.predict for this mode
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model.forward_image(x_img)
                decoded = D.decoding_output(model, x_img, out)
            feats, heat, prob = out[:, 5:].float(), out[:, :1].float().sigmoid(), out[:, 4].float().sigmoid()
            found = D.peaks(decoded, nt)
            contacts, dist, label = T.traced_pairs(prob, [p for p, _ in found])
            keep = []
            for b, (p, score) in enumerate(found):
                metric = geometry[b]['metric'].to(device) / size
                row = dict(nodes=(p / size).cpu().numpy(), node_scores=score.cpu().numpy(), pairs=np.zeros((0, 2), int),
                           edge_scores=None, curves=None, traced=None, interior=None, via=None)
                pairs = T.patch_candidates(p, metric, contacts[b], None) if len(p) >= 2 else None
                if pairs is not None and len(pairs):
                    paths, traced = T.build_paths(p, pairs, contacts[b], dist[b], label[b], None)
                    x = model.path_features(feats[b], heat[b], paths, prob[b], None)
                    interior = T.choose_interior(to_mm(paths, metric[None].expand(len(paths), 3, 3)), traced, T.voxel_size(metric))
                    curves = model.path_curves(x, paths, interior, traced)
                    if T.MODE['compress_tol']:
                        curves, kept = T.compress_curves(curves, T.MODE['compress_tol'], interior)
                        row['interior_compressed'] = kept.cpu().numpy()
                    logit0 = model.edge_head(x).squeeze(-1)
                    row.update(pairs=pairs.cpu().numpy(), edge_scores=logit0.sigmoid().cpu().numpy(),
                               curves=(curves / size).cpu().numpy(), traced=traced.cpu().numpy(), interior=interior.cpu().numpy())
                    keep.append((b, paths, traced, interior, logit0))
                record = ds.records[offset + b]
                row.update(sample_id=record.sample_id, cache=str(record.graph.parent.parent / 'spline' / f'{record.sample_id}.npz'))
                rows.append(row)
            sync()
            tm['reference'] += time.time() - t0
            # ---- 8 flips: mirrored scoring of the same candidates
            logits = {b: [] for b, *_ in keep}
            curve_sum = {b: 0.0 for b, *_ in keep}
            for dims in FLIPS:
                sync()
                t1 = time.time()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    o = model.forward_image(x_img.flip(dims) if dims else x_img)
                fk, hk, pk = o[:, 5:].float(), o[:, :1].float().sigmoid(), o[:, 4].float().sigmoid()
                sync()
                t2 = time.time()
                tm['flip_forward'] += t2 - t1
                for b, paths, traced, interior, logit0 in keep:
                    mp = mirror(paths, dims, size)
                    xk = model.path_features(fk[b], hk[b], mp, pk[b], None)
                    lk = model.edge_head(xk).squeeze(-1)
                    logits[b].append(lk)
                    curve_sum[b] = curve_sum[b] + mirror(model.path_curves(xk, mp, interior, traced), dims, size)
                    if not dims:
                        ident.append(float((lk.sigmoid() - logit0.sigmoid()).abs().max()))
                sync()
                tm['flip_scoring'] += time.time() - t2
                if anatomy and nb < ANATOMY_BATCHES and dims:  # not timed
                    back = hk.flip(dims)
                    for b, (p, _) in enumerate(found):
                        if len(p) == 0:
                            continue
                        a = anat[str(dims)]
                        a['ref'].append(sample(heat[b], p, size)[:, 0].cpu().numpy())
                        a['mirrored'].append(sample(hk[b], mirror(p, dims, size), size)[:, 0].cpu().numpy())
                        a['unmirrored'].append(sample(hk[b], p, size)[:, 0].cpu().numpy())
                        a['flipback_minus_mirrored'].append(float((sample(back[b], p, size) - sample(hk[b], mirror(p, dims, size), size)).abs().max()))
            for b, paths, traced, interior, logit0 in keep:
                r = rows[offset + b]
                r['flip_logits'] = torch.stack(logits[b], 1).float().cpu().numpy()  # (n_cand, 8), column 0 = identity
                avg = curve_sum[b] / len(FLIPS)
                if T.MODE['compress_tol']:
                    avg, kept = T.compress_curves(avg, T.MODE['compress_tol'], interior)
                    r['interior_compressed_avg'] = kept.cpu().numpy()
                r['curves_avg'] = (avg / size).cpu().numpy()
            offset += len(images)
            nb += 1
    model.train()
    n = len(rows)
    timing = {f'{k}_seconds_per_patch': v / n for k, v in tm.items()}
    timing.update(patches=n, batch=batch,
                  reference_arm_seconds_per_patch=tm['reference'] / n,
                  edge_avg_arm_as_run_seconds_per_patch=(tm['reference'] + tm['flip_forward'] + tm['flip_scoring']) / n,
                  edge_avg_arm_shared_forwards_estimate_seconds_per_patch=(tm['reference'] + tm['flip_scoring']) / n,
                  note='reference = node flip-average (8 passes) + decode + trace + score; as_run adds 8 separate passes '
                       '(identity included) and 8 mirrored scorings; shared_forwards_estimate (inference) assumes the 7 '
                       'flipped passes of the node average are reused, so only the scorings are added')
    checks = dict(identity_flip_max_abs_prob_difference=max(ident) if ident else None,
                  identity_flip_patches=len(ident))
    if anatomy:
        an = {}
        for d, a in anat.items():
            if not a['ref']:
                continue
            ref, mir, unm = (np.concatenate(a[k]) for k in ('ref', 'mirrored', 'unmirrored'))
            an[d] = dict(nodes=len(ref), mean_abs_mirrored_minus_unflipped=float(np.abs(mir - ref).mean()),
                         corr_mirrored=float(np.corrcoef(mir, ref)[0, 1]),
                         mean_abs_unmirrored_minus_unflipped_control=float(np.abs(unm - ref).mean()),
                         corr_unmirrored_control=float(np.corrcoef(unm, ref)[0, 1]),
                         mean_unflipped_heat_at_nodes=float(ref.mean()), mean_mirrored_heat_at_nodes=float(mir.mean()),
                         max_abs_flipback_volume_minus_mirrored_sample=float(max(a['flipback_minus_mirrored'])))
        checks['node_anatomy_first_batches'] = an
    return rows, timing, checks


def with_scores(rows, how, curves=False):
    """Copies of the rows with averaged edge scores (how = 'prob' | 'logit') and optionally averaged curves."""
    out = []
    for r in rows:
        r = dict(r)
        if r['edge_scores'] is not None:
            lg = r['flip_logits'].astype(np.float64)
            r['edge_scores'] = (1 / (1 + np.exp(-lg))).mean(1) if how == 'prob' else 1 / (1 + np.exp(-lg.mean(1)))
            if curves:
                r['curves'] = r['curves_avg']
        out.append(r)
    return out


def corr(rows_a, rows_b):
    a = np.concatenate([r['edge_scores'] for r in rows_a if r['edge_scores'] is not None])
    b = np.concatenate([r['edge_scores'] for r in rows_b if r['edge_scores'] is not None])
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return dict(candidates=int(len(a)), pearson=float(np.corrcoef(a, b)[0, 1]), spearman=float(np.corrcoef(rank(a), rank(b))[0, 1]),
                mean_abs_difference=float(np.abs(a - b).mean()))


def main():
    import torch
    import dense_graph_20261030 as D
    import dense_graph_traced_20261030 as T
    import node_decode_study_20261113 as S
    from node_heatmap_probe_20261030 import PLAN, make_dataset
    from protocol_report_20261106 import paired, pooled
    p = argparse.ArgumentParser()
    p.add_argument('--entry', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--batch', type=int, default=8)
    a = p.parse_args()
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    tag = a.entry + ('-smoke' if a.smoke else '')
    out_path = OUT / f'edge_flip_average_{tag}.json'
    assert a.smoke or not out_path.exists(), f'{out_path} exists; not overwriting'
    run_dir = RUNS / a.entry / RUN
    ref_json = json.loads((run_dir / REF).read_text())
    final = json.loads((run_dir / 'final.json').read_text())
    T.apply_saved_mode(ref_json)
    assert T.MODE == {**T.MODE, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in final['mode'].items()}}
    assert not T.MODE['passthrough'] and not T.MODE['roles'] and not T.MODE['two_hop'] and not T.MODE['gap_features']
    assert T.PROB_HOOK is None
    D.FLIP_AVERAGE, D.NMS = True, 5
    assert ref_json['flip_average'] and ref_json['nms'] == 5
    device = 'cuda'
    state = torch.load(run_dir / 'model.pt', map_location=device)
    model = T.TracedDenseGraph(gap_features=False, two_hop=False, passthrough=False, backbone=ref_json.get('backbone', 'unet'),
                               feature_channels=ref_json.get('feature_channels', 32)).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    plan = json.loads(PLAN.read_text())['datasets'][a.entry]
    mon_ids, val_ids = list(plan['monitor_train']), list(plan['val_ids'])
    if a.smoke:
        mon_ids, val_ids = mon_ids[:16], val_ids[:40]
    nt, et_ref = ref_json['thresholds']['node'], ref_json['thresholds']['edge']
    t_start = time.time()
    mon_rows, mon_time, mon_checks = predict_flip(model, make_dataset(a.entry, mon_ids, augment=False), device, nt, workers, a.batch, False)
    val_rows, val_time, val_checks = predict_flip(model, make_dataset(a.entry, val_ids, augment=False), device, nt, workers, a.batch, True)
    print(f'predicted {len(mon_rows)} monitor + {len(val_rows)} validation patches in {time.time() - t_start:.0f}s', flush=True)
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / f'{tag}-rows.pkl').write_bytes(pickle.dumps(dict(monitor=mon_rows, validation=val_rows)))

    # ---- check 1: reference rows vs the saved evaluation
    saved = pickle.loads((run_dir / REF.replace('.json', '-validation-rows.pkl')).read_bytes())
    saved = {r['sample_id']: r for r in saved}
    dn, de, dc, same_pairs = 0.0, 0.0, 0.0, True
    for r in val_rows:
        s = saved[r['sample_id']]
        if r['nodes'].shape != s['nodes'].shape or r['pairs'].shape != s['pairs'].shape or not np.array_equal(r['pairs'], s['pairs']):
            same_pairs = False
            continue
        dn = max(dn, float(np.abs(r['nodes'] - s['nodes']).max(initial=0)), float(np.abs(r['node_scores'] - s['node_scores']).max(initial=0)))
        if r['edge_scores'] is not None:
            de = max(de, float(np.abs(r['edge_scores'] - s['edge_scores']).max(initial=0)))
            dc = max(dc, float(np.abs(r['curves'] - s['curves']).max(initial=0)))
    check1 = dict(saved_rows_same_nodes_and_pairs=same_pairs, max_abs_node_difference=dn, max_abs_edge_score_difference=de,
                  max_abs_curve_difference=dc)

    # ---- thresholds (monitor_train only) and evaluation
    arms_rows = dict(reference=(mon_rows, val_rows),
                     edge_prob=(with_scores(mon_rows, 'prob'), with_scores(val_rows, 'prob')),
                     edge_logit=(with_scores(mon_rows, 'logit'), with_scores(val_rows, 'logit')))
    res, counts = {}, {}

    def evaluate_arm(name, et_fixed=None):
        mon, val = arms_rows[name]
        et_sel = D.select_edge_threshold(mon, nt)
        et = et_sel if et_fixed is None else et_fixed
        m, v = D.evaluate(mon, nt, et), D.evaluate(val, nt, et)
        S.CTX.update(rows=val, nt=nt, et=et)
        c = np.asarray(S.run_pool(S.stage2_counts, len(val), workers), float)
        S.CTX.clear()
        counts[name] = c
        res[name] = dict(node_threshold=nt, edge_threshold=et, edge_threshold_selected_on_monitor=et_sel,
                         train_monitor={k: m.get(k) for k in KEYS}, validation={k: v.get(k) for k in KEYS},
                         validation_pooled_counts=pooled(c))
        print(f'ARM {name} edge_t {et} (selected {et_sel}) monitor edge F1 {m["edge_f1"]:.4f} | val node {v["node_f1"]:.4f} '
              f'edge {v["edge_f1"]:.4f} branch {v["common_branch_f1"]:.4f} curve {v["common_curve_f1"]:.4f}', flush=True)

    evaluate_arm('reference', et_ref)
    evaluate_arm('edge_prob')
    evaluate_arm('edge_logit')
    f_prob, f_logit = res['edge_prob']['train_monitor']['edge_f1'], res['edge_logit']['train_monitor']['edge_f1']
    chosen = 'edge_prob' if f_prob >= f_logit else 'edge_logit'
    gain = res[chosen]['train_monitor']['edge_f1'] - res['reference']['train_monitor']['edge_f1']
    run_b = gain >= GAIN_FOR_B
    if run_b:
        how = chosen.split('_')[1]
        arms_rows['curve_avg'] = (with_scores(mon_rows, how, True), with_scores(val_rows, how, True))
        evaluate_arm('curve_avg', res[chosen]['edge_threshold'])
    ids = [r['sample_id'] for r in val_rows]
    diffs = {}
    for name in res:
        if name != 'reference':
            d = paired(counts[name], counts['reference'], ids)['difference']
            diffs[f'{name}_minus_reference'] = {k: dict(point=d[k]['point'], ci95=d[k]['ci95'], p=d[k]['p'])
                                                for k in ('node_f1', 'edge_f1', 'branch_f1', 'curve_f1', 'mae_beta0', 'mae_beta1')}
    if run_b:
        d = paired(counts['curve_avg'], counts[chosen], ids)['difference']
        diffs[f'curve_avg_minus_{chosen}'] = {k: dict(point=d[k]['point'], ci95=d[k]['ci95'], p=d[k]['p'])
                                              for k in ('branch_f1', 'curve_f1')}
    ref_v = ref_json['validation']
    check1.update(validation_node_f1=dict(this_script=res['reference']['validation']['node_f1'], json=ref_v['node_f1']),
                  validation_edge_f1=dict(this_script=res['reference']['validation']['edge_f1'], json=ref_v['edge_f1']),
                  validation_branch_f1=dict(this_script=res['reference']['validation']['common_branch_f1'], json=ref_v['common_branch_f1']),
                  train_monitor_edge_f1=dict(this_script=res['reference']['train_monitor']['edge_f1'], json=ref_json['train_monitor']['edge_f1']),
                  edge_threshold=dict(reselected_on_monitor=res['reference']['edge_threshold_selected_on_monitor'], json=et_ref),
                  pooled_counts_reproduce_evaluate={k: bool(abs(res['reference']['validation_pooled_counts'][j] - res['reference']['validation'][k]) < 6e-5)
                                                    for k, j in (('node_f1', 'node_f1'), ('edge_f1', 'edge_f1'), ('common_branch_f1', 'branch_f1'))})
    per_flip = {}
    for split, rows in (('train_monitor', mon_rows), ('validation', val_rows)):
        lg = np.concatenate([r['flip_logits'] for r in rows if r['edge_scores'] is not None])
        pr = 1 / (1 + np.exp(-lg.astype(np.float64)))
        per_flip[split] = {str(d): dict(mean_abs_prob_minus_identity=float(np.abs(pr[:, k] - pr[:, 0]).mean()),
                                        pearson_with_identity=float(np.corrcoef(pr[:, k], pr[:, 0])[0, 1])) for k, d in enumerate(FLIPS)}
    correlation = {f'{split}_{name}': corr(arms_rows['reference'][i], arms_rows[name][i])
                   for i, split in enumerate(('train_monitor', 'validation')) for name in ('edge_prob', 'edge_logit')}
    decisions = {}
    for name in ('edge_prob', 'edge_logit'):
        r0 = np.concatenate([r['edge_scores'] for r in arms_rows['reference'][1] if r['edge_scores'] is not None]) >= et_ref
        r1 = np.concatenate([r['edge_scores'] for r in arms_rows[name][1] if r['edge_scores'] is not None]) >= res[name]['edge_threshold']
        decisions[name] = dict(validation_candidates=int(len(r0)), accepted_reference=int(r0.sum()), accepted_arm=int(r1.sum()),
                               flipped_decisions=int((r0 != r1).sum()))
    result = dict(description=__doc__.split('usage:')[0].strip(), entry=a.entry, run=str(run_dir), reference_file=REF,
                  smoke=a.smoke, mode=dict(T.MODE), nms=D.NMS, flip_average_nodes=D.FLIP_AVERAGE,
                  monitor_patches=len(mon_rows), validation_patches=len(val_rows),
                  validation_subjects=len({s.split('_')[1] for s in ids}),
                  check_reference_reproduction=check1,
                  check_mirrored_scoring=dict(train_monitor=mon_checks, validation=val_checks),
                  averaging_choice=dict(train_monitor_edge_f1_prob=f_prob, train_monitor_edge_f1_logit=f_logit, chosen=chosen,
                                        train_monitor_edge_f1_gain_over_reference=gain, variant_b_run=run_b, variant_b_rule=f'gain >= {GAIN_FOR_B}'),
                  arms=res, paired_validation=diffs, score_correlation_reference_vs_averaged=correlation,
                  validation_decision_changes=decisions, per_flip_scores=per_flip,
                  timing=dict(train_monitor=mon_time, validation=val_time),
                  notes=['nodes and candidate paths are identical in all arms (single tracing on the unflipped centreline map)',
                         'node threshold from the reference file; edge thresholds selected on monitor_train only (D.EDGE_THRESHOLDS)',
                         'paired = subject bootstrap (protocol_report_20261106.paired), arm minus reference; single seed, validation only'])
    assert a.smoke or not out_path.exists()
    out_path.write_text(json.dumps(result, indent=1, default=float) + '\n')
    print('wrote', out_path, flush=True)
    print('PAIRED ' + json.dumps({n: {k: [round(v['point'], 4), *[round(x, 4) for x in v['ci95']]] for k, v in d.items()} for n, d in diffs.items()}), flush=True)


if __name__ == '__main__':
    main()
