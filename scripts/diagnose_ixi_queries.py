#!/usr/bin/env python3
"""Frozen IXI P0-A/B/C/D: full-query cache, edge attribution, oracle and NMS."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

from data.loaders import build_data_loaders, build_evaluation_loader
from models import build_model
from models.checkpoint import load_checkpoint_file
from training.engine import _move_graph_batch
from training.evaluation.inference import infer_graphs
from metrics.common_vessel_metrics import evaluate_common_vessels, aggregate_common_vessels
from scripts.controlled_spline_unfreeze import frozen_state_digest
from scripts.frozen_spline_diagnostic import training_evaluation_loader
from scripts.ixi_query_diagnostics import categorize, edge_accounting, nms, summarize

RUN = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_transfer_20261015/runs/ixi_delayed_spline')
REFINEMENT = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_node_refinement_20261018')
IDS = Path('experiments/ixi_relation_isolation/full-1278148/summary.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def table(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def load(arm):
    parent = load_checkpoint_file(RUN/'best.pt')
    adapter = None if arm == 'parent' else load_checkpoint_file(REFINEMENT/arm/'trained-heads.pt')
    cfg = copy.deepcopy(parent['config'] if adapter is None else adapter['config'])
    if adapter is not None:
        assert Path(adapter['provenance']['initial_checkpoint']).resolve() == (RUN/'best.pt').resolve()
    cfg['data'].update(train_augmentation=False, batch_size=1, validation_batch_size=1)
    cfg['runtime']['workers'] = 0
    model = build_model(cfg).cuda()
    result = model.load_state_dict(parent['net'], strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith('node_refinement.') for k in result.missing_keys)
    if adapter is not None:
        result = model.load_state_dict(adapter['parameters'], strict=False)
        assert not result.unexpected_keys
        assert set(adapter['parameters']) == set(adapter['provenance']['trainable'])
    model.eval().requires_grad_(False)
    assert model.obj_token == 256 and cfg['model']['num_classes'] == 2
    assert model.spline_head is not None and model.spline_head.curve_evidence is None
    assert not cfg['evaluation'].get('node_type_degree_caps', False)
    assert not cfg['evaluation'].get('node_type_degree_penalty', 0.)
    return model, cfg, int(parent['steps'])


def cache_patch(model, cfg, batch):
    volumes, _ = _move_graph_batch(batch, 'cuda', cfg['training']['input'], True)
    with torch.no_grad():
        tokens, pred, _ = model(volumes)
        # Recompute preliminary heads from the same tokens, without a second CNN pass.
        original_nodes = model.coord_embed(tokens[:, :model.obj_token]).sigmoid()
        original_logits = model.class_embed(tokens[:, :model.obj_token])
        calls = []
        def record(features):
            logits = model.relation_embed(features)
            calls.append(logits.detach().cpu())
            return logits
        full = infer_graphs(tokens, pred, record, object_queries=model.obj_token,
                            relation_tokens=cfg['model']['decoder']['relation_tokens'],
                            node_threshold=-1., edge_threshold=-1.)[0]
        assert len(calls) == 2
        logits = .5*(calls[0]+calls[1])
        # Normal inference remains the authority for filtering and spline orientation.
        metric = batch[4][0]['metric'].to(tokens)
        normal = infer_graphs(tokens, pred, model.relation_embed,
                              object_queries=model.obj_token,
                              relation_tokens=cfg['model']['decoder']['relation_tokens'],
                              node_threshold=cfg['evaluation'].get('node_threshold'),
                              edge_threshold=cfg['evaluation'].get('edge_threshold'),
                              spline_head=model.spline_head, spline_metrics=[metric])[0]
    out = {k: v.numpy() for k, v in full.items()}
    out.update(pair_logits=logits.numpy(), tokens=tokens[0].cpu().numpy(),
               original_nodes=original_nodes[0].cpu().numpy(),
               original_logits=original_logits[0].cpu().numpy(),
               node_logits=pred['pred_logits'][0].cpu().numpy(),
               retained=normal['query_ids'].numpy(),
               decoded_query_edges=normal['query_ids'][normal['edges']].numpy(),
               decoded_controls=normal['spline_controls'].numpy(),
               decoded_curves=normal['spline_curves'].numpy(),
               shape=np.array(volumes.shape[-3:]))
    threshold = cfg['evaluation'].get('edge_threshold')
    out['edge_selected'] = (logits.argmax(-1).numpy() == 1 if threshold is None
                            else out['edge_scores'] > threshold)
    active = np.isin(out['edges'], out['retained']).all(1) & out['edge_selected']
    expected = {tuple(sorted(map(int, e))) for e in out['decoded_query_edges']}
    actual = {tuple(map(int, e)) for e in out['edges'][active]}
    assert actual == expected, 'Full-query cache differs from normal decoded graph'
    score_lookup = {tuple(map(int, e)): s for e, s in zip(out['edges'], out['edge_scores'])}
    np.testing.assert_allclose([score_lookup[tuple(sorted(map(int, e)))] for e in out['decoded_query_edges']],
                               normal['edge_scores'].numpy(), atol=2e-6, rtol=2e-6)
    return out


def graph(cache, query_ids):
    """Delete nodes/incident edges only; preserve original curve controls and direction."""
    remap = {int(q): i for i, q in enumerate(query_ids)}
    keep = np.isin(cache['decoded_query_edges'], query_ids).all(1)
    edges = np.array([[remap[int(a)], remap[int(b)]] for a, b in cache['decoded_query_edges'][keep]], dtype=int).reshape(-1, 2)
    return dict(nodes=cache['nodes'][query_ids], edges=edges,
                spline_controls=cache['decoded_controls'][keep], spline_family='bspline')


def view(path, cache, target, mask, mapping, categories):
    import plotly.graph_objects as go
    fig = go.Figure()
    metric = target['metric']
    def lines(curves, name, color):
        xyz = []
        for c in curves:
            xyz.extend(c.tolist()+[[None]*3])
        if xyz:
            x, y, z = zip(*xyz)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=name, line=dict(color=color, width=4)))
    offsets = target['path_offsets']
    paths = target['full_paths'] @ metric.T
    lines([paths[a:b] for a, b in zip(offsets[:-1], offsets[1:])], 'GT full paths (uncertain)', 'gray')
    for kind, color in [('MM', '#168a45'), ('MU', '#e69100'), ('UU', '#cf2457')]:
        curves = []
        for e, c in zip(cache['decoded_query_edges'], cache['decoded_curves']):
            count = sum(int(i) in mapping for i in e)
            if ('MM' if count == 2 else 'MU' if count == 1 else 'UU') == kind:
                curves.append(c @ metric.T)
        lines(curves, kind+' decoded edges', color)
    q = cache['retained']; xyz = cache['nodes'][q] @ metric.T
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers', name='Retained queries',
        marker=dict(size=4), text=[f'query {i}; {categories[int(i)]}; P={cache["node_scores"][i]:.3f}' for i in q]))
    fg = np.argwhere(mask)[::max(1, int(mask.sum())//4000)] / cache['shape'] @ metric.T
    fig.add_trace(go.Scatter3d(x=fg[:, 0], y=fg[:, 1], z=fg[:, 2], mode='markers', name='Segmentation voxel centres',
                             marker=dict(size=1, opacity=.12, color='cyan'), visible='legendonly'))
    fig.update_layout(title=path.stem+' — 2 mm positional assignment; no anatomical certification',
                      scene=dict(aspectmode='data', xaxis_title='D physical mm', yaxis_title='H physical mm', zaxis_title='W physical mm'))
    fig.write_html(path, include_plotlyjs=True)


def run(args):
    started = time.monotonic()
    torch.manual_seed(191031); torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    model, cfg, step = load(args.arm)
    frozen = frozen_state_digest(model)
    root = Path(cfg['data']['datasets']['synthetic_mri']['root'])
    dataset_manifest = json.loads((root/'manifest.json').read_text())
    ids = json.loads(IDS.read_text())['ids']
    if args.full_validation:
        ids = dict(train=ids['train'], val=sorted(dataset_manifest['validation_ids']))
    if args.smoke:
        ids = {s: v[:4] for s, v in ids.items()}
    subject = lambda name: name.rsplit('_', 1)[0]
    train_subjects = {subject(x) for x in dataset_manifest['train_ids']}
    val_subjects = {subject(x) for x in dataset_manifest['validation_ids']}
    assert not train_subjects & val_subjects
    assert set(ids['train']) <= set(dataset_manifest['train_ids'])
    assert set(ids['val']) <= set(dataset_manifest['validation_ids'])
    full, _ = build_data_loaders(cfg)
    loaders = dict(train=training_evaluation_loader(full, cfg, ids['train']),
                   val=build_evaluation_loader(cfg, dataset_name='synthetic_mri', sample_ids=ids['val']))
    provenance = dict(arm=args.arm, checkpoint=str(RUN/'best.pt'), checkpoint_sha256=digest(RUN/'best.pt'),
        parent_step=step, ids=ids, dataset=str(root), dataset_manifest_sha256=digest(root/'manifest.json'),
        selection_source=str(IDS), selection_sha256=digest(IDS),
        adapter_sha256=None if args.arm == 'parent' else digest(REFINEMENT/args.arm/'trained-heads.pt'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): digest(p) for p in [Path(__file__), Path('scripts/ixi_query_diagnostics.py'), Path('training/evaluation/inference.py')]},
        config=cfg, smoke=args.smoke, full_validation=args.full_validation,
        protocol=dict(node_tolerances_mm=[1., 2.], foreground_distance_mm=1., nms_radii_mm=[.5, 1.],
                      bn='saved statistics, eval, no recalibration', augmentation=False,
                      oracle='delete unmatched retained queries using fixed baseline positional assignment; no inserted nodes or rescoring',
                      geometry='shared-guides-v2: 1mm anchors/curves, 0.25mm sampling',
                      pairs='all 256 choose 2; infer_graphs symmetric logits; default thresholds unchanged',
                      subject_split='full dataset train/val subject sets disjoint; IDs inherited from previous diagnostic'))
    save(args.output/'manifest.json', provenance)
    (args.output/'resolved-config.yaml').write_text(yaml.safe_dump(cfg))
    rows, node_rows, edge_rows, gt_rows, common = [], [], [], [], {}
    ap_data = {}
    for split, loader in loaders.items():
        (args.output/split).mkdir()
        for index, batch in enumerate(loader):
            name = loader.dataset.records[index].sample_id
            path = root/split/'spline'/f'{name}.npz'
            with np.load(path) as f:
                target = {k: f[k] for k in f.files}
            cache = cache_patch(model, cfg, batch)
            mask = batch[1][0, 0].numpy() > 0
            assert tuple(mask.shape) == tuple(cache['shape'])
            np.testing.assert_allclose(target['nodes'], batch[2][0].numpy()[:, :3], atol=1e-6)
            np.testing.assert_allclose(target['metric'], batch[4][0]['metric'].numpy(), atol=1e-5)
            save(args.output/split/(name+'-source.json'), dict(gt_sha256=digest(path),
                 raw=str(root/split/'raw'/f'{name}_data.nii.gz'), mask=str(root/split/'seg'/f'{name}_seg.nii.gz'),
                 gt=str(path), subject=subject(name)))
            np.savez_compressed(args.output/split/(name+'.npz'), **cache,
                                metric=target['metric'], gt_nodes=target['nodes'], gt_edges=target['edges'])
            physical = cache['nodes'] @ target['metric'].T
            gt_physical = target['nodes'] @ target['metric'].T
            gt_degree = np.bincount(target['edges'].ravel(), minlength=len(target['nodes']))
            types = np.where(target['boundary'], 'boundary', np.where(gt_degree >= 3, 'junction',
                       np.where(gt_degree == 1, 'interior_terminal', np.where(gt_degree == 2, 'degree2', 'isolated'))))
            for tolerance in (1., 2.):
                mappings = {}
                for population, q in [('retained', cache['retained']), ('raw', np.arange(len(physical)))]:
                    mapping, categories, distance, inside = categorize(cache['nodes'], target['nodes'], target['metric'], mask, q, tolerance)
                    mappings[population] = mapping
                    for i in q:
                        nearest = int(np.linalg.norm(gt_physical-physical[i], axis=1).argmin()) if len(gt_physical) else None
                        others = q[q != i]
                        node_rows.append(dict(split=split, sample_id=name, subject=subject(name), tolerance_mm=tolerance,
                            population=population, query=int(i), category=categories[int(i)], matched_gt=mapping.get(int(i)),
                            nearest_gt=nearest, nearest_gt_type=str(types[nearest]) if nearest is not None else None,
                            score=float(cache['node_scores'][i]), in_bounds=bool(inside[i]),
                            foreground_distance_mm=float(distance[i]) if np.isfinite(distance[i]) else None,
                            nearest_prediction_mm=float(np.linalg.norm(physical[others]-physical[i], axis=1).min()) if len(others) else None))
                base_mapping = mappings['retained']
                for j in range(len(gt_physical)):
                    gt_rows.append(dict(split=split, sample_id=name, subject=subject(name), tolerance_mm=tolerance,
                        gt=j, type=str(types[j]), retained_recovered=j in base_mapping.values(), raw_recovered=j in mappings['raw'].values(),
                        nearby_retained=int((np.linalg.norm(physical[cache['retained']]-gt_physical[j], axis=1) <= tolerance).sum())))
                policies = dict(baseline=cache['retained'], oracle=np.array(sorted(base_mapping), dtype=int),
                                nms_0p5mm=nms(physical, cache['node_scores'], cache['retained'], .5),
                                nms_1mm=nms(physical, cache['node_scores'], cache['retained'], 1.))
                for policy, q in policies.items():
                    mapping = base_mapping if policy in ('baseline', 'oracle') else categorize(
                        cache['nodes'], target['nodes'], target['metric'], mask, q, tolerance)[0]
                    result, chosen, labels, edge_types, edges = edge_accounting(q, cache['edges'], cache['edge_scores'],
                        cache['edge_selected'], mapping, target['edges'], len(target['nodes']))
                    rows.append(dict(split=split, sample_id=name, subject=subject(name), tolerance_mm=tolerance, policy=policy, **result))
                    if policy == 'baseline':
                        mm = (edge_types == 'MM') & np.isin(cache['edges'], q).all(1)
                        ap_data.setdefault((split, tolerance), []).append((labels[mm], cache['edge_scores'][mm]))
                        for i in np.flatnonzero(chosen):
                            a, b = map(int, cache['edges'][i])
                            edge_rows.append(dict(split=split, sample_id=name, tolerance_mm=tolerance, query_a=a, query_b=b,
                                endpoint_type=edge_types[i], tp=bool(labels[i]), score=float(cache['edge_scores'][i])))
                    if tolerance == 2.:
                        output_graph = graph(cache, q)
                        assert len(output_graph['edges']) == result['edges']
                        metric_row = evaluate_common_vessels(output_graph, path)
                        chord_row = evaluate_common_vessels({k: output_graph[k] for k in ('nodes', 'edges')}, path)
                        common.setdefault((split, policy), []).append(dict(sample_id=name, curved=metric_row, chord=chord_row))
                if tolerance == 2. and split == 'val' and index < 3:
                    _, categories, _, _ = categorize(cache['nodes'], target['nodes'], target['metric'], mask, cache['retained'], 2.)
                    view(args.output/(name+'.html'), cache, target, mask, base_mapping, categories)
            if (index+1) % 16 == 0:
                print(split, index+1, 'patches', round(time.monotonic()-started, 1), 's', flush=True)
    assert frozen_state_digest(model) == frozen
    comparison, subjects, ranking = [], [], []
    for split in loaders:
        for tolerance in (1., 2.):
            for policy in ('baseline', 'oracle', 'nms_0p5mm', 'nms_1mm'):
                group = [r for r in rows if r['split'] == split and r['tolerance_mm'] == tolerance and r['policy'] == policy]
                summary = dict(split=split, tolerance_mm=tolerance, policy=policy, **summarize(group))
                if tolerance == 2.:
                    geo = common[(split, policy)]
                    cg = aggregate_common_vessels([r['curved'] for r in geo])
                    ch = aggregate_common_vessels([r['chord'] for r in geo])
                    summary.update(common_branch_f1=cg['common_branch_f1'], common_curve_f1=cg['common_curve_f1'],
                                   same_endpoint_chord_curve_f1=ch['common_curve_f1'])
                comparison.append(summary)
                for sid in sorted({r['subject'] for r in group}):
                    subjects.append(dict(split=split, tolerance_mm=tolerance, policy=policy, subject=sid,
                                         **summarize([r for r in group if r['subject'] == sid])))
            y = np.concatenate([x[0] for x in ap_data[(split, tolerance)]])
            p = np.concatenate([x[1] for x in ap_data[(split, tolerance)]])
            ap = float(average_precision_score(y, p)) if y.any() else None
            ranking.append(dict(split=split, tolerance_mm=tolerance, pairs=len(y), positives=int(y.sum()), conditional_MM_ap=ap))
            if len(y) and y.any():
                precision, recall, threshold = precision_recall_curve(y, p)
                np.savez_compressed(args.output/f'{split}-{tolerance:g}mm-PR.npz', precision=precision, recall=recall, thresholds=threshold)
    for name, values in [('per-patch', rows), ('nodes', node_rows), ('decoded-edges', edge_rows),
                         ('gt-recovery', gt_rows), ('comparison', comparison), ('per-subject', subjects), ('ranking', ranking)]:
        table(args.output/(name+'.csv'), values)
    save(args.output/'geometry.json', {f'{s}/{p}': rr for (s, p), rr in common.items()})
    result = dict(arm=args.arm, ids=ids, frozen_state_unchanged=True, cache_decode_parity=True,
                  elapsed_seconds=time.monotonic()-started, comparison=comparison, ranking=ranking,
                  caveats=['Oracle uses GT and is not a deployable decoder.',
                           'Positional edge/node F1 is distinct from native box benchmark F1.',
                           'Parent versus refinement includes node-head continuation; baseline continuation is the causal control.',
                           'Validation is a reused development set. No new test-set claim.',
                           'Foreground/GT agreement does not establish anatomical accuracy.'])
    save(args.output/'summary.json', result)
    lines = ['# IXI frozen-query diagnostic', '', '|Split|Tolerance mm|Policy|Node recall|Edge F1|U fraction of FP|β0 MAE|β1 MAE|',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in comparison:
        def fmt(k):
            return 'NA' if r[k] is None else f'{r[k]:.4f}'
        lines.append(f"|{r['split']}|{r['tolerance_mm']}|{r['policy']}|"+'|'.join(fmt(k) for k in ['node_recall', 'f1', 'unmatched_fraction_of_fp', 'beta0_mae', 'beta1_mae'])+'|')
    (args.output/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(arm=args.arm, elapsed=result['elapsed_seconds'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['parent', 'baseline', 'local_image'], default='local_image')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--full-validation', action='store_true')
    run(parser.parse_args())
