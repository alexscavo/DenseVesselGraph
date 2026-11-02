#!/usr/bin/env python3
"""Straight-edge RelationFormer lineage: the B0 lineage retrained with no geometry objective.

Each stage reruns the frozen code snapshot that produced the corresponding B0 model, with
one change: loss.weights.spline = 0. TransferCriterion then returns a zero spline loss, the
zero-initialised spline/local-corrector output layers never move, and every decoded edge is
its exact endpoint chord (the project's 3D "straight" convention, cf. train_ixi_transfer).
Data, seeds, schedules, budgets, query expansion, checkpoint selection and evaluation are
those of B0. Gradient diagnostics that require spline gradients are replaced by checks that
the spline head receives none.

Stages (run from the stage's code snapshot directory, which must be passed as --code):
  s1  SyntheticMRI from scratch, fine_feature_architecture baseline (100 epochs, val-selected best.pt)
  s2  local_refinement_long B0 arm, seed 92741, 12k updates from s1 best.pt
  s3  IXI, ixi_long_20261030 recipe (100 epochs x 32) from the selected s2 checkpoint
  s4  BRAVA, D08 brava_long recipe (300 epochs x 32) from the selected s2 checkpoint
The s2 checkpoint for s3/s4 follows the D08 rule (highest SyntheticMRI validation edge F1 at
2 mm, ties to the later step) over this lineage's s2 checkpoints. No test data is read.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching-Representation')
D08 = Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/GNBM/gnbm-brava-long-20261027/experiments/brava_long_20261027')
LARGE = Path(os.environ['SCRATCH'])/'experiments/gnbm/straight_lineage_20261101'
RECORD = REPO/'experiments/straight_lineage_20261101'
OUT = {'s1': LARGE/'s1', 's2': LARGE/'s2', 's3': RECORD/'ixi', 's4': LARGE/'brava'}
S2_SEED = 92741


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def straight(cfg):
    cfg = copy.deepcopy(cfg)
    assert cfg['model']['spline']['enabled'] and float(cfg['loss']['weights']['spline']) > 0
    cfg['loss']['weights']['spline'] = 0.
    cfg['experiment']['name'] = cfg['experiment']['name']+'_straight'
    return cfg


def protocol(stage, plan, **extra):
    """Write the stage plan once; a rerun must find the identical plan (resume, never overwrite)."""
    path = OUT[stage]/'plan.json'
    if path.exists():
        assert read(path) == plan, f'{path} differs from the plan this stage would write'
    else:
        write(path, plan)
    write(OUT[stage]/'straight-stage.json', dict(stage=stage, code=os.getcwd(), plan_sha256=sha(path),
          change='loss.weights.spline 0 (B0 value replaced); everything else unchanged', **extra))


def no_spline_gradient(model):
    import torch
    grads = [p.grad for n, p in model.named_parameters() if n.startswith('spline_head.')]
    assert grads and all(g is None or not bool(torch.count_nonzero(g)) for g in grads), 'Spline head received a gradient'


def straight_diagnostic(prior):
    """fine_feature_architecture.diagnostic without its spline-gradient requirement."""
    import torch
    from scripts.multiscale_pretrained import gradients, levels

    def diagnostic(model, losses, step):
        g = gradients(model)
        for prefix in ['node_refinement.']:
            values = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            if values:
                g[prefix] = float(sum(v.square().sum() for v in values).sqrt())
        assert all(g.get(k, 0) > 0 for k in ['encoder.', 'decoder.', 'class_embed.', 'coord_embed.', 'relation_embed.'])
        no_spline_gradient(model)
        return dict(step=step, losses={k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1},
                    gradients=g, attention=levels(model))
    prior.diagnostic = diagnostic
    prior.reconstruction = lambda *args, **kwargs: None  # spline-geometry diagnostic; chords by construction


def straight_update(module):
    """Recipe update unchanged; the step-1 diagnostic asserts a spline gradient, so check the opposite."""
    original = module.update

    def update(model, *args, diagnostic=False, **kwargs):
        result = original(model, *args, diagnostic=False, **kwargs)
        if diagnostic:  # gradients of this step are still held until the next zero_grad
            no_spline_gradient(model)
            result['spline_head_gradient_none'] = True
        return result
    module.update = update


def stage_s1():
    import scripts.fine_feature_architecture as ff
    ff.ROOT = OUT['s1']
    plan = read('experiments/fine_feature_architecture/plan.json')
    plan['config'] = straight(plan['config'])
    plan['straight'] = 'B0 baseline recipe with loss.weights.spline = 0'
    protocol('s1', plan)
    straight_diagnostic(ff)
    if not (ff.ROOT/'smoke.json').exists() or not read(ff.ROOT/'smoke.json')['completed']:
        ff.smoke()
    ff.baseline(100)


def stage_s2():
    import scripts.fine_feature_architecture as ff
    import scripts.local_refinement_long as lr
    source = OUT['s1']/'runs/baseline/best.pt'
    assert read(OUT['s1']/'baseline-gate.json')['passed']
    lr.ROOT, lr.SOURCE, lr.SOURCE_SHA, lr.ARMS = OUT['s2'], source, sha(source), ['B0']
    plan = read('experiments/local_refinement_long/plan.json')
    plan['config'] = straight(plan['config'])
    plan.update(source=str(source), source_sha256=lr.SOURCE_SHA, arms=['B0'], seeds=[S2_SEED],
                straight='B0 continuation recipe with loss.weights.spline = 0; B0 arm and seed 92741 only (the D08/IXI source seed)')
    protocol('s2', plan)
    straight_diagnostic(ff)
    if not (lr.ROOT/'smoke.json').exists() or not read(lr.ROOT/'smoke.json')['completed']:
        lr.smoke()
    lr.train('B0', S2_SEED)


def selected_source():
    """D08 rule over this lineage's s2 checkpoints: max validation edge F1 at 2 mm, ties -> later step."""
    run = OUT['s2']/str(S2_SEED)/'runs/B0'
    assert read(run/'complete.json')['completed']
    candidates = []
    for step in read(OUT['s2']/'plan.json')['checkpoints']:
        summary = read(run/f'step-{step}/summary.json')
        candidates.append(dict(path=str(run/f'model-{step}.pt'), seed=S2_SEED, step=step,
                               source_validation_edge_f1_2mm=summary['metrics']['val']['2.0']['edge_f1']))
    selected = max(candidates, key=lambda x: (x['source_validation_edge_f1_2mm'], x['step']))
    return dict(selected, sha256=sha(selected['path'])), candidates


def stage_s3(end_epoch):
    import scripts.ixi_long_experiment_20261030 as ix
    ix.OUT, ix.LARGE = OUT['s3'], LARGE/'ixi'
    source, candidates = selected_source()
    plan = read(REPO/'experiments/ixi_long_20261030/plan.json')
    plan['config'] = straight(plan['config'])
    plan.update(source=source, source_candidates=candidates, large_root=str(ix.LARGE),
                experiment='IXI literature-scale straight-edge RelationFormer (B0 recipe, loss.weights.spline = 0)')
    protocol('s3', plan)
    straight_update(ix)
    ix.train(end_epoch)


def stage_s4(smoke, end_epoch):
    import scripts.brava_long_experiment as bl
    import scripts.brava_long_evaluation as ev
    bl.OUT = ev.OUT = OUT['s4']
    source, candidates = selected_source()
    plan = read(D08/'plan.json')
    plan['config'] = straight(plan['config'])
    plan.update(source=source, source_candidates=candidates, large_root=str(OUT['s4']/'large'),
                experiment='BRAVA D08 straight-edge RelationFormer (B0 recipe, loss.weights.spline = 0)')
    protocol('s4', plan)
    straight_update(bl)
    for name in ['frozen-source-sha256.json']:  # train_chunk re-verifies every frozen file against this code copy
        if not (bl.OUT/name).exists():
            shutil.copy(D08/name, bl.OUT/name)
    if smoke:
        straight_smoke_s4(bl, ev, plan)
        return
    for epoch in range(25, end_epoch+1, 25):
        if not (bl.OUT/'large/runs/synthetic'/f'chunk-{epoch:03d}.json').exists():
            bl.train_chunk('synthetic', epoch)


def straight_smoke_s4(bl, ev, plan):
    """D08 gates for the one arm/batch used: preflight, 15 finite updates, strict resume, metric path."""
    import torch
    from torch.utils.data import DataLoader
    from data.loaders.common import image_graph_collate
    if not (bl.OUT/'cpu-preflight.json').exists():
        bl.cpu_preflight()
    selection = read(D08/'batch-selection.json'); size = selection['batch_size']
    cfg, model, criterion, opt, _ = bl.setup(plan, 'synthetic', size)
    loader = DataLoader(bl.SeededDataset(bl.base_dataset(plan, 'train', plan['ids']['train'], True)),
                        batch_sampler=bl.EpochBatches(size), num_workers=4, collate_fn=image_graph_collate)
    iterator = iter(loader); rows = []
    for number in range(1, 16):
        rows.append(bl.update(model, criterion, opt, next(iterator), number, len(loader), diagnostic=number == 1))
        no_spline_gradient(model)
    assert all(torch.isfinite(torch.tensor(r['total'])) for r in rows)
    folder = bl.OUT/'large/smoke/synthetic-b32'; folder.mkdir(parents=True, exist_ok=True)
    bl.save_checkpoint(folder/'resume.pt', plan, cfg, model, opt, 0, 15, arm='synthetic', batch_size=size)
    cp = torch.load(folder/'resume.pt', map_location='cpu', mmap=True)
    reference = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(cp['net'], strict=True); opt.load_state_dict(cp['optimizer'])
    assert all(torch.equal(reference[k], v) for k, v in model.state_dict().items())
    del cp, reference, iterator, loader
    cfg, model, criterion, opt, _ = bl.setup(plan, 'synthetic', size)
    ev.snapshot(plan, cfg, model, criterion, 'synthetic', -1, smoke=True)
    result = dict(selection, passed=True, source_sha256=plan['source']['sha256'], last=rows[-1],
                  batch_size_policy='B0 D08 batch selection reused unchanged (32); straight smoke on that arm/batch only',
                  job_id=os.environ.get('SLURM_JOB_ID'))
    write(bl.OUT/'gpu-smoke.json', result); write(bl.OUT/'batch-selection.json', result)
    print('S4 straight smoke passed', json.dumps(rows[-1]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['s1', 's2', 's3', 's4'])
    p.add_argument('--code', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--end-epoch', type=int)
    a = p.parse_args()
    assert Path.cwd().resolve() == a.code.resolve(), 'Run from the stage code snapshot'
    sys.path.insert(0, str(a.code))
    import torch
    torch.set_num_threads(1)
    if a.stage == 's1':
        stage_s1()
    elif a.stage == 's2':
        stage_s2()
    elif a.stage == 's3':
        stage_s3(a.end_epoch or 100)
    else:
        stage_s4(a.smoke, a.end_epoch or 300)
    print('STAGE COMPLETE', a.stage, flush=True)


if __name__ == '__main__':
    main()
