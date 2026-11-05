#!/usr/bin/env python3
"""Plan entries for the loop A/B experiment (5 Nov 2026): train without vs with loop patches.

Adds, per dataset D in {IXI, SyntheticMRI, TopBrainMRA, TopBrainCTA}, two entries to the dense model's plan
(experiments/multidataset_continuous/plan.json; existing entries untouched, backup written first):
- D-noloops (arm A): training and monitor patches drawn only from patches without loops (no inserted anchor),
  i.e. what every cache before 5 Nov 2026 contained;
- D-loops (arm B): drawn from all patches, loop patches encoded with breaking-point anchors.
Both read the same loop cache and share ONE complete validation list (every validation patch, loops included),
scored with the loop-aware protocol (metrics/loop_aware.py). Sizes: IXI and SyntheticMRI 4096 training patches
(as the existing IXI/SyntheticMRI entries), TopBrain all training patches; monitor 256 from the arm's own training
list (thresholds are chosen there only), monitor_val 128. Selection is a fixed hash rank (seed 364505), so arm B's
list is arm A's ranking with loop patches allowed in. Test patches are never listed.
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'
D = Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets')
CACHES = {
    'IXI': (D / 'IXI/transfer_splines_k8_loops_20261105', 4096),
    'SyntheticMRI': (D / 'syntheticMRI/transfer_source_k8_25k_loops_20261105', 4096),
    'TopBrainMRA': (D / 'TopBrain_tight_20261105/transfer_splines_k8_loops_20261105/mr', None),
    'TopBrainCTA': (D / 'TopBrain_tight_20261105/transfer_splines_k8_loops_20261105/ct', None),
}


def rank(sample_id):
    return hashlib.sha256(f'364505/{sample_id}'.encode()).hexdigest()


def main():
    plan = json.loads(PLAN.read_text())
    backup = PLAN.with_name('plan.before-loop-ab-20261105.json')
    if not backup.exists():
        shutil.copy2(PLAN, backup)
    summary = {}
    for name, (root, n_train) in CACHES.items():
        manifest = root / 'manifest.json'
        rows = list(csv.DictReader((root / 'patch_index.csv').open()))
        loops = {r['sample_id'] for r in rows if int(r.get('inserted_count', 0)) > 0}
        train = sorted((r['sample_id'] for r in rows if r['split'] == 'train'), key=rank)
        val = sorted(r['sample_id'] for r in rows if r['split'] == 'val')
        monitor_val = sorted(val, key=rank)[:128]
        nodes = {r['sample_id']: int(r['anchor_count']) for r in rows}
        for arm, pool in (('noloops', [s for s in train if s not in loops]), ('loops', train)):
            ids = pool[:n_train] if n_train else pool
            if n_train and len(ids) < n_train:
                raise ValueError(f'{name}-{arm}: only {len(ids)} training patches')
            key = f'{name}-{arm}'
            plan['datasets'][key] = dict(
                root=str(root), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                train_ids=sorted(ids), val_ids=val, monitor_train=sorted(ids, key=rank)[:256], monitor_val=monitor_val,
                max_target_nodes=max(nodes[s] for s in ids + val), full_train_count=len(train), full_val_count=len(val),
                loop_training_patches=sum(s in loops for s in ids), loop_validation_patches=sum(s in loops for s in val),
                design='loop A/B 2026-11-05: ' + ('training without loop patches (arm A)' if arm == 'noloops'
                                                  else 'training with loop patches, breaking-point anchors (arm B)')
                       + '; shared complete validation list; scripts/build_loop_ab_plan_20261105.py')
            summary[key] = {k: plan['datasets'][key][k] if not isinstance(plan['datasets'][key][k], list)
                            else len(plan['datasets'][key][k]) for k in ('train_ids', 'val_ids', 'monitor_train',
                            'loop_training_patches', 'loop_validation_patches', 'max_target_nodes')}
    PLAN.write_text(json.dumps(plan, indent=1) + '\n')
    (ROOT / 'experiments/loop_subdivision_20261105/ab_plan_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    for k, v in summary.items():
        print(k, v)


if __name__ == '__main__':
    main()
