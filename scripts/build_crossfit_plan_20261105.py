#!/usr/bin/env python3
"""Cross-fitting folds for the learned node scorer (5 Nov 2026).

The scorer must learn from predictions that look like predictions on unseen data, not from a model's own training
patches (where its heatmap and edge scores are overconfident). So, per dataset D (default recipe, <D>-loops):
the training SUBJECTS are split in two by a fixed hash (seed 364505); fold k trains on half k with the usual recipe;
its out-of-fold list (oof_ids) is the other half's training patches, which it never saw. The scorer is trained on the
union of both folds' out-of-fold predictions; the deployed model (trained on all of D) is then decoded with it.
Each fold keeps D's validation list (for its own final.json only) and draws its monitor from its own half.
Adds plan entries <D>-fold0, <D>-fold1 (existing entries untouched; backup already written by the loop A/B builder).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/multidataset_continuous/plan.json'
DATASETS = ('IXI-loops', 'TopBrainCTA-loops')


def rank(x):
    return hashlib.sha256(f'364505/{x}'.encode()).hexdigest()


def main():
    plan = json.loads(PLAN.read_text())
    for key in DATASETS:
        base = plan['datasets'][key]
        subjects = sorted({s.split('_')[1] for s in base['train_ids']}, key=rank)
        halves = [set(subjects[0::2]), set(subjects[1::2])]
        for k in (0, 1):
            train = sorted(s for s in base['train_ids'] if s.split('_')[1] in halves[k])
            oof = sorted(s for s in base['train_ids'] if s.split('_')[1] in halves[1 - k])
            name = key.replace('-loops', f'-fold{k}')
            plan['datasets'][name] = dict({f: base[f] for f in ('root', 'manifest_sha256', 'val_ids', 'monitor_val', 'full_train_count', 'full_val_count')},
                                          train_ids=train, monitor_train=sorted(train, key=rank)[:256], oof_ids=oof,
                                          max_target_nodes=base['max_target_nodes'], parent=key,
                                          design=f'cross-fit fold {k} of {key} (subject halves); scorer training data = predictions on oof_ids')
            print(name, 'train', len(train), 'subjects', len(halves[k]), '| oof', len(oof), 'subjects', len(halves[1 - k]))
    PLAN.write_text(json.dumps(plan, indent=1) + '\n')


if __name__ == '__main__':
    main()
