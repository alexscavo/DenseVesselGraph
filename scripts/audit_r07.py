"""Cached CPU audit of the actual modern matched-pair supervision population."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.analyze_graph_errors import cases, load_case, geometry, match_hungarian, write, csvwrite
from scripts.r07_sampling import population

OUT = Path(__file__).resolve().parents[1] / 'experiments/r07_matched_supervision_20261027'


def main():
    rows = []
    for case in cases():
        if case['dataset'] != 'SyntheticMRI' or case['arm'] != 'A-xyz' or case['phase'] != 'final':
            continue
        for record, c in load_case(case):
            gt = geometry(record['gt']); source, target, error = match_hungarian(c, gt, case['config'])
            assert error is None or error < 2e-4
            p = population(c, gt, source, target)
            n = p['labels'] == 0; pos = ~n
            row = dict(seed=case['seed'], **record, positives=int(pos.sum()), negatives=int(n.sum()),
                       near_negatives=int((n & (p['separation'] < 5.)).sum()),
                       reliable_near_negatives=int(p['hard'].sum()),
                       strict_1mm_near_negatives=int((p['hard'] & (p['endpoint_own_mm'] <= 1.).all(1)).sum()),
                       ambiguous_negatives=int((n & p['ambiguous']).sum()),
                       potential_conflicting_negatives=int(p['potential_conflict'].sum()),
                       distant_negatives=int((n & (p['separation'] >= 20.)).sum()),
                       reliable_assigned_nodes=int(p['reliable'].sum()), assigned_nodes=len(source),
                       cap=case['config']['loss']['edge']['candidates']['max_per_graph'],
                       positive_arc_mm=[v['arc_mm'] for v in gt['geometry']],
                       positive_chord_mm=p['separation'][pos].tolist())
            row['all_negatives_enter_base_objective'] = row['positives'] + row['negatives'] <= row['cap']
            rows.append(row)
    aggregate = []
    for seed in [364505, 92741]:
        for split in ['train', 'val']:
            rr = [r for r in rows if r['seed'] == seed and r['split'] == split]
            keys = ['positives','negatives','near_negatives','reliable_near_negatives','ambiguous_negatives',
                    'potential_conflicting_negatives','distant_negatives','reliable_assigned_nodes','assigned_nodes']
            aggregate.append(dict(seed=seed, split=split, patches=len(rr), subjects=len({r['subject'] for r in rr}),
                                  all_negatives_used=all(r['all_negatives_enter_base_objective'] for r in rr),
                                  patches_with_reliable_hard=sum(r['reliable_near_negatives'] > 0 for r in rr),
                                  **{k:sum(r[k] for r in rr) for k in keys}))
    write(OUT/'supervision-audit.json', dict(completed=True, aggregate=aggregate, per_patch=rows,
          reliability='Each Hungarian query within 2mm of assigned GT, own GT nearest, next-GT distance margin >=0.5mm; negative GT chord <5mm. Stricter 1mm sensitivity also counted.',
          caveat='Possible conflicts are geometric ambiguity flags, not proven erroneous GT labels. Frozen clean eval-mode snapshots, not stochastic augmented training history.'))
    csvwrite(OUT/'supervision-audit-per-patch.csv.gz', [{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows])
    print(aggregate)

if __name__ == '__main__': main()
