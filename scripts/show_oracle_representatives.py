"""Print saved representative-oracle results; stdlib only."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'experiments/oracle_representatives_20261026'
s=json.loads((root/'summary.json').read_text())
for seed in [364505,92741]:
    print(f'\nA-xyz seed {seed}, 480 validation patches, 2 mm:')
    for condition in ['normal','nearest','confidence','confidence_nms_1mm','matched_count_confidence']:
        r=next(r for r in s['graphs'] if r['seed']==seed and r['split']=='val' and r['tolerance']==2 and r['condition']==condition)
        print(f'  {condition:26s} node recall={r["node_recall"]:.4f} edge F1={r["edge_f1"]:.4f} TP={r["edge_tp"]} FP={r["edge_fp"]} conditional recall={r["conditional_recall"]:.4f}')
    r=next(r for r in s['neighbors'] if r['seed']==seed and r['split']=='val' and r['tolerance']==2)
    print(f'  Same-neighbor AP: {r["nearest_ap"]:.4f} -> {r["confidence_ap"]:.4f}; delta CI {r["ap_difference"]["ci95"]}')
print('\nBest-supported mechanism: representative/query-role consistency and relation-relevant token content; not better localization.')
print('Next diagnostic priority: GT-free relation-aware query selection. Oracles are not deployable results.')
print('Report: '+str(root.parents[2]/'Graph-Native-Betti-Matching-Representation/docs/research/metrics/6-ixi-query-attribution.md'))
print('Evidence: '+str(root))
