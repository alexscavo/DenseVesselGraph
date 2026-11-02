"""Print completed N08 results, stdlib only."""
import json
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'experiments/gt_free_query_selection_20261026';s=json.loads((out/'summary.json').read_text())
for seed in [364505,92741]:
 print(f'\nSyntheticMRI A-xyz seed {seed}, validation 2 mm:')
 for method in ['normal','confidence_nms','local_profile','naive_global','guarded_global','protected_nms']:
  r=next(r for r in s['summary'] if r['seed']==seed and r['split']=='val' and r['tolerance']==2 and r['method']==method)
  print(f'  {method:18s} EdgeF1={r["edge_f1"]:.6f} nodeR={r["node_recall"]:.4f} TP={r["edge_tp"]} FP={r["edge_fp"]} nodes={r["nodes"]} cycle bias/patch={r["beta1_signed_error_mean"]:+.3f}')
print('\nConsistent count-controlled benefit:',s['decision']['consistent_count_controlled_benefit'])
print('Stopped further search; no GPU, training or architecture changes.')
print('Evidence:',out)
