#!/usr/bin/env python3
"""Collect the fixed-step, paired node refinement comparison."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_node_refinement_20261018'));p.add_argument('--output',type=Path,default=Path('docs/research/evidence/ixi_node_refinement_20261018'));a=p.parse_args()
arms=['baseline','local_image','zero_image'];data={k:json.loads((a.root/k/'summary.json').read_text()) for k in arms}
assert len({d['provenance']['batch_order_sha256'] for d in data.values()})==1,'Training batches differed'
assert all(d['provenance']['frozen_state_sha256']==d['provenance']['final_frozen_state_sha256'] for d in data.values())
assert all(d['initial']['validation']==data['baseline']['initial']['validation'] for d in data.values()),'Initial model scores differed'
a.output.mkdir(parents=True,exist_ok=True);rows=[]
for arm,d in data.items():
 for stage in ('initial','final'):
  report=d[stage];rel=report['relations']['2.0'];m=report['validation'];types=report['node_subtypes']['val']['2.0']
  rows.append(dict(arm=arm,stage=stage,step=report['step'],node_precision_2mm=rel['node_precision'],node_recall_2mm=rel['node_recall'],endpoint_pair_recovery=rel['endpoint_pair_recovery'],conditional_relation_recall=rel['conditional_direct_recall'],node_f1=m['native_node_f1'],edge_f1=m['native_edge_f1'],curve_f1=m['common_curve_f1'],**{k+'_recall_2mm':v['recovered']/v['total'] if v['total'] else None for k,v in types.items()}))
with (a.output/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.output/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
lines=['# IXI node refinement: fixed 1000-update comparison','','Matched batch-order hashes and identical initial validation scores verified. Frozen state unchanged in each arm.','','| Arm/stage | Node recall 2mm | Node precision 2mm | Node F1 | Edge F1 | Curve F1 |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append('|'+r['arm']+'/'+r['stage']+'|'+'|'.join(f"{r[k]:.4f}" for k in ['node_recall_2mm','node_precision_2mm','node_f1','edge_f1','curve_f1'])+'|')
lines+=['','Compare local_image against BOTH baseline and zero_image. Improvement over zero_image is needed to attribute benefit to local image evidence. These validation pilots do not establish test generalization. Relation weights stay frozen; changing selected nodes changes decoded edges.','',f'Full predictions, fixed-example PNGs, W&B offline runs and small trained-head checkpoints: `{a.root}`.','']
(a.output/'comparison.md').write_text('\n'.join(lines))
print(a.output/'comparison.md')
