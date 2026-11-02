"""Read completed graph attribution with stdlib only; no model work."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'experiments/graph_error_attribution_20261026'
p=json.loads((root/'summary.json').read_text())
print('Numerical evidence:',root)
for r in p['aggregate']:
 if r['split']!='val' or r['tolerance']!=2:continue
 if not (r['case'].endswith('/final') or r['case']=='IXI-legacy/local_image'):continue
 endpoint=1-r.get('localized_relation_miss',0)/r['edge_fn']
 print(r['case'],f"edge F1={r['edge_f1']:.4f}; unmatched-endpoint FP={100*r['false_edge_unmatched_fraction']:.2f}%; endpoint-related FN={100*endpoint:.2f}%")
print('Report: Graph-Native-Betti-Matching-Representation/docs/research/metrics/6-ixi-query-attribution.md (26 October section)')
if (root/'integration-pending.json').exists():print('Shared-report integration is pending; see',root/'integration-pending.json')
