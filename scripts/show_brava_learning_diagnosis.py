"""Display saved diagnostic results; stdlib only, no inference/training."""
import json
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'experiments/brava_learning_diagnosis_20261026'
if not (out/'analysis.json').exists():
 print('Results not collected yet. Submission:',(out/'submission.json').read_text() if (out/'submission.json').exists() else 'not submitted');raise SystemExit(0)
s=json.loads((out/'analysis.json').read_text())
for r in s['summaries']:
 if r['step']=='4000' and r['tolerance']==2 and r['population'] in ['representative_train','validation']:
  print(f'{r["population"]:22s} {r["condition"]:20s} rawR={r["raw_recall"]:.4f} filteredR={r["node_recall"]:.4f} nodeF1={r["node_f1"]:.4f} edgeF1={r["edge_f1"]:.4f} Hungarian={r["hungarian_mm"]:.3f}mm')
print('Report:',out.parents[2]/'Graph-Native-Betti-Matching-Representation/docs/research/BRAVA_experiment_plan.md')
print('Evidence:',out)
