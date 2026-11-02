"""Read saved status/results without model, array libraries or Slurm polling."""
import json
from pathlib import Path
repo=Path(__file__).resolve().parents[1];record=repo/'experiments/multidataset_continuous/submission.json'
if record.exists():
 state=json.loads(record.read_text());print(json.dumps(state['jobs'],indent=2));root=Path(state['output_root'])
 if state.get('reporting_recovery'):print('Reporting recovery:',json.dumps(state['reporting_recovery'],indent=2))
 for arm in json.loads((root/'plan.json').read_text())['arms']:
  p=root/'runs'/arm['name']/'complete.json';print(arm['name'],p.read_text() if p.exists() else 'pending or in progress')
 report=root/'results/report.md';print(report.read_text() if report.exists() else 'Aggregate report pending')
else:print('No submitted batch recorded')
print('Main report:',repo/'docs/research/multidataset_continuous_graph_experiments.md')
