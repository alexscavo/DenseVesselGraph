"""Print saved study status/results using only the standard library."""
import json
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
p=repo/'experiments/continuous_representation/submission.json'
if p.exists():
 s=json.loads(p.read_text());print(json.dumps(s['jobs'],indent=2));root=Path(s['output_root'])
 b=root/'budget.json'
 print('Budget:', b.read_text() if b.exists() else 'Pending GPU benchmark')
 for name in ['gt-report.md','learned-report.md']:
  path=root/'results'/name
  print('\n'+(path.read_text() if path.exists() else name+': pending'))
else:print('No recorded submission.')
print('\nFull methodology:',repo/'docs/research/continuous_representation_comparison.md')
