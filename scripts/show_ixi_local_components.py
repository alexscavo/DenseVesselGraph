"""Read study artifacts with stdlib only, from any current directory."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
meta=root/'experiments/ixi_local_components_20261026'
p=json.loads((meta/'plan.json').read_text());output=Path(p['output_root'])
print('Outputs:',output)
jobs=json.loads((meta/'jobs.json').read_text()) if (meta/'jobs.json').exists() else {}
audit=meta.parent/'n07_status_audit_20261028'/'status.json'
recovered=False
if audit.exists():
 status=json.loads(audit.read_text())
 recovered=bool(jobs.get('jobs')) and jobs['jobs']['smoke']['id'] not in status['replacement_jobs']
 if recovered:
  print('Recovery submitted:',jobs.get('submitted_at'),'— correctness gate and results pending.')
  print('Historical failure:',status['recorded_error'],'; earlier audit:',status['audited_at'])
 else:
  print('Execution audit as of',status['audited_at'])
  print('Scientific comparison:',status['scientific_training'],';',status['status'])
  print('Replacement job states:',status['replacement_jobs'])
  print('Smoke blocker:',status['recorded_error'])
  print('Next step:',status['next_run'])
 print('Audit:',audit)
if (meta/'jobs.json').exists():print((meta/'jobs.json').read_text())
if (output/'results.md').exists():print((output/'results.md').read_text())
else:
 for path in [output/'smoke.json']+sorted(output.glob('runs/*/progress.json'))+sorted(output.glob('runs/*/complete.json')):
  if path.exists():print(path.relative_to(output),path.read_text())
 print(('Scientific jobs did not run; see the dated execution audit.' if audit.exists() and not recovered else 'Results are not collected yet.')+' This command performs no inference or training.')
