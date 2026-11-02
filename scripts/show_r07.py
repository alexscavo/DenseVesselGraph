"""Display saved R07 results without importing ML packages or rerunning jobs."""
from pathlib import Path
import json
out=Path(__file__).resolve().parents[1]/'experiments/r07_matched_supervision_20261027'
report=out/'report.md'
if report.exists():print(report.read_text())
else:
    print('R07 results pending; no scientific outcome available yet.')
    if (out/'submission.json').exists():print(json.dumps(json.loads((out/'submission.json').read_text()),indent=2))
    print('Outputs:',out)
