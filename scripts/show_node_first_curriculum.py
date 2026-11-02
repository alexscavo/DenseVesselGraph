"""Read saved L08 status/results using only the Python standard library."""
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
meta = repo / 'experiments/node_first_curriculum_20261028'
p = json.loads((meta / 'plan.json').read_text())
root = Path(p['output_root'])
print('L08 outputs:', root)
for stage in ['pilot', 'full']:
    jobs = meta / (stage + '-jobs.json')
    if jobs.exists():
        print(stage + ' submissions:', jobs.read_text())
    report = root / stage / 'report.md'
    if report.exists():
        print(report.read_text())
    elif jobs.exists():
        print(stage + ': results pending; this reader does not query Slurm or run inference.')
        for path in sorted((root / stage).glob('*/*/progress.json')):
            row = json.loads(path.read_text())
            print(path.parent.relative_to(root), 'update', row['step'], 'edge multiplier', row['edge_multiplier'])
