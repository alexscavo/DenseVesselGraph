#!/usr/bin/env python3
"""Collect saved smoke artifacts and append the existing report; no model replay."""
import fcntl
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native2d_20261027'
if (OUT/'worktree.json').exists():OUT=Path(json.loads((OUT/'worktree.json').read_text())['output'])
MAIN=ROOT.parent/'Graph-Native-Betti-Matching-Representation'


def append(path,text,marker):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_name(path.name+'.native2d.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        content=path.read_text()
        if marker not in content:
            with path.open('a') as stream:stream.write('\n\n'+marker+'\n'+text+'\n')


def main():
    checks=json.loads((OUT/'cuda-checks.json').read_text()) if (OUT/'cuda-checks.json').exists() else None
    tiny=json.loads((OUT/'tiny-summary.json').read_text()) if (OUT/'tiny-summary.json').exists() else None
    full=json.loads((OUT/'full-architecture-check.json').read_text()) if (OUT/'full-architecture-check.json').exists() else None
    submission=json.loads((OUT/'submission.json').read_text())
    gate=bool(checks and checks['passed'] and full and full['passed'] and tiny and tiny['memorization_gate_passed'])
    lines=['### Native 2D smoke collection — 27 October 2026','',
        f"GPU correctness: **{'passed' if checks else 'incomplete/failed'}**; original-size architecture: **{'passed' if full else 'incomplete/failed'}**; tiny memorization gate: **{'passed' if gate else 'not passed'}**.",
        f"Slurm GPU `{submission['gpu']}`, collector `{submission.get('collector','pending')}`. No full training was authorized or submitted.",'']
    if tiny:
        lines+=['Two training crops only; the table is a memorization diagnostic, **not validation performance**.','',
            '| Update | Loss | Crop | Node F1 @2px | Edge F1 @2px | Branch F1 | Curve F1 |','|---:|---:|---|---:|---:|---:|---:|']
        for row in tiny['history']:
            for name,metric in zip(tiny['samples'],row['samples']):
                lines.append(f"| {row['step']} | {row['losses']['total']:.4f} | {name} | {metric['2.0']['node']['f1']:.4f} | {metric['2.0']['edge']['f1']:.4f} | {metric['common_v2']['common_branch_f1']:.4f} | {metric['common_v2']['common_curve_f1']:.4f} |")
        lines+=['',f"{tiny['steps']} updates in {tiny['runtime_seconds']:.1f}s; peak GPU allocation {tiny['peak_gpu_bytes']} bytes. Checkpoint: `{tiny['checkpoint']}`."]
    lines+=['', 'Passing this gate establishes implementation learnability on two crops only. It supplies no cross-subject generalization, baseline-versus-A-xyz benefit or journal reconstruction result. Full training remains pending subject/provenance/crossing and operating-point protocol decisions.' if gate else
        'Do not launch full training. Inspect the saved loss curves and GPU log before changing budgets, initialization or supervision. A failed/partial smoke is an implementation/feasibility result, not evidence against continuous 2D graph reconstruction.',
        '',f"Evidence: `{OUT}`. Retrieval: `python3 scripts/show_native2d.py` from the main repository."]
    report='\n'.join(lines)+'\n';(OUT/'summary.md').write_text(report)
    (OUT/'collection.json').write_text(json.dumps(dict(completed=bool(tiny),all_gates_passed=gate,
        gpu_checks=bool(checks),full_architecture=bool(full),full_training_submitted=False),indent=2)+'\n')
    append(MAIN/'research/image-to-graph-vessel.md',report,'<!-- NATIVE2D COLLECTED 20261027 -->')
    append(MAIN/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md',
        f"### Native2D-01 smoke outcome — 27 October 2026\n\nCorrectness/tiny-fit gate {'passed' if gate else 'incomplete or failed'}; no full training. See the [existing 2D report](image-to-graph-vessel.md#native-2d-smoke-collection--27-october-2026) and `{OUT}/collection.json`.",
        '<!-- NATIVE2D REGISTRY COLLECTED 20261027 -->')
    claim=ROOT.parent/'experiment_gpu_coordination/native2d-20261027.json'
    data=json.loads(claim.read_text());data.update(state='smoke collected; full training not launched',concurrent_h100=0)
    claim.write_text(json.dumps(data,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
