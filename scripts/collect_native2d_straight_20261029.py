"""Report the straight-only control using saved files; no inference or training."""
import fcntl
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native2d_straight_20261029'
MAIN=ROOT.parent.parent/'Graph-Native-Betti-Matching-Representation'

def read(path):return json.loads(path.read_text()) if path.exists() else None

def main():
    jobs=read(OUT/'submission.json') or {};ref=ROOT/'experiments/native2d_mixed_20261028'
    rows=['## Native2D-05 — original training with straight edges, 29 October 2026','',
        'Requested straight-edge control on OCTA-Synth B0: same full native2D detector, initialization seed364505, validated data/splits,50:50 mixed plant/vessel sampling,50 mixed epochs, then100 vessel-only epochs. Same node focal gamma2, matched-edge CE ratio0.15, Hungarian matching, optimizer, global gradient clipping, augmentation, query count96, thresholds0.5 and validation/checkpoint policy. Only the spline head and spline loss are disabled. Original completed curved experiments are preserved.','',
        'The existing training loop is reused unchanged through process-local configuration/output bindings. Exact source/target exposure hashes are checked against the completed curved B0 protocols. A fixed full-model tiny-overfit and production-batch finite-gradient smoke gates the long jobs. No extra sweep or automatic extension.','',
        f"Jobs: `{json.dumps(jobs)}`.",'','| Stage | State | Epoch |','|---|---|---:|']
    for name in ['octasynth_mixedpretrain_50','octasynth_mixed_B0']:
        s=read(OUT/'runs'/name/'status.json') or {}
        rows.append(f"| {name} | {s.get('state','queued')} | {s.get('epoch','—')} |")
    actual=read(OUT/'runs/octasynth_mixed_B0/final-heldout.json')
    reference=read(ref/'runs/octasynth_mixed_B0/final-heldout.json')
    rows+=['','| Model, final100 | Node mAP | Edge mAP | Node F1 @2px | Edge F1 @2px | Curve F1, positive-crop macro |','|---|---:|---:|---:|---:|---:|']
    for label,result,key in [('Curved K8 B0',reference,'spline_edge_positive_crop_macro'),('Straight-only B0',actual,'chord_edge_positive_crop_macro')]:
        if result:
            s=result['summary'];rows.append(f"| {label} | {s['node_mAP_crop_macro']:.4f} | {s['edge_mAP_crop_macro']:.4f} | {s['2.0']['node']['f1']:.4f} | {s['2.0']['edge']['f1']:.4f} | {s[key]['common_curve_f1']:.4f} |")
        else:rows.append(f'| {label} | pending | pending | pending | pending | pending |')
    rows+=['', 'The straight run retains the existing evaluator schema: its legacy `spline` metric slot also evaluates straight chords because no spline head exists; `chord` is the unambiguous straight geometry result. Dense GT paths and evaluation tolerances remain unchanged. The original global clipping rule is kept, including its interaction with the presence/absence of spline gradients.','',
        f'Artifacts: `{OUT}`. Checkpoints/epoch logs and curves: `runs/`; exact sample exposures: `exposure.npy`; gate: `gates.json`; configuration previews: `*.yaml`; comparison parity: `parity-*.json`. Test sets untouched.']
    bending=read(OUT/'displayed-octa-bending.json')
    if bending:
        rows+=['','Curvature measurement is limited to the two OCTA validation crops shown previously, not the full validation population. Maximum perpendicular deviation from each endpoint chord is measured in pixels using257 spline samples.','', '| Model | Edges | Median / p95 bend, px | Fraction below0.25px | Median arc/chord |','|---|---:|---|---:|---:|']
        for model,s in bending['results'].items():rows.append(f"| {model} | {s['edges']} | {s['median_max_lateral_deviation_px']:.4f} / {s['p95_max_lateral_deviation_px']:.4f} | {s['fraction_under_quarter_pixel']:.1%} | {s['median_arc_chord_ratio']:.6f} |")
    text='\n'.join(rows)+'\n';(OUT/'report.md').write_text(text)
    marker='<!-- NATIVE2D STRAIGHT 20261029 -->';end='<!-- END NATIVE2D STRAIGHT 20261029 -->';block=marker+'\n'+text+end
    for name in ['research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md','research/image-to-graph-vessel.md','research/experiments.md']:
        path=MAIN/name
        with Path(str(path)+'.native2d.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);old=path.read_text()
            if marker in old:a,b=old.split(marker,1);_,c=b.split(end,1);old=a+block+c
            else:old=old.rstrip()+'\n\n'+block+'\n'
            path.write_text(old)
    if actual and (read(OUT/'runs/octasynth_mixed_B0/status.json') or {}).get('state')=='complete':
        path=ROOT.parent.parent/'experiment_gpu_coordination/native2d-straight-20261029.json';claim=read(path)
        if claim:claim.update(state='completed straight control',concurrent_h100=0,reserved_after_dependency=0);path.write_text(json.dumps(claim,indent=2)+'\n')
    print(text)

if __name__=='__main__':main()
