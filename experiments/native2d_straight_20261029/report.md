## Native2D-05 — original training with straight edges, 29 October 2026

Requested straight-edge control on OCTA-Synth B0: same full native2D detector, initialization seed364505, validated data/splits,50:50 mixed plant/vessel sampling,50 mixed epochs, then100 vessel-only epochs. Same node focal gamma2, matched-edge CE ratio0.15, Hungarian matching, optimizer, global gradient clipping, augmentation, query count96, thresholds0.5 and validation/checkpoint policy. Only the spline head and spline loss are disabled. Original completed curved experiments are preserved.

The existing training loop is reused unchanged through process-local configuration/output bindings. Exact source/target exposure hashes are checked against the completed curved B0 protocols. A fixed full-model tiny-overfit and production-batch finite-gradient smoke gates the long jobs. No extra sweep or automatic extension.

Jobs: `{"global_h100_limit": 2, "concurrent_h100": 1, "queue_at_submission": "", "prior_gpu_jobs": [], "gate": "1699959", "pretrain": "1699960", "finetune": "1699962", "collector": "1699963", "gate_completed": true}`.

| Stage | State | Epoch |
|---|---|---:|
| octasynth_mixedpretrain_50 | complete | 50 |
| octasynth_mixed_B0 | complete | 100 |

| Model, final100 | Node mAP | Edge mAP | Node F1 @2px | Edge F1 @2px | Curve F1, positive-crop macro |
|---|---:|---:|---:|---:|---:|
| Curved K8 B0 | 0.8049 | 0.4447 | 0.8038 | 0.5950 | 0.6285 |
| Straight-only B0 | 0.7992 | 0.5877 | 0.7932 | 0.5903 | 0.6036 |

The straight run retains the existing evaluator schema: its legacy `spline` metric slot also evaluates straight chords because no spline head exists; `chord` is the unambiguous straight geometry result. Dense GT paths and evaluation tolerances remain unchanged. The original global clipping rule is kept, including its interaction with the presence/absence of spline gradients.

Artifacts: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_straight_20261029`. Checkpoints/epoch logs and curves: `runs/`; exact sample exposures: `exposure.npy`; gate: `gates.json`; configuration previews: `*.yaml`; comparison parity: `parity-*.json`. Test sets untouched.

Curvature measurement is limited to the two OCTA validation crops shown previously, not the full validation population. Maximum perpendicular deviation from each endpoint chord is measured in pixels using257 spline samples.

| Model | Edges | Median / p95 bend, px | Fraction below0.25px | Median arc/chord |
|---|---:|---|---:|---:|
| B0 | 37 | 0.0006 / 1.7439 | 89.2% | 1.000000 |
| Axyz | 31 | 0.0006 / 0.5755 | 93.5% | 1.000000 |
| GT_dense | 33 | 0.0000 / 2.0036 | 90.9% | 1.000000 |
