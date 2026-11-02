# Native 2D straight-edge training control — 29 October 2026

The user requested an actual model trained with straight edges in the original way, matching the completed native2D jobs. This is an OCTA-Synth **B0** control:50 mixed plants+OCTA training epochs, then100 OCTA-only epochs. The spline head is disabled and spline loss removed. The detector architecture, initial shared weights/seed364505, image/graph manifests,96 object queries,50:50 replacement sampler and twice-source pretraining epoch, D4 augmentation, focal-node gamma2, matched-edge CE ratio0.15, Hungarian matching, optimizer, global clipping0.1, schedules, batch32/4,0.5 confidence cutoffs and held-out evaluation policy are unchanged. Dense GT paths are retained for evaluation; no graph labels are replaced or simplified.

The existing `scripts/train_native2d_long.py` train loop is reused unchanged through process-local config/output/hash bindings in `scripts/train_native2d_straight_20261029.py`. These bindings do not modify the original files or completed curved experiments. Both straight pretraining and straight finetuning run from their own weights; the straight finetune does not import the curved checkpoint. Common detector initialization is checked, and pretrain/finetune sample-exposure hashes match the completed curved B0 protocols exactly. The initialization hash excludes the absent spline head. The original global clipping rule is preserved.

Worktree: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/GNBM/gnbm-native-2d-20261027`. New outputs: `experiments/native2d_straight_20261029`. Curved reference: `experiments/native2d_mixed_20261028/runs/octasynth_mixed_B0` and its corresponding mixed-pretraining stage. Epoch budgets are80,850 pretraining updates and266,200 target updates, with no extension or sweep. Test sets stay untouched.

Submitted jobs: correctness/tiny-overfit/production-batch gate **1699959** (one H100,10min maximum), mixed pretraining **1699960**, vessel finetuning **1699962** (one H100,14h maximum each), then saved-results collector **1699963**. Training stages use `afterok` dependencies and stop if the gate or previous stage fails. Only one H100 is used concurrently by this chain; the shared queue/lock is checked against the global ceiling of two. The current saved status, learning curves and checkpoints are in the new output directory; `report.md` is updated by the collector. Submission is not a completed result.

Gate1699959 completed successfully in1m18s: full-model two-crop node/edge F1@2px both1.0 after the fixed1,200 updates; real batch32 mixed and batch4 vessel-only backward smoke passed with finite gradients and no spline loss. Exact shared-core initialization and original data-exposure hashes were verified. Pretraining1699960 is queued; finetuning1699962 follows automatically. Gate weights are discarded.

## Measurement of the displayed OCTA edges

The earlier visualizations contain two validation crops selected by GT node counts. Reusing their saved predictions, sample each K8 spline at257 points and measure maximum perpendicular deviation from its endpoint chord, plus arc/chord length ratio. This measures shape directly, rather than inferring curvature from Curve F1. These numbers apply **only to the displayed crops**, not all validation predictions:

| Geometry source | Edges | Median maximum bend, pixels | 95th percentile, pixels | Fraction bending less than0.25px |
|---|---:|---:|---:|---:|
| B0 predictions |37|0.000606|1.744|89.2%|
| A-xyz predictions |31|0.000604|0.576|93.5%|
| Dense supplied GT paths |33|0|2.004|90.9%|

Median arc/chord ratio is approximately1.000000004 for both predicted models and1.0 for GT. A few edges do bend by1–3 pixels, but the majority of edges in those figures are effectively straight. The supplied graph paths are also predominantly straight in these two crops. This supports the user's visual observation within that limited sample, without establishing a full-dataset curvature distribution.

Exact measurement output: `experiments/native2d_straight_20261029/displayed-octa-bending.json`; source controls, graph membership and selected crop IDs remain in the completed campaign's `visualizations_20261029/selected-predictions.json` and `selection.json`. Original configurations and source snapshots are retained in both campaigns.
