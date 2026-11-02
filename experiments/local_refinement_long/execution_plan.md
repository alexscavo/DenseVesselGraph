# Longer local refinement study — protocol fixed 25 October 2026

Status: complete. All eight 12,000-update runs, two-seed paired analysis, image interventions and downstream consistency checks passed. Final collector1540969 completed. See the study report for conclusions; no IXI, C or A+C training launched.

## Question and bounded design

Does the image-conditioned local node head reproduce its benefit with longer
training, and does adding confidence correction help beyond coordinate correction?
Four arms, each under two fine-tuning seeds (364505 and92741): unchanged B0,
combined A, coordinate-only A-xyz, and combined no-image A-control. This is eight
runs, not a factorial sweep. All shared weights start from the clean SyntheticMRI
scratch baseline epoch95, SHA256 recorded in plan.json. No prior pilot weights.
The two seeds vary fine-tuning randomness; the source baseline remains one seed.

Each run:12,000 updates (four times the previous pilot), batch32, all25,000 existing
training patches, same64 training/13 validation subjects, all480 validation patches.
Fresh AdamW, peak LR7e-6,100-update warmup, polynomial exponent0.9 decay to1e-6,
weight decay1e-4, existing augmentation. The same schedule applies to every arm.
The old3000-update pilot had a different schedule horizon, so persistence is judged
on this study's own3000/6000/9000/12000 trajectory, not by treating it as resumed.
No loss transition: immediate node softmax focal gamma2 with weights[1,1], matched
edge CE[1,1], ratio upsampling0.15. Matcher,120 queries, relation head and K8 local
geometry with graph_only gradient route unchanged. Backbone/transformer and all
active heads train; dormant legacy relation attention stays inactive. No multiscale.

## Variant and correctness

A uses the established27 samples on the32³ early CNN map, radius4 input voxels,
128-hidden-unit residual MLP, live CNN feature gradients, detached sampling centers,
zero output initialization and five outputs. A-xyz uses the same head with three
outputs, returning the original object logits exactly; only258 parameters removed.
The legacy five-output coordinate-only layout remains the default for old models.
A-control uses the same five-output head with zero local image inputs, including
both seeds so image-versus-capacity evidence has the same training budget.

CPU: identity, unchanged coordinate-only logits, feature gradients, query permutation
equivariance, legacy shape compatibility and original focal/spatial checks.
GPU: bitwise shared-weight transfer, initial prediction parity, all active gradients,
256-update eight-patch fit per arm and batch32 forward/backward/hash smoke. Require
node F1>=0.95 and edge F1>=0.90 at2mm before releasing long runs; tiny weights discarded.
All input images and complete target structures are hashed across every update.
Hash equality is required within each seed, while seeds must differ.

## Selection and analysis fixed before training

Primary comparison is the final update12000 for every run, regardless of validation.
Secondary selection: largest full-validation physical edge F1 at2mm among updates
3000,6000,9000,12000, tie broken by node F1 at1mm, then earlier update. Save all four
checkpoints and metrics. Report primary and secondary separately; no held-out test
claim because the same validation subjects selected the original baseline.

Report raw oracle query coverage and filtered node precision/recall/F1 at1/2mm,
localization, duplicates/confidence calibration, edge precision/recall/F1 and
conditional recovery, component/cycle errors and native/common-v2 Branch/Curve F1.
Monitor fixed128 training patches and full validation at each3000-update checkpoint.
Primary pairings: A-B0, A-xyz-B0, A-xyz-A, A-A-control. Per-seed paired subject
bootstrap plus mean and range across seeds; crossed subject/seed bootstrap is
exploratory with only two seeds and one shared baseline initialization.

Operating points from cached final predictions, preserving relation scores:
- Per-patch counts equal to B0's threshold0.5 count, selecting each arm's topK.
  This is a controlled comparison, not a standalone threshold-selection policy.
- Threshold curves and validation matched-precision interpolation, explicitly
  labeled descriptive validation operating curves, not train-calibrated inference.
- Train-subset threshold calibration at precision targets0.5/0.7/0.8/0.9, separately
  at1/2mm; apply thresholds unchanged to validation and report achieved precision.
  Calibration is limited to the fixed128 training-monitor patches; no validation
  labels are used to choose these deployable thresholds. Report unattainable targets.

Final A and A-xyz: zero/roll local feature interventions on the same128 validation
patches, including native/common reconstruction. A additionally: disable confidence
correction only, or coordinate correction only, without retraining. These isolate
functional contributions but are distribution shifts, not independent trained arms.
Check query-pair correspondence, unchanged intervention relation probabilities,
original-logit identity for A-xyz, valid graph indices, and spline/curve endpoint
agreement with the refined nodes. Preserve original graph/geometry implementations.

## Scheduling and decision

One H100 smoke then an afterok array0-7%2, at most two H100s total. Each long run has
a three-hour ceiling; no automatic extension. Preserve prior study artifacts.
C is deferred, not rejected. No combined A+C, IXI transfer or other training launched.
Recommend subsequent fully unfrozen IXI transfer only if image-conditioned gains
persist at longer duration and across both seeds, supported by controls/interventions.
Prefer coordinate-only if it retains useful gains without a consistent combined-A
advantage; quantify uncertainty rather than claiming equivalence from nonsignificance.
Final report: docs/research/local_refinement_long_study.md.

## Storage failure and bounded recovery

The project inode limit (500,000 files) interrupted A-xyz at the update12000 save and A-control during the update9000 evaluation. B0/A completed; the second-seed array tasks and collector could not start. One-byte writes reproduced errno122. User/group byte quotas were unlimited, while project20305265 reported498204 file entries while new writes failed. This study's prediction NPZ files were packed byte-for-byte into verified caches.zip archives, reducing project usage to489293. The original write probe then passed. No checkpoints or prior-study artifacts were removed. cache-archive-recovery.json records archive hashes.

Readers now resolve logical NPZ cache paths from caches.zip, and future evaluation writes archive members directly. Interrupted runs resume only from intact model/optimizer update9000 checkpoints, replaying data (without optimization) to reconstruct and assert the original9000-update batch hash. Each still has a final budget of12000 learned updates. Full recovered hashes must match completed B0. The lost A-xyz last3000 computations must be repeated, which is included in resource accounting. All interrupted diagnostics are preserved. Recovery array1540960 tasks2–3 precedes second-seed array1540961 tasks4–7, both with concurrency2. No schedule, labels, objective or selection criterion changes.
