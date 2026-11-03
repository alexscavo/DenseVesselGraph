# Controlled plant transfer and spline augmentation — 2026-10-14

## Decision and experiment

User approved reusing vessel90°rotations and Gaussian noise(probability.35, std sampled up to.015), with zoom off. Plant loader retains its established quarter-turn rotations and flips. No elastic warp, crop, vessel flip, extra loss or domain-adversarial objective is introduced. Validation/test remain augmentation-free.

Two graph-only pretrainings start from the same random initialization,50epochs each:

| Arm | Unique training population | Draws/epoch |
|---|---|---:|
| pretrain_mixed |25000plants + existing10000MRI|50000, approximately half each, replacement|
| pretrain_vessels |same10000MRI|50000, replacement|

Each detector then initializes two100epoch-maximum vessel fine-tunings: straight chords and detached residual B-splines,8total controls. This gives six logical trainings. Equal pretraining updates isolate transfer at fixed compute, although MRI exposure differs by design. It is not an independent ablation of augmentation versus no augmentation. Existing no-augmentation300epoch results provide historical context, not an exposure-matched causal control.

Batch4; AdamW7e-5, weight decay1e-4, existing polynomial scheduler,1epoch warmup. Graph weights unchanged(class4, nodes2, boxes3, cardinality.8, edges6). Geometry weight0 for graph-only arms and1.4739989612791453 for spline fine-tuning, preserving previous calibration. All detector parameters train; geometry gradients detach tokens/endpoints and shared local CNN features. Fresh optimizer/scheduler for fine-tuning; stage continuation restores optimizer/scheduler and RNG.

A frozen zero-residual spline head is retained in straight controls solely to reuse the same decoder/evaluation architecture: outputs are straight chords. Pretraining transfers only detector weights; spline-specific weights start fresh with the common seed.

## Targets and augmentation correctness

New derived dataset: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014/plants-k8`.
Original uploaded data untouched.25000training/480validation accepted, no candidate excluded before reaching caps.18177 intermediate degree2 nodes removed. Existing junctions, terminations, isolates and crop anchors at original5/128 and122/128 faces retained; protected boundary nodes may remain degree2. Paths concatenate the original annotation segments, then fit endpoint-constrained8control cubic B-splines. Closed/parallel/capacity-incompatible graphs would be explicitly excluded, never silently simplified; none encountered in selected population.

Plant cache coordinates remain original PNG normalized(y,x,z=0), with pixel units. They are annotation-derived guides, not recovered anatomical centerlines or vessel-mm targets. Previously discarded curvature cannot be recovered from these files. **This experiment uses plant graph supervision only during pretraining.** Continuous plant supervision would additionally require projection/augmentation of those caches through the plant loader. Fine-tuning uses the existing MRI continuous targets.

Vessel image rotations reuse existing exact torch.rot90 transforms. Spline controls, curves and guides receive the same affine mapping, including voxel-centre offset. Controls may lie outside the image and are not clipped. For normalized transform x'=Ax+b, M'=M A^-1 preserves physical displacement in the source physical basis; the common translation cancels in geometry errors. Noise changes images only. Zoom remains rejected for spline configurations.

Implementation: `data/spline_augmentation.py`, MRI loader and config guard; optional protected anchors in existing contractor; `scripts/prepare_plant_spline_targets.py`; `scripts/train_plant_generalization.py`; `configs/plant_generalization.yaml`; three cluster submission scripts.

## Monitoring and safeguards

Unchanged1952MRI validation population, fixed128monitoring IDs and512training evaluation IDs from previous experiments. Monitoring every5epochs; training decomposition every10. Same-endpoint chord comparison separates curvature benefit from detection. Validation reports node recovery, endpoint-pair coverage, conditional long relations and nearby false connections. Two fixed projections logged to offline W&B; standard sync remains necessary.

Checkpoint selection: maximum common branch F1, native edge F1 tie-breaker, as previously used. Fine-tuning early stopping after30epochs without improvement, minimum30epochs; pretraining runs fixed50epochs. Best and final checkpoints receive full1952validation plus curve/topology/Betti/APLS/TOPO metrics. No held-out test-set selection.

Stage boundaries: pretraining25→50epochs and fine-tuning50→100epochs, preserving state. Early-completed runs exit dependent stages without extra training. Latest checkpoint each epoch. Output files are on scratch, not project storage. Full initial untrained-model evaluation proved expensive (~2.8–3hours per arm); no useful learning conclusion can be drawn from those initial metrics.

## Verification and submissions

- Four targeted tests passed: spline/image coordinate alignment, spline affine equivariance/physical distances, protected boundary contraction and closed-component handling.
- Preparation1049449 completed8m50s, including tiny target smoke.
- GPU verification1049493 completed17m09s. Four arms each passed3updates: mixed/vessel pretraining and straight/spline transfer, finite gradients and correct spline trainability. Fixed runtime solved the mutable-environment SimpleITK issue.
-1049725_0/1: pretraining through25epochs, running on separate H100s when checked.
-1049726_0/1: corresponding continuation through50epochs.
-1049727_0..3: mixed_straight, mixed_spline, vessels_straight, vessels_spline through50fine-tuning epochs.
-1049728_0..3: corresponding continuation through100epochs if not early-stopped.

Dependencies require successful preparation and GPU verification before training; corresponding stages use aftercorr. Fine-tuning waits for both pretrainings. No full300epoch run submitted.

Machine-readable job IDs: `experiments/plant_pretraining/submissions/20261014.json`.
Runs: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014/runs/`.
Next: inspect first5/10epoch learning/generalization checks; compare matched validation recovery and curvature improvement before deciding extensions or architecture changes.

## Superseding restart: domain adaptation and batch32

User corrected the intended recipe: mixed pretraining must enable domain adaptation; then explicitly selected **domain weight1.0** (overriding legacy0.1) and restart from scratch. Cancelled the original1049725/26/27/28 chain; preserve its outputs as superseded, not continuations. Cancelled provisional0.1verification1068973.

New config: `configs/plant_generalization_da_b32.yaml`. New output root ends `plant_generalization_20261014_da_b32_w1_restart`; no previous-run checkpoint is loaded. Model, optimizer and scheduler start fresh for both pretrainings. Only later stages resume within this new run; fine-tunings initialize from the newly trained detector.

Mixed pretraining enables image and instance adversarial losses plus consistency regularization, domain weight1.0, class weights[.05,.95], domain LR7e-5 and existing alpha ramp. MRI-only control and vessel-only fine-tuning disable the domain objective. The comparison now measures the combined mixed-domain adaptation recipe against MRI-only, not plant data alone. Batch32 is conditional on GPU verification passing; no silent batch fallback.50000draws/epoch gives1563updates including a final16sample batch. Original derived datasets/splits and spline augmentation policy unchanged.

Fresh GPU verification1069024 checks batch32memory/time, finite domain loss and nonzero gradients in both discriminators, then both representation fine-tunings. Two focused domain-loss/config tests passed locally. Skip the costly untrained full initial evaluation for production runs; monitoring begins at epoch5. All regular validation remains unchanged.

Replacement chain:

-1069042_0/1 pretraining to25epochs, afterok1069024.
-1069043_0/1 pretraining to50epochs, aftercorr1069042.
-1069044_0..3 fine-tuning to50epochs, afterok1069043.
-1069045_0..3 fine-tuning to100epochs, aftercorr1069044.

The original completed preparation job aged out of the scheduler dependency window, so submission uses the existing target READY guard rather than a dependency on that historical job. All training still waits for the new GPU check. Machine-readable replacement IDs: `experiments/plant_pretraining/submissions/20261014-da-b32-w1-restart.json`.


## Restart chain status (2026-10-14)

Verification1069024 completed successfully,14m47s. Batch32 mixed 3-step GPU smoke used peak9.19GiB; domain losses3.487,3.243,3.205 were finite, both domain labels were present, and gradients reached image and instance discriminators. MRI-only and both straight/spline fine-tuning smokes also passed at batch32. Two targeted local domain tests passed.

Fresh chain1069042_0/1 completed its25epoch stage from random initialization, batch32, domain weight1.0 in mixed arm. Continuation1069043_0/1 is running to epoch50. Fine-tuning arrays1069044 and1069045 are submitted with verified dependencies on those stages; scheduler inspection confirms they wait on1069043, then1069044. They will initialize only from the new pretraining checkpoints. Superseded0.1/batch4 jobs were cancelled and are not dependencies.

Job records and verified configuration: `experiments/plant_pretraining/submissions/20261014-da-b32-w1-restart.json`, `configs/plant_generalization_da_b32.yaml`.
### 2026-10-16: disable patience-based early stopping

User requested fixed 100-epoch fine-tuning for comparison. `early_stopping: false` is now explicit in plant, leaf, combined and IXI plans; best-checkpoint selection is unchanged. Completed SyntheticMRI source-comparison runs already reached 100, so they need no retraining. Three early-stopped IXI runs are resumed and retested; see `ixi-transfer.md` for job IDs and archived earlier outputs.
