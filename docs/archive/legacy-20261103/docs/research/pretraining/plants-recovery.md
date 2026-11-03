# Mixed plant pretraining recovery — 2026-10-14

## Recovered implementation

Original repository: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/Image-to-Graph-Vessel/3d`.
Exact reference snapshots and SHA256 provenance: `experiments/plant_pretraining/reference/`.
These are archival references, not a new training entrypoint. In particular, the original generator's `__main__` overrides CLI arguments with hardcoded `/data/scavone` paths; do not execute it unchanged.

- `data/dataset_mixed.py`: concatenate source plants and target MRI. With target upsampling, target sample weight is source_count/target_count, giving equal expected source/target draws and twice the source count per epoch, with replacement. Both domains have supervised graph losses.
- `configs/mixed_plants_synth_HNS_3Dsynthmr.yaml`: 25,900 plants + 3,800 SyntheticMRI requested; 50 epochs, batch32, AdamW7e-5, decay1e-4. Image/graph adversarial and consistency options enabled, domain weight0.1. These are configuration facts, not proof of the exact historical run used by the user. A separate `pretrain_plants_synth_supcl.yaml` exists; its SupCL switch is false.
- `data/dataset_plants.py`: PNG image/segmentation plus VTP graph, resize to54², project into five slices, pad to64³; image scales to[-.5,.5]. Graph coordinates transform with projection/rotation. This is planar source anatomy embedded in3D, not a full3D vessel distribution.
- `data/generate_plants_data.py`: original image-level `splits.csv`, default6x downsampling,128² crops, graph clipping/pruning, optional degree2 contraction. Hardcoded example uses overlap.35 and budgets25923/6421/6737 with contraction. Those defaults do not establish the uploaded dataset's actual generation history.
- `data/generate_plants_augmented_data.py`: preserved alternate generator; not selected or run.

## Existing implementation to reuse

Current `data/loaders/plants.py`, `data/loaders/mixed.py`, `data/loaders/discovery.py`, `configs/pretrain_mixed.yaml` already provide compatible data loading and balanced sampling. Current pretrain config differs: batch48,4000MRI, explicit validation caps; domain adaptation disabled by base config. Do not claim it exactly reproduces the legacy approach.

Current plant loader corrects the stored(y,x) coordinates for the image reader, preserves small segmentation branches through occupancy downsampling, and disables random validation augmentation. Preserve these corrections rather than copying the older loader wholesale. Current plant augmentation uses quarter-turns and flips; the legacy loader has additional continuous/elastic options whose parity is not established.

## Uploaded dataset

Use `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/plants_3d2cut/patches_3d` as loader root, **not** the top-level root or `patches` (those use different graph layouts).
Observed counts:25923train,6421val,6737test, each raw/seg/vtp. Inventory snapshot records missing/empty triplets; upload may still be active, so this is not transfer completion certification.
Eight VTPs per train/val were read successfully. Some contain degree2 nodes (0–2 per sampled graph). Resolve whether they are legitimate retained boundary nodes or intermediate nodes before harmonizing topology; do not blindly contract boundary anchors. There are no confirmed continuous spline targets for these plant patches. Preserve original uploaded data.

## Recommended experiment after compatibility checks

User's50epoch mixed pretraining then100epoch vessel fine-tuning is a reasonable transfer baseline. It changes data diversity and initialization, not merely epoch count. With25k source patches and balanced sampling there are approximately50k draws/epoch; report updates/exposure as well as epochs.

1. Implement and verify rigid augmentation of image, nodes, controls, guides and coordinate metric consistently. Current spline loader explicitly rejects augmentation; all related scratch comparisons were constrained by this. Start with right-angle rotations/flips, not elastic changes to graph geometry.
2. Reuse mixed graph pretraining with a fixed25k plant manifest and only the existing MRI training membership. Keep vessel validation/test unchanged and plant splits separated by original image. Confirm patch-to-parent provenance before claiming independent plant splits.
3. Fine-tune matched straight and detached-residual spline models from the same pretrained graph checkpoint, for up to100epochs, selecting by declared validation reconstruction metrics. Plant graph-only supervision is sufficient for this pretraining hypothesis; continuous plant supervision would require real curve targets and a separate decision.
4. Compare against a matched vessel-only initialization/training control with identical augmentation and fine-tuning conditions. This distinguishes transfer from simply fixing augmentation. Avoid simultaneously restoring legacy domain-adversarial losses, edge sampling, optimizer schedules and other changes without a specific ablation.

Expected benefit: diverse branching patterns and orientation exposure may improve graph feature transfer. This is a hypothesis, not a guarantee; slab-to-volume domain mismatch remains. More epochs or relation-negative reweighting alone did not resolve the gap. Mixed pretraining does not establish that spline training from scratch is impossible.

No plant pretraining or regeneration was launched in this task. Remaining gates: upload completion, topology/provenance harmonization, spline augmentation implementation, and a short mixed-batch forward/loss smoke before training.

Loader check limitation: direct mutable-environment item loading failed because `SimpleITK._version` is missing. VTP reads and all-file triplet inventory passed (no missing or zero-byte counterparts). Use the established fixed runtime archive for the next mixed-batch smoke; do not infer a dataset defect from this environment failure or silently substitute another image reader.

Update: the authorized preparation, augmentation implementation and training submission are recorded in [generalization-training.md](generalization-training.md). Fixed-runtime GPU smoke tests passed; the mutable-environment limitation above no longer blocks these runs.
