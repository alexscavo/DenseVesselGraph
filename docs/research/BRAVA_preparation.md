# Brava SWC graphs, skull-stripped brains and IXI-compatible patches

## Orientation repair, 20 October 2026

The original failure reproduced deterministically on BG0014. A complete screen of all **56 image subjects** found exactly **five** x-reflected SWC/image pairs: **BG0014, BG0019, BG0020, BG0021, BG0022**. Other subjects retain the existing transform. These scans share the same NIfTI axis directions as passing controls, so the correction cannot be inferred from affine signs alone. Both original and color-coded files have the same discrepancy; choosing a different trace variant does not resolve it.

The correction is explicit dataset preprocessing: apply `x = Nx - 1 - x` after the existing 0.62-unit SWC mapping for those five subjects, leaving the MRI, supplied brain mask, SWC IDs/parents/radii and source files unchanged. Input MRI, brain-mask and valid-SWC hashes are pinned in [orientation-corrections.json](evidence/brava_preparation/orientation-corrections.json). Changed files invalidate the correction and require review. This is not an automatic image-fitted reflection policy. The full screen is [all-subject-orientation-screen.json](evidence/brava_preparation/all-subject-orientation-screen.json).

|Subject|Original trace median MRI|Corrected median|Original brain support|Corrected brain support|
|---|---:|---:|---:|---:|


|BG0014|140.00|167.50|0.97585|0.99771|
|BG0019|148.00|178.00|0.97086|0.99703|
|BG0020|137.00|172.50|0.95535|0.99547|
|BG0021|155.00|190.00|0.98783|0.99784|
|BG0022|148.50|176.00|0.97623|0.99825|

All five corrections improve median image support by 18–26% and raise brain-mask support above 99%. These are alignment checks, not a claim of exact registration or complete vascular annotation. Corrected orthogonal overlays are generated in the preflight audit. No bounds, mask-support or intensity gates are weakened. No subjects are dropped, and the 39/8/9 split remains unchanged.

The audit now checks **all 111 valid subject/trace-variant combinations**, records every failure in one pass, and only then blocks downstream preparation if necessary. It writes incremental `audit/checks.json`, final `audit/failures.json` and a success-only completion marker. This prevents a new dataset case from remaining hidden behind the first failing subject. The new coordinate policy is also recorded in cached-subject markers and staged provenance; incompatible cached orientation is rejected. Four targeted regressions cover corrected/incorrect orientation, unchanged tracer scaling and collection of multiple failures. Command: `python -m unittest -v tests.test_brava_alignment`.

**Completed preflight:** all **56 subjects / 111 valid SWC variants passed**, with zero failures, before resubmission. Evidence: `experiments/brava_preparation/orientation-preflight-20261020/audit/complete.json` and [compact checks](evidence/brava_preparation/orientation-preflight-checks.json). The old BG0006 and BG0014 failures are both covered. Corrected overlays are saved for all five affected scans. The queued dataset preparation repeats the audit and retains its real graph/patch/spline smoke dependency. This verifies the coordinate issue, not completion of the full patch dataset.

## Earlier stopped run and interactive source preview

Audit **1336449 failed** on `BG0014`: the fixed orientation's sampled median intensity was140.0, versus167.5 after an x reflection. Brain-mask support was0.97585 versus0.99771. Jobs1336451,1336454,1336457,1336461,1336463 and1336466 were cancelled by the failed dependency; the full skull-stripping/graph/patch pipeline has **not completed**. The preceding0.62 tracer-scale correction remains supported, but it does not resolve this subject's apparent left–right mismatch. At that earlier review no reflection had been accepted; the subsequent all-subject diagnosis and repair are documented above.

An offline [interactive explorer](evidence/brava_preparation/brava_explorer.html) renders **BG0001, BG0006 and BG0014** directly from the original images, supplied brain masks and complete SWC annotations. It includes skull-stripped MRI slices, brain-mask surface, graph lines, junction/endpoint markers, all-sample toggle, candidate64³ patch boxes on the exact IXI grid and a display-only x-reflection toggle. A yellow box is a **candidate grid patch**, not an exported sample. The MRI preview is downsampled3× and intensity-rescaled for display; SWC coordinates remain at full resolution and the NIfTI affine defines world millimetres.

Download the HTML and open it locally; all data and Plotly are embedded and no server or original files are needed. Companion JSON records source identities, plotting resolution and alignment measurements. Projection PNGs show fixed and reflected BG0014 traces separately. This source-data preview does not assert that model-ready GT or patches exist. This viewer is the historical pre-correction snapshot; its reflection toggle remains display-only. Regenerating it from the repaired inventory uses the recorded per-subject orientation.

```bash
python scripts/render_brava_explorer.py \
  --preparation /lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057 \
  --output docs/research/evidence/brava_preparation/brava_explorer.html
```

2026-10-20. Preparation queued when accepted IDs appear below; final counts are appended only after completion.

## Inputs and requested scope

Source: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava`.

- **56 brain volumes and 56 matching brain masks.** Apply each supplied mask (`mask_hull/sub-*_space-MRA_desc-brain_mask.nii.gz > 0`) to its brain volume. Preserve source intensities inside the mask and set values outside to zero. Preserve the NIfTI affine/header, and retain links to original images and masks. No vessel segmentation is created or inferred.
- **123 SWC files from 62 canonical subjects**, after normalizing zero padding (e.g. `BG001`→`BG0001`, `BG04`→`BG0004`). Original and ColorCoded files are variants of the same subject, not independent patients or extra split members.
- Six SWC-only subjects have no corresponding brain volume: **BG0013, BH0002, BI0001, Set8, Set9, Set10**. Export their valid graph annotations in native SWC coordinates; no image alignment or patches can be claimed for them.
- `BH0027.CNG.swc` row3407 has six columns and lacks its parent ID. Do not invent the missing connection. Record the invalid original, preserve its raw file, and use the complete ColorCoded counterpart for that subject. Original variants are preferred for the other image subjects because they generally retain more coordinate precision. Both complete variants are exported separately.

## Alignment and graph GT

SWC intrinsic coordinates are **not NIfTI world coordinates**. Direct inverse-affine mapping puts most points outside the images. Four initial real-volume checks (BG0001/BG0002/BH0003/BH0040) consistently favored the fixed mapping:

```
voxel = [swc_x / 0.62, Ny - 1 - swc_z / 0.62, Nz - 1 + swc_y / 0.62]
world = nifti_affine @ [voxel, 1]
```

The tracer lattice uses **0.62 coordinate units per NIfTI voxel**, independently of the image's physical spacing. The initial implementation incorrectly divided by header spacing; the failed audit and correction are documented below. Only the actual NIfTI affine establishes physical distances. The initial pipeline applied this single mapping to every subject. The repair above adds explicit hash-verified x reflections for five subjects; it does not fit a per-case registration. An initial CPU audit checks all image/mask shapes, affines, millimetre units, SWC IDs/parent references, image bounds, brain-mask support and image intensity along traces versus reflected alternatives. Gates: ≥99% points inside the image, ≥95% within the brain mask, and the fixed orientation's median sampled intensity at least95% of the strongest reflection. Four orthogonal-projection overlays are saved. These are alignment checks, not evidence that the supplied traces cover every vessel or subvoxel registration is exact. Failures block downstream jobs.

Full-volume graph representations:

1. **Dense:** every SWC vertex and parent connection, preserving IDs/types/radii in native NPZ exports and original polyline guides in graph exports.
2. **Junction/end:** contract degree-two chains without changing their guide paths or graph components.
3. **Radius-adaptive:** the existing optimal simplifier with physical tolerance `max(0.1 mm,0.75 × physical radius)`, keeping original unsmoothed branch paths as guides. Convert the SWC radius to voxel units using0.62, then to a scalar physical radius using mean physical voxel spacing. This is a scalar approximation for anisotropic voxels and affects subdivision, not the preserved guide paths. It adds subdivision anchors where needed.

Export `nodes.csv`, `edges.csv`, `graph.vvg` in the same world-coordinate format as IXI. Verify components/cycle rank are unchanged by graph contraction. Do not skeletonize brain masks, prune manual branches, merge junctions or smooth traces. **IXI's vessel-lumen chord-containment criterion cannot be reproduced without vessel segmentations and is deliberately omitted.** The common model targets depend on the preserved guide paths rather than treating adaptive degree-two nodes as separate anatomical junctions.

SWC parent links normally encode forests. Their cycle counts do not establish that the anatomy has no anastomoses; trace coverage and tree encoding constrain topology supervision. Incomplete trace coverage must not be mistaken for negative evidence of all unannotated vessels.

## Images, patches and model targets

- Output under a new versioned `Brava/prepared_swc_<timestamp>` directory. Original files remain intact.
- Stage skull-stripped images and primary SWC graphs using numeric patient IDs with a subject mapping table. Fixed **70/15/15 subject split, seed42**, same split helper as IXI: 39 training,8 validation,9 test subjects. Brava supplies no official split here. No test model evaluation or parameter tuning is performed; preparing test GT is only dataset construction.
- Reuse `generate_synthetic_mri_dataset.py`: **64×64×64**, no padding, endpoint-distributed grid, maximum stride40, native voxel resolution and existing IXI intensity normalization/fallback. Native spacing is about0.62 mm, so physical patch size differs from other scanners while voxel conventions remain the same.
- Reuse existing endpoint-aware voxel-cell graph clipping and boundary provenance. Keep the full grid index and archive graph-free/zero-brain-mask patches through the existing all-split curation command; no source data are deleted.
- Reuse `prepare_transfer_splines.prepare` for **all compatible graph-positive patches**, without a training-patch cap: shared junction/end/crop anchors, maximum256 object nodes,8 B-spline controls and64 curve/guide samples. Same physical metric, normalized D/H/W coordinates and existing `last_inside_sample` boundary policy with exact sparse-guide fallback as IXI. Record closed/parallel/degenerate/capacity exclusions rather than silently altering GT.
- Final model root: `transfer_splines_k8/`, containing `train,val,test/{raw,seg,vtp,spline}`, manifest, exact sample IDs and READY marker.

**`seg/` contains the supplied brain-mask patches solely because the existing graph loader requires that slot. It is not vessel GT.** `brava_semantics.json` and the target manifest state this explicitly. Use image-input graph supervision. Do not use these masks for vessel Dice, vessel-supported query categories, vessel-mask-input oracles or vessel segmentation training. They are not synthesized vascular masks. Original brain masks and skull-stripped full images remain separately accessible.

## CPU execution and verification

All jobs use the existing Jean-Zay CPU partition/account, four CPUs per task, no GPU and no unsupported memory directives. Large outputs and temporary runtime staging are on SCRATCH. Dependencies:

`all-volume alignment/native exports → real-data smoke → 56-volume graph/skull-strip array (%4) → staging/split → patch grid → spline targets → collector`.

The smoke prepares one BG and two BH images, runs the actual shared patch generator, curation and six real spline targets across the three construction splits, checks finite controls and query capacity. Its completed full-volume outputs are reused by the subsequent array. Smoke/audit failures block the full dataset. Completed-subject markers verify source hashes before reuse. No existing IXI training job is cancelled or changed.

Wall allocations: audit1h, smoke1h, per-volume array30min, staging15min, patch generation4h, spline conversion4h, collector30min. Compact inventory, alignment plots and final counts are copied to `docs/research/evidence/brava_preparation/`. The collector appends the completed summary here. Patch counts are intentionally unknown before extraction finishes.

Implementation: `scripts/prepare_brava.py`, `cluster/jean_zay/{submit_brava.py,prepare_brava.slurm}`. Submission record: `experiments/brava_preparation/submission.json`.

```bash
python3 cluster/jean_zay/submit_brava.py --submit
```

## Failed initial submission and coordinate-scale correction

Audit1335951 failed on BG0006; jobs1335952–1335957 were automatically cancelled by their failed dependency, before skull stripping or patch extraction. No running job was cancelled manually. Original submission provenance is archived as `experiments/brava_preparation/submission-1335951.json` on retry.

The error was conflating intrinsic tracer units with physical NIfTI units. BG0006 has0.5625-mm in-plane spacing; dividing SWC coordinates by that spacing pushed39 points outside its256-voxel x dimension. Under the fixed0.62-unit tracing lattice, its in-image fraction changes0.9865→1.0000, brain-mask support0.8326→0.9990 and sampled median image intensity131.54→196.50. BG0001 also improves in sampled median intensity160.72→169.50. A simple −8-voxel crop-offset hypothesis did not recover brain support or image agreement. The original99%/95% audit gates remain unchanged, and the NIfTI affine continues to define physical geometry.

Targeted replay of the original BG0006 failure and a BG0001 control passes after the correction. All56 image subjects pass the unchanged coordinate-bound gate. The queued all-volume image/brain support audit still gates downstream preparation. The BG0006 overlay is saved in `docs/research/evidence/brava_preparation/BG0006-tracer-scale.png`.

## Initial jobs (failed/cancelled; retained for provenance)

Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057`.

|Stage|Job ID|afterok|
|---|---|---|
|alignment_and_native_exports|1335951|none|
|real_graph_and_patch_smoke|1335952|1335951|
|full_volume_graphs_and_skull_stripping|1335953|1335952|
|subject_split_and_staging|1335954|1335953|
|ixi_grid_patches|1335955|1335954|
|model_compatible_spline_targets|1335956|1335955|
|collection|1335957|1335956|

Retrieve without waiting or rerunning preparation:

```bash
python3 scripts/prepare_brava.py show --output /lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057
```

## Accepted jobs

Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057`.

|Stage|Job ID|afterok|
|---|---|---|
|alignment_and_native_exports|1336449|none|
|real_graph_and_patch_smoke|1336451|1336449|
|full_volume_graphs_and_skull_stripping|1336454|1336451|
|subject_split_and_staging|1336457|1336454|
|ixi_grid_patches|1336461|1336457|
|model_compatible_spline_targets|1336463|1336461|
|collection|1336466|1336463|

Retrieve without waiting or rerunning preparation:

```bash
python3 scripts/prepare_brava.py show --output /lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057
```

## Accepted jobs

Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057`.

|Stage|Job ID|afterok|
|---|---|---|
|alignment_and_native_exports|1339525|none|
|real_graph_and_patch_smoke|1339526|1339525|
|full_volume_graphs_and_skull_stripping|1339528|1339526|
|subject_split_and_staging|1339529|1339528|
|ixi_grid_patches|1339530|1339529|
|model_compatible_spline_targets|1339531|1339530|
|collection|1339532|1339531|

Retrieve without waiting or rerunning preparation:

```bash
python3 scripts/prepare_brava.py show --output /lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057
```

## Completed preparation

```json
{
  "completed": true,
  "output": "/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057",
  "images": 56,
  "native_swc_subjects": 62,
  "native_swc_files": 123,
  "orphan_swc_subjects": [
    "BG0013",
    "BH0002",
    "BI0001",
    "Set10",
    "Set8",
    "Set9"
  ],
  "patch_counts": {
    "train": 8768,
    "val": 1796,
    "test": 2043
  },
  "exclusion_counts": {},
  "no_vessel_segmentations": true,
  "segmentation_role": "brain_mask_only_not_vessel_GT",
  "limitation": "SWC parent-link forests do not establish absence of anatomical vascular cycles; trace coverage may be incomplete."
}
```


## Bounded representation audit follow-up — 26 October 2026

The completed fully trainable study is recorded in [BRAVA_experiment_plan.md](BRAVA_experiment_plan.md#fully-trainable-representation-follow-up--26-october-2026). It reuses the validated SWC transform, reflection corrections, immutable target manifest and subject split; it does not constitute a new full-dataset alignment audit.

On 1,504 development patches, actual degree-2 subdivision preserves all 15,718 parent paths and graph topology, increases node count from 24,863 to 30,446, and stays within 120 queries (maximum 69 nodes). All branches satisfy the existing splitter's 0.5-mm criterion. The audit distinguishes geometric face location from clipping provenance: 270 valid nodes outside the sigmoid voxel-center range have `boundary=False`; no targets were relabeled or clamped. Maximum sigmoid clamp displacement is 0.4402 mm. Numeric evidence: [target-summary.json](../../experiments/brava_representation_20261026/target-summary.json) and [target-audit.json](../../experiments/brava_representation_20261026/target-audit.json).

Both tiny learnability gates passed, but larger-cohort detector reconstruction remains weak. Subdivision improves partial curve coverage and GT-conditioned geometry while increasing cycle errors; K24/fixed C1 improve fitting bounds without a clear learned reconstruction gain. Consult the experiment report for paired uncertainty, oracle-versus-inference distinctions, checkpoints and limitations. These results do not establish annotation completeness or absence of anatomical cycles.
