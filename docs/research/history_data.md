# Ground-truth extraction, dataset production and storage — consolidated history (to 29 Oct 2026)

> **Status.** Covers how the IXI and TopBrain vessel graphs and 64³ patch sets were extracted,
> produced, filtered and stored (Sept – Oct 2026). Consolidated on 3 Nov 2026 from documents
> now archived under `docs/archive/legacy-20261103/`. The produced datasets are still the ones in
> use; the models trained on them are superseded by the
> [dense traced model](dense_traced_model_20261102.md). Representation choices (junction-only vs
> adaptive vs dense, simplification policy, query and spline capacity) are in
> [`history_representation.md`](history_representation.md).

## Context

The question was how to turn a binary vessel segmentation into a graph target that a direct
image-to-graph model (then RelationFormer, a DETR-style transformer with a fixed number of object
queries) could learn from raw MRA/CT, and how to cut it into patches without changing the target
from one crop to another. Work was done on 220 paired image/segmentation volumes: 170 IXI MRA,
25 TopBrain MR, 18 TopBrain CT training and 7 labelled TopBrain CT test volumes
([tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)). The patient
split is 154/33/33 train/val/test patients, with the seven official TopBrain CT test cases kept in
test.

Other datasets have their own reference documents, which this file does not repeat:
- early IXI extraction work, header repair, brain masks and the dead `IXI_patches_final` set:
  [`../IXI_DATASET.md`](../IXI_DATASET.md);
- SyntheticMRI generation and audits: [`../DATA_GENERATION.md`](../DATA_GENERATION.md),
  [`../SYNTHETIC_MRI_DATASET_REPORT.md`](../SYNTHETIC_MRI_DATASET_REPORT.md),
  [`../DATA_AUDIT.md`](../DATA_AUDIT.md);
- BRAVA SWC traces and patches: [`BRAVA_preparation.md`](BRAVA_preparation.md).

## Shorthand

- **Dense / adaptive / junction-only.** The three graph products derived from one extraction
  (definitions in [`history_representation.md`](history_representation.md)). Adaptive is the
  training target.
- **D01–D08.** Entries of the decision log of the implementation tracker. D07 (patch eligibility)
  is described here; the others in the representation history.
- **T01–T11.** Task IDs of the same tracker (T02 annotation overrides, T09 the 300-epoch IXI
  RelationFormer run, etc.).
- **Legacy backend / `vedo_original` backend.** The two centreline extractors. Legacy is the
  project's own skeletonisation; `vedo_original` runs the unmodified original
  `vedo_extractor_original.py` plus a mask-repair wrapper.
- **`optimal_radius_0p75x_c095`.** The selected adaptive simplification policy: exact
  minimum-node simplification, deviation ≤ 0.75 × local lumen radius, ≥ 95% of each straight
  chord inside the segmentation.
- **Exact-boundary crop.** Cropping a full-volume graph by clipping edges at voxel-cell faces and
  inserting a flagged boundary node at each crossing.
- **Graph-positive patch.** Segmentation foreground > 0, graph nodes > 0 and graph edges > 0.
- **β₀ / β₁.** Number of connected components / independent cycles.

## 1. Source inventory and annotation provenance

**[DONE] Annotation overrides (T02).** For IXI638 and IXI661 the planning run selects the
`nii(1).gz` segmentation and records the row as `preferred_duplicate`
([tracker E03](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md),
[evidence/T02](evidence/T02/annotation_selection.csv)).

**[DONE] Completion markers.** Each volume gets an atomic, configuration-fingerprinted
`complete.json`; relaunching skips volumes whose marker and nine graph files (three files ×
three representations) match. Forced reruns invalidate stale markers before overwriting, so an
interrupted regeneration cannot look complete.

Caveat kept throughout: the segmentations are inputs, not anatomical truth. Their provenance is
unknown for IXI (see [`../IXI_DATASET.md`](../IXI_DATASET.md) §8).

## 2. Extraction pipeline

**[DONE] Full volume first, crop afterwards.** The first IXI122/TopBrain counts were made by
cropping the segmentation before extracting a graph. This was corrected: each full segmentation is
skeletonised once into an immutable dense centreline, the three representations are derived from
it, and only then is the saved graph cropped. Patch-level re-extraction or re-simplification is
not allowed, so overlapping and neighbouring patches share the same graph
([tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)).

**[DONE] Exact boundary intersections.** Cropping is done in voxel coordinates (needed for
TopBrain's oblique affine), clipping against `[start − 0.5, start + crop_size − 0.5]`. Each
crossing gets a synthetic node that keeps `source_edge_index` and a boundary flag, so neighbouring
patches share the same face node. The alternative "stop at the last inside sample" policy gave the
same topology and token counts on the densest IXI122 and TopBrain-MR-001 patches but ended edges
about 0.42 voxel inside the face, so exact clipping was chosen
([tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)).

**[DONE] Other fixes found by validation, not by luck:**
- six-decimal world coordinates could move a junction by ~1e-7 voxel across a cell face after
  reloading; node CSVs now use 12 decimals, with a regression test;
- the legacy intensity normaliser cast float MRA to integers, which made its median/MAD threshold
  zero on IXI371; a 99.5th-percentile fallback is used only when that threshold is invalid;
- a `2.94e-7` image/label affine difference on TopBrain MR-002 led to a shared `1e-4` tolerance
  for patches and full volumes;
- rewriting `patient_split.csv` with LF instead of CRLF line endings broke a hash check; reuse now
  compares per-patient assignments, not file bytes
  ([Vedo run](../archive/legacy-20261103/docs/research/VEDO_PRODUCTION_RUN.md)).

## 3. Centreline backend: legacy vs Vedo

**[ABANDONED: worse on real IXI] Re-implemented Vedo-style backend (18 Sept).** A backend that
re-implemented the historical Vedo approach (surface-geodesic validation, orphan handling,
sub-voxel smoothing, local cycle cleanup) made the IXI002 crop worse: adaptive graph 244 nodes /
276 edges / β₁ = 37 became 363 / 411 / β₁ = 49, with visible tangles. The legacy extractor was
restored as default and the evidence removed at the user's request
([tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)).

**[DONE] Full-volume CT-001 comparison with the exact original Vedo code (22 Sept).** On TopBrain
`topcow_ct_001` (284 × 327 × 243 voxels), same mask and affine
([evidence](evidence/vedo_original_ct001/full_volume/README.md)):

| Full-volume dense centreline | Legacy | Original Vedo |
|---|---:|---:|
| Components / β₁ | 5 / 73 | 4 / 45 |
| Endpoints / junctions | 37 / 144 | 102 / 178 |
| Dense nodes outside segmentation | 0 | 78 |
| Edge samples outside segmentation | 0.37% | 2.31% |
| Extraction time | 5.6 s | 190.3 s |

- The original Vedo centreline could not pass the 95%-containment graph policy unchanged. A repair
  wrapper moved 78 points into the mask (median 0.41 mm, max 0.60 mm), rerouted 52 crossing edges
  through the mask and dropped 5 with no local route. After that, the final adaptive graph was
  646 nodes / 684 edges / β₁ = 44, vs 581 / 649 / β₁ = 73 for legacy.
- Vedo looked smoother, but applying the legacy pipeline's own mask-constrained smoothing to the
  legacy centreline gave the same smoothness (median turn 45.0° raw → 6.34°, Vedo 6.56°) without
  leaving the mask. Smoothing cannot change β₁; Vedo's 28 fewer cycles come from different
  pruning.
- The report's own conclusion: fewer cycles could mean false loops removed, true loops removed, or
  both. One case, no centreline ground truth.

**[DONE] Production switched to `vedo_original` with mask repair (22–23 Sept).** All 220 volumes
were re-extracted with the original Vedo centreline plus the repair wrapper, then the same
`optimal_radius_0p75x_c095` policy
([Vedo run](../archive/legacy-20261103/docs/research/VEDO_PRODUCTION_RUN.md)). The research index
records the decision as "Choose Vedo-original centerlines with documented mask repair"; no
written rationale beyond the CT-001 comparison was found. The single-volume IXI013 smoke gave
1,375 nodes / 1,415 edges with zero dropped edges. Nine large TopBrain cases were killed for lack
of memory: one CPU on Jean Zay comes with only 4 GB, and the original Vedo code holds several
full-volume arrays at once. One finished on an `--exclusive` retry; the other eight were rerun
with 16 CPUs (62.5 GB). An earlier claim that they exceeded a full node's memory was wrong.

## 4. Production runs

**[SUPERSEDED by the Vedo production] Legacy-backend production (20–21 Sept).** All 220 volumes
were extracted with the selected policy. An independent inventory found all nine files for every
subject and identical full-volume β₀/β₁ across the three representations for every subject. That
confirms that the representations do not change topology; it does not show the segmentation
topology is correct. The complete exact-boundary grid had **50,613 patches**: nodes median 7, P95
100, P99 142, max 239; 1,271 patches above 120 nodes and 35 above 192, all IXI; TopBrain's
28,611 patches peak at 133
([tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md),
[evidence](evidence/full_dataset_production/summary.json),
[per-stratum counts](evidence/full_dataset_production/patch_strata.csv)). The per-stratum file
shows IXI MRA is the dense tail (train P95 123 nodes), while no TopBrain CT/MR split has a node
P95 above 58. Graph-only full-volume views of IXI033 (1,198 nodes / 1,305 edges) and TopBrain CT-010
(287 / 290) are in [sample_views](evidence/full_dataset_production/sample_views/). These
legacy-backend graph and patch directories were **deleted** after the Vedo production passed
validation (`old_directories_removed: true` in
[final_validation.json](evidence/vedo_production/final_validation.json)). The capacity numbers
above therefore describe a dataset that no longer exists, although the patch grid is the same.

**[DONE] Vedo-original production (23 Sept), the current dataset.** All 220 graph markers, all
220 patch-patient markers, all six active indices and every active raw/seg/graph triplet passed
the final read-only check
([final_validation.json](evidence/vedo_production/final_validation.json)). Location:
`/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/{IXI,TopBrain_Data_Release_Batches1n2_081425}/`
`vascular_{graphs,patches}/optimal_radius_0p75x_c095_vedo_original/`. All three representations
are kept per volume. Current launchers are documented in `scripts/README.md`.

## 5. Patch grid and eligibility

**[DONE] Deterministic grid.** 64³ crops, per-axis starts evenly spaced with
`ceil((length − 64)/40)` intervals (maximum stride 40), Cartesian product, no random positions and
no foreground preselection. Adjacent crops overlap by at least 24 voxels (37.5%) per axis.
`patch_index.csv` keeps the exhaustive grid as an audit record
([tracker E11](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)).

**[DONE] Eligibility audit on the legacy grid (E11).** 30,778 graph-positive patches, 18,903 with
an empty mask and 932 with foreground but no graph (928 without nodes, 4 with nodes but no edges).
Reopening all 932 segmentation crops showed that in 922 of them no foreground lies ≥ 8 voxels from
every face. This *suggests* most are boundary slivers of vessels whose graph lies in the next
patch; it is not proof for the other ten
([evidence](evidence/full_dataset_production/patch_eligibility/patch_eligibility.json)).

**[SUPERSEDED by the per-split rule below] D07: filter training only (21 Sept).** The loader got
an opt-in `patch_selection: graph_positive` that filters *before* any sample cap. Training crops
were curated: IXI kept 13,736 and archived 2,092 (1,966 empty, 126 foreground-without-graph);
TopBrain kept 7,379 and archived 11,124. Validation and test grids were left exhaustive so that
negatives stayed in evaluation. Rejected triplets were moved, not deleted, to `excluded_train/`
([IXI capacity audit](evidence/ixi_capacity_192_256/README.md)).

**[DONE] Same eligibility in every split (Vedo production).** Train, validation and test are all
restricted to graph-positive patches; ineligible triplets are archived under
`excluded_{train,val,test}` and `active_{train,val,test}_index.csv` names the sets the model uses
([Vedo run](../archive/legacy-20261103/docs/research/VEDO_PRODUCTION_RUN.md)):

| Dataset | Grid patches | Active train / val / test | Excluded train / val / test |
|---|---:|---:|---:|
| IXI | 22,002 | 13,742 / 2,961 / 2,384 | 2,086 / 467 / 362 |
| TopBrain | 28,611 | 7,391 / 1,650 / 2,685 | 11,112 / 2,350 / 3,423 |

- Eligibility counts on the Vedo grid: IXI 19,087 graph-positive / 2,755 empty / 160
  foreground-without-graph; TopBrain 11,726 / 16,148 / 737
  ([IXI](evidence/vedo_production/ixi/patch_eligibility.json),
  [TopBrain](evidence/vedo_production/topbrain/patch_eligibility.json)).
- One IXI training patch (`sample_000050_0096`) had a 2-node graph on a mask with zero foreground
  voxels; it was excluded.
- No active patch has more than 256 graph nodes.
- Stated caveat: patch metrics are **conditional on a vessel being present**. They say nothing
  about empty regions; any full-grid or whole-volume evaluation must be reported separately.

**[DONE] IXI training subset.** The IXI-only 256-query config
(`configs/finetune_ixi_vessels_q256.yaml`) takes a deterministic 8,000 of the graph-positive
training crops instead of re-extracting with less overlap; TopBrain's ~7.4k is used as is
([IXI capacity audit](evidence/ixi_capacity_192_256/README.md)).

**[DONE] Spline-target exclusions.** The K8 spline exporter (a K8 spline is a cubic B-spline with
8 control points; see the representation history) excludes the whole patch when it contains a
closed or parallel branch the decoder cannot represent. Such a branch would otherwise appear as a
duplicate undirected edge and crash preparation. Exclusions are listed in the target manifest.
Spline training was capped at 10,000 training and 2,000 fixed validation candidates
([spline representation](../archive/legacy-20261103/docs/SPLINE_REPRESENTATION.md)).

## 6. False loops in the segmentation-derived topology

**[PROPOSED, never run] Raw-MRA validation of false bridges.** 3D review of real IXI crops showed
dense tangles of cycles. These come from the segmentation (fused neighbouring vessels or
26-neighbour digital contacts), not from simplification: radius-adaptive simplification fixed
off-vessel chords but left β₁ unchanged
([loop mitigation](../archive/legacy-20261103/docs/research/loop_failure_mitigation.md)). The
proposal:
1. contract degree-2 chains and compute a cycle basis;
2. consider only edges whose removal lowers β₁ without raising β₀;
3. score each bridge by locally normalised raw-image support, radius bottleneck and tangent
   continuity (absolute intensity thresholds are forbidden across sites);
4. remove a bridge only when clearly weaker than both arms, and log every removal.

Required before use: a manually labelled stratified sample, precision-first false-bridge
detection (so that the Circle of Willis is not cut), and before/after metrics. The tracker said
training-data generation would stay paused until this was done. In practice production went
ahead with the mask-containment rule only, and no cleaning was ever implemented. The same open
question appears as "Q33 loop policy: undecided" in [`../IXI_DATASET.md`](../IXI_DATASET.md) §8.
The CaravelMetrics-style repairs (orphan linking, Laplacian smoothing, triangle deletion) were
explicitly *not* adopted as graph repair (tracker D08).

## 7. Storage and retention

**[DONE] Evidence curation (21–23 Sept).** Superseded pilots, staging copies, duplicated Vedo
smoke graphs and redundant viewers were removed from `docs/research/evidence/` (about 25 MB),
keeping the 30-volume study, CT-001 comparison, production checks, capacity benchmarks and metric
checks ([tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)).

**[DONE] SCRATCH cleanup, 16 Oct.** 70 reviewed files deleted (100.38 GiB, mostly smoke and
cancelled-run checkpoints). 43 startup or abandoned run directories (1,925 files) were packed into
a byte-verified archive, with `ARCHIVED.md` restore notes left in place. 119 retained checkpoints
were verified unchanged. No dataset, target cache, comparison checkpoint or final test prediction
was removed. The archive is on SCRATCH, so it is **not a durable backup**
([cleanup 16 Oct](../archive/legacy-20261103/docs/research/storage/cleanup-20261016.md)). The
retained runs (source-transfer, overfit, scratch-ablation and frozen-head pilots under
`$SCRATCH/experiments/gnbm/spline_learning/`) are listed in the
[retained register](../archive/legacy-20261103/docs/research/storage/retained-experiments.md).

**[DONE] WORK file-count cleanup, 25 and 27 Oct.** The WORK *file-count* quota, not bytes, was
the limit: 489,714 / 500,000 entries at the start, of which this account owned at most 83,042,
mostly editor installs. 8,128 loose NPZ prediction files were zipped into three SHA256-checked
archives and an old copied runtime was removed, bringing WORK to 480,466 entries. On 27 Oct, 204
untracked secondary files (old launchers, images, SWC exports) went into
`experiments/storage_cleanup/20261027/secondary-artifacts.zip`, with a `restore.py` that refuses
to overwrite differing files
([cleanup 25 Oct](../archive/legacy-20261103/docs/research/storage/cleanup-20261025.md)).
SCRATCH was at 0.34% of bytes and 2.91% of entries, so deleting SCRATCH data would not have
helped WORK.

**[DONE] Environment rules learned the hard way (Sept).** A `uv` download cache on WORK added about
93,000 files and caused `EDQUOT`. Since then `UV_CACHE_DIR` and `TORCH_EXTENSIONS_DIR` go on
SCRATCH, and the ~31k-file venv lives on SCRATCH behind a WORK symlink
([old repo map](../archive/legacy-20261103/docs/repo-graph.md)).

**Retention rules adopted** (cleanup 16 Oct, still the policy):
- keep config, dataset IDs, initialisation, metrics and interpretation for every comparison,
  negative results included;
- keep best and final/resume checkpoints of current comparisons, with optimiser state;
- delete startup-only checkpoints after a successful smoke test, keeping its log;
- archive superseded attempts only after checksum verification;
- large arrays on SCRATCH, compact reports in Git.

## What carried forward / what was dropped

Carried forward to the current dense model:
- the Vedo-original, `optimal_radius_0p75x_c095` IXI and TopBrain graphs and patch sets, with
  full-volume extraction, exact-boundary crops and patient splits;
- the original dense centreline paths (carried in the spline target caches as
  `full_paths`/`path_offsets`) as the geometric ground truth for curve metrics;
- per-split graph-positive eligibility, so reported patch metrics are conditional on vessels
  being present;
- the storage rules above.

Dropped:
- the legacy-backend production (deleted);
- training-only filtering;
- crop-first extraction and last-inside-sample cropping;
- any automatic loop repair. False segmentation loops remain an open, unmeasured source of label
  noise in IXI/TopBrain GT topology, and β₁ metrics inherit it.

## Sources

Archived originals (post-move paths):
- [IMPLEMENTATION_TRACKER.md](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)
  — task list T01–T11, decision log D01–D08, evidence log, dated extraction/production notes.
- [VEDO_PRODUCTION_RUN.md](../archive/legacy-20261103/docs/research/VEDO_PRODUCTION_RUN.md) —
  Vedo-original production, memory retry, per-split eligibility and final validation.
- [loop_failure_mitigation.md](../archive/legacy-20261103/docs/research/loop_failure_mitigation.md)
  — failure classes (off-vessel chords vs false topology) and the unexecuted bridge-validation plan.
- [storage/cleanup-20261016.md](../archive/legacy-20261103/docs/research/storage/cleanup-20261016.md)
  — SCRATCH cleanup and retention rules.
- [storage/cleanup-20261025.md](../archive/legacy-20261103/docs/research/storage/cleanup-20261025.md)
  — WORK file-count audit, NPZ archiving, 27 Oct decluttering.
- [storage/retained-experiments.md](../archive/legacy-20261103/docs/research/storage/retained-experiments.md)
  — register of retained spline-learning runs and checkpoint counts.
- [SPLINE_REPRESENTATION.md](../archive/legacy-20261103/docs/SPLINE_REPRESENTATION.md) — K8
  spline target rules and patch exclusions; its links now point to
  `projects/archive/graph-representation/docs/`.
- [repo-graph.md](../archive/legacy-20261103/docs/repo-graph.md) — early repository map of the
  fork (dataset copies, WORK inode lessons); skimmed.

Evidence kept in place: [vedo_original_ct001](evidence/vedo_original_ct001/full_volume/README.md),
[vedo_production](evidence/vedo_production/final_validation.json),
[full_dataset_production](evidence/full_dataset_production/summary.json),
[ixi_capacity_192_256](evidence/ixi_capacity_192_256/README.md),
[T02](evidence/T02/annotation_selection.csv).
