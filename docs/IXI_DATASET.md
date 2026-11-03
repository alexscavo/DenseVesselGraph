# IXI vessel-graph dataset

Real brain-MRA target domain built from the IXI collection: full-resolution MRA
volumes, vessel segmentations and brain masks, reduced to sparse vessel graphs
and cut into 64-cubed patches in the layout the SyntheticMRI loader already
consumes.

**Historical research note.** Sections 1–8 describe the earlier IXI dataset
and extraction experiments. The old commands in sections 9–10 are superseded
and their one-off scripts were removed. The current 170-volume IXI and
50-volume TopBrain workflow, with both selectable centerline backends, is
documented in [the scripts guide](../scripts/README.md) and
[the production run tracker](archive/legacy-20261103/docs/research/VEDO_PRODUCTION_RUN.md). The current
Vedo graphs and single-patch train/validation/test sets are complete.

## 1. Purpose and scope

IXI is the **real-vessel target domain** — the dataset finetuning runs on and
final numbers are reported against. It replaces or complements `synthetic_mri`
in that role; it is not a pretraining source like `plants`.

That decision sets the fidelity bar. Because IXI supplies the reported test
number, its ground truth has to be defensible on its own terms, and its
conventions have to match `synthetic_mri` closely enough that a transfer number
measures a domain gap rather than a convention mismatch. Both constraints drove
the parameter choices in §7.

Two properties of the target repository shape the ground truth:

- `training.input: image` in `configs/base.yaml` — the model is fed the **raw
  MRA**, never the segmentation. Ground truth therefore has to describe what is
  in the image, not what is in the mask. Where the two disagree, the mask is a
  lossy annotation rather than the phenomenon.
- `decoder.object_queries: 120` — a hard ceiling on representable nodes per
  patch. Every node-density choice in §7 is measured against it.

## 2. Source data inventory

```text
$SCRATCH/datasets/IXI_dataset/
├── volumes/          569 files   IXI<n>-<site>-<id>-MRA.nii.gz        (~15.6 MB each)
├── segmentations/    405 files   IXI<n>-<site>-<id>-MRA.nii.gz        (binary vessel labels)
└── Brain_masks/      570 files   IXI<n>-<site>-<id>-MRA_mask.nii.gz   (binary, uint8)
```

`volumes/` was flattened by hard-link out of the nested upload tree
(`<n>_<site>/MRA/NIfTI/<subject>-MRA.nii.gz`); the inner filenames already
carried the canonical subject id that matches the annotation naming. Hard-links
cost no extra disk and left the original tree intact.

**397 subjects are complete** (volume + segmentation + brain mask):

| site | complete subjects |
|---|---|
| Guys | 149 |
| HH   | 177 |
| IOP  | 71  |
| **total** | **397** |

Volume geometry is heterogeneous — 326 volumes at `(512, 512, 100)`, 72 at
`(1024, 1024, 92)`, 1 at `(1024, 1024, 91)`.

### Data-quality findings

| finding | detail | resolution |
|---|---|---|
| `IXI371-IOP-0970` | segmentation and brain mask present, **no raw volume** | excluded; genuinely unusable |
| `IXI035AAA-IOP-0873` | **byte-identical** to `IXI035-IOP-0873` | redundant copy, no ambiguity |
| six `*-MRA.nii(1).gz` | browser duplicate-download suffix; each has a properly-named twin already present | see below |

The `(1)` files add no subjects. Four are noise-level duplicates of their twin;
**two are materially different annotations of the same subject**:

| subject | `(1)` voxels | proper voxels | IoU |
|---|---|---|---|
| IXI522-HH-2453 | 66,188 | 66,188 | 1.0000 |
| IXI524-HH-2412 | 60,362 | 60,104 | 0.9952 |
| IXI577-HH-2661 | 58,538 | 58,405 | 0.9944 |
| IXI586-HH-2451 | 60,965 | 60,997 | 0.9807 |
| **IXI638-HH-2786** | **91,747** | **66,407** | **0.4721** |
| **IXI661-HH-2788** | **73,598** | **60,229** | **0.5339** |

For `IXI638` and `IXI661` the `(1)` version is the better annotation on
image-intensity evidence: its exclusive voxels sit on bright MRA signal
(135.4 and 102.7 mean intensity, comparable to the 125.5 of voxels both
versions agree on), whereas the properly-named file's exclusive voxels sit at
or below the background 99th percentile (57.7 and 62.6 — dark, not vessel).
Despite being 38% and 22% larger, the `(1)` versions also carry a *smaller*
fraction of dark labelled voxels (16.9% vs 26.3%; 21.3% vs 23.7%).

This is a heuristic from image intensity, not ground truth. **Decision: use the
`(1)` version for those two subjects.** No files have been renamed or deleted;
the selection is enforced by `prepare_ixi_sources.segmentation_path` and recorded
as `source_segmentation` plus `segmentation_variant` in generated provenance.

## 3. Header repair

**167 of 405 segmentations had their spatial metadata stripped** to unit
spacing. The true geometry survives in the matching brain mask:

| segmentation zooms (mm) | brain-mask zooms (mm) | subjects |
|---|---|---|
| (0.469, 0.469, 0.8) | same | 175 |
| **(1.0, 1.0, 1.0)** | **(0.469, 0.469, 0.8)** | **147** |
| (0.264, 0.264, 0.8) | same | 56 |
| **(1.0, 1.0, 1.0)** | **(0.264, 0.264, 0.8)** | **15** |
| **(1.0, 1.0, 1.0)** | **(0.488, 0.488, 0.8)** | **4** |
| **(1.0, 1.0, 1.0)** | **(0.264, 0.264, 0.809)** | **1** |

Verification: unit-spacing segmentations have a differing affine in **167/167**
cases and a **matching array shape in 167/167** cases. Among the 231
segmentations that kept real spacing, zooms agree with the brain mask in
**231/231** cases.

The stripped headers also concealed spacings that the segmentation files alone
never revealed (0.488 in-plane; 0.809 through-plane).

**Repair rule** (`prepare_ixi_sources.load_geometry`): when a segmentation
reports exactly unit spacing and its array shape matches its brain mask, adopt
the brain mask's affine. Recorded per subject as `header_repaired` in the
completion marker.

A residual: 64 subjects have agreeing zooms but still-differing affines (smaller
discrepancies, not characterised).

## 4. Brain masks and extracranial vessels

Roughly **a quarter of all annotated vessel voxels lie outside the brain mask**
(mean 24.3%, median 22.1%, range 0.1–68.3% over 25 subjects).

This is **anatomy, not misregistration**:

- Subjects whose segmentation/mask affines *agree*: **28.6%** outside.
  Subjects whose affines *disagree*: **27.6%** outside. Indistinguishable — if
  misregistration were the cause, the identical-affine group would sit near zero.
- **83–86% of the outside voxels sit in the lowest quarter of slices**, 0–2% in
  the highest quarter. That is the carotid and vertebral territory at the neck
  and skull base, exactly what brain extraction is built to strip.

The annotations were made on the **full field of view, skull and neck
included**. Annotated voxels outside the brain mask are **14.6× brighter than
their surrounding background** — genuine MRA flow signal, and typically brighter
than the vessels *inside* the mask (546 vs 378, 804 vs 552, 747 vs 526), because
the extracranial feeders are the largest, fastest-flow vessels in the
acquisition.

Two subjects have *more* annotated vessel outside the mask than inside:
`IXI050-Guys-0711` (44,580 vs 16,040) and `IXI019-Guys-0702` (72,812 vs
19,778). Brain masking would destroy roughly three-quarters of their annotation.

**Decision: `--brain-mask none` by default.** Masking is not a clean-up step
here — it deletes about a quarter of the ground truth, the most reliable
quarter. "Intracranial vessels only" remains a legitimate benchmark definition,
so the option exists (`--brain-mask zero`, applied to **both** raw and
segmentation so the target never asserts vessel where the image is blank), but
it is a scoping decision, not preprocessing.

## 5. The extraction algorithm

`scripts/ixi_vessel_graph.py`, `build_vessel_graph()`. Stages in order:

1. **Skeletonization** — `skimage.morphology.skeletonize` on the binary
   segmentation. Homotopy-preserving: on the reference subject the skeleton's
   topology matches the mask's exactly (β₁ = 161 for both).

2. **Topology-faithful adjacency** — 26-neighbour adjacency, then
   **local-redundancy pruning**: an edge is removed when its endpoints remain
   joined by a path of length two or three confined to their shared 3×3×3
   neighbourhood. Such an edge is redundant by construction, so removal cannot
   disconnect anything or alter global topology.

   Necessary because naive 26-connectivity makes every diagonal run a clique —
   three voxels stepping diagonally are all mutually adjacent, so a straight
   segment becomes a triangle. Effect: **β₁ 732 → 168** against a true 161,
   564 edges removed, all 22 components preserved.

3. **Spur pruning** (`spur_length = 4` voxels) — terminal branches of at most
   four voxels ending in a junction are dropped, iteratively. Skeletonization
   produces short hairs at vessel boundaries; without this the junction count
   roughly doubles (mean 50 anchors vs 29).

4. **Bounded junction clustering** (`max_junction_extent_mm = 1.5`) —
   26-adjacent degree-3+ voxels collapse into one node, but only while the
   cluster's spatial extent stays within the bound.

   Unbounded merging fuses *distinct* junctions wherever two vessels touch. On
   the reference subject one cluster reached **20 voxels spanning 4.79 mm with
   19 branches attached** — a fabricated hub joining branches that were never
   mutually connected. Bounding it drops max node degree **13 → 7**.

5. **Artifact self-loop removal** (`max_artifact_self_loop = 4` voxels) —
   clustering turns a branch that leaves and immediately re-enters the same
   cluster into a self-loop. Measured: 98 such loops, median length 2 voxels.
   Effect: **β₁ 257 → 162**. Self-loops longer than the threshold are kept —
   only two exceeded 6 voxels (max 31), and those are genuine small vascular
   loops.

6. **Branch extraction** — walk degree-2 chains between anchors (junctions and
   terminations). A ring made entirely of degree-2 voxels has no anatomical
   anchor, so it is represented by three deterministic degree-2 geometric nodes
   and three ordered arcs. This keeps it compatible with the downstream simple
   adjacency matrix while preserving its component and β₁ contribution.

7. **Constrained centerline smoothing** (`smooth_iterations = 5`,
   `smooth_alpha = 0.5`) — Laplacian smoothing with endpoints pinned and any
   point that would leave the mask reverted. See §7 for the sweep.

8. **RDP curvature nodes** — Douglas-Peucker selects interior points as
   degree-2 nodes so the retained polyline never deviates from the centerline by
   more than the tolerance. The tolerance is a hard geometric error bound, not a
   tuned threshold.

### Node placement fixes

- **Junction nodes snap to a real skeleton voxel** (the cluster member nearest
  the centroid), not to the centroid itself. An L-shaped or elongated junction's
  centroid can land off-vessel; that put **32 centerlines (2.6%) partly through
  background**, one as low as 50% contained. After the fix: **0**.
- **Degenerate zero-length edges removed** — two nodes at identical positions
  joined by an edge, carrying no geometry and breaking any length-normalised
  quantity downstream.

### Output format

`nodes.csv` / `edges.csv` / `graph.vvg`, the inherited Voreen source format,
with every position in **world coordinates** via the subject's repaired affine —
matching how the SyntheticMRI source graphs are stored (verified against
`upz73jr/datasets/syntheticMRI/graphs/10`). Round-trip through
`audit_synthetic_mri_grid.SourceGraph.from_directory` is verified.

## 6. The topology bug chain

Ground truth is trained against by H0/H1 Betti topology losses
(`configs/losses/betti.yaml`). A cycle-count error in the ground truth
corrupts precisely the term this repository is built around — and it would be
very hard to attribute later, because the graphs *look* correct in projection.

| stage | β₁ | mask truth |
|---|---|---|
| raw 26-adjacency | **732** | 161 |
| + local-redundancy pruning | **168** | 161 |
| − artifact self-loops | **162** | 161 |

Verified arithmetically: 671 edges − 530 nodes + 21 components = 162.

The mask's own topology was obtained from the Euler characteristic
(`skimage.measure.euler_number`, connectivity 3): χ = −138 with 23 components,
giving β₁ − β₂ = 161; the skeleton reproduces it exactly.

Both defects were digital-quantisation artifacts, and **both were invisible in
the maximum-intensity-projection overlay**, which collapses slices and hides
cycles. Visual checks and the Euler number catch different classes of error.

An intermediate attempt — a distance-stratified adjacency rewrite — made things
*worse* (β₁ 509 → 631) by removing within-pass filtering that was doing real
work. It was replaced by the local-redundancy rule above.

## 7. Parameter choices with evidence

| parameter | value | evidence |
|---|---|---|
| `spur_length` | 4 voxels | reproduces the reference junction distribution (mean 28.7 / p99 62 / max 64 vs Voreen's 26.0 / 61 / 64) |
| local-redundancy pruning | 3×3×3 neighbourhood | β₁ 732 → 168 vs true 161, components preserved |
| `max_artifact_self_loop` | 4 voxels | β₁ 257 → 162; threshold sweep: ≤3 → 165, ≤4 → 162, ≤6 → 160 |
| `max_junction_extent_mm` | 1.5 | max node degree 13 → 7 (see sweep below) |
| `smooth_iterations` / `smooth_alpha` | 5 / 0.5 | turn angle 45.0° → 5.6°, bracketing SyntheticMRI's 6.5° |
| `rdp_voxels` | 2.0 (0.937 mm) | fits the 120-query budget (see below) |
| `radius_fraction` | 0.0 (off) | **open** — see §8 |
| `brain_mask` | `none` | §4 |

### Centerline smoothing sweep

Reference: SyntheticMRI source centerlines turn by a median **6.5°** per step
with 0.1% of turns above 30°; a raw voxel skeleton turns by **45.0°** with
72.1% above 30°, because a 26-connected skeleton can only change direction in
coarse lattice increments. Douglas-Peucker selects nodes by deviation from a
chord, so on a raw staircase it samples quantisation corners rather than
anatomical bends.

| config | nodes | turn median | >30° | inside mask | β₁ |
|---|---|---|---|---|---|
| none | 1351 | 45.0° | 72.1% | 99.69% | 162 |
| 3 iter, α=0.5 | 1130 | 7.4° | 1.3% | 99.68% | 162 |
| **5 iter, α=0.5** | **1104** | **5.6°** | **1.0%** | **99.68%** | **162** |
| 8 iter, α=0.5 | 1074 | 4.5° | 1.1% | 99.68% | 162 |
| 5 iter, α=0.8 | 1077 | 4.6° | 1.0% | 99.68% | 162 |

β₁ is unchanged in every configuration (smoothing moves geometry, not topology)
and mask containment is unchanged (the constraint holds). Node count falls 18%,
which helps the query budget for free. There was no trade-off to make.

### RDP tolerance vs the 120-query budget

Whole volume, `IXI143-Guys-0785` (98,247 vessel voxels):

| configuration | nodes | edges | term | deg-2 | 3-way | 4-way | 5+ | β₀ | β₁ |
|---|---|---|---|---|---|---|---|---|---|
| junctions + terminations | 530 | 671 | 171 | 0 | 295 | 48 | 16 | 21 | 162 |
| degree-2, fixed 2.0 vox | 1102 | 1243 | 171 | 572 | 295 | 48 | 16 | 21 | 162 |
| degree-2, adaptive 0.8× r | 1658 | 1799 | 171 | 1128 | 295 | 48 | 16 | 21 | 162 |
| degree-2, adaptive 0.6× r | 1922 | 2063 | 171 | 1392 | 295 | 48 | 16 | 21 | 162 |

Per non-empty 64-cubed patch (including boundary nodes) — the figure that
matters, since `object_queries = 120`:

| configuration | mean | p99 | max | over 120 |
|---|---|---|---|---|
| junctions + terminations | 17.1 | 54 | 56 | 0.0% |
| **degree-2, fixed 2.0 vox** | **27.5** | **84** | **91** | **0.0%** |
| degree-2, adaptive 0.8× r | 37.2 | 119 | 131 | 1.0% |
| degree-2, adaptive 0.6× r | 42.0 | 136 | 151 | 3.4% |

For reference, SyntheticMRI ground truth averages 20.5 nodes per patch
(p99 35, max 43). The fixed 2.0-voxel setting places IXI at 27.5 — close enough
that pretrained checkpoints transfer into a familiar regime — with the whole
dataset inside the query budget.

The junction breakdown is identical across all four rows, as it must be: adding
degree-2 nodes subdivides edges without touching junctions or terminations.

### Junction-extent sweep

| max extent | nodes | edges | β₁ | max degree | deg≥5 | deg≥7 |
|---|---|---|---|---|---|---|
| unbounded | 1102 | 1243 | **162** | **13** | 16 | 3 |
| 2.0 mm | 1106 | 1251 | 166 | 8 | 17 | 4 |
| **1.5 mm** (default) | 1113 | 1258 | 166 | 7 | 15 | 3 |
| 1.0 mm | 1133 | 1279 | 167 | **6** | **7** | **0** |

Bounding the extent fixes implausible hubs but **raises** β₁, because merging
junction voxels into one node *collapses* small cycles into a point and hides
them. Unbounded clustering was not preserving correct topology; it was
concealing tangles by fusing them. This is a representation choice, not a
correctness fix. See §8.

### Geometric containment

Straight node-to-node edges are the ground truth (`write_vtp_graph` emits
one two-point line per edge); the dense centerline only places degree-2 nodes
and finds patch-face crossings.

| | junctions-only | degree-2, fixed 2.0 vox |
|---|---|---|
| median chord inside mask | 80% | **100%** |
| edges <50% inside | 30.6% | **4.6%** |
| max chord deviation | 14.24 mm | 0.937 mm |

Junctions-only ground truth is geometrically poor: nearly a third of its edges
spend more than half their length outside the vessel, and the worst chord misses
the centerline by 14 mm.

The residual 4.6% has a specific cause — the 0.937 mm tolerance **exceeds the
0.80 mm median vessel radius**, so a chord can satisfy its bound and still leave
the lumen. Radius-adaptive tolerance addresses it (4.60% → 0.56% at 0.8× local
radius) at the cost of 50% more nodes. Undecided; see §8.

Precision of the final configuration (`IXI143-Guys-0785`):

| metric | value |
|---|---|
| centerline points inside the vessel mask | 99.66% |
| median centredness (1.0 = perfectly centred in lumen) | 0.843 |
| max RDP deviation vs its 0.703 mm bound | 0.701 mm |
| mean RDP deviation | 0.263 mm |
| median vessel radius sampled | 0.80 mm |

Centredness spreads below 1.0 because at 0.80 mm radius these vessels are 3–4
voxels across and frequently have no unique centre voxel.

## 8. Known limitations and open questions

### Properties of the segmentation, not of the extractor

**162 loops per volume.** The mask's own topology is β₁ = 161; the extracted
graph reports 162. One cycle of disagreement across an entire brain — the loops
are in the annotation. But 161 loops far exceeds real cerebral anatomy (the
circle of Willis and a handful more), so the segmentation is very likely fusing
vessels that merely pass close together.

| cycle property | value |
|---|---|
| independent cycles | 162 |
| median size | 4 edges |
| median perimeter | 12.8 mm |
| tiny loops (<5 mm perimeter) | 17 (10%) |
| large loops (>30 mm) | 44 (27%) |
| waist ratio (thinnest/thickest edge) | median 0.506 |

**Centerlines connect vessels that should not be connected.** Observed by the
user in the 3D viewer. Since a skeleton can only follow mask voxels, this means
the segmentation fuses those vessels. Same root cause as the loops, and **not
fixable in the extractor**.

**16 nodes of degree 5+.** Anatomically implausible — real vessels bifurcate
3-way, occasionally 4-way. Partly addressed by bounding junction clustering
(§7), partly the same fusion artifacts.

### Open decisions

| question | options | status |
|---|---|---|
| **Q33 — loop policy** | keep loops (faithful to mask) / break tiny ones by perimeter / break by waist ratio / fix the segmentation upstream | **undecided** |
| RDP tolerance | fixed 2.0 vox (budget-safe, 4.18% of edges badly outside) vs adaptive 0.8× radius (0.56% outside, p99 119 against the 120 ceiling) | **undecided** |
| junction extent | 1.5 mm (max degree 7, β₁ 166) vs 1.0 mm (max degree 6, no degree-7+, β₁ 167) | **undecided**, 1.5 mm is the current default |

On Q33, no clean criterion has been established. The waist ratio was the first
instinct and the data refuted it: a unimodal distribution centred at 0.506 is
equally consistent with natural calibre taper along a branching tree, so it
offers no threshold to stand on.

A test that could separate the classes has been **proposed but not run**: sample
raw MRA intensity along each loop's thinnest edge, normalised against the two
vessels it joins. A genuine vessel bridge carries bright flow signal; a
segmentation bleed sits on dark voxels. A bimodal result yields a defensible
threshold; a unimodal one argues for leaving the loops alone and documenting
that IXI ground-truth topology inherits the segmentation's.

Note this does not conflict with the "mask is the sole authority" rule (§5):
that settles where vessels *are*; this uses the image to validate a topological
claim the mask makes. It is also supported by `training.input: image` — the
model never sees the mask, so ground truth diverging from the mask is not
automatically wrong.

Consequences worth weighing before deciding: the two error directions are
asymmetric (a kept false loop teaches the model to hallucinate connections; a
broken real loop teaches it to miss anastomoses, and for vessel graphs the
former is usually worse); breaking loops drives most patches to β₁ = 0, leaving
the H1 term with correct but sparse signal; and filtering IXI loops while
SyntheticMRI keeps its own makes β₁-based metrics mean different things across
the two domains.

### Untested

- **The 73 subjects at `(1024, 1024, 92)`** have never been run. They are 4× the
  voxels of the tested geometry, and their spacing is 0.264 mm rather than
  0.469 mm — so `spur_length` (4 voxels) and `max_junction_extent_mm` (1.5 mm)
  correspond to different physical and lattice scales there. Runtime is also
  unmeasured for them.
- **Patch generation on IXI sources.** `generate_synthetic_mri_dataset.py` has
  not been confirmed to run on this source tree. Specific risks: IXI affines
  carry rotation terms, unlike SyntheticMRI's clean `diag(-1, -1, 1)`, and the
  generator asserts normalised coordinates fall in `[-0.01, 1.01]`;
  `normalize_like_legacy` casts to `int32` and uses a MAD threshold tuned for
  SyntheticMRI intensities, while IXI MRA is `uint16` with a different
  distribution; and a 64-cubed voxel patch spans 30×30×51 mm here rather than
  isotropic.
- **Data-loader smoke test.** Patches have not been loaded through
  `SyntheticMRIDataset`, and no forward pass has been run.
- **Patch count and storage.** Unmeasured. At roughly 500 grid candidates per
  512² subject and 1350 per 1024² subject, expect on the order of 250,000
  candidates across 397 subjects, against 14,045 in the superseded
  `IXI_patches_final`. Run the stage-2 preflight before committing storage and
  inodes.
- **Mask provenance.** Unknown; the dataset originated as
  `/data/scavone/vesselFM_datasets/IXI_patches_final` on another machine. No
  record exists of whether the segmentations are manual, model-generated or
  filter-based. This belongs in any write-up as a caveat.

### Superseded

`$SCRATCH/datasets/IXI_patches_final` (14,045 pre-cut 64-cubed patches) is
**dead**. It has no subject provenance — no mapping from its integer directory
ids back to an IXI subject, its per-split ids restart from zero and collide, and
its ids are a filtered subsequence with gaps. Its splits cannot be reproduced
and its patient-disjointness cannot be verified. Nothing in this pipeline reads
from it.

## 9. Pipeline

Two stages. **Stage 1 emits exactly the layout
`audit_synthetic_mri_grid.discover_sources()` expects, so stage 2 reuses
`generate_synthetic_mri_dataset.py` unchanged** — inheriting its audited
endpoint grid, boundary-aware graph cropping, MAD normalisation, manifests,
fingerprints and per-patient resume, rather than a parallel implementation that
could drift from it.

```text
scripts/ixi_vessel_graph.py        graph extraction library
scripts/prepare_ixi_sources.py     stage 1: per-subject, shardable, resumable
scripts/make_ixi_split.py          site-stratified subject split
scripts/ixi_graph_report_3d.py     interactive 3D QC report
cluster/jean_zay/extract_ixi_sources.slurm   stage 1 array job
cluster/jean_zay/submit_ixi_extraction.sh    stage 1 launcher
cluster/jean_zay/generate_ixi_patches.slurm  stage 2 job
cluster/jean_zay/submit_ixi_patches.sh       stage 2 launcher
```

### Stage 1 — sources

```text
$SCRATCH/datasets/IXI_sources/
├── raw/<numeric_id>.nii.gz           float32, repaired affine
├── seg/<numeric_id>.nii.gz           uint8
├── graphs/
│   ├── junction_only/<numeric_id>/{nodes.csv, edges.csv, graph.vvg}
│   ├── adaptive/<numeric_id>/{nodes.csv, edges.csv, graph.vvg}
│   └── dense/<numeric_id>/{nodes.csv, edges.csv, graph.vvg}
├── subject_map.csv                   numeric id <-> IXI subject, site, provenance
└── .complete/<numeric_id>.json       per-subject marker with counts and Betti numbers
```

The three graph directories are separate comparison products generated with the
same topology settings. `junction_only` retains anchors, `adaptive` uses the
smoothed centerline plus RDP-selected degree-2 nodes, and `dense` preserves every
raw centerline sample after topology cleanup but before smoothing or RDP. The
patch generator automatically selects `graphs/adaptive/` as the compact training
target; the other representations are evaluation controls until the
representation study justifies a final choice.

Subject ids map to integers because the patch generator requires numeric patient
ids (`patient_token`). The mapping is deterministic (sorted subject order) and
recorded in `subject_map.csv` together with the originating file paths, so the
provenance gap that killed `IXI_patches_final` cannot recur.

```bash
cd $WORK/projects/Graph-Native-Betti-Matching

IXI_DRY_RUN=1 bash cluster/jean_zay/submit_ixi_extraction.sh   # preflight only

export IXI_ARRAY="0-19"                 # 20 shards
export IXI_CPUS=5                       # x 5 cpus = 100 parallel
export IXI_TIME="03:00:00"
export IXI_EXTRACT_ARGS="--workers 5"
bash cluster/jean_zay/submit_ixi_extraction.sh
```

Shard `i` of `n` handles subjects whose index is congruent to `i` mod `n`; each
subject writes its own files and completion marker, so shards never contend and
re-runs skip completed work. Within a shard, `--workers` drives a process pool.

Runs on `vnc@cpu` / `cpu_p1` / `qos_cpu-t3`. **The project venv's interpreter
runs standalone on `cpu_p1` with no modules loaded** — verified by `srun`:
numpy 1.24.4, scipy 1.10.1, skimage 0.24.0, nibabel 5.3.2 all import. This stage
needs no torch, so `cluster/jean_zay/env.sh` (which loads `arch/h100`) is not
used.

Cost: ~35 s per 512×512×100 subject, dominated by `skeletonize` and
`distance_transform_edt`. About 3.9 CPU-hours for all 397.

### Stage 2 — patches

```bash
$IXI_PYTHON scripts/make_ixi_split.py \
    --sources $IXI_SOURCES --output $IXI_SOURCES/ixi_split.csv

IXI_DRY_RUN=1 bash cluster/jean_zay/submit_ixi_patches.sh   # prints candidate count
bash cluster/jean_zay/submit_ixi_patches.sh
```

The split is **subject-level and stratified by site**. IXI was acquired at three
sites on different scanners, so an unstratified draw can concentrate one site in
the test split and turn a domain shift into apparent generalisation failure.
Sizes match what the generator validates: floor(70%) train, floor(15%) val,
remainder test — for 397 subjects, 277/59/61.

**This stage is not yet verified on IXI sources.** See §8.

## 10. QC artefacts

```text
$SCRATCH/experiments/ixi_qc/
├── A_precision_<subject>.png     slab overlay + centredness, RDP error, summary
├── B_mesh3d_<subject>.png        graph inside the vessel mesh, 3 viewpoints
├── graph3d_<subject>.html        self-contained interactive 3D report (~6 MB)
├── viz_precision.py  viz_mesh3d.py  viz_graph.py  count_patch_nodes.py
└── graph_counts.py               node/edge counts by degree, whole-volume and per patch
```

The interactive report carries independently toggleable layers, for each of the
junction-only, adaptive, and dense graph variants:

| layer | appearance | default |
|---|---|---|
| vessel surface mesh | grey, semi-transparent | on |
| edges (straight node-to-node, = the GT) | red | on |
| highlight loops | thick magenta, drawn **over** the full edge set | on |
| centerline (reference, **not** GT) | dotted teal | off |
| junctions / terminations / degree-2 nodes | orange / blue / green | on |

Hover any node for its id, degree and local vessel radius.

```bash
source cluster/jean_zay/env.sh
python scripts/ixi_graph_report_3d.py IXI019-Guys-0702
python scripts/ixi_graph_report_3d.py IXI002-Guys-0828 \
    --half 70 --half-z 25 --rdp-voxels 1.5 --radius-fraction 0.8
```

`--half`/`--half-z` size the region, `--mesh-step` decimates the surface for a
smaller file, `--full` renders the whole volume (tens of megabytes; a full
512×512×100 mesh is roughly 300k faces).

A full-brain maximum-intensity projection hides cycles — the topology bugs in §6
were invisible in it while the graphs looked correct. Use the Euler
characteristic and the loop-highlight layer alongside it, not instead of it.
