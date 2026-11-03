# Vedo production extraction run

Started 2026-09-22; completed and validated 2026-09-23. The exact original `vedo_extractor_original.py` centerline
algorithm is selected with `--centerline-backend vedo_original`. A wrapper
computes all orphan centroids in one SciPy call (same centroid definition),
projects outside centerline nodes into the vessel segmentation, routes short
invalid connections within the segmentation, and records any connection that
has no local route. The original file is unchanged. The graph derivation
policy is the same optimal adaptive radius 0.75 / contained chord 0.95 policy
as before. The alternative remains `--centerline-backend legacy`.

## Jobs and outputs (historical submission IDs)

| Stage | Slurm ID | State at submission | Output |
| --- | --- | --- | --- |
| All 220 full-volume graphs | `58593` (`0-219%8`, 1 CPU/task, 30 min/task) | initial run; nine TopBrain cases needed memory retry | `IXI/vascular_graphs/optimal_radius_0p75x_c095_vedo_original/`, `TopBrain.../vascular_graphs/optimal_radius_0p75x_c095_vedo_original/` |
| IXI patches | `58715` (12 CPUs, 1 h) | superseded by completed patch run | `IXI/vascular_patches/optimal_radius_0p75x_c095_vedo_original/` |
| TopBrain patches | `58727` (8 CPUs, 1 h) | superseded by completed patch run | `TopBrain.../vascular_patches/optimal_radius_0p75x_c095_vedo_original/` |

Dataset root: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets`.
Logs: `logs/vedo-original/real-graphs-58593_*.out` and
`logs/vedo-original/vessel-patches-{58715,58727}.out`.
Manifest: `docs/research/vedo_original_extraction_manifest.csv`.

The single-volume IXI smoke test (`IXI013-HH-1212-MRA_IXI-HH`) completed:
adaptive graph 1,375 nodes / 1,415 edges, zero dropped edges during repair.
Its copied graph files were removed from research evidence after the complete
dataset extraction passed validation; the production graph is in the IXI
dataset folder and the final audit is in
`docs/research/evidence/vedo_production/final_validation.json`.
The earlier simultaneous local CT and IXI smoke tests exhausted the local
command environment and killed the CT test. The standalone Slurm CT-001 Vedo
comparison had already completed; all 220 production cases now run as separate
CPU tasks.

## Resume and validation gates

1. Check `squeue -j 58593,58715,58727` and `sacct -j 58593,58715,58727`.
   A failed array task prevents the dependent patch jobs from starting.
2. Count and validate `complete.json` markers against all 220 manifest rows;
   every marker must say `centerline_backend=vedo_original`, and all three
   graph representations must have `nodes.csv`, `edges.csv`, and `graph.vvg`.
   Review projected/rerouted/dropped-edge counts by dataset and inspect
   outliers in a segmentation mesh viewer.
3. Check both patch `generation_summary.json` files, source manifests,
   `patch_index.csv`, active train index, and `training_view.json`. Confirm
   patient splits, nonempty active train graphs, and node counts against the
   model's object-token capacity. Inspect a few real 3D patch views.
4. Only after the new graphs and patches pass those gates, remove the four old
   `optimal_radius_0p75x_c095` graph/patch directories under IXI and
   TopBrain. (No old dataset-local graph-source directories exist.) They are
   retained for now because the patch jobs hard-link their unchanged raw/seg
   files while generating the new graph patches.
5. Keep only the final production QC and a small number of representative
   3D comparisons in `docs/research/evidence`; remove intermediate probe
   files after the full run is verified.

## 2026-09-23 memory retry

The first array completed 211/220 cases. Nine TopBrain cases failed with Slurm
OOM kills. A first retry using `--exclusive` completed only one of them: Slurm
accounting still showed `ReqMem=4000M`, because the job requested one CPU.
The earlier claim that these cases exceeded the full 191 GB node memory was
incorrect. The original Vedo code holds a float64 volume, an isosurface mesh,
and several full-volume distance transforms and masks at once; the one-CPU
4 GB allowance was insufficient for these larger cases.

Array `88235` reruns the eight missing shards
`170,171,173,176,185,189,190,210` with 16 CPUs per task (Slurm confirms
`ReqTRES=cpu=16,mem=62.50G`), at most two concurrent tasks. The new IXI and
TopBrain patch jobs are `88288` and `88290`, respectively, each depending on
successful completion of `88235`. The obsolete patch jobs with unsatisfiable
dependencies were canceled. The previous 212 successful outputs are retained.
Use `REAL_GRAPH_CPUS` with the manifest launcher for future large-volume runs;
its default remains one CPU for smaller cases.

The later validation and cleanup below supersede this in-progress checkpoint.

## Patch restart after graph completion

All 220 full-volume graph markers are now present (IXI 170, TopBrain 50).
The first dependent patch jobs `88288` and `88290` failed before producing
patches: staging rewrote a copied `patient_split.csv` with LF rather than the
old file's CRLF endings. Both files had identical patient assignments, but
their SHA-256 values differed and the hard-link reuse check rejected them.
Staging now preserves an existing validated split file, and the patch job
restores the old file bytes before staging. The failed jobs did not delete or
modify the old patch datasets. A second attempt (`89129`, `89133`) revealed
that the old generation configuration hashes the original combined-cohort
split CSV, whereas the dataset folders store filtered split CSVs. Reuse now
checks each patient's assignment against the old patch index and checks the
grid and normalization settings; it does not compare those incomparable byte
hashes. Corrected patch jobs `89202` (IXI) and `89203` (TopBrain) are queued.

## Patch evaluation scope

For the current research phase, train, validation, and test are evaluated as
individual vessel-containing patches. The same eligibility condition is used
for each split: segmentation foreground > 0, graph nodes > 0, and graph edges
> 0. Ineligible triplets are archived under `excluded_train`, `excluded_val`,
and `excluded_test`; `active_{train,val,test}_index.csv` names the patch sets
used by the model. The complete `patch_index.csv` remains an audit of what was
sampled, not an evaluation set. Any later full-grid or whole-volume evaluation
must be reported separately, because patch metrics conditional on a vessel
being present do not measure performance on empty regions.

The completed active patch sets contain IXI 13,742 train / 2,961 val / 2,384
test and TopBrain 7,391 train / 1,650 val / 2,685 test. All six active indices
have zero rows with empty foreground or missing graph nodes/edges, and zero
patches above 256 graph nodes. Physical `raw` file counts match the indices.
IXI QC records one train patch (`sample_000050_0096`) with an empty mask but a
2-node/1-edge graph; the mask was checked directly and contains zero foreground
voxels. That patch is archived from the active set. TopBrain has no such case.
The evidence is in `docs/research/evidence/vedo_production/{ixi,topbrain}/`.

The read-only final handoff check passed for all 220 graph markers, all 220
patch-patient markers, all six active indexes, and every active raw/seg/graph
triplet. Its report is `docs/research/evidence/vedo_production/final_validation.json`.
Cleanup job `90171` completed the repeated validation and removed the four
superseded `optimal_radius_0p75x_c095` graph/patch directories. The report now
says `old_directories_removed: true`. The current entry points and Slurm
launchers are documented in `scripts/README.md`; obsolete one-off research
scripts were removed after this run.
