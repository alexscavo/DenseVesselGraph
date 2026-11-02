# Native 2D two-stage dense vessel-graph model — 30 Oct 2026

Port of the 3D WS3 dense model (`DenseVesselGraph/scripts/dense_graph_20261030.py`,
design doc `docs/research/network_proposals_20261030.md` §3–4) to native 2D, to test whether it
fixes real 2D data (AV-WIDE, REAVER) the way it fixed IXI/BRAVA in 3D.

- Code: `scripts/dense_graph_2d_20261030.py` (single file; `--cpu-checks` mode for the CPU/oracle checks)
- Launcher: `cluster/jean_zay/dense_graph_2d.slurm <dataset> [--run-name ...] [--updates ...]`
- Big outputs (model.pt, training.jsonl, per-crop validation pickles):
  `$SCRATCH/experiments/gnbm/dense_graph_2d_20261030/<dataset>/<run>/`
- Small outputs here: `cpu-checks.json`, `results/<dataset>-<run>-final.json`, `logs/`

## Data and evaluation (shared with the native2D RelationFormer long study)

- `StudyDataset`, `DATA`, splits, D4 augmentation (`native2d.data.transform`), 128² crops,
  `x/255 − 0.5`, `max_queries=96` exactly as `scripts/train_native2d_long.py`.
- Metrics: per crop `native2d.evaluation.positional` at 1/2/4 native units and
  `common_metrics` (spline = K8 `spline_controls`, chord = straight), aggregated with
  `train_native2d_long.aggregate` (same micro node/edge F1, same crop-macro K8 Branch/Curve F1).
  Pooled (micro) common-v2 Branch/Curve F1 is also reported (`spline_pooled`).
- Units: pixels (avwide, octasynth), micron (REAVER), as the native2D evaluation.
- Validation population: the full validation split (as `final-heldout.json`).
  Threshold-selection monitor: `subset()` of the training split (16 subjects × 4 crops;
  REAVER has 4 training subjects → 16 crops). Validation never selects anything; no test data.
- Reported: validation at fixed 0.5/0.5, validation at monitor-selected thresholds,
  and the training monitor at both.

## Deviations from the 3D design (all forced by 2D or by the native2D data/eval)

1. Input 3 channels (RGB PNGs), `x/255 − 0.5`; outputs 4 + 32 channels (2 offsets instead of 3).
2. 3×3 NMS; 3×3 centreline dilation; Gaussian σ = 1.5 px.
3. Corridor 24 × 11 samples (±12 px) in a 2D frame (t, u = rot90(t)); 2D corridor CNN
   `Conv2d(32→64) → Conv2d(s2) → Conv2d(s(1,2))` gives 64·12·3 = 2304 features
   (3D: 6912); pair vector 2304 + 3 (|d|, d) + 64 + 2 = 2373.
4. Geometry head outputs 6 × 2 local offsets (t, u) × 12 px; same clamped cubic K8 basis.
5. Coordinates are XY (x = column) so `grid_sample` needs no axis flip.
6. Geometry reference = the stored dense GT path per edge (`full_paths`/`path_offsets`),
   resampled to 64 points by arc length (the 3D cache stored 64-point guides); the
   centreline target rasterizes the dense path resampled to 256 points per edge.
7. Matching tolerance for labels = 2 native units (3D: 2 mm), GT jitter σ = 0.5 native units
   (3D: 0.5 mm). For REAVER (0.207 µm/px) this is 9.6 px and 2.4 px respectively; the
   chamfer loss is in native units (pixels, or µm for REAVER), as 3D used mm.
8. Node/edge mAP are computed with the native2D helper but on what this model scores:
   node boxes of the fixed 0.2 GT width at all peaks ≥ 0.10, edge boxes over the 16-NN
   candidate pairs of the retained nodes (RelationFormer: predicted box widths, all
   query pairs). `ranking` (RelationFormer-specific) is not computed.
9. Common-v2 curve metrics are computed in a CPU process pool (`--metric-workers`); the
   metric code itself is unchanged. Inference pair features are chunked (4096 pairs).
10. REAVER has 256 training crops; the same 15k × 8 budget = 120k exposures ≈ 469 passes
    (avwide ≈ 87, octasynth ≈ 11). Kept for budget comparability, heavy repetition noted.
