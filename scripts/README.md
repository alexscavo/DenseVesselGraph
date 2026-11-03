# Vessel data scripts

There are three entry points for the current IXI/TopBrain workflow:

| Task | Command | Output |
| --- | --- | --- |
| Full-volume graphs | `extract_vessel_graphs.py` | Three graph representations per volume in its dataset folder |
| Vessel patches | `extract_vessel_patches.py` | 64³ image/segmentation/graph triplets; ineligible triplets archived for every split |
| 3D review | `visualize_vessel_graphs.py` | Interactive HTML and a PNG with the segmentation mesh and graph node types |

`extract_vessel_graphs.py --centerline-backend vedo_original` is the chosen
pipeline. `--centerline-backend legacy` is the comparison option. Both use the
same graph derivation policy. The original author-provided
`vedo_extractor_original.py` and `vedo_extractor_batch.py` are retained as
reference sources; `vedo_original_backend.py` adapts the former to the current
segmentation geometry and records mask repairs.

The other current helpers have one purpose each: `make_real_graph_manifest.py`
validates image/label pairs, `ixi_vessel_graph.py` implements graph derivation,
`prepare_real_graph_patch_sources.py` stages completed graphs,
`audit_real_patch_eligibility.py` writes patch QC, and
`curate_vessel_training_patches.py` creates the active single-patch sets.
`extract_vessel_patches.py` runs these patch stages in order and reuses the
fork's `generate_synthetic_mri_dataset.py` for boundary-aware cropping.

To run all volumes on Jean Zay:

```bash
REAL_GRAPH_CENTERLINE_BACKEND=vedo_original bash cluster/jean_zay/submit_real_graph_manifest.sh
```

The launcher uses separate arrays: one CPU per IXI volume and 16 CPUs per
TopBrain volume, because Slurm memory scales with requested CPU count. The
manifest is `docs/research/vedo_original_extraction_manifest.csv`.

After the graph arrays succeed, generate each dataset's patches with:

```bash
python scripts/extract_vessel_patches.py --dataset ixi
python scripts/extract_vessel_patches.py --dataset topbrain
```

For Slurm, use `REAL_PATCH_DATASET=ixi` or `topbrain` with
`cluster/jean_zay/submit_real_graph_patches.sh`. The active patch sets are the
physical `train/`, `val/`, and `test/` triplets, described by
`active_{train,val,test}_index.csv`. Each active patch has foreground, at
least one graph node, and at least one graph edge. The original
`patch_index.csv` records the sampled grid for audit, not evaluation.

For a full-volume interactive review:

```bash
python scripts/visualize_vessel_graphs.py SUBJECT \
  --graph-dir DATASET/vascular_graphs/optimal_radius_0p75x_c095_vedo_original/MODALITY/SPLIT/SUBJECT/graphs/adaptive \
  --segmentation LABEL.nii.gz --output-dir REVIEW_DIR
```

For an individual patch, use `--patch-vtp PATCH_graph.vtp` and the matching
`--segmentation PATCH_seg.nii.gz`; the viewer converts normalized model
coordinates back to patch voxels and highlights boundary-intersection nodes.

```bash
python scripts/visualize_vessel_graphs.py PATCH_ID \
  --patch-vtp DATASET/vascular_patches/optimal_radius_0p75x_c095_vedo_original/test/vtp/PATCH_ID_graph.vtp \
  --segmentation DATASET/vascular_patches/optimal_radius_0p75x_c095_vedo_original/test/seg/PATCH_ID_seg.nii.gz \
  --output-dir REVIEW_DIR
```

The fork's original scripts remain in this directory:
`audit_synthetic_mri_grid.py`, `build_deformable_attention.sh`,
`compare_model_forward.py`, `diagnose_synthetic_mri_boundaries.py`,
`evaluate_random_patches_3d.sh`, `generate_synthetic_mri_dataset.py`,
`run_finetune_mri_experiment.sh`, and `visualize_graph_prediction_3d.py`.

The directory also holds the research scripts of the September–November
studies. The current dense model is `dense_graph_20261030.py` (stage 1, 16-NN
stage 2, training, evaluation), `dense_graph_traced_20261030.py` (traced
connectivity, adaptive K) and `dense_graph_2d_20261030.py`; the metric scripts
are listed in [`../docs/EVALUATION.md`](../docs/EVALUATION.md). Older study
scripts are kept because their recorded results refer to them; what each study
found is in [`../docs/research/HISTORY.md`](../docs/research/HISTORY.md).
