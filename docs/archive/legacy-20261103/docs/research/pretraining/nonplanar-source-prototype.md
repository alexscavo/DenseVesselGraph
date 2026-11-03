# Two-sample nonplanar plant prototype

## Question

Can we move plant nodes away from their original plane while applying the same deformation to the raw image and segmentation, then use the existing random rotations?

This is a separate rendering experiment. Existing datasets, training configurations and results are unchanged. It does not evaluate a model or demonstrate a transfer-learning improvement.

## Construction

Use two plant training patches selected deterministically: first two sample IDs in lexical order among existing accepted patches with at least six nodes and five edges. This structural eligibility makes nonplanarity observable; no selection by model performance. Read each through the actual `PlantsDataset`, including the original five-slice slab embedding, padding and axis convention.

1. Sample independent Gaussian depth offsets at nodes (standard deviation 5 voxels), then remove their mean.
2. Fit a smooth Gaussian radial-basis displacement field over the image plane (length scale 10 voxels, ridge 0.05). Smoothing means realized node shifts need not exactly equal the independent draws.
3. Apply one common amplitude scale if necessary to bound absolute displacement at 10 voxels and maximum depth slope at 1. This limits extreme stretching from neighbouring nodes with opposing shifts. Requested and realized offsets are recorded.
4. Deform every graph node and densely sampled reference path using `F(x,y,z)=(x,y,z+h(x,y))`. The graph's edge indices remain unchanged.
5. Resample the raw volume with the inverse mapping `I_new(x,y,z)=I_old(x,y,z-h(x,y))`, using linear interpolation. Warp the segmentation with the same mapping and nearest-neighbour interpolation. Never independently move nodes while leaving image pixels fixed.
6. Apply the existing quarter-turn and flip implementation identically to the image, mask, nodes and sampled paths. Coordinates follow the repository's voxel-index/axis-size convention.

The continuous warp has unit Jacobian determinant and an explicit inverse, so it does not fold space. Voxel resampling can still alter tiny details, and the prototype reports guide support in the masks before/after. Graph connectivity is preserved exactly, but branch lengths and directions change intentionally. Crop-face nodes on the original in-plane boundaries stay on those boundaries before rotation.

## Scope and limitations

This bends a planar slab into a nonplanar **height-field sheet**. It is a useful first augmentation but does not create arbitrary 3D anatomy or depth-separated projected crossings. Thickness remains the existing five-slice extrusion, now warped; no noise or tubular re-rendering is added in this experiment.

Dense paths are warped directly. Do NOT simply apply this nonlinear map to B-spline control points and assume the resulting spline is the exact transformed curve: a future training integration should refit controls to warped paths and recheck fit/coordinate units. The prototype exports nodes and warped paths, not misleading transformed K8 spline targets.

## Artifacts and reproduction

Code: `experiments/source_augmentation/prototype_nonplanar_plants.py`.
Run in the project's fixed training environment:

```bash
python experiments/source_augmentation/prototype_nonplanar_plants.py
```

Output (separate from original datasets): `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/source_nonplanar_20261016/`.

- `nonplanar-plants.html`: self-contained interactive 3D comparison; download this single file and open locally. Original, warped, and warped+rotated panels. Select the sample; drag/zoom and toggle raw surface/reference paths/nodes in the legend.
- Per-sample PNG: raw-image isosurface + graph, plus side projection of the actual raw volume.
- Per-sample NPZ: original, warped and rotated raw/mask/node/path arrays, original edge indices and depth displacement field.
- `report.json`: parameters, realized displacements, guide-mask support and distance from the best-fit plane.

## Measured results (2026-10-16)

Seed: `364505`. Both samples completed; finite arrays, nodes within bounds, nonplanarity and rotation consistency checks passed.

| Sample | Nodes / edges | Guide points inside mask: original → warped → rotated | RMS distance from best-fit plane |
| --- | --- | --- | --- |
| sample_000001 | 8 / 6 | 98.96% → 98.96% → 98.96% | 1.80 voxels |
| sample_000002 | 10 / 7 | 93.08% → 93.08% → 93.08% | 2.23 voxels |

Actual node depth offsets span −2.94 to +3.02 voxels and −4.13 to +3.71 voxels, respectively. These are realized shifts after smoothing and amplitude limiting, not the initial Gaussian draws.

Both PNG renders were inspected: the raw structure and graph bend together and remain aligned after rotation. Existing imperfect guide-mask agreement remains; this augmentation does not repair source annotations. The quantitative check measures sampled guide support in the binary mask, not perfect raw-image correspondence or topology after voxelization.

Conclusion: this shared deformation is feasible on these two examples. It demonstrates consistent nonplanar augmentation, not improved learning or transfer. Before training integration, refit continuous targets to warped paths and check the resulting fit error. Keep a matched planar control when assessing any training benefit.

The small machine-readable result is also tracked at `experiments/source_augmentation/report.json`; large visualizations and arrays remain in the output directory above.
