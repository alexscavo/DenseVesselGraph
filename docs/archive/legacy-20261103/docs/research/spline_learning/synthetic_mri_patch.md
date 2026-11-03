# SyntheticMRI: the same continuous-branch viewer as IXI

Download [synthetic_mri_patch.html](../experiments/synthetic_mri_patch/results/synthetic_mri_patch.html) and open it in your local browser. This single file embeds Plotly, the MRI patch, mask mesh, and fits; no server, assets folder, or internet connection is needed. Size: approximately 42 MiB after adding Catmull–Rom.

The initial representation is the **clamped cubic B-spline with 10 total controls per branch**. Two controls are the fixed graph endpoints; the eight internal controls add **24 scalar geometry values per 3D branch**. They are edge parameters, not additional graph nodes.

## Controls and comparisons

The layout and interactions match the [IXI experiment](ixi_patch.md): rotate/zoom the 3D view, switch branches and curve families, compare the original and mildly smoothed guides, show MRI slices and the vessel mask, enlarge or label nodes, focus a node, and edit internal control coordinates under **Edit edge parameters**. Pink points are spatial curve controls; gold, blue, and purple markers are junctions, terminations, and crop boundaries. The sine and Chebyshev alternatives edit coefficient vectors rather than spatial points.

The same 23 variants and metric definitions are used, including B-splines and centripetal Catmull–Rom with 6, 8, 10, 12, and 16 total points. B-splines use least-squares fits; Catmull–Rom interpolates equal arc-length guide samples. Whole-patch tables use unweighted means across branches. Saved automatic fits remain in the comparison tables after a manual edit; **Restore fit** resets the edited branch.

## Data and coordinate checks

- SyntheticMRI subject `13`, validation patch `sample_000013_0002`. Selection before fitting: first validation index row with 20–40 stored graph nodes and more than 1,500 foreground voxels. This is an illustrative patch, not a dataset benchmark.
- Actual source: `/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary`. That path describes the original study; the dataset has since been restored under `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI`. The Catmull–Rom update reuses the exact embedded original case and does not re-read or re-extract it.
- The prepared input is 64³ voxels: a 54³ source crop starting at `(0, 0, 78)`, with five zero-padding voxels on each side. It contains 2,215 foreground voxels.
- Physical coordinates use the source MRI affine, with 1 mm voxel spacing, and the padded crop origin. The prepared patch's identity affine is not used to locate source graph paths.
- The stored generation configuration is version 3. Its closed voxel-center crop convention is reproduced, rather than applying the later version 4 voxel-cell convention. Reconstructed sparse node positions are checked against the stored VTP coordinates, including their normalization by 64.
- Full generator edge polylines are clipped and retained before contracting degree-two chains. Replacing them with node-to-node chords would discard the very bends under study.

There are **167 dense path nodes / 164 edges**, contracted to **22 anchors / 19 continuous branches**. Seven anchors are crop boundaries. Three connected components and zero cycles are retained; no isolated nodes or tangent contacts were removed. The stored training graph also has 22 nodes here, so contraction does not reduce its node count in this particular patch. The experiment supplies topology and endpoints; it does not test prediction by a network.

## Observations on this patch

Against the original generator guide, the 10-control B-spline gives:

| Quantity | Unweighted mean across 19 branches |
| --- | ---: |
| Symmetric guide distance | 0.0153 mm |
| Absolute tortuosity difference | 0.0014 |
| Curve samples inside existing mask | 99.63% |
| Geometry values over the patch | 456 |

For context, the single cubic Bézier gives 0.0715 mm mean guide distance with 114 geometry values. The 16-control B-spline gives 0.0045 mm with 798 values. The initially selected branch is the most tortuous original guide among branches at least 5 mm long, chosen independently of fit errors. Even this branch has tortuosity only about 1.03: this patch is a useful transfer check, but a weak stress test for highly tortuous vessels.

These are agreement measurements against **supplied generator geometry**, not proof of image accuracy. One guide has only about 93% mask support; disagreement should remain visible rather than being hidden or treated as a fitting failure. The existing mask is not an independent anatomical reference. The alternate guide uses the same five mask-constrained Laplacian iterations as IXI, with weight 0.35 and fixed endpoints. It is a sensitivity check, not a corrected annotation.

## Files and reproduction

Documentation lives here; experiment code and outputs are separate:

- [Build script](../experiments/synthetic_mri_patch/src/build.py), reusing the IXI fitting functions and viewer sources.
- [Image and fit overview](../experiments/synthetic_mri_patch/results/overview.png) and [node overview](../experiments/synthetic_mri_patch/results/nodes.png).
- [Metrics](../experiments/synthetic_mri_patch/results/metrics.csv): 684 records (19 branches × 2 guides × 18 variants).
- [Provenance and source hashes](../experiments/synthetic_mri_patch/results/provenance.json).

From the projects directory on the cluster:

```bash
MPLCONFIGDIR=/tmp/gnbm-curve-mpl "$WORK/venvs/vascular-graph-extraction-h100-torch231/bin/python" graph-representation/experiments/synthetic_mri_patch/src/build.py
```

The builder accepts `--dataset-root`, `--sample`, and `--output`. It expects this prepared version 3 dataset and source graph files. It writes only experiment outputs and does not run training.

Verification: source/patch coordinate and mask-count checks passed; image overlays and node projections were inspected. A JavaScript smoke check confirmed the initial spline, eight editable internal controls, edit/reset, family switching, and node focus. The JavaScript spline agreed with the Python fit at checked points within numerical precision. Browser/WebGL rendering was not tested on the cluster.

## Catmull–Rom comparison

The equal-budget orange overlay and editable interpolation points are now included. See [the construction, comparison metrics and download instructions](catmull-rom-single-cases.md). These are fitting demonstrations, not learned predictions.
