# IXI patch: curve-valued branches with uncertain centerlines

Open [the interactive experiment](../experiments/ixi_patch/results/ixi_patch.html). This HTML is **self-contained**: download this one file and open it locally, with no assets folder or internet connection required. The expanded file is approximately 30 MiB because it embeds Plotly, image data, the vessel mesh, and all fitted curves.

Other outputs: [visual overview](../experiments/ixi_patch/results/overview.png), [per-branch metrics](../experiments/ixi_patch/results/metrics.csv), [source provenance and hashes](../experiments/ixi_patch/results/provenance.json).

## What was selected

- IXI033-HH-1259-MRA_IXI-HH, validation patch `sample_000001_0007`.
- Selection before fitting: first active validation patch from patient `000001` with 35–70 adaptive graph nodes and more than 1,500 foreground mask voxels. This is one illustrative patch, not a representative dataset sample.
- Native crop: start `(0, 107, 36)`, size `64 × 64 × 64`; source voxel spacing approximately `(0.46875, 0.46875, 0.8)` mm.
- The patch files have identity affines. Physical coordinates therefore come from the **source MRI affine and crop origin**, not the patch header.
- The existing dense graph was clipped with the project's voxel-cell crop convention, then degree-two chains were contracted while retaining their complete ordered paths. There are no rings in this patch.

The patch contains **272 dense nodes / 270 edges**, reduced to **14 anchors / 12 branches**, preserving two connected components and zero cycles. Three anchors are crop-boundary intersections and are explicitly marked in purple. The existing adaptive training graph has 45 nodes.

## Centerlines are guides, not ground truth

The study fits the available extracted centerlines to assess representation capacity. It does **not** establish whether those paths are anatomically correct. Fitting residuals measure agreement with extraction, and tortuosity differences measure agreement with its morphology.

The viewer includes the MRI volume as a movable slice and the existing vessel segmentation as a translucent surface. Mask support is the fraction of samples along the fitted curve inside that segmentation. The mask also informed graph extraction, so this is **not an independent anatomical validation**. Missing vessels, incorrect junctions, or biased centerlines can remain wrong even with excellent residual and mask support.

The alternate fitting guide applies five Laplacian smoothing iterations with weight 0.35. Endpoints are fixed, and an interior point moves only when its new position is inside the existing mask. This is an explicitly chosen sensitivity check, not a corrected annotation. Both extracted and smoothed fits remain available.

## Fitting and metrics

All representations share the same branch endpoints and connectivity:

| Representation | Geometry values per 3D branch, excluding graph anchors |
| --- | ---: |
| Straight chord | 0 |
| Single cubic Bézier | 6 |
| Cubic B-spline with `n` total controls | `3(n − 2)` |
| `s` piecewise cubic Bézier segments | `3(3s − 1)` |
| Sine series with `K` modes | `3K` |
| Endpoint-constrained Chebyshev basis with `K` terms | `3K` |

B-spline control counts are 6, 8, 10, 12, and 16; piecewise segment counts are 2, 3, and 4. The single cubic is equivalent to a clamped four-control cubic B-spline.

The added sine and Chebyshev families each use 2, 4, 6, and 8 coefficient vectors. There are now 18 variants and 432 fit records (12 branches × 2 guides × 18 variants). All 240 original fit records were checked and are unchanged.

## Additional function families

The sine representation adds endpoint-vanishing basis functions to the endpoint chord:

\[
\Gamma(t)=(1-t)A+tB+\sum_{k=1}^{K}a_k\sin(k\pi t).
\]

Each `a_k` is a 3D coefficient vector in mm. Low modes describe broad bends; higher modes permit finer oscillations. Changing a coefficient affects the whole branch. These coefficients are not spatial control points, so the viewer edits their X/Y/Z components without drawing them as pink points.

The polynomial alternative uses a Chebyshev basis with an endpoint constraint:

\[
\Gamma(t)=(1-t)A+tB+t(1-t)\sum_{k=0}^{K-1}a_kT_k(2t-1).
\]

Here `T_0(x)=1`, `T_1(x)=x`, and `T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)`. This spans polynomials of degree up to `K+1` with fixed endpoints. Two terms give exactly the same curve space as the single cubic Bézier; their fitted curves were checked for numerical equality. Higher degrees add flexibility, with potential oscillations. The Chebyshev basis itself supplies no smoothing penalty.

Both new families use the same 257 guide samples and endpoint-constrained least-squares objective as the single Bézier and B-spline fits. No regularization or tuning on mask support was added. This keeps the comparison about representation capacity.

At **18 free values per branch**, the selected long branch gives:

| Function | Residual to extracted guide (mm) | Fitted tortuosity | Existing mask support |
| --- | ---: | ---: | ---: |
| B-spline, 8 controls | 0.231 | 1.846 | 91.8% |
| Sine series, 6 modes | 0.242 | 1.838 | 94.2% |
| Chebyshev, 6 terms | 0.319 | 1.846 | 82.2% |

At this budget on this branch, the sine fit has slightly higher mask support while the B-spline has slightly lower guide residual. The polynomial fit has lower mask support despite a similar tortuosity ratio. These measurements do not identify a general winner or establish anatomical correctness. Global coefficients may be harder to adjust locally than B-spline controls; that is another useful distinction to explore interactively.

## Node visibility

The viewer now separates **junctions (gold)**, **terminations (blue)**, and **crop boundaries (purple diamonds)**, with larger adjustable markers and node IDs. Selected branch endpoints are labelled A/B. Click a node or use the node selector to inspect its coordinates, degree, and incident branches. Node IDs retain their indices in the cropped dense graph, so they are not consecutive 0–13 IDs.

Use **Nodes only** to remove occluding surfaces and lines, **Zoom to node** for a 6 mm neighborhood, and **Show full patch** to restore the extent. Mask opacity is adjustable and defaults to 10%. The [node overview](../experiments/ixi_patch/results/nodes.png) provides three projections; nearby nodes can overlap in a projection, so use the interactive inspector for those clusters.

## Sampling and metrics

Endpoint-constrained least squares uses 257 samples uniformly spaced along the guide's arc length. Piecewise fits use equal arc-length intervals with 129 samples per interval, sampling their internal joins from the guide and counting those joins as geometry parameters. Their joins guarantee position continuity only; smooth tangent matching is not imposed.

Metrics use physical coordinates and samples no more than 0.1 mm apart. Symmetric residual is the average of the two directed mean point-to-polyline distances. Tortuosity is length / endpoint chord. Mask membership uses the voxel cell containing each sample. Fitted curves are evaluated as dense polylines for these measurements. The patch summary uses unweighted means over all 12 branches; the selected-branch table makes long-branch failures visible that a mean over short branches can conceal.

## Initial observations

The viewer opens on branch 5, selected as the most tortuous extracted branch among those at least 5 mm long, without using curve-fitting results. Its extracted path length is 50.70 mm and tortuosity is 1.875.

Fits to this branch's extracted guide:

| Curve | Geometry values | Mean residual to guide (mm) | Fitted tortuosity | Inside existing mask |
| --- | ---: | ---: | ---: | ---: |
| Straight chord | 0 | 4.644 | 1.000 | 3.3% |
| Cubic Bézier | 6 | 1.149 | 1.647 | 28.4% |
| B-spline, 8 controls | 18 | 0.231 | 1.846 | 91.8% |
| Piecewise Bézier, 3 segments | 24 | 0.143 | 1.851 | 98.6% |

The extracted guide itself has 99.6% mask support. Mild smoothing changes its tortuosity from 1.875 to 1.859; the main bends persist. This supports testing richer curve families for this branch, without validating the extracted path as anatomy. The static overview shows MRI projections and how one cubic misses the larger bends.

Across all 12 branches, cubic Bézier has mean fitting residual 0.286 mm and mean mask support 85.6%; an eight-control B-spline gives 0.034 mm and 99.2%. The corresponding total edge geometry counts are 72 and 216 scalar values. All cases keep 14 object anchors. This reduces the anchor count relative to both dense (272) and adaptive (45) graphs, but **does not prove lower total network cost**: an eight-control spline on every short branch spends many unnecessary edge parameters.

The representation remains worth investigating. A single cubic is insufficient for the long winding branch in this patch; splines or several cubic segments can represent its large bends. The next useful study is a small set of additional patches, with image review and sensitivity to guide extraction/smoothing, followed by selecting curve complexity per branch. This one patch cannot settle anatomical accuracy, learnability, or generalization.

## Interaction and verification

Select a branch, fitting guide, and curve family. Toggle the MRI slice and change its orientation/index to inspect the vessel in depth. Click a pink control, or choose it under **Edit edge parameters**, and edit its X/Y/Z coordinates in physical mm. The curve and selected-branch measurements update. Restore fit returns to the saved least-squares result; comparison tables always show the saved fits.

Source images and datasets were read only. Endpoint constraints, graph edge coverage after contraction, patch dimensions, units, and foreground counts were checked during generation. JavaScript curve evaluation, residuals, and mask support were compared numerically with Python for the selected branch's cubic, spline, and piecewise fits; control editing and restoration were exercised with a mocked Plotly/DOM. The generated MRI/curve overview was visually inspected. A browser was unavailable on the cluster, so the full WebGL viewer still needs a local browser check.

For the added families, JavaScript and Python residuals, curve lengths, and mask support also matched for 2- and 6-term/mode fits. Coefficient edits/restoration and node selection, isolation, zoom, and extent reset were exercised with mocked Plotly/DOM. The node projections were visually inspected; the full browser interaction limitation still applies.

Reproduce from the shared projects directory using the existing scientific environment:

```bash
MPLCONFIGDIR=/tmp/gnbm-curve-mpl \
  "$WORK/venvs/vascular-graph-extraction-h100-torch231/bin/python" \
  graph-representation/experiments/ixi_patch/src/build.py
```

The builder, HTML template, and viewer code are in `experiments/ixi_patch/src/`; generated artifacts are in `experiments/ixi_patch/results/`. Documentation remains here in `docs/`.

## Catmull–Rom comparison

Centripetal Catmull–Rom with 6/8/10/12/16 total points is now available, alongside an equal-budget B-spline overlay. Volumes, guides, mesh, topology and original fits are unchanged. See [the construction and comparison notes](catmull-rom-single-cases.md).
