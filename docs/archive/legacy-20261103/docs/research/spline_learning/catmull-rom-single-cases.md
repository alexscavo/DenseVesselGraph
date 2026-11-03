# Catmull–Rom versus B-spline: existing single cases

## Open the experiments

Download just these two HTML files and open them locally in a browser:

- [SyntheticMRI](../experiments/synthetic_mri_patch/results/synthetic_mri_patch.html)
- [IXI](../experiments/ixi_patch/results/ixi_patch.html)

They embed the plotting library, MRI volume, mask mesh, guides, fit data and
viewer code. No assets folder, server or network connection is required.

Select **Catmull–Rom: 8 interpolation points** or **10 interpolation points**.
The orange overlay shows the B-spline with the same point budget; switching
back to B-spline reverses the comparison. Pink points belong to the editable
family, orange points to the saved comparison. Click a pink point and edit its
X/Y/Z coordinates under **Edit edge parameters**. Catmull–Rom follows that
point exactly. **Restore fit** returns to the guide-point interpolation.
The ordinary branch/guide selectors, MRI slices, mask and graph nodes remain.

Static comparisons, on the original selected branch, are also available:
[SyntheticMRI image](../experiments/synthetic_mri_patch/results/catmull_rom_comparison.png)
and [IXI image](../experiments/ixi_patch/results/catmull_rom_comparison.png).

## Representation and parameter count

Catmull–Rom passes through its ordered interpolation points. An interior
B-spline control is a handle and usually does not lie on its curve.
These Catmull–Rom points are **edge geometry parameters**, not extra graph
junction/termination nodes or extra object tokens.

Use centripetal knots:

`t[i+1] = t[i] + ||P[i+1] - P[i]||^0.5`

Distances are measured in physical millimetres. Each interval is a cubic
Hermite curve; its tangents are computed from neighboring points and knot
intervals. Two reflected endpoint ghost points are derived, not learned.
Both graph endpoints stay fixed. There are no separately predicted tangents,
knots or alpha values in this simulation.

| Total points including endpoints | Free 2D geometry values | Free 3D geometry values |
|---|---:|---:|
| 8 | 12 | 18 |
| 10 | 16 | 24 |

This is the same coordinate budget as an 8/10-control B-spline with fixed
endpoints. The total coordinates including both endpoints are 16/20 in 2D
and 24/30 in 3D; endpoint coordinates are shared graph-node parameters.
The viewers also include 6, 12 and 16 interpolation points.

## Why it is worth testing

On-curve parameters are interpretable landmarks and may simplify geometric
supervision or editing. Catmull–Rom has continuous tangents but generally not
continuous curvature at its interpolation points. A cubic B-spline with simple
interior knots has continuous curvature. Interpolation can also reproduce
uncertain guide noise rather than smoothing it.

Centripetal Catmull–Rom avoids cusps and self-intersections **within an individual
segment** for distinct neighboring points. This does not guarantee global
non-intersection or vessel-mask containment. See
[Yuksel, Schaefer and Keyser, Parameterization and Applications of Catmull–Rom Curves](https://www.cemyuksel.com/research/catmullrom_param/).
Repeated neighboring points use a minimum knot increment of 1e-9 in this
double-precision prototype; the segment guarantee does not apply to that
degenerate case. A future learned implementation would need appropriate
spacing/order checks and numerical handling in its training precision.

## Construction and measured example results

The existing B-splines retain their endpoint-constrained least-squares fits.
Catmull–Rom interpolates equal arc-length samples from the selected uncertain
guide. Its positions are not optimized by least squares. The families share
endpoints and scalar budget but use different target-construction methods.
This is an illustrative capacity/editing comparison, not proof of the best
possible Catmull–Rom approximation and not a trained-model comparison.

Original-guide results: unweighted mean across all branches in each case.

| Case | Family / total points | Guide distance mm | Absolute tortuosity error | Mask support |
|---|---|---:|---:|---:|
| IXI, 12 branches | B-spline / 8 | 0.03391 | 0.01058 | 99.214% |
| IXI, 12 branches | Catmull–Rom / 8 | 0.05460 | 0.01925 | 98.151% |
| IXI, 12 branches | B-spline / 10 | 0.02410 | 0.00896 | 99.735% |
| IXI, 12 branches | Catmull–Rom / 10 | 0.03065 | 0.01567 | 99.505% |
| SyntheticMRI, 19 branches | B-spline / 8 | 0.02295 | 0.00163 | 99.633% |
| SyntheticMRI, 19 branches | Catmull–Rom / 8 | 0.02459 | 0.00228 | 99.633% |
| SyntheticMRI, 19 branches | B-spline / 10 | 0.01529 | 0.00135 | 99.633% |
| SyntheticMRI, 19 branches | Catmull–Rom / 10 | 0.01449 | 0.00118 | 99.633% |

The existing alternate smoothed-guide fits are also available. Two cases do
not establish which representation learns better. This SyntheticMRI patch is
only mildly tortuous, so it is a weak stress test for severe bends. Neither
guide agreement nor image-derived mask support is independent anatomical
validation.

## Reproducibility and scope

The update reused the exact embedded cases: IXI `sample_000001_0007` and
SyntheticMRI `sample_000013_0002`. Volumes, masks, mesh, anchors, branch
guides and every non-Catmull numerical fit were verified unchanged. Existing
source paths in provenance describe the original extraction; they were not
re-read or newly verified during this update. No dataset, training target,
checkpoint or training configuration was changed.

- Shared Python evaluator: `DenseVesselGraph/scripts/catmull_rom.py`.
- Update command: `python scripts/update_single_patch_curve_viewers.py` from
  that repository, using the usual verified runtime.
- Existing shared experiment builder and viewer sources are in
  `experiments/ixi_patch/src/`; SyntheticMRI uses those same functions/templates.
- CSV measurements and construction notes are in each case's `results/`.
- Three interpolation/tangent/degeneracy regression checks passed.
- Python and browser-JavaScript evaluations agreed within 1.4e-15 mm on
  nonuniform and repeated-point fixtures. Simulated DOM interactions checked
  family switching, equal-budget overlay, point editing, restore and guide
  switching. These checks did not execute WebGL in a real browser on the cluster.
- Both static comparison figures were visually inspected.

If proceeding to learning, start with the existing eight-point budget and
training/validation split, and predefine shared topology/geometry evaluation.
Assess spacing, point order, guide noise sensitivity and supervision calibration
before interpreting any difference from B-spline learning.

Implementation committed as `f64280a` on the `representation` branch.
