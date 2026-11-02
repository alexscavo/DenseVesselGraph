# SyntheticMRI held-out prediction viewers

Two standalone offline HTML files display the actual saved decoded outputs from the validation-selected checkpoint evaluated on the test set. Predictions were not refitted, smoothed, repaired or regenerated. The checkpoint and per-example metrics are recorded in `selection.json`.

## Examples

- [Subject 20, patch 0470](../experiments/synthetic_mri_test_predictions/results/sample_000020_0470.html).
- [Subject 24, patch 0478](../experiments/synthetic_mri_test_predictions/results/sample_000024_0478.html).
- [MRI projection inspection preview](../experiments/synthetic_mri_test_predictions/results/preview.png).

Selection: sorted test subjects, first two IDs, midpoint of their sorted eligible sample IDs. No prediction scores or geometric errors were used to select examples. Equal node/edge counts do not imply correct connectivity.

## Interaction

Download each HTML and open it locally in a browser; all data and Plotly are embedded. Drag to rotate, scroll to zoom. Cyan curves and blue nodes are predictions; gray fitted splines, gold full original guides, purple reference anchors and the translucent vessel-mask mesh are uncertain references. Toggle layers and node labels, adjust mesh opacity, select a predicted edge, and display its eight controls/control polygon and prediction confidence. Six internal controls encode 18 learned 3D offsets. Axes are patch-local physical millimetres; reference and predicted node labels use separate indices.

## Verification and reproduction

Verified exported metric transforms against target caches, exact predicted endpoint/control agreement, expected curve/edge counts and absence of embedded placeholders. Inspected real-MRI projections for coordinate/mask alignment. JavaScript trace construction, layer toggles, control counts, edge selection, opacity and camera reset were checked with DOM/Plotly stubs. Browser/WebGL rendering cannot be inspected on this cluster.

Builder: `DenseVesselGraph/scripts/build_spline_test_viewers.py`. Pass `--run`, `--targets` and `--output`; it streams the large prediction export and stops after finding the selected samples. No GPU inference or training is repeated.
