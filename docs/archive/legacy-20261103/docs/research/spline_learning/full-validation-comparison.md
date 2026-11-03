# Full validation comparison — 2026-10-09

The user authorized validating the observed connectivity/geometry trade-off
before further training. Evaluate three fixed checkpoints on **all 1,952
validation patches** from the original comparison manifest:

| Arm | Checkpoint | Reason |
|---|---|---|
| `frozen_residual` | Frozen residual head after 15k updates, 817904_0 | Best pilot curve F1 |
| `detached_head` | Graph-only detector gradients + adapting detached head, 824964_0 | Best pilot connectivity |
| `endpoints_detached` | Joint token gradients, endpoint path detached, 824964_1 | Possible geometry/connectivity compromise |

No checkpoint selection within this evaluation, training, optimizer updates,
BN recalibration, threshold changes, or test-set scoring. All models run in
eval mode with gradients disabled; the full model state is hashed before and
after. Source checkpoint configurations supply the common evaluation protocol.
Initialization is strict; the same ordered validation IDs are asserted for all
three arms. Explicit IDs bypass the earlier diagnostic cap of 128.

## Measures

- Native node/edge F1 and existing native metrics.
- Common anchor/branch/whole-curve metrics, Betti errors, object counts and
  count errors, using the existing representation-independent guide protocol.
- Each decoded graph's own same-endpoint straight chords, to separate curve
  benefit from differing detection/connectivity.
- GT-matched positive-pair raw guide, fitted-target, residual and endpoint-neutral
  geometry; keep these distinct from metrics on full decoded predictions.
- Decoded matched-branch tortuosity error. Guides remain uncertain references.

The primary result is the full validation aggregate. It includes the 128 cases
used for diagnostics, and these subjects were also used for earlier checkpoint
selection. This is broader model-selection evidence, not an unbiased final test
or a new seed. Keep the separate test set untouched until the recipe is fixed.

## Fixed-case visualizations

Select six subjects using seed 364505, then one validation patch per selected
subject, without examining model predictions. Use exactly those six IDs in all
arms. For each case, export the existing three orthogonal MRI/mask projections
with guide nodes/paths above predicted nodes/curves. Also export identical-node
straight-chord versions. The gallery shows models side by side with a case
selector and curves/chords toggle. Projected crossings are not necessarily
junctions; these images do not replace 3D topological evaluation.

When all evaluations complete, the last task verifies paired protocols and
automatically writes:

- `comparison.csv`: full compact metrics for all three checkpoints.
- `comparison.md`: readable main comparison table.
- `comparison.json`: table, evaluation IDs and paired-protocol audit.
- **`comparison.html`**: self-contained offline gallery with embedded images.
  Download this file alone and open it locally; no server or extra assets.

Each arm retains full predictions, native/common summaries, per-case metrics,
GT-matched per-branch errors and PNGs. W&B stays offline, storing metrics and
the selected validation projections. Large outputs remain outside `projects`.

## Execution

Config: `configs/spline_full_validation.yaml` in
`Graph-Native-Betti-Matching-Representation`.
Runner: `scripts/evaluate_spline_validation.py`, reusing existing evaluation,
geometry, plotting and tracking code. Slurm:
`cluster/jean_zay/spline_full_validation.slurm`, three independently scheduled
H100 tasks, one-hour ceiling each. Each task first evaluates two fixed cases
in a separate smoke directory, then restarts the full evaluation from its
checkpoint. No original artifact is overwritten.

Output root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/full_validation_20261009/`.
Subdirectories: each arm and its `smoke-` counterpart. The four comparison
files above are generated at this root once all three summaries exist.

After completion, inspect paired tables and the fixed-case gallery to decide
whether a longer pretrained continuation is justified. No automatic training
is attached to these jobs.

## Submission

Submitted **826121_0/1/2** for frozen residual / detached head / endpoints detached respectively. All checkpoints load, configurations validate, evaluation/model/input/root settings agree, and the exact 1,952 validation IDs are disjoint from train/test IDs. Python and Slurm syntax checks passed. GPU smoke and full evaluations are queued. Logs: `experiments/spline_learning/logs/full-validation-826121_<task>.{out,err}`. Compact evidence: `experiments/spline_learning/metadata/full-validation/submission.json`.

Selected cases: `sample_000014_0919`, `sample_000016_0843`, `sample_000041_0761`, `sample_000070_0552`, `sample_000075_0287`, `sample_000092_0813`. No result claimed at submission.

## Completed results

All jobs completed with exit 0:0: **826121_0 9m50s**, **_1 9m47s**,
**_2 10m20s**. Each scored exactly 1,952 validation patches and 31,934
GT-matched positive branches. Ordered IDs, protocols and visual cases agree;
all model states stayed unchanged, with no BN recalibration or test scoring.

### Decoded graph metrics (F1 higher is better)

| Model | Node F1 | Edge F1 | Common branch F1 | Curve F1 | Own-chord curve F1 |
|---|---:|---:|---:|---:|---:|
| Frozen residual | **0.91041** | 0.86363 | 0.27625 | **0.68441** | 0.64702 |
| Graph-only detector + detached head | 0.90910 | **0.87198** | 0.29183 | 0.67955 | 0.64762 |
| Joint tokens, endpoints detached | 0.90889 | 0.87096 | **0.29236** | 0.68296 | **0.64766** |

Native edge F1 and common branch F1 have different matching rules and should
not be read as interchangeable measures. Splines improve whole-curve F1 over
their own chords for every model. The frozen model has the largest such gain
(0.03739), versus 0.03194/detached and 0.03530/endpoints-detached.

### Geometry and topology (errors lower is better)

| Model | GT-matched raw guide mm | Endpoint-neutral guide mm | Neutral gain vs chord | Betti-0 count error | Betti-1 count error | Anchor count error |
|---|---:|---:|---:|---:|---:|---:|
| Frozen residual | 0.89484 | 0.46094 | 10.48% | **0.56301** | 1.65984 | 2.48002 |
| Detached head | 0.89051 | 0.46107 | 10.46% | 0.83914 | **1.00512** | 1.94723 |
| Endpoints detached | **0.88814** | **0.45990** | **10.69%** | 0.84990 | 1.06096 | **1.92828** |

Raw guide errors improve 6.67% / 6.69% / 6.90% over each model's own chords.
Decoded topology-matched branch tortuosity error is 0.02547 / 0.02413 / 0.02534.
Mean predicted node counts are 21.32 / 22.20 / 22.23; predicted common branch
counts 16.07 / 15.85 / 15.93 against mean reference 16.36. Count agreement
alone does not establish correct connectivity. Betti-0 absolute count error
worsens after unfreezing even as edge/branch F1 and Betti-1 error improve:
topology gains are mixed. Absolute component-count error alone does not tell
whether the model has too many or too few components.

### Visual inspection and gallery

Inspected source projections for `sample_000014_0919` across all three models,
and `sample_000075_0287` for frozen/endpoints-detached curves and the latter's
chords. Predictions broadly localize many vessel segments but retain missed
connections, spurious structures and notably under-recovered bends. On these
cases, predicted curves often remain close to chords; numerical gains should
not be presented as solved tortuosity reconstruction. This is a limited
qualitative inspection, not a visual review of all validation cases, and 2D
projected crossings cannot certify 3D connectivity.

All 36 embedded PNGs (six cases × three models × curves/chords) were checked
against their source files. No browser was available for an interactive
rendering test. The HTML has no external image or script dependency.

**Download only this file and open it in your local browser:**

`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/full_validation_20261009/comparison.html`

Size: **9,808,153 bytes** (~9.8 MB). Tables are alongside it as
`comparison.csv`, `comparison.md`, and `comparison.json`.

### Decision and limits

- Full validation confirms the pilot trade-off, with a smaller difference
  between the two unfrozen alternatives. Endpoints-detached is the tentative
  compromise: nearly the detached model's edge F1, slightly better common
  branch F1 and guide geometry, and curve F1 closer to the frozen model.
- It is **not an overall winner**: frozen remains best on curve F1, node F1
  and component-count error; fully detached has the best edge F1 and cycle
  count error. Endpoint-neutral shape differences between these candidates
  are small (~0.001 mm). No statistical superiority is established.
- Keep all three fixed checkpoints. The residual representation provides
  modest transferable shape benefit, but unfreezing has not clearly increased
  curvature learning, and visual underfitting remains. Before a long run,
  discuss whether to prioritize topology or improving geometry beyond the
  current token-conditioned head. These results do not justify claiming the
  original scratch-training failure is explained.

No new training was submitted during this results review. Compact summaries,
tables and completion/visualization audit are saved in the code repository at
`experiments/spline_learning/metadata/full-validation/`. Larger artifacts and
offline W&B data remain in the run root outside `projects`.

Dataset-context follow-up (2026-10-10): the user highlighted that SyntheticMRI
has limited bends. A fitted-target audit confirms this, so small average gains
and predictions close to chords alone cannot establish curvature-learning
failure. See the [local-feature experiment and curvature audit](local-geometry-diagnostic.md)
for the corrected interpretation and IXI follow-up constraints.
