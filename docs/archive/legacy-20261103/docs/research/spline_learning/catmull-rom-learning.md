# Eight-point Catmull–Rom training — 2026-10-09

## Question and protocol

Does supervising points that lie on the represented curve make learning easier
than supervising B-spline handles? The user requested a 300-epoch experiment
with the same control budget. This run starts **from scratch**, matching the
original 300-epoch B-spline comparison. It is separate from the pretrained,
image-informed connectivity pilots.

| Setting | Catmull–Rom run |
|---|---|
| Dataset | SyntheticMRI vessels only |
| Membership | Same 10,000 training, 1,952 validation, 13,071 test patches |
| Graph nodes | Existing junctions, terminations, boundary nodes; no added degree-2 nodes |
| Representation | Centripetal Catmull–Rom, alpha 0.5, physical millimetres |
| Budget | Eight points including graph endpoints; six internal points = 18 predicted scalars per 3D edge |
| Head | Same pair-token MLP and learned parameter count as the B-spline head |
| Initialization | Seed 364505; random weights; zero geometry residual gives a chord |
| Optimization | Whole network, 300 epochs, batch 4, 750,000 updates |
| AdamW | LR 7e-5, weight decay 1e-4, betas 0.9/0.999 |
| Schedule | 2,500 warmup updates; original polynomial schedule, power 0.9 |
| Loss | Original control-coordinate Smooth L1 in mm, beta 1; weight 1.4739989612791453 |
| Other weights | Boxes 3, class 4, cardinality 0.8, nodes 2, edges 6 |
| Validation | Every five epochs, same graph-loss checkpoint selection, four projections |
| Tracking | W&B offline, same comparison group; distinct Catmull–Rom run name/tags |
| Resources | One H100; 20-hour allocation; latest full-state checkpoint each epoch |

No image-evidence correction, loss ramp, pretraining, or new connectivity loss
is added to this experiment. Its purpose is to isolate the representation and
corresponding target construction under the original optimization settings.
The nominal geometry weight is preserved, not recalibrated to equalize
gradient magnitudes between representations.

## Implementation and targets

`models/catmull_rom.py` evaluates differentiable cubic Hermite segments with
centripetal knots computed from the predicted points. Reflected endpoint ghost
points and tangents are derived, not learned. A small distance floor handles
coincident points. The head copies predicted graph endpoints exactly into both
the controls and sampled curves. Model configuration selects `bspline` (the
backward-compatible default) or `catmull_rom`.

The existing full guide of each branch is resampled at eight equally spaced
physical arc-length positions. These are the supervised interpolation points;
64 curve samples are also cached for monitoring. This differs deliberately
from the B-spline target's endpoint-constrained least-squares handles. The
existing guide is an uncertain reference, not perfect anatomy. Interpolation
does not guarantee vessel-mask containment or correct connectivity.

Images, segmentations, graphs, splits, branch orientation, full guides, and
boundary-node rules are inherited. Only the new family-specific controls and
curves are generated. The dataset loader rejects mismatched curve families.
Original caches and checkpoints are not overwritten.

The unchanged control loss directly supervises internal point coordinates.
This tests easier target learning; it does not add a new image-containment or
topology objective. The earlier scratch-training collapse could still recur.

## Jobs and verification

- **805038**: CPU target preparation, with independent target cache.
- **805044**: after successful preparation, verify exact memberships,
  identical initial learned tensors/parameter counts and unchanged optimizer,
  then three real GPU updates, validation, projections, and inference.
- **805047**: 300-epoch training, starts only after **805044** succeeds.

All 13 focused local tests passed before submission. These cover agreement
with the existing NumPy viewer implementation, interpolation at knot positions,
direction reversal, finite gradients including coincident points, physical
metric handling, parameter counts, endpoint anchoring, target preservation,
and existing B-spline/inference behavior. Python compilation, shell syntax,
and `git diff --check` also passed. GPU verification is a queued prerequisite;
its submission is not evidence of success. The additional native/common metric
regression and inference tests also passed (eight tests in the second check).
Saved predictions now explicitly carry the curve family, and both metric paths
reconstruct the appropriate physical curve. Target preparation produced all
10,000 / 1,952 / 13,071 caches successfully; GPU verification remains pending.

## Files and next steps

Code repository: `Graph-Native-Betti-Matching-Representation`.

- Config: `configs/scratch_comparison_catmull_rom_300ep.yaml`.
- Preparation: `scripts/prepare_catmull_rom_targets.py` and
  `cluster/jean_zay/prepare_catmull_rom.slurm`.
- Training/check: `cluster/jean_zay/train_catmull_rom.slurm`.
- Compact provenance: `experiments/spline_learning/metadata/catmull-rom/`.
- Logs: `experiments/spline_learning/logs/catmull-*.{out,err}` and
  `scratch-catmull-rom-805047.{out,err}`.

Dataset cache:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep/catmull_rom/`.

Run:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_catmull_rom_k8_10k_300ep/`.

After training, evaluate both the graph-loss-selected best checkpoint and the
fixed epoch-300 checkpoint with the same BN policy and representation-independent
full-guide geometry/topology protocol used for the three existing runs. Do not
rank representations by agreement with their own fitted curve targets alone.
No test evaluation is launched in this submission chain.
