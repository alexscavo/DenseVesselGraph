# Training the RelationFormer pipeline

This page covers `train.py`, the YAML-driven trainer for the 3D RelationFormer baseline, and its
loss stack. It is **not** how the current dense traced model is trained: that model is trained by
its own scripts ([`research/dense_traced_model_20261102.md`](research/dense_traced_model_20261102.md)
§2.5 and §3). For Jean Zay launchers, every CLI flag and the YAML reference, see
[`RUNNING_THE_MODEL.md`](RUNNING_THE_MODEL.md).

```bash
export PLANTS_DATASET="$SCRATCH/datasets/plants_3d2cut/patches_3d"
export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/new_patches_boundary"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm"

python train.py --config configs/pretrain_mixed.yaml
```

Run this inside a GPU job, never on a login node. Dataset and output paths come from the
environment; scientific settings stay in version-controlled YAML.

The baseline uses images as model input, AdamW, and the legacy per-iteration polynomial schedule
with linear warmup. Fields that the legacy trainer never used were removed rather than kept as
misleading no-ops: gradient clipping, minimum learning rate, the 3D encoder/decoder freeze flags,
automatic-resume metadata and EMA.

Operational overrides are limited to device, output directory, run name, batch size, worker
count and checkpoint paths. The fully resolved YAML is written to the run directory.

## Experiment tracking

W&B tracking is configured by `tracking` in `configs/base.yaml`. Training logs every loss
component and the learning rate against `train/iteration`, and validation losses against
`validation/epoch`. TensorBoard is not used.

```bash
wandb login
source cluster/jean_zay/wandb_env.sh
export WANDB_RUN_GROUP=my-group  # optional
```

`wandb_env.sh` sets the default destination
`alessandrobenvenuti2002-politecnico-di-torino/focal-loss` (inherited from the upstream
repository); override it with `WANDB_ENTITY` / `WANDB_PROJECT` before sourcing. It contains no
API key. Never store `WANDB_API_KEY` in configuration files. Compute nodes have no internet, so
jobs normally use `WANDB_MODE=offline` and are synced later; `WANDB_MODE=disabled` suppresses
tracking for a smoke test.

When `--resume` targets the same run directory, `wandb-run.json` supplies the original W&B run
ID. W&B cannot resume offline; sync an offline run before continuing it online. Training state
always comes from the checkpoint passed to `--resume`.

## Checkpoints

Every training checkpoint contains strict model, optimizer, scheduler, epoch and iteration
state. Resume an interrupted run with:

```bash
python train.py \
  --config configs/pretrain_mixed.yaml \
  --resume /absolute/path/to/latest_checkpoint.pt
```

Load a trusted checkpoint as initialization, without its optimizer or scheduler:

```bash
python train.py \
  --config configs/finetune_synthetic_mri.yaml \
  --initial-weights "$GNBM_MRI_CHECKPOINT"
```

| Checkpoint slot | Selection | Metadata |
|---|---|---|
| `best_checkpoint.pt` | Lowest validation loss | Training checkpoint state |
| `best_metric_checkpoint.pt` | Configured metric, normally edge mAP (not recall) | `best-metric.json` |
| `best_node_f1_checkpoint.pt` | Highest validation node F1 | `best-node-f1.json` |
| `best_edge_f1_checkpoint.pt` | Highest validation edge F1 | `best-edge-f1.json` |
| `latest_checkpoint.pt` | Latest scheduled save; always saved on early stop | Training checkpoint state |

Each slot is a complete model and resume state. `best_only` keeps these replaceable slots
rather than an epoch archive; slots are replaced atomically and share a hard-linked payload when
several win in the same epoch. With `best_only` there are at most five slots, about 6.8 GB if all
come from different epochs for the current model, plus headroom for temporary writes.

The F1 used for selection is the **native box-based** F1 of `evaluate.py`
([`EVALUATION.md`](EVALUATION.md) §2.2), not the physical 2 mm edge F1 reported in the current
results.

## Epoch patience and F1 checkpoints

The full-dataset MRI fine-tuning recipes enable these settings; older recipes default to
disabled:

```yaml
training:
  epochs: 250
  early_stopping:
    enabled: true
    monitor: edge_mAP
    mode: max
    patience_epochs: 50
    min_epochs: 0
    min_delta: 0.0
evaluation:
  training_metrics:
    enabled: true
    selection_metric: edge_mAP
    selection_mode: max
    save_best_checkpoint: true
    save_f1_checkpoints: true
  protocol:
    f1_iou_threshold: 0.5
```

- Patience counts **epochs**, not validation calls. With validation every five epochs and the
  last improvement at epoch 105, training stops at epoch 155 absent further gains. With
  irregular intervals it stops at the first validation at least `patience_epochs` after the last
  improvement.
- Ties do not improve the monitor. A positive `min_delta` requires a gain strictly larger than
  that amount relative to the last significant improvement.
- The first finite validation initializes the reference. A missing or non-finite monitored
  metric raises an error.
- `monitor` may also be `edge_f1` or `node_f1` (and the other keys listed in
  `RUNNING_THE_MODEL.md` §9). This does not change the independent AP slot. A node-F1 gain does
  not reset an edge-AP or edge-F1 patience clock.
- F1 uses micro-aggregated TP/FP/FN over the whole validation split at the configured node and
  edge thresholds (normally 0.5), with all retained predictions and no top-40 cap. Changing
  thresholds changes F1 and its winner.

`validation-metrics.jsonl` and W&B include `node_precision`, `node_recall`, `node_f1` and their
edge equivalents; JSON summaries add `node_tp_total`, `node_fp_total`, `node_fn_total` and the
edge totals. W&B records selected epochs under `checkpoints/<metric>` and the patience clock
under `stopping/*` (values only, not weight files).

On early stop, rank zero broadcasts the decision so all DDP workers stop together after the final
`latest_checkpoint.pt` is saved. `early-stopping.json` records the clock and
`training-status.json` reports `early_stopping` or `max_epochs`. Patience and F1 winners are
stored in the checkpoint, so resuming does not reset them. Changed stopping settings are rejected
on strict resume. Older checkpoints without this state start the trackers at the first new
validation.

**Budget is not continuation policy.** For a new run the scheduler is built for `epochs`.
Extending a completed run needs an explicitly chosen continuation learning-rate policy: changing
`epochs` while restoring the old scheduler is not a validated extension. Loading
`--initial-weights` starts a new fine-tuning experiment, not a resume.

```bash
python -m unittest -v tests.test_early_stopping tests.test_f1_metrics \
  tests.test_training_selection tests.test_training tests.test_tracking
```

(The distributed test needs two local Gloo workers.)

## Loss stack

The implementation is under `training/losses/`; `training.losses.build_criterion(config, model)`
is the entry point.

- `criterion.py`: matching, baseline supervision, edge candidates and the opt-in terms below;
- `focal.py`: focal classification and hard-negative selection;
- `betti_h0.py`, `betti_h1.py`: graph-native topology losses;
- `relation_distance.py`: a pilot-only reweighting of nearby GT non-edges (used by
  `scripts/train_relation_distance_pilot.py`, not by the main criterion).

`loss.enabled` lists the optimized terms, each with a weight in `loss.weights`. The total is the
weighted sum of the enabled terms, plus the degree term and the non-log-only Betti terms when
those are enabled.

### Baseline terms

The default configuration (`enabled: [boxes, class, cardinality, nodes, edges]`) preserves the
established 3D objective:

1. Hungarian assignment combines object probability with L1 distance over the first three
   predicted node coordinates.
2. `class`: weighted cross-entropy with `[0.25, 0.75]`, class `1` being the object class.
3. `nodes`: summed L1 error divided by the number of target nodes.
4. `boxes`: the last three node-head values are predicted sizes, against a fixed target size of
   `0.2`, scored by 3D generalized IoU.
5. `edges`: matched nodes, one random orientation per undirected pair, cross-entropy and legacy
   ratio upsampling.
6. `cardinality`: normalized L1 error of the predicted object count.

Disabled extensions do not score candidates, consume random numbers or enter the objective.

### Opt-in terms

| Term | Switch | What it does |
|---|---|---|
| `node_types` | `loss.enabled` + `model.node_type.enabled: true` | 4-class degree head on matched queries: isolated, patch terminal, degree-2, junction |
| `spline` | `loss.enabled` + `model.spline.enabled: true` | Edge-shape loss for the B-spline (or Catmull-Rom) head on matched GT edges, against the full-path target cache; `loss.spline.mode` is `controls` or `curve` |
| `domain` | `loss.enabled` + `domain_adaptation.enabled: true` | Adversarial domain loss (below) |
| degree consistency | `loss.degree.enabled` with a positive weight | Matches expected node degrees from relation probabilities to GT degrees, correcting for sampled negatives; warmup and ramp configurable |
| Betti H0 / H1 | `topology.betti_h0`, `topology.betti_h1` | Graph-native topology losses on the complete graph over Hungarian-matched nodes. Every undirected pair is scored in both endpoint orders and averaged; persistence matching uses detached probabilities while the selected filtration values keep gradients. Warmup, ramp, weight, normalization and `log_only` are configurable |
| Focal classification | `loss.node.classification.name` / `loss.edge.classification.name: focal` | Focal loss, optionally with a gamma curriculum |
| Hard-negative mining | `loss.edge.candidates.include_unmatched: true` (with focal edges) | See below |

The training-time matcher can also be replaced by semi-relaxed Fused Gromov-Wasserstein
matching (`configs/matchers/fgw.yaml`; see [`../configs/README.md`](../configs/README.md)).

**Hard-negative mining** belongs to the focal edge candidates:

```yaml
loss:
  edge:
    classification:
      name: focal
    candidates:
      include_unmatched: true
```

Active unmatched queries form candidates with matched and other active unmatched queries. Each
candidate is evaluated in both endpoint orders and the two logit vectors are averaged, so
selection and focal supervision are undirected. Selection ranks detached edge probabilities but
indexes the original symmetric logits, keeping gradients through both relation-head evaluations
and their endpoint tokens. Caps, threshold, weight, warmup and ramp are configurable.

Diameter smoothing is not part of this stack, and there is no dormant diameter option.

## Adversarial domain adaptation (opt-in)

**Status.** Off by default (`domain_adaptation.enabled: false` in `configs/base.yaml`). It is
enabled by:

- `configs/pretrain_plants_ixi_domain_types_50ep.yaml` and its matched control
  `configs/pretrain_plants_ixi_domain_no_types_50ep.yaml` (Plants source, IXI target, run through
  `train.py`);
- the October study plans `configs/ixi_transfer.yaml`, `configs/ixi_transfer_source_aug.yaml`,
  `configs/{plant,leaf,combined}_generalization_da_b32.yaml` and their `_source_aug` variants.
  These are not `train.py` configs: they are read by `scripts/train_ixi_transfer.py` and
  `scripts/train_plant_generalization.py`, which turn domain adaptation on only for the mixed
  source+MRI pretraining arm and off for MRI-only fine-tuning.

Results of these runs are summarised in [`research/HISTORY.md`](research/HISTORY.md).

Source and target labels are always in the model-facing batch. They describe dataset roles and
may drive sampling and diagnostics; their presence does not enable adaptation.

**Objective.** Domain-adversarial training in the style of Domain-Adaptive Faster R-CNN. A
gradient-reversal layer precedes each discriminator:

```text
L_domain = w_image * NLL(image_domain_prediction, domain)
         + w_graph * NLL(graph_domain_prediction, domain)
         + w_consistency * MSE(mean_image_prediction, graph_prediction)

L_total = L_graph + w_domain * L_domain
```

During backpropagation the reversal multiplies the feature gradient from `L_domain` by `-alpha`,
pushing features towards domain invariance while the discriminators learn source versus target.

**Implementation** (`models/domain_adaptation.py`, `RelationFormer`, `criterion.py`): two
2048-wide discriminators, one on the projected image features and one on the flattened decoder
tokens; two-pass gradient reversal; weighted image/instance NLL; consistency MSE; a logistic
reversal schedule; and a separate discriminator learning rate. The target is sampled with
replacement at the reference's expected 50% rate, using a combined-dataset epoch length.

```yaml
domain_adaptation:
  enabled: true
  image_adversarial: true
  graph_adversarial: true
  consistency_regularization: true
  alpha_coefficient: 1.0
  learning_rate: 0.00007
  hidden_size: 2048
  class_weights: [0.05, 0.95]

loss:
  enabled: [boxes, class, cardinality, nodes, edges, node_types, domain]
  supervise_target_graphs: true
  weights:
    domain: 1.0
```

Configuration validation enforces the contract: `enabled: true` requires `domain` in
`loss.enabled`, a positive domain weight, a source and a target dataset,
`supervise_target_graphs: true`, both discriminators and consistency regularization; `domain`
in `loss.enabled` without `enabled: true` is rejected. When disabled, no discriminator is built,
run or optimized and no domain metric is logged. `tests/test_domain_adaptation.py` checks
discriminator and feature gradients, the optimized loss, the disabled state, the sampler and
checkpoint compatibility.

**Why it was off in the inherited baseline.** The pre-focal reference repository built image-
and instance-level discriminators and computed a domain loss, but its committed 3D configs
enabled only `boxes`, `class`, `cards`, `nodes` and `edges`, and the trainer back-propagated only
that total. The domain loss was logged but never trained anything. References:
[model discriminators](https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/models/relationformer.py#L69-L71),
[domain loss](https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/losses.py#L749-L777),
[optimized total](https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/losses.py#L850-L868),
[3D loss list](https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/configs/mixed_plants_synth_HNS_3D.yaml#L138-L145),
[backward call](https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/trainer.py#L318-L345).
A separate cluster run did activate it, with `DOMAIN_LR: 0.00007`,
`DOMAIN_WEIGHTING: [0.05, 0.95]`, both adversarial flags, consistency regularization,
`ALPHA_COEFF: 1.0`, `domain` in `TRAIN.LOSSES` and `W_DOMAIN: 1.0`; the current defaults follow
those values. The old `UPSAMPLE_TARGET_DOMAIN` and `COMPUTE_TARGET_GRAPH_LOSS` fields are now
`data.mixed_sampling.balance_source_target` and `loss.supervise_target_graphs`.

## Smoke runs and cluster tests

The smoke configuration keeps the full model and loss path but uses two training samples and one
validation sample per dataset:

```bash
python train.py --config configs/smoke_mixed.yaml
```

The resolved configuration and checkpoints should appear under `$GNBM_OUTPUT_DIR/smoke_mixed/`.

After building deformable attention (`bash scripts/build_deformable_attention.sh`), run the CUDA
forward and backward tests inside a GPU job:

```bash
python -m unittest -v tests.test_model_cluster.CudaModelTests
```

This covers the CUDA operator, model forward, baseline criterion backward, and combined focal +
H0/H1 Betti backward. Dataset integration tests are described in
[`DATA_PIPELINE.md`](DATA_PIPELINE.md).

To compare the baseline criterion with the original repository:

```bash
python scripts/compare_model_forward.py \
  --legacy-root /absolute/path/to/original/3d \
  --checkpoint "$GNBM_MRI_CHECKPOINT" \
  --compare-losses
```

It fixes the relation-candidate random seed and compares the five weighted baseline components,
cardinality and total loss, in addition to model outputs. Focal and Betti objectives are covered
by their focused gradient tests, because they are opt-in deviations rather than parity targets.
