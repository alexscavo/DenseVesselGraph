# 3D RelationFormer model contract

This page documents the 3D RelationFormer built by `models.build_model(config)` and used by
`train.py` / `evaluate.py`. It is the baseline the project started from (called B0 in the
results when it has the K8 spline head). It is **not** the current model: the dense two-stage
model with traced connectivity is described in
[`research/dense_traced_model_20261102.md`](research/dense_traced_model_20261102.md), and its
code lives in `scripts/dense_graph_20261030.py` and `scripts/dense_graph_traced_20261030.py`.
Why RelationFormer was replaced is summarised in [`research/HISTORY.md`](research/HISTORY.md).

## Architecture

```text
[B,1,D,H,W]
    -> 3D SE-ResNet encoder
    -> 1x1x1 input projection
    -> 3D sine positional encoding
    -> deformable transformer encoder/decoder (one feature level by default)
    -> object and relation tokens
    -> node-class and coordinate heads
```

Implementation:

- `models/seresnet.py`;
- `models/position_encoding.py`;
- `models/deformable_transformer.py`;
- `models/relationformer.py`;
- `models/ops/` for the 3D deformable-attention operator.

`models.build_model(config)` is the construction entry point. With `data.spatial_dims: 2` it
builds the separate native 2D model from `native2d/` instead.

### Optional modules

All are off in `configs/base.yaml` and are built only when enabled, so existing checkpoints
keep their exact shapes:

| Module | Switch | Code |
|---|---|---|
| Node-type (degree class) head | `model.node_type.enabled` | `node_type_embed` in `models/relationformer.py` |
| Edge-shape spline head (B-spline or Catmull-Rom, K controls) | `model.spline.enabled` | `models/spline.py`, `models/catmull_rom.py`, `models/curve_evidence.py`, `models/local_spline_geometry.py` |
| Local node refinement from early encoder features | `model.node_refinement.enabled` | `models/node_refinement.py` |
| Second (finer) feature level for the decoder | `model.decoder.feature_levels: 2` | `models/relationformer.py` |
| Image and instance domain discriminators | `domain_adaptation.enabled` | `models/domain_adaptation.py` (see [`TRAINING.md`](TRAINING.md)) |

Research variants built by their own scripts, not through `build_model`, include the
heatmap-anchored RelationFormer (`models/anchored_relationformer.py`, used by
`scripts/anchored_relationformer_20261030.py`).

## Forward interface

The input is a float tensor `[B,C,D,H,W]`. The model returns:

```python
tokens, predictions, projected_features = model(samples)
```

- `tokens`: `[B,Q,H]`, all object, relation and dummy tokens;
- `predictions["pred_logits"]`: `[B,Q_object,num_classes]`;
- `predictions["pred_nodes"]`: `[B,Q_object,6]`, passed through a sigmoid;
- optional entries such as `pred_node_types` and the domain log-probabilities when the
  corresponding modules are enabled;
- `projected_features`: `[B,H,D',H',W']`, kept for diagnostics.

The first three coordinate channels are normalized node positions. The other three are box
sizes, used by the box loss and kept for checkpoint compatibility.

The relation classifier `relation_embed` is a model submodule but is not called by
`RelationFormer.forward`. The loss and the inference code call it on pairs of object tokens
plus the relation tokens.

## Baseline architecture corrections

Two legacy configuration fields did not describe the model that was actually built:

1. `ENC_LAYERS` was configured as 4 but never passed by the builder, so the transformer used its
   default of 6 encoder layers. The configuration now says `encoder_layers: 6`, and the builder
   reads it.
2. The first value of encoder `STRIDES` was ignored because layer 1 always used stride 1. The
   configuration now records the effective stage strides `[1,2,2,2]`.

These are schema corrections, not architecture changes: the effective model and checkpoint
shapes are unchanged.

Legacy `DROP_RATE`, `DROP_PATH_RATE`, `LABEL_SMOOTHING`, encoder `CELL_SIZE` and decoder
`TWO_STAGE` were removed because this 3D path never read them. Transformer dropout remains
configurable through `model.decoder.dropout`.

The legacy relation-attention branch is preserved exactly: when enabled, deformable
cross-attention is applied to every query except the final token. With two relation tokens,
only the final relation token skips cross-attention. This is unusual, but changing it would
alter the baseline.

## Not ported

The port does not include: 2D/pre-2D backbones and padding paths in this 3D model,
general-transformer adapters, MoCo, diameter smoothing, model-embedded debug printing and random
visualization, or compiled objects and previous build directories.

`models.checkpoint.checkpoint_compatibility` accepts checkpoints that contain domain
discriminator tensors when the current model was built without them; those keys are reported as
ignored. Missing active keys, other unexpected keys and shape mismatches are hard failures.

## Deformable-attention implementations

The model has two execution paths with the same trainable state:

- `model.decoder.use_cuda_extension: true` uses the compiled CUDA extension (training);
- `false` uses PyTorch `grid_sample`, for CPU tests and debugging only.

Build the extension on Jean Zay from the repository root, inside a GPU job or after loading the
H100 environment:

```bash
bash scripts/build_deformable_attention.sh
```

The build needs the CUDA toolkit of the active PyTorch installation. Build products are ignored
by Git.

## Tests

```bash
python -m unittest -v tests.test_model_components   # CPU-safe
python -m unittest discover -v                      # all tests
python -m unittest -v tests.test_model_cluster.CudaModelTests   # after building the extension
```

The CUDA test compares the CUDA operator with the PyTorch fallback and runs a small end-to-end
forward on GPU.

### Checkpoint verification

```bash
export GNBM_MRI_CHECKPOINT=/absolute/path/to/checkpoint.pt
python -m unittest -v \
  tests.test_model_cluster.RealCheckpointTests.test_mri_checkpoint_matches_all_active_model_tensors
```

The test accepts legacy `module.` wrappers and reports old discriminator tensors as ignored.
Every active tensor must exist with the expected shape.

### Numerical comparison with the original repository

Run the original and refactored models in separate processes with the same checkpoint and a
deterministic input:

```bash
python scripts/compare_model_forward.py \
  --legacy-root /absolute/path/to/original/3d \
  --checkpoint "$GNBM_MRI_CHECKPOINT"
```

The script derives both architectures from `configs/pretrain_mixed.yaml` and compares the
decoder tokens, node logits, node coordinates and projected encoder features. A nonzero exit
status means an output is outside the reported tolerance. Each repository needs its own
deformable-attention extension built for the active PyTorch/CUDA environment. `--compare-losses`
and `--compare-inference` extend the comparison to the criterion and to graph decoding
([`TRAINING.md`](TRAINING.md), [`EVALUATION.md`](EVALUATION.md)).
