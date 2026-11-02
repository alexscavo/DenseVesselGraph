# IXI adaptive-GT 300-epoch run and held-out test

The initial Slurm job **93539** failed during checkpointing at epoch 76/100
because the shared Work filesystem quota was exceeded while creating its
runtime-state directory. Its last complete full-state checkpoint was epoch 74
(iteration 37,000), with both distributed ranks' RNG states intact.
Continuation job **101450** resumed from that checkpoint on Scratch and
completed epoch 100, including validation. It ended successfully
(exit 0, 44m21s). Job **113928** then trained through epoch 300, adding 200
epochs, and completed successfully (exit 0, 5h53m41s). The full held-out IXI
test results and paired real-patch views are in [test_results.md](test_results.md).

| Setting | Frozen value |
|---|---|
| Config | `configs/finetune_ixi_vessels_q256.yaml` |
| Train | 8,000 seeded IXI patches from 13,742 active training patches |
| Validation | all 2,961 active IXI validation patches |
| Test | all 2,384 active IXI test patches; job 138721, exit 0 |
| Split unit | source volume/patient; no patch sharing across splits |
| Object queries | 256 |
| Hardware | two H100 GPUs, batch 8 per GPU (global batch 16) |
| Initial job 93539 | 12 hours, `qos_gpu_h100-t3`; failed on Work quota at epoch 76 |
| Resume job 101450 | 2 hours, `qos_gpu_h100-dev`; completed 100 epochs on Scratch |
| Extension job 113928 | 10 hours, `qos_gpu_h100-t3`; completed in 5h53m41s on Scratch |
| Epoch target | 300 total (200 additional after epoch 100) |
| Evaluation | validation loss and graph metrics every 10 epochs; Branch F1 in physical mm |
| Checkpoints | best validation, best Branch F1, and replace-in-place latest every two epochs |
| Initialization | random, fixed seed 364505; not resumed from the one-epoch smoke/pilot |
| Extension schedule | resume model and optimizer; restart polynomial decay at `7e-6` (10% of original `7e-5`) through epoch 300 |

The one-GPU I/O/metric pilot was job `93034`: 512 train / 64 validation,
batch 8, one epoch, exit 0 in 89 seconds. Its measured training portion was
25.25 seconds for 512 samples (20.28 samples/s; 3.541 GiB peak allocated).
This is a **one-GPU timing anchor**, not a measured two-GPU throughput or a
convergence claim. The pilot confirmed that Branch F1 appears in
`validation-metrics.jsonl`; at one epoch from scratch it was 0, as expected.
The completed 100-epoch run's final validation loss was 3.923579; Branch F1
was 0.0056663, node F1 0.3743385, and edge F1 0.2963719. The Branch F1
selection metric and validation checkpoint state are carried into the
extension.

The original Work outputs are `runs/ixi_q256_100ep_20260923/`; the completed
100-epoch resume record is `$SCRATCH/experiments/gnbm/ixi_q256_100ep_20260923_resume/`.
The 300-epoch continuation writes to
`$SCRATCH/experiments/gnbm/ixi_q256_extend200_20260923/`.
Cluster stdout/stderr:
`/lustre/fswork/projects/rech/vnc/upz25mj/logs/graph-native-betti-matching/gnbm-train-93539.{out,err}`.
The root error was `OSError: [Errno 122] Disk quota exceeded` at
`models/.runtime_states`; this was a filesystem-quota failure, not GPU memory
or model instability. The checkpointing output is moved to Scratch for the
resume. The extension config and scheduler-rebase helper are
`configs/finetune_ixi_vessels_q256_extend200.yaml` and
`cluster/jean_zay/rebase_resume_checkpoint_lr.py`. The helper preserved the
full optimizer and distributed state, then prepared a separate Scratch
checkpoint with a smooth 200-epoch polynomial decay starting at 10% of the
original peak LR.

The offline W&B runs are `9gfnprri` (initial job), `k6foj18b` (100-epoch
resume), and `g9hzhs3h` (300-epoch extension). From a
network-connected Jean Zay login node, upload the Work run with
`bash cluster/jean_zay/sync_wandb_offline.sh runs`, the completed 100-epoch
resume with `bash cluster/jean_zay/sync_wandb_offline.sh /lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_q256_100ep_20260923_resume`,
and the completed extension with
`bash cluster/jean_zay/sync_wandb_offline.sh /lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_q256_extend200_20260923`,
after `wandb login`.

The chart was regenerated with the project H100 virtualenv's Python
(the system Python does not have Matplotlib installed). Regenerate it with
`/lustre/fswork/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python docs/research/evidence/T09_long/plot_progress.py`.
It combines all job segments and JSONL records, deduplicating repeated epochs
75–76 in favor of resumed values. It includes all 300 training epochs and 30
validation epochs. The validation-selected checkpoint is from epoch 270
(Branch F1 0.0067563).
