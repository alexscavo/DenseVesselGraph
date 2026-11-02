### Native 2D smoke collection — 27 October 2026

GPU correctness: **passed**; original-size architecture: **passed**; tiny memorization gate: **passed**.
Slurm GPU `1637307`, collector `1637308`. No full training was authorized or submitted.

Two training crops only; the table is a memorization diagnostic, **not validation performance**.

| Update | Loss | Crop | Node F1 @2px | Edge F1 @2px | Branch F1 | Curve F1 |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 11.3064 | wide01_00 | 0.1000 | 0.0000 | 0.0000 | 0.0831 |
| 0 | 11.3064 | wide01_01 | 0.1000 | 0.0000 | 0.0000 | 0.1023 |
| 1 | 10.1148 | wide01_00 | 0.1333 | 0.0000 | 0.0000 | 0.0941 |
| 1 | 10.1148 | wide01_01 | 0.1333 | 0.0000 | 0.0000 | 0.0971 |
| 10 | 8.3737 | wide01_00 | 0.1667 | 0.0000 | 0.0000 | 0.0000 |
| 10 | 8.3737 | wide01_01 | 0.1667 | 0.0000 | 0.0000 | 0.0000 |
| 50 | 6.0358 | wide01_00 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| 50 | 6.0358 | wide01_01 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| 100 | 4.8604 | wide01_00 | 0.3333 | 0.3333 | 0.0000 | 0.1739 |
| 100 | 4.8604 | wide01_01 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| 250 | 2.2569 | wide01_00 | 0.6667 | 0.3333 | 0.0000 | 0.2122 |
| 250 | 2.2569 | wide01_01 | 0.8333 | 0.3333 | 0.0000 | 0.6148 |
| 500 | 0.4029 | wide01_00 | 1.0000 | 1.0000 | 1.0000 | 0.9765 |
| 500 | 0.4029 | wide01_01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1000 | 0.1200 | wide01_00 | 1.0000 | 1.0000 | 1.0000 | 0.9789 |
| 1000 | 0.1200 | wide01_01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

1000 updates in 22.9s; peak GPU allocation 659497472 bytes. Checkpoint: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261027/runtime/tiny-last.pt`.

Passing this gate establishes implementation learnability on two crops only. It supplies no cross-subject generalization, baseline-versus-A-xyz benefit or journal reconstruction result. Full training remains pending subject/provenance/crossing and operating-point protocol decisions.

Evidence: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261027`. Retrieval: `python3 scripts/show_native2d.py` from the main repository.


<!-- NATIVE2D FINAL REVIEW 20261028 -->
## Native 2D completion review — 28 October 2026

**DONE — measured (implementation/tiny memorization only).** GPU 1637307 completed in45 seconds at 04:05; CPU collector 1637308 completed at 04:06, both exit0. Experiment IDs and the older automatically generated headings use the27 October launch date; actual completion was28 October (Europe/Paris). The2D work waited for BRAVA to release the two globally reserved H100s. No full 2D training or test/validation performance evaluation was launched.

Eight focused regressions and13 real-image CPU/CUDA conditions passed, including shared-weight initialization parity, unchanged class logits under A-xyz, relation symmetry, exact spline endpoints, graph-only gradient isolation, finite backward and strict in-memory legacy-core replay. The original-width **ResNet101 /512D /3 encoder+3 decoder layers /80 object+5 relation queries** also passed one forward/backward on a 64² real crop (**72,440,408 parameters; zero optimizer updates**). This is architecture correctness, not convergence or128² production throughput.

The separate tiny correctness model is **ResNet18 /64D /one encoder+one decoder layer /16 object+two relation queries**, optional A-xyz enabled, K8 `graph_only`, all detector parameters trainable. It trained 1,000 AdamW updates on **two 64² crops from the same AV-WIDE training image**, without augmentation or checkpoint selection. Training took **22.9 seconds**, 43.62 updates/s; peak allocation about 629 MiB. By the logged 500-update checkpoint, both crops had perfect node/edge/Branch F1. Final total loss fell **11.3064→0.1200**.

| Training crop | Pred/GT nodes | Pred/GT edges | Node/edge F1 @1px | Node/edge F1 @2px | Branch F1 | Curve F1 | Mean matched localization, px |
|---|---:|---:|---|---|---:|---:|---:|
| wide01_00 | 6/6 | 5/5 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 | 0.9789 | 0.209 |
| wide01_01 | 6/6 | 4/4 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 | 1.0000 | 0.181 |

No duplicate nodes or false edges remain in these two memorized graphs; raw coverage and conditional relation recall/AP are1.0. The imperfect first-crop Curve F1 remains visible despite exact graph recovery. Distances are **pixels**, not invented millimeters. These results establish that the implemented detector, local coordinate module, matched relations and detached spline head can jointly fit this tiny input/target set. They do **not** isolate an A-xyz benefit, validate cross-subject generalization, establish a spline-versus-chord gain, or compare the tiny model with the original-width architecture.

**What remains:** no complete 2D training population, trained B0/A-xyz comparison, held-out performance or final patient/specimen-disjoint split exists yet. AV-WIDE supplies paired images/graphs and REAVER supplies documented raw/manual masks/skeletons; OCTA-Synth raw/mask folders remain empty. Physical calibration is dataset-specific, biological grouping is unresolved, and ambiguous projected crossings must be audited before full training. Use native2D entry points; the old 3D training CLI remains unchanged. The Python 3.10 compiled 2D attention binary is still incompatible with the active Python 3.11 runtime; smoke uses the original PyTorch reference, so tiny timing is not a production-kernel benchmark. No new architecture or loss was introduced during reporting.

Main2D source and data-provenance report is the preceding section. Original numeric evidence and checkpoint are preserved in sibling `gnbm-native-2d-20261027/experiments/native2d_20261027/`, with checkpoint runtime on SCRATCH. The consolidated review JSON is `experiments/completed_result_review_20261028/review.json`. Print saved results with:

```bash
python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/scripts/show_native2d.py
```
