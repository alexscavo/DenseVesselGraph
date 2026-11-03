# Image-to-Graph-Vessel — legacy project inventory

<!-- NATIVE2D CURRENT POINTER 20261028 -->
> Latest native2D result: **Native2D-03 corrected-target pilot completed** (B0/A-xyz6,000 updates each; no filtered graph recovery at0.5). [Final comparison, dataset readiness and stopping decision](../docs/research/native2d_heldout_20261028.md#completed-corrected-target-comparison). Native2D-02 remains preserved with its unstable-target and post-training-timeout caveats.


Latest reviewed result: [native 2D completion](#native-2d-completion-review--28-october-2026), 28 October 2026. Correctness/tiny learnability passed; full training remains unlaunched.

Audited 23 October 2026. Companion project: [Image-to-Graph-Vessel](../../../../../Image-to-Graph-Vessel). This records source code, prepared configurations and archived observations separately. It does not replace the continuous-representation experiment history.

## Scope and architecture

The sibling repository contains self-contained **2D and 3D image-to-node/edge RelationFormer pipelines**. Its [README](../../../../../Image-to-Graph-Vessel/2d/README.md) describes mixed-domain and cross-dimension learning; that description alone does not establish which options were active in any published experiment.

| Component | Source | Observed role |
|---|---|---|
| Entrypoints | [3D training](../../../../../Image-to-Graph-Vessel/3d/train.py), [2D training](../../../../../Image-to-Graph-Vessel/2d/train.py) | Separate dimensional pipelines and uppercase YAML configurations |
| Detector | [3D RelationFormer](../../../../../Image-to-Graph-Vessel/3d/models/relationformer.py), [adapters](../../../../../Image-to-Graph-Vessel/3d/models/adapter.py) | CNN/deformable transformer, object and relation tokens; projection/inflation/group cross-dimension adapters |
| Data | [mixed loader](../../../../../Image-to-Graph-Vessel/3d/data/dataset_mixed.py), [plant loader](../../../../../Image-to-Graph-Vessel/3d/data/dataset_plants.py) | Supervised source/target mixtures; plants embedded as a five-slice slab in a 64³ volume |
| Objectives | [criterion](../../../../../Image-to-Graph-Vessel/3d/training/losses.py), [contrastive objective](../../../../../Image-to-Graph-Vessel/3d/training/losses_supcl.py) | Node weighted CE `[0.25,0.75]`; edge CE with configurable sampling; optional domain, hard-negative, EMA and supervised-contrastive pathways |
| Evaluation | [batch inference](../../../../../Image-to-Graph-Vessel/3d/run_batch_inference_eval.py), [BN calibration](../../../../../Image-to-Graph-Vessel/3d/run_eval_with_bn_calib.py) | Native detection AP/AR, SMD and graph diagnostics; optional BN adaptation changes evaluation state |

No continuous spline head was found in the inspected 3D model/training sources. Continuous edge geometry, adaptive K and piecewise splines belong to the newer representation repository. Checkpoint transfer requires explicit architectural and preprocessing checks; neither YAML keys nor metrics should be assumed interchangeable.

Dependencies include PyTorch, MONAI, NumPy, PyVista, MedPy and custom 3D deformable-attention CUDA operators (see [requirements](../../../../../Image-to-Graph-Vessel/requirements.txt)). Existing compiled artifacts include Python 3.10 builds and are not evidence of compatibility with the current protected Python 3.11 runtime.

## Prepared experimental families

These are **configuration facts**, not completion claims. Counts are requested source populations, not audited unique-subject exposures.

| Family | Configurations / protocol |
|---|---|
| Source identity | `ablation_plants_only_3d.yaml`, `ablation_halvs_only_3d.yaml`, `ablation_roads_only_3d.yaml` |
| Source quantity and balance | `ablation_all_available_3d.yaml`: plants 25,923 + HALVS 7,146; `ablation_equal_mix_3d.yaml`: 7,146 + 7,146. These change total source counts as well as composition |
| Contrastive pretraining | `ablation_all_available_supcl_3d.yaml`: `USE_SUPCL=True`, weight 0.1, temperature 0.3, warmup 10 epochs |
| Legacy named SupCL recipe | `pretrain_plants_synth_supcl.yaml` / `finetune_plants_synth_supcl.yaml`: 50 / 100 epochs, but **USE_SUPCL=false in both**. Filename is not proof of active contrastive learning |
| Target fine-tuning | `finetune_target_only_synthmr_3d.yaml`: 100 epochs; separate target-only and contrastive variants exist |
| Broader targets / methods | IXI, SyntheticMRI, TopCoW CT/MRA, MiniVess and HiP-CT configurations; adversarial, consistency, EMA/HNM and MoCo code paths. Presence does not establish a measured benefit |

Config directory: [3D configurations](../../../../../Image-to-Graph-Vessel/3d/configs). The inspected source-ablation recipes commonly use batch 32, 50 epochs, AdamW LR 7e-5, Hungarian class/node weights 2/5 and ratio-upsample edge sampling 0.15. Exact launched settings must be recovered per run.

## Archived observations and provenance

The [job archive notice](../../../../../job-runs/README.md) preserves August jobs at:

`/lustre/fsstor/projects/rech/vnc/upz25mj/archives/2026-10-06/job-runs-2026-08.tar.gz`

This audit indexed **271 archived files** and extracted final printed metric blocks from **32 evaluation logs**, without executing models. See [archive index](evidence/image-to-graph-vessel/archive-index.csv) and [all recovered metric rows](evidence/image-to-graph-vessel/historical-evaluation-metrics.csv). Each recovered row retains its archive member, content SHA256 and a saved-results marker. A marker is not a current Slurm-state check. Logs without a metric block remain in the archive index.

Examples of historical output (native legacy definitions; no new evaluations):

| Archived evaluation | Node mAP | Node mAR | Edge mAP | Edge mAR | SMD |
|---|---:|---:|---:|---:|---:|
| eval3d-from-baseline-697088.out | 0.092929 | 0.158859 | 0.157178 | 0.247482 | 0.014735 |
| eval3d-from-plants-only-695050.out | 0.082584 | 0.139860 | 0.148330 | 0.221700 | 0.015538 |
| eval3d-from-halvs-only-695049.out | 0.036843 | 0.088113 | 0.042579 | 0.109616 | 0.018172 |

**Interpretation limits:** these are raw historical log summaries, not a newly harmonized ranking. “Baseline” is an archived run label, not proof of no pretraining. Exact checkpoints, data membership, BN calibration, seeds and aggregation must be reconciled before causal source comparisons. Logged standard deviations are not subject-bootstrap confidence intervals. Do not compare these native AP/SMD values directly to common-v2 physical Branch/Curve or newer relation AP. No new test-set selection or evaluation was performed.

The archive also contains later `wd1`, `lambda1` and SupCL-stage variants; all recovered metric blocks are retained in the CSV rather than selecting only favorable runs. Their names alone do not define their actual loss weights.

## Decisions and relevance to the journal

- Preserve the legacy code/archive as provenance; do not overwrite or silently port it into current training.
- Establish any GrEVe-era claim from the actual paper/run records. Roads/plants versus no pretraining is a different comparison from additional vascular-domain optimization.
- Do not activate domain adversarial, SupCL, HNM or BN adaptation merely because the legacy repository implements them. Each would change a controlled comparison.
- Original plant inputs were generated from segmentation-derived vessel-like rendering, not simply the original RGB photograph. Preserve the current loader coordinate/boundary corrections rather than copying the legacy loader wholesale.
- Some YAMLs/generator entrypoints still contain `/data/scavone` paths; `AUTO_RESUME` is explicitly noted as unwired in source-ablation configurations. These are archival recipes, not ready-to-launch commands.
- BN calibration on validation inputs is adaptation/transductive evaluation and must be declared separately from a frozen zero-shot reference.

Detailed verified source recovery and rendering discussion: [plant recovery](../docs/research/pretraining/plants-recovery.md), [textured-source provenance](../docs/research/pretraining/textured-source-prototype.md), [domain adaptation](../../../experiments/domain_adaptation.md).

## Priorities and pending work

This inventory launches no jobs and does not alter the current detector-recovery priority. If legacy experiments are needed for the journal, first reconcile archived launch scripts, resolved configs/checkpoints and split membership, then select the necessary reference comparisons. Keep 2D transfer, 3D source composition, contrastive learning and continuous representation as distinguishable research questions. Existing modern transfer results remain in the [experiment register](experiments.md); they are not relabeled as results from this sibling repository.


<!-- NATIVE2D IMPLEMENTATION 20261027 -->
## Native 2D journal extension — 27 October 2026

**Scope and status:** native 2D implementation and bounded correctness study, not a full training campaign. The original `Image-to-Graph-Vessel/2d` and the current roadmap, architecture, decisions, registry, modern local-refinement study and completed N06/N08/R06/R07 evidence were reviewed. No completed oracle, selector, relation-adapter, mining or geometry-capacity study was repeated. Modern SyntheticMRI supports optional A-xyz; its benefit is still a hypothesis in native 2D.

**Isolation:** branch `research/native-2d-20261027`, worktree `../gnbm-native-2d-20261027`. New `native2d/`, config, tests and scripts are also available in the main repository. Only an explicit `spatial_dims == 2` dispatch was added to `models.build_model`; default 3D construction, parameter names, checkpoint files and existing implementations remain unchanged. `shared-before.json`, `integration.json` and source manifests record the audit. Existing uncommitted changes were preserved.

### Reuse, implementation and incompatibilities

| Component | Native 2D implementation / contract |
|---|---|
| Detector core | Original ResNet + deformable encoder/decoder, sine positional encoding, XY/WH boxes, object/relation queries and pair MLP. `native2d/backbone.py`, `transformer.py`, `position_encoding.py`, `model.py` retain legacy core parameter names. No new attention architecture. |
| Attention runtime | Reuse the original 2D PyTorch `grid_sample` reference implementation with the same trainable matrices. The available legacy extension is a Python 3.10 binary, incompatible with the current Python 3.11 environment. Native 2D correctness runs require no CUDA extension and do not alter the 3D extension. Compiling/benchmarking a matching optimized 2D kernel is a later performance task. |
| Initial architecture | `configs/native2d.yaml`: original 2D ResNet101, hidden512, 8 attention heads, 3 encoder /3 decoder layers, four feature levels, 80 object /5 relation queries. Image input RGB, grayscale repeated into RGB; `/255 - 0.5`, no resizing in this smoke. No automatic ImageNet download or undocumented pretraining. |
| Modern graph objectives | Reuse `GraphCriterion` and actual `HungarianMatcher(dimensions=2)`: cost `5*L1 - 2*object_probability`, node softmax focal gamma2 weights `[1,1]`, matched-pair edge CE weights `[1,1]`, ratio upsampling0.15/tolerance0.02, cap9999, no positive cap/unmatched mining. Existing box/class/cardinality/node/edge weights3/4/0.8/2/6. 2D GIoU replaces the dimension-specific box operation. |
| A-xyz | Optional zero-output-initialized coordinate-only residual. **Nine** samples in a native3×3 XY neighborhood at −4/0/+4 pixels replace the 3D27-sample neighborhood. ResNet early `relu` has64 channels, stride2 and input centers0,2,4,… . Exact pixel-center `align_corners=False` conversion; CNN samples and sampling centers detach. Decoder token/coordinate gradients remain connected. Original class logits and relation tokens remain unchanged. Optional-head construction preserves shared-weight RNG initialization. |
| Geometry | Endpoint-constrained clamped uniform cubic K8 B-spline, unchanged 64-sample basis and physical chord-residual parameterization. Two coordinates/control, six internal controls. Canonical physical endpoint ordering makes endpoint reversal invariant. Geometry tokens/endpoints detach under `graph_only`; no additional detector loss. Existing residual-control Smooth-L1 is averaged per coordinate and edge; geometry weight1.4739989612791453. Local CNN-conditioned spline correction is **not included** in this minimal port; fine CNN sampling is used by optional A-xyz. |
| Compatibility | Fresh/state-dict round-trip and strict legacy-core loader checks passed. Added-head keys are the only permitted missing keys; extra/mismatched core keys fail. No actual archived 2D checkpoint was found in the inspected 2D implementation directory, so cross-run 2D checkpoint parity is not yet established. 3D convolutions/coordinates/control tensors cannot be loaded into 2D by truncation. |
| Training integration | Dedicated native2D config/data/loss/check entry points; old 3D `train.py` and 3D config validator are unchanged. Use the native entry points below; pointing the existing 3D CLI at this YAML is not supported. |

### Data inventory and provenance

Root: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets`.

| Dataset | Actually present | Readiness / supervision |
|---|---|---|
| AV-WIDE |30 `.mat` files; audited images01/02 contain `I2`, dense `G.V` and `G.A` | Paired retinal images and supplied AV-labelled graphs. Preserve graph connectivity; these are **not independently hand-traced centerline GT**. Ordered graph polylines are the geometric reference; no mask is fabricated. Physical calibration and biological patient IDs were not supplied in the inspected metadata. |
| AV-DRIVE |20 training /20 test graph MATs | Source README explicitly excludes original images; missing paired raw images. No test MAT was loaded. |
| AV-INSPIRE |40 graph MATs | Original images excluded by source README; not image-to-graph-ready. |
| REAVER |36 raw /36 Manual TIFFs, quantitative CSV | Local DOCX confirms R=manual binary segmentation, G=manual-analysis skeleton, B=raw image. One development scan verified: B is bitwise identical to raw; skeleton mask coverage100%; TIFF0.207194025 microns/pixel agrees with CSV. Contract the provided skeleton; labels are derived graph connectivity from manual-analysis outputs, **not independent anatomical graph annotation**. Automated-method folders are predictions, never GT. Specimen IDs/splits require clarification. |
| OCTA-Synth |480/220/2000 graph directories; raw and label folders **empty** | Cannot train from current local copy. Recover paired source images/masks and generator/split provenance before use. Test directory names counted only. |
| HALVS |144 train /6 validation image-mask pairs | Botanical source, not vascular GT. Graphs from masks would be derived labels. Preserve leaf/source grouping before any transfer study. |
|20cities2D |Prepared directories exist but are empty | No ready road source pairs in this local copy. |
| RETA (`16960855`) |Image/MAT/model/support RAR archives plus README | Not extracted or validated in this task. README describes54 train /27 test; public test vessel masks unavailable. Do not treat archive names or supplied segmentation models as graph GT. |

Only **two64×64 training crops from AV-WIDE01 and two development validation crops from AV-WIDE02** were prepared. They have6/6 and2/7 nodes, respectively. All crops from a source image stay together. This is **image-disjoint**, not a proven subject-disjoint journal split; patient/specimen identity remains a blocker. REAVER has a separate128×128 alignment-only crop and is not mixed with retinal training. Source hashes, crop starts, normalization, units and label provenance are in `manifest.json`, `dataset-audit.json` and the per-crop JSONs. No existing source annotation, split or mask was changed.

### Extraction, clipping, crossings and annotation artifacts

`native2d/targets.py` supports binary-mask skeletonization and separately accepts an existing centerline without reskeletonizing it. The pixel graph uses8-connectivity, removing a diagonal shortcut only where an orthogonal pixel route already exists. Preserve terminal/junction identities, contract degree2 chains while keeping every ordered dense reference path, and retain pure cycles with three deterministic anchors. Parallel paths receive necessary degree2 anchors to fit the existing simple-graph output; they are not deleted. Isolates and short components remain visible rather than silently pruning apparent artifacts.

Crop the **dense path**, not a straight endpoint chord: clip every segment, including outside/outside crossings and exit/re-entry fragments. Source anchor identities are reused; newly clipped boundary endpoints are edge-specific, so two spatially crossing but disconnected vessels are not merged. Boundaries use pixel-center extents0…size−1; normalized coordinates are XY/axis size, consistent with modern geometry sampling. Save ragged original paths, offsets, controls,2×2 physical metric and explicit units.

Four-way skeleton junctions and adjacent multi-pixel junction clusters are flagged for QC. No automatic anatomical resolution of artery/vein crossings is defensible from a binary projection. Provided graph adjacency remains authoritative even where strokes cross. The inspected REAVER skeleton has19 junction pixels, no degree≥4 pixels, no adjacent-junction clusters and no isolates; this does **not** certify the other35 images. Full crossing/short-component/connectivity QC is required before deriving a training benchmark from masks. No graph-labelled claim is made for image-only sources.

Five overlays show cyan original paths, dashed magenta K8 fits and white anchors. Retinal green-channel contrast is changed **only for display**, not the saved RGB network input. REAVER includes the independently available manual mask/skeleton overlay. Inspect `prepared/*-overlay.png`; dense references remain separate from fitted K8 targets.

### Coordinates and metrics

- Image tensors BCHW, model/GT coordinates **XY=(column/width,row/height)**. MATLAB `G.V` is XY with1-based image pixel centers; subtract1 exactly once, following the supplied plotting convention. Raw source and crop provenance remain recorded.
- Unknown retinal spacing stays **pixels**; no invented millimeter scale. REAVER's micron calibration is converted to millimeters in saved metrics. For future microscopy evaluation, predeclare meaningful micron/pixel tolerances; inheriting MRI1/2mm thresholds would be scientifically inappropriate.
- The current retinal smoke uses **1/2 pixels** for one-to-one maximal-cardinality positional node/edge matching. Equivalent1/2mm evaluation is supported only for genuinely millimeter-calibrated inputs. Confidence/edge cutoffs remain0.5. Report raw query coverage, filtered counts, duplicates, localization, conditional relation recall/AP with explicit populations, and signed components/cycles.
- Native2D box area IoU, detection AP/AR and original normalized straight-edge SMD are retained separately from physical graph metrics. AP uses predicted XYWH boxes and the legacy40-detection cap; physical graph counts are uncapped. Physical squared-distance SMD is separately labelled; geometric coverage/SMD alone cannot establish correct adjacency or punish every overprediction adequately.
- Reuse common-v2 Branch/Curve definitions through an **isometric zero-plane metric adapter**, not a3D network: distances, graph connectivity and arc sampling are unchanged. Unit-depth detection boxes give exactly2D IoU. Common-v2 remains1-unit anchors/coverage with0.25-unit arc sampling; labels explicitly say pixel when spacing is unknown. Curves are compared to original supplied dense paths, not fitted K8 targets.

### Verification, bounded execution and exit gates

Eight focused regressions pass: crossing/boundary clipping, cyclic topology, anisotropic augmentation, exact CNN sampling centers, zero refinement, endpoint-constrained spline/gradient behavior, common metric identity/units, maximal-cardinality matching and unchanged3D dispatch. Real-image CPU forward/backward passes with finite node/refinement/relation/spline gradients; geometry-only backward leaves every detector gradient absent/zero. Relation logits are symmetric before softmax; spline endpoints and endpoint reversal are exact. Optional A-xyz and baseline share initialization; eval forwards do not mutate state. CPU code and imports were also checked in the main repository.

A five-update CPU rehearsal reduced total loss11.3396→9.0380. It is **not a successful overfit or a reconstruction result**. Its files remain under `cpu-five-step/` and `cpu-five-step-runtime/`.

The requested GPU smoke is one job: (1) repeat correctness on CUDA, (2) one original-size ResNet101/512D forward/backward with **zero optimizer updates**, (3) at most**1000 updates** on the same two real training crops, ResNet18/64D one-layer correctness model, batch2, AdamW2e−4/weight decay1e−4, clip0.1, seed364505, no augmentation and no validation selection. Fixed checkpoints/metrics at0/1/10/50/100/250/500/1000. Predeclared memorization gate: each crop node F1≥0.9 and edge F1≥0.8 at2px. Failure is recorded, not followed by automatic longer training. This tiny model is a correctness fixture, not the proposed journal architecture. All graph/backbone parameters are trainable; spline gradients stay isolated.

One H100 maximum,20-minute wall allocation, followed by a CPU collector using `afterany` so failures are reported too. Existing BRAVA work already occupies the two-H100 ceiling; serialize the smoke after its submitted GPU chains. No running job was cancelled or modified. Checkpoint/runtime files use SCRATCH `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/native2d_20261027`.

### Reproduce and subsequent controlled plan

From the main repository, after sourcing the standard Jean-Zay environment:

```bash
python -m unittest discover -s tests -p test_native2d.py -v
python scripts/check_native2d.py --device cpu
python3 scripts/show_native2d.py
```

The last command reads saved evidence only, follows the isolated worktree pointer and requires no scientific Python environment. Source hashes/archive, CPU/CUDA checks, dataset manifest, numeric learning curves and Slurm records live in `../gnbm-native-2d-20261027/experiments/native2d_20261027/`. GPU results are pending until its collector appends the next dated section. No scientific benefit is claimed at submission.

**Next experiments, prepared conceptually but not launched:** first resolve source-image recovery, patient/specimen grouping, retinal scale and ambiguous-crossing policies; freeze source-disjoint manifests and a full extraction QC audit. Then compare native B0 versus optional A-xyz with the same fresh or truly compatible2D source initializer, trainable set, augmentation, focal/CE objective, patch order and update checkpoints. Start K8 `graph_only`; compare learned splines versus straight chords on **identical** predicted nodes/edges. Separate graph-target representation and geometry effects. Choose operating points from training subjects only, evaluate held-out validation once with paired subject uncertainty, and keep final test sources untouched. Do not add R06/R07/N08 changes, a new matcher, multiscale architecture or new detector losses. Only request a bounded training budget after the tiny-fit and annotation/split gates pass; no full2D training is authorized by this implementation study.



<!-- NATIVE2D ACCEPTED JOBS 20261027 -->
### Native 2D accepted bounded jobs — 27 October 2026

**SUBMITTED — verify:** GPU smoke **1637307**, one H100,20-minute allocation; original-size architecture backward plus1000-update memorization on two training crops. CPU collector **1637308**, `afterany:1637307`, appends the outcome automatically even if smoke fails. GPU dependency: `afterany:1637025:1637027:1637032:1637034:1637036:1637037:1637039:1637040:1637042:1637044:1637045:1637046`. These are existing BRAVA GPU chains, ending at1637046; the dependency preserves the global two-H100 limit and may delay the small2D check until that longer batch ends. No existing job was cancelled, no full2D training submitted.

Completed before submission: eight unit regressions,13 real-image CPU checks, two training/two validation development crops plus one REAVER alignment overlay. Final main-repository replay passed; existing3D implementation hashes remain unchanged. CUDA and tiny memorization results remain pending, not claimed passed.

Evidence: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261027`; large runtime/checkpoint output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/native2d_20261027`. Source/config hashes and archive are frozen in `code-identities.json` / `source.zip`. Read saved results with:

```bash
python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/scripts/show_native2d.py
```



<!-- NATIVE2D COLLECTED 20261027 -->
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

<!-- NATIVE2D HELDOUT LIVE 20261028 -->
## Native2D-02: held-out original-width AV-WIDE study — 28 October 2026

Authorized bounded B0 versus A-xyz pilot, following Native2D-01. All measurements below refer to image-disjoint validation, never test data. Patient independence is unverified. Distances are pixels.

Cohort: 17 train / 6 validation images; six reserved test images have header-only inspection. Image 07 excluded for its diagonal self-loop. 1385 train / 506 validation 128² tiles; 96 object queries selected from contracted training GT. All spatial tiles retained, including empty graphs.
**Target-quality limitation:** the original irregular-sampling K8 least-squares fit produces pathological controls (one verified128² tile reaches 56,890 px), causing shared loss spikes. This pilot is preserved but does not support a reliable geometry conclusion. A separately recorded single corrected-target pair uses uniform arc-length samples of unchanged dense paths; no graph labels, image splits, architecture or loss policy change.
Full ResNet101/512D, 3 encoder + 3 decoder layers, 5 relation tokens: gates PASSED; 1200 optimizer updates on two 128² training tiles before the experiment. Shared initialization and graph-only gradient checks passed. Budget fixed to 6000 updates per arm, batch 4, no automatic extension.
Throughput (existing PyTorch deformable attention): B0 0.0575s/update, 69.52 images/s, Axyz 0.0578s/update, 69.25 images/s.
Slurm: gates `1653554`, Axyz `1653991`, collector `1653994`, aborted_B0 `1653987`, B0 `1654241`.
Scheduler outcome: A-xyz completed normally. B0 completed all 6,000 updates and all 506-crop final metrics in about 436s, then its redundant initialization-checkpoint curve evaluation hit the1h45 Slurm limit. Its job state is **TIMEOUT**, despite complete primary final-checkpoint evidence. The initialization artifact is preserved as `B0/excluded-initialization.pt`; the trained-checkpoint index was repaired from the genuine final checkpoint/metrics, without retraining or new inference. No budget extension or threshold tuning was performed.

Primary comparison: **final fixed-budget checkpoints**, all validation tiles. Branch/Curve columns use GT edge-positive crops; all-crop aggregates are also retained in JSON, where empty/empty graphs can receive perfect scores.

| Arm | State | Final update | Node F1 @2 px | Edge F1 @2 px | Branch F1 @1 px | K8 Curve F1 @1 px | Chord Curve F1 @1 px |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | complete | 6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Axyz | complete | 6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Arm | Node mAP | Node mAR | Edge mAP | Edge mAR | Raw node recall @2 px, micro | Filtered node recall @2 px |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.0846 | 0.4316 | 0.0305 | 0.3001 | 0.2353 | 0.0000 |
| Axyz | 0.0833 | 0.4286 | 0.0289 | 0.3187 | 0.2240 | 0.0000 |

**Negative bounded pilot:** both final checkpoints retain zero nodes/edges at the predeclared0.5 confidence thresholds, missing 7,613 GT nodes and 6,308 GT edges. Conditional relation performance is undefined because no filtered endpoint pairs remain; zero duplicates are a consequence of empty predictions. There is no evidence of A-xyz benefit or spline superiority. The8 edge-empty target crops inflate all-crop Branch/Curve macro F1 to0.01581; on 498 edge-positive crops these metrics are exactly zero. Pixel-space raw coverage and unthresholded AP remain visible, and no validation-driven operating-point change is made. Stop at the fixed budget; no automatic extension.

Intermediate validation uses 24 fixed spatial tiles (4/image); final and selected trained checkpoints are evaluated on every validation tile. Curve points at final therefore use a larger population; do not interpret that discontinuity as a training change. Selection uses the same 24 tiles throughout, excludes initialization, and resolves trained ties toward later updates. Best and final outcomes are both saved.

Detection AP/AR uses unthresholded query scores, IoU .50:.05:.95, node limit equal to query capacity and edge top300; reported AP is crop macro. Physical node/edge F1 is micro over crops at 1/2/4 px, including crop-boundary anchors. Branch/Curve F1 is crop macro at 1 px, .25 px sampling. Spline/chord use identical inferred nodes/edges; topology cannot change. Raw/filtered recall, relation AP populations, duplicates and Betti errors are retained per crop. No subject-level uncertainty claim, full-image stitching, independent manual topology or pretraining benefit is established.

Evidence and checkpoints: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261028`. [Protocol and dataset readiness](../docs/research/native2d_heldout_20261028.md). Learning curves: `learning-curves.png`; full metrics: `B0/best-heldout.json`, `Axyz/best-heldout.json`; final metrics: `<arm>/validation-<budget>.json`; weights/optimizer: `<arm>/best.pt`, `<arm>/latest.pt`. Failed gates and jobs are reported without automatic resubmission.
<!-- END NATIVE2D HELDOUT LIVE 20261028 -->

<!-- NATIVE2D TARGETFIT LIVE 20261028 -->
## Native2D-03: corrected-target held-out AV-WIDE study — 28 October 2026

Authorized bounded B0 versus A-xyz pilot, following Native2D-01. All measurements below refer to image-disjoint validation, never test data. Patient independence is unverified. Distances are pixels.

Cohort: 17 train / 6 validation images; six reserved test images have header-only inspection. Image 07 excluded for its diagonal self-loop. 1385 train / 506 validation 128² tiles; 96 object queries selected from contracted training GT. All spatial tiles retained, including empty graphs.
Correction: uniform64 arc-length samples of the same polyline before K8 fitting. All graph/image/split/reference-path tensors are exactly unchanged. Refit 25263 branches; control max absolute coordinate 2007485.8→131.2 px, uniform design condition number4.55. One fresh fixed-budget pair only; original checkpoints are not resumed.
Full ResNet101/512D, 3 encoder + 3 decoder layers, 5 relation tokens: gates PASSED; 1200 optimizer updates on two 128² training tiles before the experiment. Shared initialization and graph-only gradient checks passed. Budget fixed to 6000 updates per arm, batch 4, no automatic extension.
Throughput (existing PyTorch deformable attention): B0 0.0558s/update, 71.72 images/s, Axyz 0.0559s/update, 71.59 images/s.
Slurm: gates `1659581`, B0 `1659729`, Axyz `1659730`, collector `1659731`.

Primary comparison: **final fixed-budget checkpoints**, all validation tiles. Branch/Curve columns use GT edge-positive crops; all-crop aggregates are also retained in JSON, where empty/empty graphs can receive perfect scores.

| Arm | State | Final update | Node F1 @2 px | Edge F1 @2 px | Branch F1 @1 px | K8 Curve F1 @1 px | Chord Curve F1 @1 px |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | complete | 6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Axyz | complete | 6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Arm | Node mAP | Node mAR | Edge mAP | Edge mAR | Raw node recall @2 px, micro | Filtered node recall @2 px |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.0880 | 0.4497 | 0.0375 | 0.3138 | 0.2374 | 0.0000 |
| Axyz | 0.0920 | 0.4120 | 0.0267 | 0.2918 | 0.2142 | 0.0000 |

**Negative bounded pilot:** both final checkpoints retain zero nodes/edges at the predeclared0.5 confidence thresholds, missing 7,613 GT nodes and 6,308 GT edges. Conditional relation performance is undefined because no filtered endpoint pairs remain; zero duplicates are a consequence of empty predictions. There is no evidence of A-xyz benefit or spline superiority. The8 edge-empty target crops inflate all-crop Branch/Curve macro F1 to0.01581; on 498 edge-positive crops these metrics are exactly zero. Pixel-space raw coverage and unthresholded AP remain visible, and no validation-driven operating-point change is made. Stop at the fixed budget; no automatic extension.

Intermediate validation uses 24 fixed spatial tiles (4/image); final and selected trained checkpoints are evaluated on every validation tile. Curve points at final therefore use a larger population; do not interpret that discontinuity as a training change. Selection uses the same 24 tiles throughout, excludes initialization, and resolves trained ties toward later updates. Best and final outcomes are both saved.

Detection AP/AR uses unthresholded query scores, IoU .50:.05:.95, node limit equal to query capacity and edge top300; reported AP is crop macro. Physical node/edge F1 is micro over crops at 1/2/4 px, including crop-boundary anchors. Branch/Curve F1 is crop macro at 1 px, .25 px sampling. Spline/chord use identical inferred nodes/edges; topology cannot change. Raw/filtered recall, relation AP populations, duplicates and Betti errors are retained per crop. No subject-level uncertainty claim, full-image stitching, independent manual topology or pretraining benefit is established.

Evidence and checkpoints: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261028_targetfix`. [Protocol and dataset readiness](../docs/research/native2d_heldout_20261028.md). Learning curves: `learning-curves.png`; full metrics: `B0/best-heldout.json`, `Axyz/best-heldout.json`; final metrics: `<arm>/validation-<budget>.json`; weights/optimizer: `<arm>/best.pt`, `<arm>/latest.pt`. Failed gates and jobs are reported without automatic resubmission.
<!-- END NATIVE2D TARGETFIT LIVE 20261028 -->

<!-- NATIVE2D LONG LIVE 20261028 -->
## Native2D-04 — mixed plant/vessel pretraining and longer vessel training, 28 October 2026

User clarified that pretraining must mix plants and vessels. The superseded pure-plant gate/launcher1670808/1670809/1670810 were cancelled while pending; no pure-plant training occurred. Use50 mixed-pretraining epochs, then100 vessel-only epochs, with unchanged confidence cutoffs0.5. Previous pilot artifacts and the superseded protocol archive are retained.

Protocol: [native2d_long_20261028.md](../docs/research/native2d_long_20261028.md). Queued/running states are not completed experimental results.
GPU gate job `1671559`; conditional training launcher `1671560`. The launcher submits training only after all dataset/GPU gates pass.
Full native ResNet101/512D,3+3 layers,5 relation tokens, common96 object queries from contracted training GT distributions. Each vessel stage:100 exact passes, batch4, identical exposure/augmentation/optimizer schedule within each B0/A-xyz pair.
For each vessel dataset, pretrain one B0 on plants plus ONLY that target training split. Use the existing50:50 domain-balanced sampling with replacement and a twice-source epoch (51,734 draws), batch32, for50 epochs. The final mixed checkpoint for that target initializes both B0/A-xyz; A-xyz starts with zero coordinate residuals. Fresh optimizer/scheduler for100 vessel-only epochs, batch4; no plant replay. All pretraining geometry and target-validation distances use pixels; REAVER fine-tuning/final evaluation restores its original micron calibration. Pretraining validation/selection uses target validation only, never pooled plant/vessel metrics.

| Dataset | Train / validation crops | Train maximum nodes | Units |
|---|---:|---:|---|
| avwide | 1385 / 506 | 76 | pixel |
| reaver | 256 / 128 | 16 | micron |
| plants | 25867 / 6398 | 28 | pixel |
| octasynth | 10648 / 4840 | 52 | pixel |

GPU gates passed: initialization/gradient/strict shared-weight transfer checks, fixed1,200-update full-model two-crop overfit per cohort and production-batch throughput. Gate weights are never reused for training.

| Dataset / benchmark | Batch | Measured images/s | Estimated optimizer hours/run |
|---|---:|---:|---:|
| avwide | 4 | 71.48 | 0.54 |
| reaver | 4 | 74.79 | 0.10 |
| octasynth | 4 | 77.84 | 3.80 |
| plants | 32 | 214.03 | 1.68 |
| mixed_avwide | 32 | 213.65 | 3.36 |
| mixed_reaver | 32 | 219.89 | 3.27 |
| mixed_octasynth | 32 | 206.83 | 3.47 |

| Run | Slurm | State | Epoch | Node mAP | Edge mAP | Node F1 @2 units | Edge F1 @2 units |
|---|---|---|---:|---:|---:|---:|---:|
| avwide_mixedpretrain_50 | 1673568 | complete | 50 | 0.2655 | 0.0695 | 0.1887 | 0.0103 |
| avwide_scratch_B0 | 1673549 | complete | 100 | 0.0788 | 0.0296 | 0.0000 | 0.0000 |
| avwide_scratch_Axyz | 1673551 | complete | 100 | 0.0757 | 0.0281 | 0.0000 | 0.0000 |
| avwide_mixed_B0 | 1673572 | complete | 100 | 0.2760 | 0.0774 | 0.2045 | 0.0123 |
| avwide_mixed_Axyz | 1673574 | complete | 100 | 0.2713 | 0.0691 | 0.1963 | 0.0088 |
| reaver_mixedpretrain_50 | 1673557 | complete | 50 | 0.5239 | 0.0959 | 0.3309 | 0.0859 |
| reaver_scratch_B0 | 1673553 | complete | 100 | 0.1135 | 0.0219 | 0.0000 | 0.0000 |
| reaver_scratch_Axyz | 1673555 | complete | 100 | 0.1181 | 0.0065 | 0.0000 | 0.0000 |
| reaver_mixed_B0 | 1673560 | complete | 100 | 0.4737 | 0.1190 | 0.7455 | 0.4085 |
| reaver_mixed_Axyz | 1673562 | complete | 100 | 0.1815 | 0.0075 | 0.2040 | 0.0000 |
| octasynth_mixedpretrain_50 | 1673564 | complete | 50 | 0.7379 | 0.5746 | 0.7262 | 0.4807 |
| octasynth_scratch_B0 | 1673576 | complete | 100 | 0.0548 | 0.0377 | 0.0000 | 0.0000 |
| octasynth_scratch_Axyz | 1673578 | complete | 100 | 0.1613 | 0.0580 | 0.0000 | 0.0000 |
| octasynth_mixed_B0 | 1673566 | complete | 100 | 0.8049 | 0.4447 | 0.8038 | 0.5950 |
| octasynth_mixed_Axyz | 1673580 | complete | 100 | 0.8513 | 0.5736 | 0.8101 | 0.5923 |

Primary comparisons use the final fixed-epoch checkpoints on complete validation populations; selected checkpoints are secondary. Selection uses fixed interim source-image groups, excludes initialization and resolves trained ties toward later epochs. Pixel and micron metrics are reported separately. Raw versus filtered recall, conditional relation populations/AP, duplicates, components/cycles and same-graph K8/chord geometry remain in per-run validation JSON. Mixed-pretraining validation reports target-vessel detection/graph metrics in pixels. Plant graphs supply sparse source supervision; plant validation is not pooled into target metrics.

No test arrays or predictions are used. AV-WIDE remains17 training /6 validation images with unknown patient independence; image07 excluded. REAVER is the separately verified diaphragm60x4/2-image cohort in microns. OCTA uses official source-image membership and regenerated128² crops with coordinate/crop-boundary QC. Plant parent IDs are disjoint across official train/val/test membership.

Mixed pretraining adds vessel exposures as well as biological data. Relative to scratch100, a gain cannot isolate a biological-pretraining causal effect; no extra vessel-only-pretraining control or sweep is launched. Realized per-domain draws and source/target memberships are recorded in protocol.json and exposure.npy.

Artifacts: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028`; validated datasets reused read-only from `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_long_20261028`. Source/config hashes: `source.zip`, `code-identities.json`; epoch losses and full state checkpoints: `runs/<name>/`; final metrics: `final-heldout.json`; plots: `curves-<dataset>.png` and separate pretraining curves. Stop at50 mixed /100 vessel epochs; exceptions stop jobs without automatic retries, new sweeps or budget extensions.
<!-- END NATIVE2D LONG LIVE 20261028 -->

<!-- NATIVE2D FINAL RESULTS 20261029 -->
## Native2D-04 completed — 29 October 2026

All15 runs completed. [Final results and validation overlays](../docs/research/native2d_results_20261029.md): scratch100 yields empty filtered graphs at0.5; mixed-pretrained OCTA B0/A-xyz edge F1@2px0.5950/0.5923, REAVER0.4085/0.0000 at2µm, AV-WIDE0.0123/0.0088 at2px. A-xyz improves tighter OCTA localization and Branch/Curve metrics but increases duplicates/false cycles; no cross-domain superiority or isolated botanical-effect claim. Tests untouched; budgets stopped.
<!-- END NATIVE2D FINAL RESULTS 20261029 -->

<!-- NATIVE2D STRAIGHT 20261029 -->
## Native2D-05 — original training with straight edges, 29 October 2026

Requested straight-edge control on OCTA-Synth B0: same full native2D detector, initialization seed364505, validated data/splits,50:50 mixed plant/vessel sampling,50 mixed epochs, then100 vessel-only epochs. Same node focal gamma2, matched-edge CE ratio0.15, Hungarian matching, optimizer, global gradient clipping, augmentation, query count96, thresholds0.5 and validation/checkpoint policy. Only the spline head and spline loss are disabled. Original completed curved experiments are preserved.

The existing training loop is reused unchanged through process-local configuration/output bindings. Exact source/target exposure hashes are checked against the completed curved B0 protocols. A fixed full-model tiny-overfit and production-batch finite-gradient smoke gates the long jobs. No extra sweep or automatic extension.

Jobs: `{"global_h100_limit": 2, "concurrent_h100": 1, "queue_at_submission": "", "prior_gpu_jobs": [], "gate": "1699959", "pretrain": "1699960", "finetune": "1699962", "collector": "1699963", "gate_completed": true}`.

| Stage | State | Epoch |
|---|---|---:|
| octasynth_mixedpretrain_50 | complete | 50 |
| octasynth_mixed_B0 | complete | 100 |

| Model, final100 | Node mAP | Edge mAP | Node F1 @2px | Edge F1 @2px | Curve F1, positive-crop macro |
|---|---:|---:|---:|---:|---:|
| Curved K8 B0 | 0.8049 | 0.4447 | 0.8038 | 0.5950 | 0.6285 |
| Straight-only B0 | 0.7992 | 0.5877 | 0.7932 | 0.5903 | 0.6036 |

The straight run retains the existing evaluator schema: its legacy `spline` metric slot also evaluates straight chords because no spline head exists; `chord` is the unambiguous straight geometry result. Dense GT paths and evaluation tolerances remain unchanged. The original global clipping rule is kept, including its interaction with the presence/absence of spline gradients.

Artifacts: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_straight_20261029`. Checkpoints/epoch logs and curves: `runs/`; exact sample exposures: `exposure.npy`; gate: `gates.json`; configuration previews: `*.yaml`; comparison parity: `parity-*.json`. Test sets untouched.

Curvature measurement is limited to the two OCTA validation crops shown previously, not the full validation population. Maximum perpendicular deviation from each endpoint chord is measured in pixels using257 spline samples.

| Model | Edges | Median / p95 bend, px | Fraction below0.25px | Median arc/chord |
|---|---:|---|---:|---:|
| B0 | 37 | 0.0006 / 1.7439 | 89.2% | 1.000000 |
| Axyz | 31 | 0.0006 / 0.5755 | 93.5% | 1.000000 |
| GT_dense | 33 | 0.0000 / 2.0036 | 90.9% | 1.000000 |
<!-- END NATIVE2D STRAIGHT 20261029 -->
