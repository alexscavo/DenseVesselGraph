# Literature and resource check for the journal paper (4 Nov 2026)

Scope: baselines, datasets, non-medical biological source data and prior pretraining work for a
MedIA-style journal paper on 3D vessel-graph extraction. The paper's model is dense
detect-and-trace with continuous B-spline edges, and it is pretrained on non-medical biological
branching structures.

**Method.** Web search plus primary sources: arXiv abstract/HTML/PDF text, PMLR/CVF/MICCAI pages,
GitHub READMEs, and Zenodo records read through the Zenodo API. Numbers are quoted as printed.
"Not found" means I looked and did not find it; it is not a claim that the thing does not exist.
Section C was compiled by a delegated web-survey agent from primary pages. Its rows marked
"(search)" come from search summaries, not opened pages.

**Shorthand used below.**
- **SMD**: Street Mover Distance, the Wasserstein distance between points sampled on predicted and
  GT edges.
- **mAP / mAR**: mean average precision / recall of node or edge "boxes" over IoU 0.5:0.95.
- **β₀ / β₁**: Betti numbers (connected components / independent loops).
- **TEASAR**: the skeletonisation algorithm used by Kimimaro.
- **SWC / RSML**: neuron and root trace file formats.
- **GT**: ground truth.

---

## Key facts at a glance

1. **One synthetic dataset dominates the 3D image-to-graph literature.** The DeepVesselNet /
   Tetteh et al. synthetic angiography set (136 volumes of 325×304×600; vascular trees from
   Schneider et al. 2012) is used by:
   - Relationformer ("Synthetic vessel");
   - Vesselformer;
   - Berger et al. ("Synthetic MRI", our local `syntheticMRI`);
   - Trexplorer;
   - Vesselpose ("Multi-Tree Synthetic").

   The only real 3D image-to-graph benchmarks in this line are:
   - VesSAP mouse-brain microscopy (Berger);
   - ATM'22 airways and PARSE 2022 pulmonary arteries, with Kimimaro-derived GT (Trexplorer
     Super, RefTr, Vesselpose).
2. **No image-to-graph baseline ships pretrained weights.** Vesselformer, Trexplorer, Trexplorer
   Super, RefTr and DeformCL have **no stated code license**. Relationformer is Apache-2.0;
   cross-dim_i2g (Berger), Vesselpose, clDice and DSCNet are MIT.
3. **Reproducibility is weak.** Vesselpose (MIDL 2026) states that its authors "were unable to
   reproduce [the baselines'] published results", so they copied numbers from Naeem et al. 2025.
4. **Strong limited-data evidence already exists.**
   - Relationformer's own ablation (Table 5): training on 25% of the synthetic-vessel data drops
     node mAP from 78.5 to 17.0.
   - Berger et al.: 2D→3D pretraining roughly doubles node mAP on Synthetic MRI (0.162 → 0.356).
     Pretraining on 2D OCTA beats pretraining on 2D roads for VesSAP (node mAP 0.548 vs 0.483).
5. **Closest prior work to the dense two-stage design: Mathys et al., ICLR 2025 SynthData
   workshop.** It uses a 3D U-Net node detector, a second network that predicts the full adjacency
   matrix, and pretraining on ~1M synthetic graph→mask→image samples. With 5 real labelled patches,
   edge F1 rises from 0.496 to 0.626. It must be cited and contrasted. No code was found.
6. **The user's own MICCAI 2026 paper** (Scavone et al., "Structural Congruence Matters", provisional
   accept) and MSc thesis (PoliTo, March 2026) are public. The reviews ask for three things that the
   journal extension must deliver:
   - medically relevant baselines;
   - more than one botanical source (only grapevines were used);
   - image-to-graph paradigms other than RelationFormer.

   See §D.
7. **TopBrain provides voxel labels only, not graphs.** It has 50 public training volumes
   (25 CTA/MRA pairs). An Aug 2026 Zenodo v3 adds 36-class "v2" labels. Graph-native public
   targets are:
   - CoW centerline graphs (Musio et al., 250 graphs, CC BY-NC 4.0);
   - BraVa SWC traces;
   - CASILab/TubeTK centerlines;
   - VesselGraph (whole mouse brain).

---

## A. Baselines

### A.0 Summary table

| # | Method (venue, year) | Task | Dim | Datasets | Code / license | Weights | Retraining difficulty |
|---|---|---|---|---|---|---|---|
| 1 | Relationformer (ECCV 2022) | Image→graph, set prediction | 2D+3D | 20 US Cities, Toulouse, synthetic vessel (Tetteh), Visual Genome | github.com/suprosanna/relationformer, Apache-2.0 | None | Deformable-attention CUDA op; 64³ patches; bounding-box node/edge formulation |
| 2 | Berger et al. cross-domain/cross-dim (WACV 2025) | Image→graph transfer (Relationformer-based) | 2D→3D | 20 US Cities → Agadez/Munich, synth. OCTA, OCTA-500, Synthetic MRI, VesSAP | github.com/AlexanderHBerger/cross-dim_i2g, MIT (2D/3D branches) | None found | As Relationformer, plus a domain-adversarial head |
| 3 | Vesselformer (MIDL 2023, PMLR 227) | Image→graph with radius, plus patch merging | 3D | Synthetic vessel (Tetteh) only | github.com/chinmay5/vesselformer, no license | None | CUDA ops; 150 epochs; RefTr reports its post-processing takes "hours" |
| 4 | Trexplorer (MICCAI 2024) | Recurrent DETR centerline-tree tracking | 3D | Synthetic vessel (Tetteh) only | github.com/RomStriker/Trexplorer, no license | None | Tree-only, needs roots; fixed bifurcation degree |
| 5 | Trexplorer Super (MICCAI 2025) | As 4, plus STT/FCA/TA | 3D | Single-tree synthetic (SVT), ATM'22, PARSE 2022 | github.com/RomStriker/Trexplorer-Super, no license; data Zenodo 15888958 (academic-only) | None | 4×A100, ~2M iterations, 5 runs per dataset; single-tree |
| 6a | RefTr (arXiv 2511.20823; MICCAI 2026 per Chalmers record) | Producer/refiner trajectory refinement | 3D | Same 3 as Trexplorer Super | github.com/RomStriker/RefTr, no license; README "cleaning up the code" | None | Same as 5 |
| 6b | Vesselpose (MIDL 2026, PMLR 315) | Seg + voxel direction vectors → guided TEASAR | 3D | Single-tree synthetic, PARSE 2022, multi-tree synthetic (Tetteh), private micro-CT rat heart | github.com/Kainmueller-Lab/Vesselpose, MIT | `trained_models_and_results/mt_syn` dir (contents unverified) | U-Net-based; easiest of the learned graph methods to retrain |
| 6c | DeformCL (CVPR 2025) | Deformable centerline graph from coarse seg | 3D | HaN-Seg, ImageCAS (+2 not named in README) | github.com/barry664/DeformCL, no license | None | PyTorch 1.10 / Detectron2 0.6 / PyTorch3D 0.7.3 stack |
| 6d | Mathys et al. (ICLR 2025 SynthData workshop) | Node U-Net + adjacency net, synthetic pretraining | 3D | Synthetic + 10 real TPLSM patches | Not found | — | Must be reimplemented |
| 6e | Musio et al. CoW centerline baseline (arXiv 2510.13720, 2025) | U-Net skeleton + A* graph | 3D | TopCoW 2024 | github.com/fmusio/CoW_Centerline_Extraction (license not stated) | n/a | CoW-specific |
| 7 | nnU-Net + skeleton→graph (Voreen / VesselVio / skan / Kimimaro), with clDice / Skeleton Recall / DSCNet | Seg + post-processing | 2D/3D | — | See A.7 | nnU-Net trains per dataset | Standard; the "segment-then-skeletonise" reference all papers compare to |

### A.1 RelationFormer

- **Citation:** Shit, Koner, Wittmann, Paetzold, Ezhov, Li, Pan, Sharifzadeh, Kaissis, Tresp, Menze.
  "Relationformer: A Unified Framework for Image-to-Graph Generation." ECCV 2022, LNCS 13697,
  pp. 422–439. arXiv 2203.10202.
- **Task:** one-stage set prediction of [obj] tokens (nodes as boxes) plus an [rln] token for
  pairwise relations. Built on a Deformable-DETR encoder/decoder. Covers 2D roads, 3D vessels and
  scene graphs.
- **Datasets (Table 1).** None of these is real 3D vessel data.

  | Dataset | Patch size | Train / val / test patches |
  |---|---|---|
  | Toulouse (2D binary) | 64×64 | 80k / 12k / 19k |
  | 20 US Cities (2D RGB) | 128×128 | 124k / 13k / 25k |
  | Synthetic vessel [Tetteh/DeepVesselNet] (3D grayscale) | 64×64×64 | 54k / 6k / 20k |
  | Visual Genome | — | — |

  The supplement describes the synthetic vessel set as "136 3D volumes of size 325x304x600", with
  40 used for train/val and 10 for test. Its GT graphs come from the volumes' centerlines, with
  degree-2 nodes pruned by angle.
- **Metrics:** SMD; TOPO score (precision/recall/F1); node and edge mAP/mAR.
- **3D result (Table 2):**

  | Method | SMD | node mAP | node mAR | edge mAP | edge mAR |
  |---|---|---|---|---|---|
  | Relationformer | 0.01107 | 78.51 | 84.34 | 78.10 | 82.15 |
  | U-Net + heuristics | 0.01982 | 18.94 | 29.81 | 17.88 | 27.63 |

- **Data-size ablation (Table 5, synthetic vessel):**

  | Training data | node mAP | node mAR | edge mAP | edge mAR |
  |---|---|---|---|---|
  | 25% | 17.0 | 32.1 | 11.5 | 28.3 |
  | 50% | 39.2 | 53.5 | 33.3 | 48.9 |
  | 100% | 78.5 | 84.3 | 78.1 | 82.1 |

  This is a ready-made limited-data argument for the journal paper.
- **Code:**
  - github.com/suprosanna/relationformer, Apache-2.0.
  - Branches `vessel_graph`, `road_network`, `road_network_rgb`, `scene_graph`.
  - The 3D README needs CUDA ≥ 9.2 and PyTorch ≥ 1.7.1, and compiles `models/ops`
    (MultiScaleDeformableAttention).
  - No checkpoints are linked.
- **Retraining issues:**
  - The compiled CUDA op.
  - Node "boxes" with a fixed radius.
  - Patch-only graphs (no whole-volume merging).
  - Degree-2 pruning conventions.
  - Fixed token budget (Berger 3D: 120 [obj] tokens).

  We already retrain it in-house (`dense_traced_model_20261102.md` §3).

### A.2 Cross-domain and cross-dimension learning for image-to-graph transformers

- **Citation:** Berger, Lux, Shit, Ezhov, Kaissis, Menten, Rueckert, Paetzold. WACV 2025,
  pp. 64–74. arXiv 2403.06601 (v1 Mar 2024, v2 Dec 2024).
- **Contributions:**
  1. A regularised edge-sampling loss.
  2. Supervised domain-adversarial alignment of image- and graph-level features.
  3. A 2D→3D projection: downsample the 2D image, place it as the z = 0.5 slice of an empty
     volume, set node z = 0.5, then apply the same random 3D rotation to the volume and the graph.
- **Datasets (Table 13):**

  | Dataset | Patch size | Train / val / test |
  |---|---|---|
  | 20 US Cities | 128² | 99.2k / 24.8k / 25k |
  | Agadez | 128² | 480 / 120 / 290 |
  | Munich | 128² | 440 / 110 / 220 |
  | Synthetic OCTA (Menten) | 128² | 480 / 120 / 2k |
  | OCTA-500 | 128² | 1.6k / 400 / 2.2k |
  | Synthetic MRI [Schneider; Tetteh] | 64³ | 4k / 1k / 5k |
  | Microscopy / VesSAP [Todorov] | 50³ | 4k / 1k / 1.2k |

  - Synthetic MRI GT graphs: "obtained with the method described by Drees et al." (Voreen).
  - OCTA-500 graphs: Drees et al. applied to sparse manual vessel maps.
  - VesSAP: the image + segmentation + graph of the VesselGraph / ogbl-vessel benchmark.
- **Main results (Table 1), node mAP / node mAR / edge mAP / edge mAR / SMD:**

  | Target | No pretraining | Self-supervised (MoCo v3) | Ours (2D roads) |
  |---|---|---|---|
  | Synthetic MRI (3D) | 0.162 / 0.250 / 0.125 / 0.201 / 0.013 | 0.162 / 0.252 / 0.120 / 0.193 / 0.014 | 0.356 / 0.450 / 0.221 / 0.322 / 0.013 |
  | VesSAP (3D) | 0.231 / 0.308 / 0.249 / 0.329 / 0.017 | 0.344 / 0.404 / 0.363 / 0.425 / 0.017 | 0.483 / 0.535 / 0.523 / 0.566 / 0.017 |

  - Supplement: VesSAP pretrained **on OCTA** instead of cities gives 0.548 / 0.583 / 0.588 /
    0.615 / 0.016. A closer domain beats a bigger one.
  - "Our framework is especially useful if target data is scarce." The gap narrows as target
    size grows (Sec. 6.5).
- **Training:** 100 epochs, batch 32, for each fine-tune.
  - Hyperparameters were tuned in three stages (architecture, then pretraining, then
    fine-tuning). The paper does not say whether this tuning used validation or test data.
  - Results are reported as mean ± std over five folds of the test set.
- **Code:**
  - github.com/AlexanderHBerger/cross-dim_i2g, MIT, with separate `2D` and `3D` branches.
  - "Largely based on the relationformer repository."
  - The appendix still says "will be made publicly available upon acceptance".
  - Pretrained weights: not found.
- **Notes:**
  - 3D SMD is identical across arms (0.017), so SMD does not discriminate here. Our BRAVA SMD
    caveat is the same.
  - Supervised pretraining in 3D is reported as "technically not possible" without the
    projection.

### A.3 Vesselformer

- **Citation:** Prabhakar, Shit, Paetzold, Ezhov, Koner, Li, Kofler, Menze. "Vesselformer: Towards
  Complete 3D Vessel Graph Generation from Images." MIDL 2023, PMLR 227:320–331 (2024 volume).
- **Task:** Relationformer plus a [rad] token for edge radius, plus "Vesselmerger", which merges
  non-overlapping patch graphs with a checkerboard parallel scheme.
- **Data:** "a publicly available dataset (Tetteh et al., 2020)", the synthetic set with GT radius.
  No real data.
- **Training:**
  - 150 epochs, batch 48, AdamW at lr 1e-4.
  - 3D SE-ResNet backbone; 80 [obj] tokens.
  - The README example uses 3 GPUs.
- **Metrics:** SMD, β₀/β₁ relative error (%), node/edge mAP/mAR, radius MAE.
- **Results (Table 1):**

  | Method | SMD | β₀ err | β₁ err | node mAP | node mAR | edge mAP | edge mAR | radius MAE |
  |---|---|---|---|---|---|---|---|---|
  | Patch level: Vesselformer | 0.01260 | 0.1018 | 0.0996 | 76.56 | 82.83 | 76.04 | 80.65 | 0.51 |
  | Volume level: Vesselformer + Vesselmerger | 0.01381 | 0.2188 | 0.2054 | 72.32 | 80.11 | 72.19 | 76.24 | 0.52 |
  | Volume level: Voreen | 0.03071 | 0.2955 | 0.2766 | 36.17 | 43.35 | — | — | 1.79 |

- **Code:**
  - github.com/chinmay5/vesselformer: train/eval scripts, CUDA ops, PyTorch ≥ 2.0.
  - **No license stated**; no weights.
- **External view of retraining:**
  - Trexplorer Super trained Vesselformer for ~12M iterations versus ~2M for its own model.
  - RefTr notes Vesselformer's patch runtime is 23 ms, but its post-processing "takes hours".

### A.4 Trexplorer

- **Citation:** Naeem, Hagerman, Svensson, Kahl. "Trexplorer: Recurrent DETR for Topologically
  Correct Tree Centerline Tracking." MICCAI 2024, LNCS 15011, pp. 744–754.
  doi:10.1007/978-3-031-72120-5_69.
- **Task:** recurrent DETR that tracks a centerline tree breadth-first from a root. Outputs a 3D
  centerline graph (tree) with radius.
- **Data:** the synthetic vessel centerline set from the DeepVesselNet wiki only. The test set has
  "10 volumes, 166 vessel trees, 654,387 nodes, and 654,221 edges" (rebuttal).
  - The authors call it "the only publicly available vessel centerline dataset".
  - Reviewers asked for real data. The authors tried IRCADb-01 but withheld the results.
- **Baseline:** Vesselformer only.
- **Code:** github.com/RomStriker/Trexplorer (PyTorch 2.2 / CUDA 11.8; builds on DETR and
  Trackformer). **No license stated**; no weights.
- **Limitations (authors and reviewers):**
  - Duplicate branches and premature termination.
  - Fixed bifurcation degree (2).
  - No variance reported.

### A.5 Trexplorer Super

- **Citation:** Naeem, Hagerman, Alvén, Svensson, Kahl. "Trexplorer Super: Topologically Correct
  Centerline Tree Tracking of Tubular Objects in CT Volumes." MICCAI 2025, LNCS 15967,
  pp. 595–605. arXiv 2507.10881.
- **What it adds:**
  - **Super Trajectory Training:** chained sub-trajectories, with query outputs carried across
    them.
  - **Focal Cross Attention:** high-resolution features over a large region, but cross-attention
    restricted to a small focal region.
  - **Target Augmentation:** Laplace-shifted bifurcation targets.
  - **Three new centerline datasets:**

    | Dataset | Train / val / test | Source |
    |---|---|---|
    | Single-tree synthetic | 368 / 32 / 100 | Synthetic Vascular Toolkit |
    | ATM'22 airways | 220 / 16 / 60 | CC BY-NC |
    | PARSE 2022 pulmonary arteries | 72 / 8 / 20 | Paper says "CC BY-NC-ND 6.0" (sic) |

    Real-data GT: resampled to 0.5 mm, roots found with VMTK, centerlines with **Kimimaro**
    (automatic, not manual).
- **Metrics:**
  - Point precision/recall/F1 at a 1.5-voxel radius, plus radius MAE.
  - Branch F1 (a branch matches if ≥ 80% of its points match).
  - β₀/β₁ MAE.
- **Results (point F1 / branch F1):**

  | Dataset | Trexplorer Super | Other baselines (point F1) |
  |---|---|---|
  | Synthetic | 77.83 ± 1.89 / 77.12 ± 1.59 | Vesselformer 48.18 ± 5.62 |
  | ATM'22 | 60.45 ± 1.03 / 41.15 ± 1.28 | Trexplorer 3.34 ± 0.30 |
  | PARSE | 39.46 ± 1.93 / 23.46 ± 1.09 | Vesselformer 16.43 ± 0.78 |

  - β₀/β₁ MAE is 0.00 by construction (output is always a tree).
- **Compute:** 4×A100, mixed precision, ~2M iterations, five runs per dataset.
- **Code and data:**
  - Code: github.com/RomStriker/Trexplorer-Super, **no license stated**.
  - Data: Zenodo 15888958, 7.6 GB, custom "Academic Research Use Only" terms; redistribution
    needs written permission.
  - For ATM/PARSE only the graphs are provided; images must come from the challenges.
- **Retraining issues:**
  - Tree-only and single-root; cannot represent loops (e.g. the Circle of Willis).
  - Large compute budget.
  - Authors report "complete fails" on some test samples.
  - Vesselpose notes training and evaluation are "restricted to single-tree structures".

### A.6 Other recent direct graph / centerline methods (2023–2026)

1. **RefTr.** Naeem, Hagerman, Alvén, Kahl. arXiv 2511.20823 (v2 Mar 2026). A Chalmers record lists
   it in MICCAI 2026 proceedings (venue not verified on a MICCAI page).
   - **Method:** a Producer proposes "confluent trajectories" and a shared Refiner refines them;
     Tree NMS merges duplicate branches.
   - **Results (point rF1, radius-aware, averaged over thresholds):**

     | Dataset | RefTr | Trexplorer Super |
     |---|---|---|
     | Synthetic | 84.64 | 77.98 |
     | ATM'22 | 66.78 | 61.45 |
     | PARSE | 42.15 | 40.01 |

     Branch rBF1 is **lower** than Trexplorer Super, e.g. PARSE 31.48 vs 46.57.
   - **Compute:** 4×A100, 1.92M iterations. 2.4× fewer decoder parameters.
   - **Code:** github.com/RomStriker/RefTr, no license, "cleaning up the code".
2. **Vesselpose.** Palaniappan, Karg, Navarro-Arambula, Hirsch, Kraeker, Mais, Kainmueller. MIDL 2026,
   PMLR 315. arXiv 2605.00538.
   - **Method:** a 3D U-Net predicts a mask plus voxel-wise rootward direction vectors, then a
     direction-guided TEASAR builds the graph. Also proposes false-split / false-merge metrics.
   - **Data:** single-tree synthetic and PARSE (from Naeem et al.), multi-tree synthetic (Tetteh),
     and a private rat-heart micro-CT set (3 crops of 400³).
   - **Results (point F1 / branch F1):**

     | Dataset | Vesselpose | Trexplorer Super |
     |---|---|---|
     | Single-tree synthetic | 92.25 / 81.50 | 77.83 / 77.12 |
     | PARSE | 57.52 / 35.33 | 39.46 / 23.46 |

     PARSE β₀ error is 1.85. On multi-tree synthetic, the baselines are U-Net + TEASAR and
     vesselFM + TEASAR.
   - **Code:** github.com/Kainmueller-Lab/Vesselpose, MIT.
   - **Relevance:** the closest "dense map + tracing" competitor to our traced model, and the
     easiest learned baseline to retrain.
3. **Mathys, Plesner, Elmiger, Wattenhofer.** "Synthetic Data for Blood Vessel Network Extraction."
   ICLR 2025 SynthData workshop. arXiv 2504.11858.
   - **Method:** a 3D U-Net regresses ≤ 32 node coordinates with confidences (Hungarian
     matching). A second U-Net-like net samples features at the nodes and outputs a full
     adjacency matrix.
   - **Pretraining data:** synthetic graph→Bézier-tube mask→microscopy-like image; "Varied" set of
     1024 configurations × 1024 samples.
   - **Real data:** 10 TPLSM (two-photon laser-scanning microscopy) patches, 5 fine-tune / 5 test.
     - Edge F1 0.496 → 0.626 after fine-tuning.
     - Training on the 5 real scans only gives node F1 0.067.
   - **Code:** not found.
   - **Relevance:** conceptually the nearest prior to dense two-stage detection plus synthetic
     pretraining; must be discussed.
4. **DeformCL.** Zhao, Zhang, Liu, Zhang, Yu, Wang, Wang. CVPR 2025. arXiv 2506.05820.
   - **Method:** deformable centerline graph (nodes = centerline points) refined from a coarse
     segmentation in a cascade.
   - **Data:** four 3D datasets; the README names HaN-Seg and ImageCAS (LAD/LCX/RCA masks
     released).
   - **Code:** github.com/barry664/DeformCL, no license, no weights.
5. **GraphMorph.** Zhang, Zhao, Wang, Wang. NeurIPS 2024. arXiv 2502.11731.
   - **Method:** a graph decoder plus SkeletonDijkstra morphing to a centerline mask.
   - **Code:** "will be released soon"; not found as of today.
6. **CoW centerline graphs + baseline.** Musio, Juchler, Yang, Shit, Prabhakar, Menze, Hirsch.
   arXiv 2510.13720 (2025).
   - **Method:** U-Net skeletonisation + A* connection.
   - **Reported result:** "F1 = 1" for graph topology on the CoW. The data is in §B.
7. **Not an image-to-graph baseline:** "3D Vessel Graph Generation Using Denoising Diffusion"
   (Prabhakar et al., MICCAI 2024, arXiv 2407.05842; code github.com/chinmay5/vessel_diffuse) is
   **unconditional** graph generation with no image input.
   - Its VesSAP (24 volumes, Voreen graphs) and CoW (390 MRA graphs: 300 CROWN + 90 TopCoW)
     graph sets are useful priors.
   - It could be a synthetic-graph source for pretraining.

### A.7 Classic pipeline baselines

| Tool | Citation | What it does | License | Use in vessel-graph papers |
|---|---|---|---|---|
| **nnU-Net** | Isensee et al., *Nature Methods* 18:203–211 (2021) [prior knowledge, not re-fetched] | Self-configuring segmentation | Apache-2.0 [prior knowledge] | The TopBrain top teams built on nnU-Net (~80% Dice, medRxiv 2026). Vesselpose appendix C.9 uses it |
| **Voreen vessel graph** | Drees, Scherzinger, Hägerling, Kiefer, Jiang. *BMC Bioinformatics* 22:346 (2021), doi:10.1186/s12859-021-04262-w | Segmentation → annotated graph (any topology). One dimensionless refinement parameter; ~1 TB volumes on commodity hardware | In Voreen 5.1 (license not checked) | **De facto GT generator:** Berger (Synthetic MRI, OCTA-500), VesselGraph, the diffusion paper, and our IXI `graph.vvg` format. Vesselformer's "Voreen" row is the classic baseline |
| **VesselVio** | Bumgarner & Nelson. *Cell Reports Methods* 2:100189 (2022) | GUI/py analysis of binarised vasculature → graphs + features | GPLv3 | Analysis tool, rarely a baseline |
| **skan** | Nunez-Iglesias, Blanch, Looker, Dixon, Tilley. *PeerJ* 6:e4312 (2018) | nD skeleton image → graph / branch statistics | BSD-3 | Lightweight skeleton→graph |
| **Kimimaro** (TEASAR) | seung-lab; TEASAR: Sato et al., PCCGA 2000 | Labelled volume → skeleton (SWC) with radius | GPL-3.0 | GT generator for ATM'22/PARSE graphs (Trexplorer Super, RefTr, Vesselpose). Vesselpose's baselines are U-Net/vesselFM + TEASAR |
| **clDice** | Shit, Paetzold et al. CVPR 2021, doi:10.1109/CVPR46437.2021.01629 | Soft-skeleton topology loss + clDice metric, 2D/3D | MIT | Standard topology-aware segmentation baseline |
| **Skeleton Recall** | Kirchhoff, Rokuss, Roy et al. ECCV 2024, arXiv 2404.03010 | Tubed-skeleton recall loss, computed on CPU; multi-class | Apache-2.0; nnU-Net trainer `nnUNetTrainerSkeletonRecall` | Evaluated on TopCoW; cheapest nnU-Net-integrated topology baseline |
| **DSCNet** | Qi et al. ICCV 2023, arXiv 2307.08388 | Dynamic snake convolution | MIT (2D + 3D code) | 3D evaluation uses private CCTA + KIPA22 |

**Recommended "segment-then-skeletonise" baseline set:**
- nnU-Net (Dice + CE);
- nnU-Net + Skeleton Recall;
- optionally clDice;
- each followed by **one** skeleton→graph tool. Kimimaro matches the Trexplorer/Vesselpose GT
  pipeline; Voreen matches the Berger/IXI GT pipeline.

Add vesselFM zero/few-shot + TEASAR, as in Vesselpose. **Caution:** when GT graphs were made by
Voreen or Kimimaro, the same tool applied to a predicted mask is favoured by convention matching.
Say so in the paper.

---

## B. Datasets (3D targets)

### B.0 Summary table

| Dataset | Modality / species | Size | Labels | Graphs provided? | License / access |
|---|---|---|---|---|---|
| **TopBrain 2025** | CTA + MRA, human (TopCoW patients, USZ) | 50 public training volumes (25 pairs); 90 annotated in total (medRxiv); test = 20 hidden TopCoW patients | v1: CTA 40 / MRA 42 classes (34 shared); v2 (Aug 2026): 36 modality-agnostic classes | **No** (voxel masks only) | opendata.swiss "Open use. Must provide the source. Use for commercial purposes requires permission"; Zenodo, open |
| TopCoW 2024 | CTA + MRA | 125 pairs (250 images) + 4 external test sets (incl. MRA_IXI_HH, MRA_Lausanne) | CoW multiclass | No | Zenodo 15692630, open (2023 release was CC BY-NC; 2024 terms in License.txt, not checked) |
| **CoW Centerline Graphs** (Musio 2025) | Derived from TopCoW 2024 | 250 graphs (paper also says 400 from 200 patients) | Labelled centerline graphs, meshes, morphometrics (VTK) | **Yes** (CoW only) | Zenodo 17358162, **CC BY-NC 4.0** |
| **BraVa** (Wright et al., *NeuroImage* 82:170–181, 2013) | 3T TOF-MRA, 0.62 mm iso., healthy | 61 subjects (traces); our local copy: 56 image subjects | Manual arterial-tree reconstructions (ACA/MCA/PCA) | **Yes** (SWC traces) | cng.gmu.edu/brava; images via ICBM (NITRC "icbmmra" / LONI IDA); terms not verified |
| IXI MRA | TOF-MRA, healthy | ~570 MRA (raw IXI) | No official vessel labels. Third-party: Bizjak et al. SPIE 2022 (570 reference segs); VesselVerse (45 IXI, MICCAI 2025) | No | IXI CC BY-SA 3.0 [prior knowledge]. **Our IXI mask provenance is recorded as "Unknown"** (`docs/IXI_DATASET.md` §8); must be resolved before publication |
| CASILab / TubeTK | MRA + T1, healthy, 3T | 100 pairs; centerlines for a subset (Musio et al.) | Vessel centerlines with radii | **Yes** (subset) | data.kitware.com; free for medical research; non-medical use needs Bullitt's approval |
| DeepVesselNet synthetic ("Synthetic MRI") | Synthetic angiography (Schneider 2012 trees) | 136 volumes, 325×304×600 | Seg, centerline, bifurcations, radius | Centerlines; graphs via Voreen (Berger) or the repos' scripts | github.com/giesekow/deepvesselnet wiki; license not found |
| Single-tree synthetic (Naeem 2025) | Synthetic (SVT) | 368/32/100 | Centerline trees | Yes | Zenodo 15888958, academic-only |
| ATM'22 | Chest CT, airways | 500 (300/50/150); 296 used by Trexplorer Super | Airway masks | Kimimaro graphs via Zenodo 15888958 | Registration; Trexplorer Super says CC BY-NC |
| PARSE 2022 | CTPA, pulmonary arteries | 200 (100 train / 30 val / 70 test) | Artery masks | Kimimaro graphs via Zenodo 15888958 | "CC BY-NC-ND 6.0" (sic); free download since 2025-09-10 per site |
| VesSAP / VesselGraph | Light-sheet, cleared whole mouse brain | VesselGraph: 12 whole-brain graphs (e.g. BALBc1: 3,538,495 nodes, 5,345,897 edges) | CNN segmentation → Voreen graphs | **Yes** | github.com/jocpae/VesselGraph; OGB `ogbl-vessel`. NeurIPS 2021 Datasets & Benchmarks [venue from prior knowledge] |
| ImageCAS | Coronary CTA | 1,000 patients | Coronary masks | No (DeformCL releases LAD/LCX/RCA masks) | GitHub XiaoweiXu/ImageCAS |
| OCTA-500 (2D) | Retinal OCTA | 300 scans (6×6 mm), 400×400 en-face | Sparse vessel masks | Graphs via Drees (Berger) | Not checked |
| vesselFM D_real | 17 datasets (README lists 16, e.g. SMILE-UHURA, BvEM, MSD8, TubeTK, tUbeNet, TopCoW, VesSAP, DeepVesselNet, HR-Kidney, 3D-IRCADb-01, DeepVess, CSD, VesselExpress, MiniVess, HiP-CT) | — | Masks | No | Per-dataset licenses; repo does not redistribute |

### B.1 TopBrain details

- **Name:** "TopBrain 🔝🧠 Segmentation Challenge for Whole Brain Vessel Anatomy" (TopBrain 2025),
  the successor to TopCoW. MICCAI 2025 challenge event on 23 Sep 2025, with the SWITCH workshop.
- **Organisers:** UZH (Kaiyuan Yang, Bjoern Menze et al.).
- **Summary paper:** Yang K, Shi P, Huang H, Musio F, et al. "TopBrain segmentation challenge for
  whole brain vessel anatomy." medRxiv 2026.05.28.26354312 (posted 30 May 2026).
  - "90 annotated volumes", "48 landmark vessel classes".
  - 15 teams submitted, out of > 220 registered.
  - Top nnU-Net teams reached ~80% Dice.
- **Zenodo versions** (Zenodo API, 4 Nov 2026):

  | Record | Date | Content |
  |---|---|---|
  | 16623496 | 2025-07-30 | Batch 1 |
  | 16878417 | 2025-08-14 | Batches 1+2; **our local copy** `TopBrain_Data_Release_Batches1n2_081425` |
  | **21972006** | 2026-08-17 | Batches 1+2 + `labelsTr_topbrain_v2_topaneu36class`: 36 classes, adds infraclinoid ICA C1–C5 vs C6–C7, modality-agnostic, made for the TopAneu 2026 challenge |

  - Podium Docker images: Zenodo 21959166.
  - Required citations: the TopBrain medRxiv preprint and the TopCoW NEJM AI paper (Yang et al.,
    *NEJM AI* 3(8):AIdbp2500994, 2026).
- **Format:** NIfTI, 16-bit signed, LPS+, file pattern `topcow_{ct|mr}_{id}_0000.nii.gz`. Voxel
  size not stated on the page.
- **Graphs:** none. Challenge metrics are topology-oriented (centerline, connected components,
  invalid neighbours, contamination, caliber along centerlines). Graph GT must be derived by us, as
  we already do: `docs/IXI_DATASET.md` lists 50 TopBrain volumes inside the IXI target set.
- **Caution:** the 25 TopBrain patients are TopCoW patients. If TopCoW or CoW-graph data are used
  as extra training data, check that test subjects do not overlap.

---

## C. Non-medical biological source datasets (images + branching annotation)

The delegated survey opened most rows from primary pages; "(search)" marks rows that come from
search summaries only. Neurons and roots are **trees**. Leaf veins, *Physarum* and fungal mycelia
contain **loops**. No dataset below was confirmed to fill a per-node radius field. SWC and RSML can
carry radius, so check before relying on it.

### C.0 Summary table

| Dataset | Structure | Modality, dim | Size | Annotation | License / access |
|---|---|---|---|---|---|
| **BigNeuron Gold166** (Manubens-Gil et al., *Nat Methods* 20:824–835, 2023) | Neurons, multi-species | Light microscopy, 3D | 166 stacks | Manual gold SWC (6–7 annotators, 2015 workshop); also 7,978 algorithm reconstructions | github.com/BigNeuron/Data releases (A*STAR + Zenodo mirrors); license not found |
| **Janelia MouseLight** (Winnubst et al., *Cell* 179:268, 2019) | Mouse neurons (axons) | 2-photon serial tomography, 3D, 0.3×0.3×1.0 µm | > 1,000 neurons | Semi-automatic, two annotators per neuron; SWC/JSON | `s3://janelia-mouselight-imagery` (anonymous), **CC BY 4.0** |
| SEU-ALLEN / Peng et al. (*Nature* 598:174, 2021) | Mouse neurons | fMOST, 3D | 1,741 neurons, 53 brains (search) | SWC | Brain Image Library (doi:10.35077/g.73); image route unconfirmed |
| HiNeuron blocks (Zenodo 10.5281/zenodo.20501092, Jun 2026) | Neurons | Optical, 3D blocks | "8,092,547 difficulty-graded data blocks" (28.1 GB subset on Zenodo) | Image + SWC + seeds | **CC BY 4.0** |
| fMOST axon set (*Front Neuroinform* 2025) | Mouse axons | fMOST, 192³ blocks | 852 (676/85/91), 11 brains | SWC | Zenodo 15372438; license not found |
| DIADEM (Brown et al., *Neuroinformatics* 9:143, 2011) | Neurons, 6 collections | Confocal/2-photon, 3D | Small (e.g. 9 stacks) | Gold SWC | diadem.janelia.org; license not found |
| NeuroMorpho.Org | Neurons | **Traces only** (FAQ excludes voxel data) | 300,495 cells (v8.7) | SWC | Graph-only prior (render synthetic volumes) |
| **RootNav 2.0** (Yasrab et al., *GigaScience* 8:giz123, 2019) | Wheat roots | 2D photos | 3,630 images (+277 *Arabidopsis*, +120 rapeseed) | **RSML traces** with hierarchy | GigaDB doi:10.5524/100651; license not verified |
| PRMI (Xu et al., 2022) | Roots in soil | 2D minirhizotron | > 72K images, 63K masked | Masks | Dryad, CC0 (per Dataset Ninja) |
| ChronoRoot 2.0 | *Arabidopsis*, tomato | 2D IR | 1,393 images | 6-class masks | HF ngaggion/ChronoRoot2, **CC0** |
| Lobet simulated roots (Zenodo 61739) | Simulated roots | Images (2D/3D not stated) | 10,000 | RSML GT | **CC0** |
| TopoRoot (Zeng et al., *Plant Methods* 17:127, 2021) | Maize root crowns | **3D X-ray CT**, 109 µm | 45 scans | No skeletons released | github.com/danzeng8/TopoRoot, CC0 |
| Selzner et al. (*Plant Phenomics* 5:0076, 2023) | Lupine roots | **3D MRI** | 8 scans | Manual VR → **RSML** | **On request only** |
| **3D2cut Single Guyot** (ViNet, *Comput Electron Agric* 2023) | Grapevine | 2D photos, 4032×3024 | 1,511 (1,254/257) | Branch graphs (nodes, terminations), JSON | Zenodo 7679898, **CC BY-NC-SA 4.0**. **This is the MICCAI-paper source** (local `plants_3d2cut`) |
| **HALVS** (Liu W. et al., IJCAI 2024, arXiv 2405.10041) | Leaf veins (soybean, sweet cherry, London planetree) | 2D transmission scans | 5,057 leaves | Primary/secondary/tertiary vein masks | Signed agreement (email). Local `HALVS_DATASET` |
| Blonder et al. (*Ecology* 100:e02844, 2019) | Leaf veins, 295 species | 2D, ~50 MP | 726 leaves | Hand-traced vein masks (ROI) | ORA Oxford, ODC-By |
| LeafVeinCNN (Xu et al., *New Phytol*) | Same images as Blonder | 2D | — | GT + CNN predictions | Zenodo 4008614 |
| LVD2021 | Leaf veins, 36 types | 2D | 4,977 | Masks | Request form |
| Fungal Network Analysis (Fricker et al., *ISME Commun* 2:2, 2022) | Mycelium | 2D | n.f. | Software-extracted networks | Zenodo 5725751, CC BY 4.0 |
| Oyarte Gálvez et al. (*Nature* 639, 2025) | Arbuscular mycorrhizal networks | 2D robot imaging | > 500,000 tracked nodes | Network tracking | Data statement not retrieved |
| Slime Mold Graph Repository (Dirnberger, Mehlhorn et al., 2016) | *Physarum* | 2D image series | 54 usable series (search) | NEFI graphs (automatic), planar, loopy | Edmond doi:10.17617/3.XWST2Q, open |
| MyceliumSeg (*Sci Data* 2025) | Fungal colonies | 2D macro | 20,176 images, 567 masked | Colony masks | Zenodo 15224240; not hyphal graphs |

### C.1 Takeaways for source selection

- **Best 3D sources.** Neurons:
  - MouseLight: CC BY 4.0 on anonymous S3, terabyte-scale.
  - HiNeuron blocks: ready-cut image+SWC, CC BY 4.0.
  - BigNeuron Gold166: the most diverse, but small.

  All three are tree-only and much thinner (relative to voxel size) than MRA arteries.
- **Roots in 3D:** no public 3D CT/MRI root dataset with traced architecture was found. TopoRoot
  (CC0 CT) needs our own skeletonisation; Selzner (MRI + RSML) is on request.
- **Best 2D sources to lift to 3D:**
  - HALVS (loops, three-level vein hierarchy);
  - Blonder (open, loopy);
  - RootNav 2.0 (true RSML graphs);
  - 3D2cut (already used);
  - *Physarum* (loopy, automatic graphs).

  Loopy sources matter because CoW and capillary beds contain cycles. Trexplorer-family models
  cannot represent them.
- **Reviewer-driven need:** the MICCAI 2026 reviews criticised "only one botanical data type
  (grapevines)". The journal version should add at least one loopy 2D source (HALVS or leaves)
  and one native-3D source (neurons). A non-biological control (20 US Cities, following Berger)
  separates "biological structure" from "any graph pretraining".

---

## D. Prior work on non-medical / cross-dimension pretraining

### D.0 Summary table

| Work | Source domain → target | Task | Evidence |
|---|---|---|---|
| **Berger et al., WACV 2025** | 2D roads (20 US Cities) or 2D OCTA → 3D Synthetic MRI / VesSAP (via the projection Π) | Image→graph | Node mAP 0.162→0.356 (Synth. MRI); 0.231→0.483 (VesSAP); 0.548 with OCTA as source |
| **Scavone et al., MICCAI 2026** (user's own; provisional accept) | 2D grapevine skeleton graphs (3D2cut) → 2D/3D vascular benchmarks | Image→graph (RelationFormer/Berger framework) | "Botanical pretraining outperforms infrastructure-based (road network) pretraining"; code github.com/erc-caravel/Vascular-Graph-Extraction (MIT) |
| Scavone, MSc thesis, PoliTo (March 2026), "Topology-Aware and Multi-Target Transfer Learning for Vascular Graph Extraction" | Botanical → vessels, multi-target, zero-shot | Image→graph | Public on webthesis.biblio.polito.it/39935; cite as prior disclosure |
| Mathys et al., ICLR-W 2025 | Procedural synthetic graphs → TPLSM microscopy | Two-stage node + adjacency | Edge F1 0.496→0.626 with 5 real patches |
| **DeepBranchAI** (Maltsev, Hartnell, Ferrucci; *Front. AI* 9, Sep 2026) | FIB-SEM mitochondrial networks (human muscle) → 3D-IRCADb veins, plant-root CT, AeroPath airways | nnU-Net segmentation | Relative clDice gain +11.7% / +5.5% / +1.9%. On IRCADb, clDice 0.629 vs vesselFM 0.360. Code + weights on GitHub. **Closest published "non-vascular branching → vessels" pretraining** |
| vesselFM (Wittmann et al., CVPR 2025) | 17 real vessel datasets + domain-randomised synthetic + flow-matching samples | 3D vessel segmentation foundation model | Zero/one/few-shot across 4 modalities. Code GPLv3; weights OpenRAIL++-M (non-commercial) on HF |
| VesselSim (Rainville et al., arXiv 2605.26277; "MICCAI 2026 proceedings" per authors) | ~16,500 simulated 3D angiograms → real MR/CT | Segmentation, zero-shot + test-time adaptation | Competitive with foundation models (abstract); code not found |
| VessShape (Comin & Galvão, arXiv 2510.27646) | 2D procedural tubes + random textures → 2D vessels | Few-shot segmentation | Fine-tuning needs 4–10 samples (abstract) |
| FreeCOS (Shi et al., ICCV 2023) | Fractal L-system curves blended into unlabelled images | 2D curvilinear segmentation, self-supervised | Gains on XCAD, DRIVE, STARE, CrackTree |
| DeepVesselNet (Tetteh et al., *Front. Neurosci.* 14, 2020) | Synthetic vessels → TOF-MRA | Segmentation | Dice 86.39% (scratch) vs 86.68% (synthetic pretrain): small gain (search summary) |
| Stolt Ansó (arXiv 2001.00666, 2020) | Synthetic vessels blended into head CT | CTA segmentation | Improvement claimed; no numbers in abstract |
| DINeuro (Cheng et al., ISBI 2025) | 2D natural-image ViT → 3D neuron ViT | Neuron segmentation (BigNeuron Janelia) | +4.53% mean Dice. A 2D→3D weight-transfer analogue |
| Root-segmentation benchmark (PMC13227879, 2026) | Generic pretraining → nine root datasets | Root segmentation | Mean Dice 0.623 → 0.666 with pretraining (search summary) |
| General 2D→3D lifting | I3D filter inflation (Carreira & Zisserman, CVPR 2017); ACS convolutions (Yang et al., IEEE JBHI 2021) [prior knowledge, not re-fetched] | Weight lifting | Alternatives or complements to Berger's data-level projection Π |

### D.1 What is and is not covered

- **Covered:**
  - Roads → vessels for image-to-graph, including 2D→3D (Berger).
  - Grapevines → vessels for image-to-graph (the user's MICCAI paper).
  - Procedural synthetic graphs → vessels (Mathys; Vesselformer and Trexplorer train on
    synthetic only).
  - Non-vascular biological branching → vessels for **segmentation** (DeepBranchAI).
- **Not found:**
  - Image-to-graph or centerline-tracking pretraining on **neurons** (BigNeuron, MouseLight) or
    **roots/leaves** for vessels, other than the user's own work.
  - Any study comparing **several** biological sources against roads and against procedural
    synthetic graphs under one protocol.
  - 2D→3D lifting other than Berger's slice-plus-rotation Π used for graph pretraining. For
    example: extruding 2D graphs into 3D tubes with sampled radius, or curving the 2D plane.
  - Pretraining of a **dense** (heatmap/tracing) graph extractor, as opposed to a set-prediction
    transformer. Mathys is the nearest, and it pretrains on synthetic data, not biological data.

  These gaps are the journal's novelty space.
- **Required controls**, given Berger and the MICCAI reviews:
  - no pretraining;
  - roads (Berger Π);
  - procedural synthetic graphs (Mathys-style or the DeepVesselNet synthetic set);
  - each biological source, alone and mixed.

  Run all at matched pretraining compute, with subject-paired CIs on the real targets (IXI,
  BRAVA, TopBrain-derived).
- **vesselFM caveat:** it is the obvious "foundation model" comparison reviewers raise. The
  MICCAI rebuttal argued it targets segmentation, not graphs. Vesselpose shows the cheap fix:
  vesselFM + TEASAR as a baseline row.

---

## Sources (primary pages used)

- **Relationformer:**
  - arXiv 2203.10202 (PDF text read);
  - ECCV supplement: ecva.net/papers/eccv_2022/papers_ECCV/papers/136970414-supp.pdf;
  - github.com/suprosanna/relationformer (+ `vessel_graph` branch).
- **Berger:**
  - arXiv 2403.06601v2 (PDF text read);
  - openaccess.thecvf.com/content/WACV2025/html/Berger_Cross-Domain_and_Cross-Dimension_Learning_for_Image-to-Graph_Transformers_WACV_2025_paper.html;
  - github.com/AlexanderHBerger/cross-dim_i2g.
- **Vesselformer:**
  - proceedings.mlr.press/v227/prabhakar24a.html (PDF text read);
  - github.com/chinmay5/vesselformer.
- **Trexplorer:**
  - papers.miccai.org/miccai-2024/801-Paper2112.html;
  - github.com/RomStriker/Trexplorer.
- **Trexplorer Super:**
  - arxiv.org/html/2507.10881v1;
  - github.com/RomStriker/Trexplorer-Super;
  - zenodo.org/records/15888958.
- **RefTr:** arxiv.org/html/2511.20823v1; github.com/RomStriker/RefTr.
- **Vesselpose:**
  - arxiv.org/html/2605.00538;
  - proceedings.mlr.press/v315/palaniappan26a.html;
  - github.com/Kainmueller-Lab/Vesselpose.
- **Mathys et al.:** arxiv.org/abs/2504.11858 and HTML v1.
- **DeformCL:** arxiv.org/abs/2506.05820; github.com/barry664/DeformCL.
- **GraphMorph:** arxiv.org/abs/2502.11731.
- **Vessel graph diffusion:** arxiv.org/html/2407.05842v1.
- **CoW centerline graphs:** arxiv.org/abs/2510.13720 and HTML v1; zenodo.org/records/17358162.
- **TopBrain:**
  - topbrain2025.grand-challenge.org (+ /data/);
  - zenodo.org/records/16878417 and the Zenodo API for versions 16623496 / 16878417 / 21972006;
  - medRxiv 10.64898/2026.05.28.26354312v1.
- **TopCoW:** zenodo.org/records/15692630; topcow24.grand-challenge.org/data/.
- **VesselGraph:** arXiv 2108.13233 (PDF text read).
- **BraVa:** PMC3971907 and PMC4082472 (search summaries).
- **CASILab / TubeTK:** cmake.org/Wiki/TubeTK/Data; marron.web.unc.edu/?p=218 (search summaries).
- **IXI segmentations:** Bizjak et al. SPIE 2022 (search summary); VesselVerse,
  papers.miccai.org/miccai-2025/1004-Paper0087.html (search summary).
- **ATM'22:** arXiv 2303.05745 (search summary).
- **PARSE:** parse2022.grand-challenge.org (search summary).
- **ImageCAS:** arXiv 2211.01607 (search summary).
- **Classic tools:**
  - Drees et al.: arXiv 2102.03444 / BMC Bioinformatics;
  - VesselVio: PMC9046271;
  - skan: PeerJ e4312;
  - github.com/seung-lab/kimimaro;
  - github.com/jocpae/clDice;
  - github.com/MIC-DKFZ/Skeleton-Recall;
  - github.com/YaoleiQi/DSCNet.
- **vesselFM:**
  - arXiv 2411.17386;
  - openaccess.thecvf.com/content/CVPR2025/html/Wittmann_vesselFM_A_Foundation_Model_for_Universal_3D_Blood_Vessel_Segmentation_CVPR_2025_paper.html;
  - github.com/bwittmann/vesselFM (+ `vesselfm/d_real`).
- **User's own work:**
  - MICCAI 2026: papers.miccai.org/miccai-2026/1001-Paper2313.html;
  - github.com/erc-caravel/Vascular-Graph-Extraction;
  - thesis: webthesis.biblio.polito.it/39935/1/tesi.pdf.
- **Other pretraining work:**
  - DeepBranchAI: frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1862872/full;
  - DINeuro: arXiv 2410.22078;
  - Stolt Ansó: arXiv 2001.00666;
  - VesselSim: arXiv 2605.26277;
  - VessShape: arXiv 2510.27646;
  - FreeCOS: arXiv 2307.07245.
- **Section C:** URLs are given in the delegated survey. Key records:
  - registry.opendata.aws/janelia-mouselight;
  - github.com/BigNeuron/Data;
  - zenodo.org/records/20501092 (HiNeuron), 7679898 (3D2cut), 61739 (Lobet), 5725751 (fungi);
  - github.com/WeizhenLiuBioinform/HALVS-Hierarchical-Vein-Segment;
  - ora.ox.ac.uk/objects/uuid:de65fc07-4b8f-4277-a6c4-82836afbdeb3 (Blonder);
  - neuromorpho.org/myfaq.jsp;
  - github.com/danzeng8/TopoRoot;
  - huggingface.co/datasets/ngaggion/ChronoRoot2;
  - doi:10.17617/3.XWST2Q (*Physarum*).
