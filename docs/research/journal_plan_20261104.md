# Journal plan: curved vessel graphs from limited labels (draft, 4 Nov 2026)

Target: **Medical Image Analysis** (MICCAI-extension special issue), submission in about 3 months
(early Feb 2027). This is the journal extension of the MICCAI 2026 paper *"Structural Congruence
Matters"* (provisional accept). Status labels: DONE / RUNNING / PROPOSED. Literature and resources:
[`journal_literature_20261104.md`](journal_literature_20261104.md). The current model and its results:
[`dense_traced_model_20261102.md`](dense_traced_model_20261102.md).

Shorthand:
- **Dense model**: the two-stage detect-and-trace extractor (stage 1 U-Net maps; stage 2 traced candidates, edge head, B-spline geometry).
- **Source**: pretraining data. **Target**: vessel data the model is fine-tuned and evaluated on.
- **Budget**: the fraction of target training subjects used.
- **Regimes**: scratch / sequential (pretrain on the source, then fine-tune) / mixed (source + target co-training, then fine-tune; the MICCAI setting).

## 1. Story and claims

One story with two pillars: **learning curved vessel graphs from limited labels.**

1. **Method (main contribution).** A dense detect-and-trace graph extractor with continuous B-spline edges.
   - It beats set-prediction transformers and tree trackers on real 3D vessels.
   - It is competitive with segment-then-skeletonise pipelines.
   - Curved edges are learnable and better than straight ones.
   - The traced connectivity recovers long, tortuous edges.
   - The failure analysis of set prediction on real data is the motivation, not a separate claim.
2. **Data (MICCAI extension).** With few vessel labels, pretraining on *non-medical biological* branching structures helps, and *which* source matters more than pretraining itself. Two claims back this up:
   - Biological sources must beat a fabricated (procedural) source rendered identically, so the gain is attributed to biological structure, not to the rendering.
   - Transfer gain is related to measured structural congruence between source and target.

These claims address the MICCAI reviews: medically relevant baselines; more than one botanical source; image-to-graph methods beyond RelationFormer; the missing ablation, whose exact item is to be checked in the meta-review.

## 2. Datasets

**3D targets** (fine-tuning and evaluation). Graph GT extraction must start from very good segmentations.

| Dataset | Modality | GT | Status |
|---|---|---|---|
| SyntheticMRI (DeepVesselNet / Tetteh) | synthetic MRA | provided graphs | DONE (in use); the shared benchmark of RelationFormer, Vesselformer, Berger, Trexplorer, Vesselpose |
| IXI | MRA | graphs from vessel masks (Voreen/Vedo pipeline) | DONE (in use); **mask provenance undocumented, must be settled** |
| BRAVA | MRA | manual SWC traces | DONE (in use) |
| TopBrain (MICCAI 2025 challenge) | CTA + MRA | graphs from its segmentations with our pipeline | PROPOSED |
| VesSAP | mouse-brain microscopy | graphs as in Berger et al. | PROPOSED |
| Circle of Willis centreline graphs (Musio et al., 250, CC BY-NC) | MRA/CTA | provided graphs | PROPOSED |
| Others with excellent segmentations | — | — | open |

**2D targets:** retinal fundus (AV-WIDE; DRIVE-like sets), REAVER, OCTA (OCTA-Synth plus a real OCTA set).
Status: AV-WIDE, REAVER and OCTA-Synth DONE (prepared); the rest PROPOSED. More datasets are coming from
the user in a few weeks.

**Sources** (pretraining):

| Source | Dim | Structure | Status |
|---|---|---|---|
| Leaves (HALVS; Blonder venation) | 2D | loops | HALVS DONE; Blonder PROPOSED |
| Plants (3D2cut grapevines, CC BY-NC-SA) | 2D | trees | DONE |
| Neurons (Janelia MouseLight / HiNeuron / BigNeuron) | 3D microscopy | trees | PROPOSED |
| Physarum / fungal networks | 2D | loops | PROPOSED |
| Synthetic vessels (SyntheticMRI) | 3D | trees | DONE |
| **Fabricated control** (procedural random trees, same renderer) | 3D | trees / loops | PROPOSED |
| Roots | — | — | no public 3D traced set found; skipped unless one appears |

## 3. 2D → 3D lifting of 2D sources

- **MICCAI:** a flat 5-voxel slab at a random depth.
- **17 Oct** (`data/source_augmentation.py`, opt-in): a smooth RBF height field bends the slab; the nodes and GT paths are warped consistently; multi-scale noise texture is added.

Proposed default (PROPOSED):
1. **Lift the 2D GT graph into 3D.** Smooth out-of-plane bending, then an arbitrary 3D rotation (0–360° on all axes).
2. **Render round tubes.** Radii come from the 2D widths. The appearance is MRA-like: vessel intensity profile, bias field, Rician noise, blur.
3. **Compose 2+ independently rotated trees per volume.**
   - Reject near-coplanar pairs.
   - Branches from different trees may pass over and under each other. A crossing is not a junction, because the trees remain separate graphs.
   - Reject placements where tubes actually intersect.
4. **Rotation validity.** Clip the rotated graphs at the volume boundary with the same boundary-node convention as real patches. Reject a placement only when too little of the graph remains. This avoids biasing towards compact trees, which "rotate until valid" would do.

Ablation: flat slab vs bent textured sheet vs tube rendering vs tube rendering with multi-tree composition.

## 4. Baselines

| Baseline | Type | Notes |
|---|---|---|
| RelationFormer (Shit 2022) | set prediction | Apache-2.0; already run at literature scale (DONE on IXI/BRAVA) |
| Berger et al. 2024/25 (cross-domain, cross-dimension) | set prediction + transfer | MIT; also a pretraining baseline (roads/OCTA to 3D) |
| Vesselformer | set prediction | no stated license (research comparison) |
| Trexplorer Super (MICCAI 2025) | tree tracking | tree-only (cannot represent loops); heavy (4×A100, ~2M iterations); no license; evaluated where GT is a tree, with the limitation stated |
| Vesselpose (MIDL 2026) | mask + direction field + guided TEASAR | MIT; strong on real data |
| nnU-Net (± Skeleton Recall) → Kimimaro/Voreen | segment + skeletonise | the classic pipeline reviewers asked for; GT-tool bias noted (the GT generator favours itself) |
| clDice / DSCNet segmentation → same skeletonisation | topology-aware segmentation | optional |
| Mathys et al. 2025 (U-Net detector + adjacency) | dense, closest prior | no code found: cite and contrast; reimplement only if time allows |

All baselines are trained on our splits with their published recipes, at literature-scale budgets. All are
evaluated by our metric suite (§5) at patch level; full volume is added when available.

## 5. Metric suite

- **Geometry:**
  - node F1 / edge F1 at a mm tolerance (2 mm);
  - Curve F1 and guide distance (mm);
  - centreline point/branch F1 as in Trexplorer / Vesselpose (for alignment with that literature);
  - HD95 of centrelines.
- **Topology:** Branch F1; β₀/β₁ errors (absolute; relative as in GrEVe in the supplement); loops per patch; SMD; long-edge recovery by arc length.
- **Representation efficiency:**
  - nodes and edges per patch versus GT;
  - control points per edge (adaptive K);
  - bits or parameters per graph;
  - compression of dense paths at a fixed geometric error.
- **Compute efficiency:** model parameters, training GPU-hours, inference time per patch and per volume, peak memory.

Each metric is computed at patch level and, when available, at full-volume level (stitched graphs).

## 6. Protocol

- Fixed subject-level train/val/test splits per dataset. The test sets stay untouched until a single
  final locked evaluation.
- Thresholds, checkpoints and hyperparameters are chosen on the train monitor only (current rule).
- Three seeds for every main and ablation run. Report mean ± sd and subject-paired bootstrap 95% CIs on paired differences.
- Equal total update counts across regimes, so "more exposure" is not mistaken for transfer.
- Literature-scale training budget for every method before any option is dismissed.

### 6.1 Decoding and threshold protocol (fixed 6 Nov 2026, before any locked evaluation)

Implemented in `scripts/protocol_tables_20261106.py` and `scripts/protocol_report_20261106.py`
(results: `experiments/evaluation_protocol_20261106/report.md`).

- **Two kinds of claim are kept apart.**
  - Model claims (architecture, edge representation, loop training, pretraining, sources) rest on threshold-free
    metrics: node AP @2 mm, edge ranking AP, box mAP/mAR (RelationFormer/GrEVe protocol), PR curves; plus F1 and
    topology at the single global operating point below.
  - Decoding claims (tree-rule hysteresis V4 vs single thresholds V0) are a separate ablation on the same models.
- **Four operating points, always labelled:**
  1. *global, leave-one-dataset-out (headline)*: settings chosen on the training monitors of the other datasets
     only (macro average). Nothing is tuned on the reported dataset; this is what an unseen dataset gets.
  2. *global, pooled*: chosen on all training monitors; the frozen setting for any future dataset, including small
     ones, which then need no tuning at all.
  3. *per dataset*: chosen on the dataset's own training monitor (best achievable without validation data).
  4. *oracle*: best on the evaluated set; an upper bound only, never claimed.
- **Rules, identical for every dataset, operating point and method with a tunable decoder:**
  - V0: node threshold by node F1, then edge threshold by edge F1.
  - V4 (rule A): maximise edge F1 subject to |β₀| error, |β₁| error ≤ V0 and Curve F1 ≥ V0 on the same data.
  - The unconstrained V4 (edge F1 only) is reported too, to show the constraint does not create the gains.
- **Primary endpoints:** edge F1, Curve F1, |β₀| and |β₁| errors at the headline operating point; node AP @2 mm and
  box edge mAP as threshold-free endpoints. Everything else is secondary.
- **Statistics:** subject-paired bootstrap (2000 resamples), two-sided bootstrap p, Holm across datasets per
  comparison and primary endpoint. Non-inferiority margins fixed in advance: F1 −0.01, Betti error +0.10 per patch.
  "No loss" is claimed only when the CI clears the margin.
- **Robustness evidence:** sensitivity curves (metric vs each threshold, descriptive), and the variance of
  per-dataset choices when the rule runs on 100- and 256-patch monitor subsamples.
- **Reporting:** all datasets, including unfavourable ones, in the main table; per-patch results and exact settings
  released with the code. Validation is used for development only; the locked test runs once (Phase 6).

## 7. Experiments

**E1. Main comparison (method pillar).**
- Final dense model vs all baselines.
- 3D targets: SyntheticMRI, IXI, BRAVA, TopBrain, VesSAP / CoW. 2D targets: fundus, REAVER, OCTA.
- 3 seeds, full budget.

**E2. Continuous vs straight edges.**
- The same model with straight, polyline and spline edges (fixed K vs adaptive K), with nodes identical.
- Geometry and representation-efficiency metrics.
- DONE in part: single seed, on SyntheticMRI/IXI/BRAVA.

**E3. Architecture ablations of the dense model:**
- centreline channel on/off;
- candidate generation (16-NN vs traced vs traced + fallback);
- soft tracing; gap features;
- neighbourhood offsets;
- pass-through decision;
- adaptive-K rule;
- backbone: U-Net / U-KAN / DSCNet (RUNNING) with capacity-matched controls.

Many of these are DONE on a single seed; they need 3 seeds on the final configuration.

**E4. Pretraining pillar**, pruned as agreed:
- Full label-budget curves on 1–2 2D targets and 1–2 3D targets:
  - budgets 1, 2, 5, 10, 25 and 100% of subjects;
  - all sources plus the fabricated control and scratch;
  - all three regimes;
  - 3 seeds.
- Other targets: the best regime and best source, against scratch and the fabricated control, with 3 seeds. Open: at which budgets.
- Zero-shot as a secondary diagnostic only.

**E5. 2D → 3D lifting ablation** (§3), on one 3D target at small budgets.

**E6. Structural congruence.** Per-source structural statistics (branch angles, segment lengths, tortuosity, loop density, radius distribution) versus per-target statistics, related to the transfer gain from E4.

**E7. Full-volume evaluation.** Stitched whole-subject graphs. A section of its own if strong, otherwise supplementary.

## 8. Compute (rough)

| Item | Runs | GPU-hours |
|---|---|---|
| One dense run (stage 1 + traced) | 1 | ~2.5–3 |
| E4 curves, per target (7 arms × 6 budgets × 3 seeds) | ~126 | ~350 |
| E4 curves, 3–4 targets | | ~1,000–1,400 |
| E4 regime selection: done on 1 target at 2 budgets first, then the best regime only for the curves | | |
| One-off source pretraining runs | | ~100 |
| Baselines | | ~300–600 (Trexplorer Super dominates) |
| E1–E3 with 3 seeds | | ~300 |

At today's usage rate (~50 GPU-h/day) this fits in about 2 months, if the H100 queue cooperates.

## 9. Timeline (13 weeks)

| Weeks | Work |
|---|---|
| 1–2 | Lock the architecture (backbone results; pass-through/offset decision). Build the GT pipelines for TopBrain / VesSAP / CoW. Settle IXI provenance. Write the 3D renderer and multi-tree composer and the fabricated control. Implement the metric suite. Set up baselines. |
| 3–6 | E1 (all targets, 3 seeds). E4 regime selection. One-off source pretraining. |
| 6–9 | E4 budget curves; E2/E3/E5 with 3 seeds; E6 analysis. |
| 9–11 | E7 (if the full-volume pipeline is ready); locked test evaluation; figures. |
| 11–13 | Writing and internal review. |

## 10. Open items and risks

- **IXI mask provenance.** Required for publication. Candidates: Bizjak et al. 2022, VesselVerse.
- **Missing ablation in the MICCAI meta-review:** read the reviews and add it explicitly.
- **E4 "other targets" protocol:** which budgets, or all budgets.
- **Baseline reproducibility:** Vesselpose reports that it could not reproduce the published baselines. Use published recipes and report our numbers alongside theirs.
- **Licenses:** CC BY-NC sets are research-only. Trexplorer Super data is academic-only, and several codebases have no license.
- **GT-tool bias:** if Voreen or Kimimaro produced the GT, the skeletonise baseline using the same tool is favoured. Use a different tool, or report both.
- **Backbone:** U-KAN shows no gain at our budget. DSCNet is RUNNING; its capacity is 3× the U-Net's, so a straight-branch control is needed before any claim.

## 11. Checklist (ordered)

Work through the phases in order. Within a phase, items can run in parallel unless marked *(needs X)*. `[x]` = done. Owner: **U** = user, **C** = Claude.

### Phase 0: decisions and inputs (week 1)
- [ ] **U** Document the provenance of the IXI vessel masks (source, method, version).
- [ ] **U** Identify the "missing ablation" in the MICCAI meta-review and add it to §7.
- [ ] **U** Confirm the MedIA special-issue deadline and pin the timeline to it.
- [ ] **U** Decide the E4 protocol for the "other targets": full budget only, or a few budgets too.
- [ ] **U** Decide whether to add the same-size DSCNet control (straight, frozen-offset branches).
- [ ] **U** Hand over the additional 2D datasets when available.

### Phase 1: lock the architecture (weeks 1–2)
- [x] **C** Dense model, traced connectivity, adaptive-K splines (single seed, 3 datasets).
- [x] **C** Soft tracing, gap features, two-hop, pass-through and offset fix evaluated (single seed).
- [x] **C** U-KAN (concat), MLP control and U-KAN (add) trained on both stages.
- [ ] **C** DSCNet (kernels 3 and 9) on both stages — RUNNING.
- [ ] **C** Full paired backbone comparison: U-Net / U-KAN concat / U-KAN add / MLP / DSCNet 3 / DSCNet 9, with shared thresholds, long edges and topology.
- [ ] **C** Re-run the U-Net baseline on current code, to close the 30 Oct vs 3 Nov code gap.
- [ ] **C** Fix the BRAVA node-threshold confound (one fixed node threshold with re-traced evaluation) for all comparisons.
- [ ] **U+C** Freeze the final model configuration: backbone, offsets, pass-through yes/no, candidate rule.
- [ ] **C** Merge the `KanKan` / `DSCNet` work into the main line; push after U approves.

### Phase 2: data and ground truth (weeks 1–3, parallel with Phase 1)
- [ ] **C** TopBrain: download, check segmentation quality, extract graph GT with our pipeline, make patches, define splits.
- [ ] **C** VesSAP: obtain the data, extract graphs as in Berger et al., make patches, define splits.
- [ ] **C** Circle of Willis graphs (Musio et al.): obtain the images, align with the provided graphs, make patches, define splits.
- [ ] **C** Search for further 3D vessel datasets with excellent segmentations; U approves each one.
- [ ] **C** 2D: check AV-WIDE / REAVER / OCTA-Synth splits; add a real OCTA set and a DRIVE-like fundus set.
- [ ] **C** Write a held-out test split for every dataset, and lock it (no access until Phase 6).
- [ ] **C** GT quality audit per dataset: alignment with the image, as in the BRAVA check; skeleton-tool bias.

### Phase 3: sources and 2D→3D rendering (weeks 2–4)
- [ ] **C** Acquire neuron data (MouseLight / HiNeuron / BigNeuron) and convert it to image + graph patches.
- [ ] **C** Acquire Physarum / fungal networks and Blonder leaf venation, and convert them to graphs.
- [ ] **C** Fabricated control: a procedural tree / loop generator with matched size statistics.
- [ ] **C** 3D lifting: graph bending, arbitrary rotation, and boundary clipping with boundary nodes.
- [ ] **C** Tube renderer with MRA-like appearance: profile, bias field, Rician noise, blur.
- [ ] **C** Multi-tree composer: independent rotations, rejection of coplanar pairs and of intersecting tubes.
- [ ] **C** Visual and statistical QA of rendered sources against real targets. **U** reviews the samples.
- [ ] **C** Structural statistics (angles, segment lengths, tortuosity, loops, radii) per source and per target, for E6.

### Phase 4: metric suite and baselines (weeks 2–5)
- [ ] **C** Metric suite in one module, unit-tested on toy graphs:
  - node/edge F1;
  - Curve F1;
  - Trexplorer point/branch F1;
  - HD95;
  - β₀/β₁;
  - SMD;
  - long-edge recovery;
  - representation size;
  - compute.
- [ ] **C** Patch-to-volume stitching (for E7), if the full-volume pipeline is ready.
- [ ] **C** Baselines, on our splits and with published recipes:
  - [x] RelationFormer (IXI, BRAVA, SyntheticMRI);
  - [ ] RelationFormer on the new datasets;
  - [ ] Berger et al.;
  - [ ] Vesselformer;
  - [ ] Trexplorer Super, on tree targets only;
  - [ ] Vesselpose;
  - [ ] nnU-Net and nnU-Net + Skeleton Recall, each followed by Kimimaro / Voreen.
- [ ] **C** Sanity check each baseline: reproduce its own published number on its own dataset where possible before training on ours.

### Phase 5: experiments (weeks 3–9)
- [ ] **C** E1: final model vs all baselines, all targets, 3 seeds *(needs Phases 1, 2, 4)*.
- [ ] **C** E4a: regime selection (scratch / sequential / mixed) on one 3D target at 2 budgets, 3 seeds *(needs Phase 3)*.
- [ ] **C** One-off source pretraining runs, for each source and the fabricated control.
- [ ] **C** E4b: full label-budget curves (1, 2, 5, 10, 25, 100%) on 1–2 2D and 1–2 3D targets, all sources, best regime, 3 seeds.
- [ ] **C** E4c: other targets, scratch vs best source + regime vs fabricated control, 3 seeds.
- [ ] **C** E2: straight vs polyline vs spline (fixed vs adaptive K), with nodes identical, 3 seeds.
- [ ] **C** E3: architecture ablations on the final configuration, 3 seeds.
- [ ] **C** E5: 2D→3D lifting ablation (slab / bent sheet / tubes / multi-tree), small budgets, 3 seeds.
- [ ] **C** E6: relate the structural-congruence statistics to the E4 gains.
- [ ] **C** E7: full-volume evaluation of the final model and the main baselines *(needs stitching)*.

### Phase 6: locked evaluation (weeks 9–11)
- [ ] **U+C** Freeze every model, threshold and checkpoint (chosen on the train monitor only).
- [ ] **C** Run the single locked test evaluation for E1, E4 and E7, with subject-paired CIs.
- [ ] **C** Figures: qualitative 2D/3D examples, budget curves, congruence plot, representation-efficiency plot, 3D viewers.
- [ ] **C** Results tables (main text and supplement), with every caveat stated.

### Phase 7: writing (weeks 11–13)
- [ ] **U+C** Outline mapped to the claims (§1) and the reviewer requests.
- [ ] **U** Draft the introduction and related work (C provides the literature notes).
- [ ] **C** Draft the methods, experiments and results from the docs; U revises.
- [ ] **U+C** Internal review against the MICCAI reviews and the MedIA guidelines; reproducibility package (code, configs, splits).
- [ ] **U** Submit.
