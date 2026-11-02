## D08: full-cohort BRAVA long-training initialization comparison — 27 October 2026

**Status: completed through epoch 300; reviewed 28 October 2026; graph reconstruction remains weak.**

This is distinct from the older full-data 20–40-pass weighted-node-CE/edge-focal runs and the modern 1,248-patch/4k-update pilots. It compares scratch against compatible SyntheticMRI initialization under one fixed modern recipe. It does not isolate every difference from those historical studies.

Population: **8,768 training patches / 39 subjects**, full **1,796 validation patches / eight disjoint subjects**. Fixed representative train160 and val256 monitors are reused for learning curves. No test images/targets are loaded. Shared 256 queries, original K8 cubic B-splines, two relation tokens, single-level detector and existing `graph_only` route; no node refiner, token adapter, extra loss or unmatched supervision.

Node softmax focal gamma 2 / equal weights; matched-edge CE / equal weights / ratio upsampling 0.15; all established graph/spline coefficients and physical metrics retained. Both arms use the existing three-plane quarter turns plus Gaussian noise p=.35 and std 0–.015, no zoom. Augmentation is deterministically seeded per epoch/sample. Every epoch visits every patch once; complete augmented input/target stream hashes must match between arms.

Shared AdamW: LR 7e-5, encoder 3.5e-5, weight decay 1e-4, betas (.9,.999), eps 1e-8. Five full epochs of linear warmup, followed by cosine to 1e-6 over **300 full epochs**. Optimizer/global-step/LR continuity is preserved through 25-epoch Slurm chunks. No early stopping or validation-based LR choice. Float32; no AMP/accumulation.

Source: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/local_refinement_long/92741/runs/B0/model-12000.pt`; SHA256 `8b90d07292761244991f7f1e2fbe07d52c1a0fb93175c1fa863176a2c6337afe`. Selected by source-only SyntheticMRI validation edge F1 (**0.596715**) among the documented compatible B0 continuations; source parent scored 0.5881. Refined A/A-xyz variants require architecture changes and are excluded. Preserve 120 learned object-query rows; move the two learned relation rows from 120/121 to 256/257; retain the 136 added rows from the identical seeded scratch model. Every other source tensor must transfer exactly. Fresh target AdamW for both arms.

CPU audit and GPU smoke passed. Selected **batch 32**, four data workers, **274 updates/epoch**, **82,200 target updates/arm**. Benchmark estimate: **7.89 training GPU-hours/arm**, excluding evaluation/startup/queue. Measured costs are recorded separately. Benchmark optimizer updates are discarded.

| Arm | Batch | Seconds/update | Patches/s | Peak reserved GiB |
|---|---:|---:|---:|---:|
|scratch|16|0.1965|81.42|4.58|
|synthetic|16|0.1718|93.12|6.84|
|scratch|32|0.3455|92.61|9.86|
|synthetic|32|0.3137|102.02|10.85|

Predefined milestone evaluations: **25, 50, 100, 150, 200, 300 epochs**. Primary comparison is epoch 300 at unchanged .5 node/edge thresholds. Saved-BN, clean evaluation batch eight. Positional 1/2-mm metrics and common-v2 (1-mm anchors/curve support, .25-mm sampling) remain separate. Report node/edge TP/FP/FN/F1, raw recall, localization/Hungarian distances, conditional connectivity/AP and prevalence, duplicate/count biases, components/cycles, Branch/Curve F1, losses and paired subject uncertainty.

| Epoch | Arm | Train node / edge F1 (2mm) | Full val node / edge F1 | Val raw recall | Val conditional recall | Branch / Curve F1 |
|---:|---|---|---|---:|---:|---|
|25|scratch|0.00000 / 0.00000|0.00000 / 0.00000|0.33103|NA|0.00000 / 0.00000|
|25|synthetic|0.18246 / 0.01285|0.17416 / 0.01278|0.51075|0.49135|0.00028 / 0.03006|
|50|scratch|0.00376 / 0.00000|0.00213 / 0.00000|0.39073|0.00000|0.00000 / 0.00001|
|50|synthetic|0.27246 / 0.02640|0.25999 / 0.02762|0.55480|0.41101|0.00201 / 0.05563|
|100|scratch|0.00000 / 0.00000|0.00006 / 0.00000|0.42750|NA|0.00000 / 0.00005|
|100|synthetic|0.30836 / 0.05234|0.28721 / 0.04082|0.58991|0.60023|0.00301 / 0.07119|
|150|scratch|0.01668 / 0.00119|0.01279 / 0.00040|0.46040|0.28571|0.00000 / 0.00213|
|150|synthetic|0.34526 / 0.07265|0.31325 / 0.04997|0.61879|0.63336|0.00293 / 0.09322|
|200|scratch|0.02213 / 0.00000|0.02242 / 0.00040|0.47184|0.22222|0.00000 / 0.00272|
|200|synthetic|0.38119 / 0.09041|0.34843 / 0.06542|0.64347|0.61164|0.00492 / 0.11495|
|300|scratch|0.02282 / 0.00000|0.02544 / 0.00110|0.48288|0.40741|0.00000 / 0.00302|
|300|synthetic|0.39449 / 0.08989|0.35572 / 0.06794|0.65530|0.61979|0.00647 / 0.12174|

Results are descriptive pending review of the complete trajectory. Paired intervals resample eight whole validation subjects and do not measure training-seed uncertainty. All graph discrepancies are relative to incomplete SWC annotations. Brain masks are not vessel labels.

![BRAVA long-training fixed-monitor learning curves](../../../gnbm-brava-long-20261027/experiments/brava_long_20261027/learning-curves.png)

Interpretation predefined before training: increasing train **and** validation recovery supports a benefit from additional optimization/exposure under this recipe; strong train fit with persistently weak validation supports a generalization bottleneck; weak fit in both retains optimization/assignment/supervision alternatives. Tiny-set overfit does not prove the cohort needs only more epochs. Initialization gains are conditional on this shared schedule; pretraining adds source-data compute. No architecture/loss follow-up launches automatically.

Isolated worktree: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph`; artifacts: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/brava_long_20261027` with large checkpoints/caches at `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/brava_long_20261027`. `plan.json`, `cpu-preflight.json`, `gpu-smoke.json`, `batch-selection.json`, frozen code hashes/archive, per-epoch histories/input hashes, immutable checkpoint identities, prediction ZIPs, per-patch/subject metrics and Slurm accounting preserve provenance.

Submission record:
```json
{
  "preflight": "1636355",
  "smoke": "1636935",
  "smoke_prior_gpu_jobs": [],
  "smoke_queue": "",
  "frozen_source_manifest_sha256": "477645e496e8ee5a93d71536b8f8825c0df9d46c26ccbb2027f3c3ec9af01031",
  "previous_smokes": [
    {
      "job": "1636588",
      "status": "CANCELLED by 307254|0:0",
      "reason": "Shorten multiprocessing temporary socket path; correct aggregation of one-off gradient diagnostics. No long-training updates or scientific settings changed."
    }
  ],
  "training_arrays": [
    "1637025",
    "1637027",
    "1637032",
    "1637034",
    "1637036",
    "1637037",
    "1637039",
    "1637040",
    "1637042",
    "1637044",
    "1637045",
    "1637046"
  ],
  "arrays_by_end_epoch": {
    "25": "1637025",
    "50": "1637027",
    "75": "1637032",
    "100": "1637034",
    "125": "1637036",
    "150": "1637037",
    "175": "1637039",
    "200": "1637040",
    "225": "1637042",
    "250": "1637044",
    "275": "1637045",
    "300": "1637046"
  },
  "collectors": [
    "1637026",
    "1637029",
    "1637035",
    "1637038",
    "1637041",
    "1637047"
  ],
  "collector_epochs": {
    "25": "1637026",
    "50": "1637029",
    "100": "1637035",
    "150": "1637038",
    "200": "1637041",
    "300": "1637047"
  },
  "global_h100_limit": 2,
  "training_prior_gpu_jobs": [],
  "training_queue": "",
  "chunk_walltime": "01:20:00",
  "selected_batch": 32
}
```


<!-- BEGIN D08 COMPLETION REVIEW 20261028 -->
## D08 completion review — 28 October 2026

Both arms completed **300 full epochs / 82,200 target updates**, with all 24 training tasks and all six milestone collectors exiting successfully. All epoch losses were finite; the paired patch-order and complete augmented image/target hashes agree for every one of the 300 epochs. Final checkpoint SHA256 values were rechecked against the actual files. No additional training, threshold tuning or test evaluation was run during this review.

**Additional training helps the pretrained arm but does not recover a functional graph detector.** Its 2-mm representative training edge F1 rises from 0.01285 at epoch 25 to 0.08989 at 300; full validation rises from 0.01278 to 0.06794. Improvement is small after epoch 200 (train 0.09041→0.08989; val 0.06542→0.06794). At 300, train/val node F1 is 0.39449/0.35572 and raw node recall 0.71774/0.65530. A train–validation gap remains, but both reconstructions are poor. These measurements support a persistent cohort-fitting/optimization limitation alongside a generalization gap; they do not establish a purely generalization-only bottleneck or show that further epochs will solve it.

**Scratch remains strongly underpredictive under this shared schedule.** Train/val 2-mm node F1 is 0.02282/0.02544 and edge F1 0/0.00110 despite decreasing loss. It retains only 1,680 nodes against 30,786 validation GT nodes. The pretrained arm retains 18,793 nodes. The experiment compares initializations under one shared schedule and one target seed; it does not establish an optimal scratch learning rate or budget.

Final full-validation physical metrics (unchanged .5 node/edge thresholds):

| Arm | Tolerance | Node F1 | Edge TP / FP / FN | Edge F1 | Raw node recall |
|---|---:|---:|---|---:|---:|
| Scratch | 1 mm | 0.00702 | 1 / 489 / 19,445 | 0.00010 | 0.13051 |
| Scratch | 2 mm | 0.02544 | 11 / 479 / 19,435 | 0.00110 | 0.48288 |
| SyntheticMRI | 1 mm | 0.14740 | 167 / 9,706 / 19,279 | 0.01139 | 0.29932 |
| SyntheticMRI | 2 mm | 0.35572 | 996 / 8,877 / 18,450 | 0.06794 | 0.65530 |

For the pretrained arm at 2 mm, **17,839/18,450 missed edges (96.69%) lack a recovered retained endpoint pair**. Conditional relation recall is 0.61979 and conditional AP 0.60471 (32,880 candidates / 1,607 positives; prevalence 4.887%). All-retained AP is 0.17897 (158,454 candidates / 1,607 positives; prevalence 1.014%). About 89.40% of false edges touch a spatially unmatched endpoint. Raw-versus-retained node recall (0.65530 vs 0.28643) warrants examining localization, confidence and assignment together; it does not by itself prove a calibration fix. Common-v2 Branch/Curve F1 is 0.00647/0.12174, node/edge count bias −6.678/−5.330 per patch, component MAE 2.626 and cycle bias +1.059. Thus the improvement is incomplete graph reconstruction, with substantial missed structure and false loops.

The paired eight-subject bootstrap gives pretrained-minus-scratch 2-mm node F1 +0.33027 [0.32262, 0.33879] and edge F1 +0.06684 [0.06245, 0.07032]. These are subject-resampling intervals on reused development subjects, not training-seed uncertainty. Clean train/full-val total losses are 4.42483/4.59476 for scratch and 3.75477/4.21577 for pretraining. All graph errors remain relative to incomplete SWC annotations; brain masks are not vessel masks. K8 capacity and raw-image quality were held fixed and are not isolated causal explanations here.

Measured allocation: **7.9694 H100-hours scratch + 8.0197 pretrained = 15.9892 H100-hours** for the two long arms, including within-job evaluation/startup. Smoke/recovery adds 0.08639, giving 16.07556 H100-hours total; source pretraining and CPU collection are excluded. Both arms ran in parallel subject to the global two-H100 limit. Final accounting and the review are saved as `completion-slurm-accounting.psv` and `completion-review.json` in the isolated experiment outputs.

Final checkpoint identities: scratch `large/runs/scratch/model-epoch-300.pt`, SHA256 `7d19b56e91666ce6aabec1e56396cf23abb096011e3af7d64aa6d8ae7c9854c6`; pretrained `large/runs/synthetic/model-epoch-300.pt`, SHA256 `aeab859b5c8bc435ae1205baf598b9aca24b91ddde2a4d181364e0fab611fe59`. The source checkpoint and all milestone artifacts remain preserved at the paths documented above.

**Next recommendation:** review cached assignment/localization/confidence and annotation-relative target coverage before another training extension or representation change. This is a recommendation only. No follow-up, new architecture/loss, or held-out-test evaluation was launched.

<!-- END D08 COMPLETION REVIEW 20261028 -->


<!-- D08 FINAL REVIEW 20261028 -->
## Reviewed final D08 results — 28 October 2026

**DONE — measured.** Both K8 training arms completed **300 epochs / 82,200 updates**. All 24 production GPU tasks and six milestone collectors exited successfully; final GPU array 1637046 completed at 04:04 and collector 1637047 at 04:08 on 28 October (Europe/Paris). The earlier cancelled smoke 1636588 remains a documented pre-production infrastructure recovery, not a failed training arm. No inference or training was repeated for this review.

The primary comparison uses the predeclared final epoch, unchanged 0.5 node/edge thresholds, **1,796 validation patches from eight subjects**, with **8,768 training patches from 39 subjects**. Clean training numbers below use the fixed160-patch representative monitor, not every training patch. Node focal gamma2/equal weights, matched-edge CE ratio0.15, original K8 geometry and detached `graph_only` are unchanged. Full augmented sample streams and data order match across both arms.

| Final validation metric | From scratch | SyntheticMRI initialization |
|---|---:|---:|
| Node F1, 1 mm | 0.00702 | 0.14740 |
| Positional edge F1, 1 mm | 0.00010 | 0.01139 |
| Node F1, 2 mm | 0.02544 | 0.35572 |
| Positional edge F1, 2 mm | 0.00110 | 0.06794 |
| Node precision, 2 mm | 0.24583 | 0.46922 |
| Filtered node recall, 2 mm | 0.01342 | 0.28643 |
| All-query node recall, 2 mm | 0.48288 | 0.65530 |
| Edge precision, 2 mm | 0.02245 | 0.10088 |
| Full-GT edge recall, 2 mm | 0.00057 | 0.05122 |
| Conditional relation recall, 2 mm | 0.40741 | 0.61979 |
| Hungarian mean assignment distance, mm | 2.42702 | 2.25564 |
| Component-count MAE | 5.69321 | 2.62584 |
| Cycle-count MAE | 0.03396 | 1.05902 |
| Duplicate candidates per patch, 2 mm | 0.00668 | 0.34577 |
| Branch F1, common-v2 | 0.00000 | 0.00647 |
| Curve F1, common-v2 | 0.00302 | 0.12174 |
| Edge TP / FP / FN, 2 mm | 11 / 479 / 19435 | 996 / 8877 / 18450 |

**Paired whole-subject uncertainty:** SyntheticMRI minus scratch at 2 mm gives node F1 **+0.33027 [0.32262, 0.33879]**, edge F1 **+0.06684 [0.06245, 0.07032]**, and raw node recall **+0.17242 [0.16780, 0.17741]**. These are existing 1,000 paired resamples of eight reused validation subjects; they do not quantify training-seed uncertainty. One target seed 271039 was run. Pretraining has additional source optimization/data exposure.

### What improved, and what still fails

**Initialization matters under this controlled recipe.** Scratch remains poor on representative training data too: node/edge F1 at 2 mm is **0.02282 /0.00000**, versus **0.39449 /0.08989** after SyntheticMRI initialization. Thus the scratch failure is not explained solely by held-out-domain generalization. Training-loop time was 7.51/7.54 GPU-hours, respectively (excluding evaluation/startup). The transferred model progresses from validation node/edge F1 **0.17416 /0.01278** at 25 epochs to **0.35572 /0.06794** at 300. Its edge F1 moves only 0.06542→0.06794 between 200 and 300; these curves do not justify another automatic duration extension. This study does not isolate epochs alone from its schedule, population and initialization differences relative to older4k pilots.

**Node coverage and selection remain the dominant measured missed-edge bottleneck.** The transferred model has raw 2 mm coverage **65.53%**, but only **28.64%** after confidence filtering. It retains18,793 nodes over 1,796 patches (10.46/patch), versus 30,786 GT nodes (17.14/patch). Only **1,607/19,446 GT edges (8.26%)** have both endpoints recovered under that filtered spatial association. Of18,450 missed edges, **17,839 (96.69%)** lack a recovered endpoint pair;611 have both endpoints but fail the relation decision. Even perfect edge decisions on those fixed associations cannot exceed the8.26% direct-edge recall ceiling. This accounting is diagnostic, not a deployable oracle or proof of the cause of confidence failure.

The scratch model retains only 0.94 nodes/patch: raw coverage48.29% collapses to filtered recall1.34%. Its tiny cycle error therefore cannot be treated as improved topology. The transferred model also underpredicts total nodes/edges, despite many annotation-relative false detections: it predicts5.50 edges/patch versus10.83 GT, with **996 TP /8,877 FP**. About **89.40%** of its FP edges touch at least one spatially unmatched node. Component MAE decreases5.693→2.626, while cycle MAE increases0.034→1.059; neither scalar alone establishes correct anatomy. BRAVA traces are incomplete and SWC forest topology is annotation-relative.

**Connectivity is improved but remains weak in absolute terms.** Transferred edge precision/recall is10.09%/5.12%; only 86 branches match common-v2, giving Branch F1 **0.00647**. Curve F1 **0.12174** combines32.11% geometric precision with7.51% coverage recall. Better curve coverage does not establish a sufficiently reconstructed vascular graph or select a better spline representation.

### Relation ranking and calibration limits

Transferred conditional relation AP at 2 mm is **0.60471** on **32,880 pairs /1,607 positives** (prevalence 4.887%); all-retained AP is **0.17897** on **158,454 pairs /1,607 positives** (1.014%). Scratch values are0.37041 on 433/27 and0.02390 on 6,851/27. These arms select different query populations: this is **not** a matched-candidate proof that the relation scorer alone improved. Scratch's 1 mm conditional AP/recall of1.0 rests on **one** recovered positive edge and cannot support a robustness claim. Report AP, denominators and node recovery together.

Only the fixed 0.5 operating point is evaluated in D08; training-selected thresholds, matched-count controls and oracle endpoint substitutions were not performed here. No validation threshold was tuned. The next inexpensive diagnostic is a **training-only node/edge operating-point audit using these caches**, followed by unchanged validation application and candidate-aware error accounting. Do not conflate the raw-versus-filtered recall gap with a guaranteed threshold fix, or repeat the failed generic NMS/verifier/mining studies. No further job was submitted.

### Current decision and evidence

Use the SyntheticMRI-initialized K8 detector as the stronger **measured BRAVA reference under this recipe**. Preserve the scratch negative result. Do not present D08 as validating K24, adaptive K, C1, degree2 splitting, node refinement or a new relation architecture: none changes in this comparison. Prioritize confidence/coverage and assignment consistency before another geometry or long-training sweep. No final journal representation is selected.

Final checkpoint SHA256: scratch `7d19b56e91666ce6aabec1e56396cf23abb096011e3af7d64aa6d8ae7c9854c6`; transferred `aeab859b5c8bc435ae1205baf598b9aca24b91ddde2a4d181364e0fab611fe59`. Source SHA256 remains `8b90d07292761244991f7f1e2fbe07d52c1a0fb93175c1fa863176a2c6337afe`. Earlier protocol, milestone tables, source-selection history and checkpoint manifests are preserved above.

Review tables and input hashes: `experiments/completed_result_review_20261028/{review.json,brava-milestones.csv,brava-final-per-subject.csv,provenance.json}`. Original analysis and predictions: sibling `gnbm-brava-long-20261027/experiments/brava_long_20261027/`; learning curves remain linked in the D08 section. To print the saved full report without rerunning a model:

```bash
cat /lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/brava_long_20261027/report.md
```

