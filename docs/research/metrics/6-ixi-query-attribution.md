# 6. IXI full-query, duplicate and edge attribution — 2026-10-19

## Scope and status

First implementation of P0-A/B/C/D in the
[diagnostic roadmap](../IXI_graph_extraction_diagnostic_roadmap.md).
The eight-patch GPU smoke and all three frozen-model pilot evaluations completed.
The jobs failed in the final collector; corrected CPU collection then completed
for every arm from the saved caches, without repeating GPU inference.
No model training or test-set evaluation is part of these jobs.

| Arm | Meaning | Pilot job |
|---|---|---|
| parent | IXI-only delayed-spline best checkpoint, step 28170 | 1291324 |
| baseline | Same parent's node-head-only continuation, 1000 updates | 1291328 |
| local_image | Same continuation with early-CNN node refinement | 1291329 |

These are evaluations of existing checkpoints. Comparing local_image with the
parent alone would confound refinement with continued node-head training;
the baseline continuation is included for that reason. The previously completed
zero-image control remains part of the original refinement evidence, but is not
repeated in this first attribution pilot.

## Frozen protocol

Reuse the exact 256 training / 128 validation IDs from relation-isolation job
1278148. The full training and validation subject sets are checked for overlap,
not merely the selected patches. No augmentation, BN recalibration, parameter
update, edge-threshold search, GT regeneration, or split modification.

For each patch, save all 256 queries, original/refined coordinates and object
logits, decoder tokens, all 32,640 unordered pair logits/probabilities, decoded
query identities/edges, and spline controls/curves. Use `infer_graphs` itself
for both unfiltered and normal scoring. Require their selected query-edge sets
and probabilities to agree. Cache geometry for decoded edges only: deletion
policies never introduce an edge and can reuse its original controls exactly.

Record checkpoint/adapter/manifest/code hashes, resolved configuration, parent
step, file identifiers, per-target hashes, metric transforms and patch shapes.
Directories refuse overwrite. Caches/checkpoints stay on SCRATCH.

## Node and edge accounting

At both 1 and 2 mm, reuse the existing maximum-cardinality, minimum-distance
positional assignment without degree matching. Repeat independently for raw
queries and retained queries. Reserve accepted matches, then categorize extras:
near an already recovered GT node; near an unrecovered GT node; within 1 mm
of foreground voxel centres but not near GT; or off-vessel/outside. These are
diagnostic categories, not declarations of anatomical falsity.

Report confidence distributions, raw versus retained recovery, nearby-query
multiplicity, nearest-neighbour distance, GT subtype and per-subject counts.
Boundary has priority over junction/degree-2/interior-terminal labels.
Foreground support uses the physical cache metric and valid voxel-face bounds.

An M–M edge is correct only when its assigned GT endpoints are adjacent.
M–U/U–U edges are false positives under this declared scoring rule. Record
TP/FP/FN, conditional recall/AP/PR, missing endpoints versus missing relations,
edge subtype/confidence, degree histograms, isolates and Betti count errors.
The collector adds acquisition-site summaries, training-defined arc-length
quartiles and the conditional false-positive rate for nonadjacent GT nodes
closer than 5 mm. Closed/parallel GT is rejected rather than silently collapsed.

## Deletion controls

- **Baseline:** existing confidence/class decisions.
- **Oracle:** retain precisely the baseline one-to-one spatially matched queries.
  Keep their assignments fixed; delete all other nodes and incident edges. No
  missing nodes are inserted, and no scores, coordinates or curves change.
- **NMS 0.5 / 1 mm:** confidence-ordered suppression in physical coordinates,
  with query ID breaking score ties. GT-free and fixed before results. Recompute
  diagnostic matching after suppression. Closely spaced legitimate anchors
  may be lost; report recall rather than treating fewer nodes as success.

Keep isolated retained nodes in component counts. Report original common-v2
Branch/Curve F1 and identical-endpoint chords in addition to positional edge
F1. Common-v2 retains its 1 mm/0.25 mm geometry protocol even when the
separate positional accounting uses 2 mm. Per-subject paired bootstrap intervals
use 1,000 fixed-seed resamples; they describe this reused development population.

Three validation examples are selected by fixed manifest order, before looking
at results. Offline Plotly views show uncertain GT paths, query IDs/categories,
segmentation voxel centres and M–M/M–U/U–U edges in 3D.

## Verification

Smoke **1291009** completed the three crafted CPU checks and four training/four
validation model forwards, cache/decoder parity, physical-coordinate checks,
all deletion controls, geometry scoring and frozen-state check. Diagnostic
execution took 50.6 seconds (excluding environment setup).

Saved validation reproduction passed on all four smoke cases with exactly the
same retained query IDs and decoded edges. The larger pilot exposed historical
numerical differences and threshold-sensitive decisions. Historical outputs used
batch sizes 4/32, while this diagnostic uses batch 1; batch size is a plausible
cause, not isolated experimentally. The original 0.03 mm/exact-decision historical
gate failed and is **not** claimed to have passed. Historical comparison now
records all differences descriptively instead of asserting equivalence.

For local_image, historical retained query IDs agree, but two additional edges
cross 0.5 (scores 0.5000819 and 0.5000812). Maximum common-query displacement is
0.07552 mm and maximum compared edge-probability difference is 0.0002821.
Against the previous **batch-1** threshold diagnostic, all **384 patches** exactly
reproduce retained node counts, positional TP/FP/FN and recoverable-edge counts
at 2 mm. Same-forward cache/normal-decoder parity passes separately at 2e-6
probability tolerance, and model state remains unchanged.

Jobs 1291324/1291328/1291329 have Slurm status FAILED because collection failed
after the model stage (nested parent reference path, historical decision gate,
and empty-node shape respectively). Those issues were corrected; CPU collectors
subsequently exited 0 for all three arms. Model stages took approximately
295–300 seconds per arm. No GPU rerun was needed.

## Pilot results

These are 128 reused development-validation patches from 26 subjects, with
3,517 GT nodes and 2,977 GT edges. Positional accounting below uses 2 mm;
common-v2 Branch/Curve metrics retain their original geometry protocol.

| Frozen checkpoint | Node recall | Positional edge F1 | Common Branch F1 | Common Curve F1 |
|---|---:|---:|---:|---:|
| Parent | 0.5073 | 0.0596 | 0.0061 | 0.1172 |
| Continuation control | 0.5118 | 0.0583 | 0.0050 | 0.1179 |
| Local-image refinement | 0.5627 | 0.0737 | 0.0122 | 0.1686 |

### Local-image deletion controls

| Policy | Node recall | Edge TP / FP / FN | Positional edge F1 | Common Curve F1 |
|---|---:|---:|---:|---:|
| Baseline decoding | 0.5627 | 142 / 733 / 2835 | 0.0737 | 0.1686 |
| Oracle remove unmatched nodes | 0.5627 | 142 / 65 / 2835 | 0.0892 | 0.0718 |
| NMS 0.5 mm | 0.5374 | 116 / 611 / 2861 | 0.0626 | 0.1543 |
| NMS 1 mm | 0.4822 | 90 / 478 / 2887 | 0.0508 | 0.1334 |

- **91.1% of false edges involve unmatched nodes** (668/733). However, only
  262/733 (35.7%) touch a duplicate candidate: unmatched does not mean duplicate.
- Oracle deletion improves edge precision but leaves true-edge recall at 4.77%.
  Of all GT edges, 62.0% lack a recovered endpoint, 33.2% have both endpoints
  recovered but no correct predicted relation, and 4.77% are correctly predicted.
  Conditional relation recall is only 12.57% (142/1,130).
- Retained nodes recover 56.27% of GT; independent matching of all 256 raw queries
  recovers 83.62%. This is a diagnostic upper bound using GT, not evidence that
  lowering the confidence threshold alone will achieve that recall.
- Of 3,827 retained nodes, 584 are duplicate candidates, 672 are foreground-supported
  extras and 592 are off-vessel/outside. Duplicate and matched confidence distributions
  overlap strongly (medians 0.9521 and 0.9632).
- Both fixed NMS radii lose true nodes and edges. Subject-bootstrap 95% intervals
  for edge-F1 change are oracle [+0.0123, +0.0195], NMS 0.5 mm
  [−0.0170, −0.0041], and NMS 1 mm [−0.0301, −0.0152].
- Oracle deletion lowers component-count error (19.29→9.40) but also lowers Curve
  F1 (0.1686→0.0718). Spatially unmatched nodes can still support geometric overlap;
  positional graph accuracy and curve coverage must both be reported.

These observations support investigating node selection/recovery and relation
recall together. They do not support simple geometric suppression as a fix, nor
establish that unmatched-negative training alone will resolve connectivity.

Compact evidence is preserved in
[the evidence directory](../evidence/ixi_query_attribution_20261019/README.md).

## Reproduction and outputs

From the repository root:

```bash
sbatch --parsable --time=00:30:00 cluster/jean_zay/ixi_query_diagnostics.slurm smoke local_image
# After smoke and saved-prediction reproduction have succeeded:
python3 cluster/jean_zay/submit_ixi_query_diagnostics.py --smoke-job JOB_ID
python3 cluster/jean_zay/submit_ixi_query_diagnostics.py --smoke-job JOB_ID --submit
```

The submitter defaults to dry-run and rejects an existing submission record.
Each pilot requests one H100/six CPUs for at most two hours and depends on the
smoke succeeding. Existing fixed runtime staging is reused. A CPU collector
runs at the end of each diagnostic without repeating model inference.

Code: `scripts/{diagnose_ixi_queries,ixi_query_diagnostics,collect_ixi_query_diagnostics}.py`;
checks: `tests/test_ixi_query_diagnostics.py`.
Submission record: `experiments/ixi_query_diagnostics/submission.json`.
Logs: `experiments/spline_learning/logs/ixi-query-audit-JOBID.{out,err}`.

Output root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_query_diagnostics_20261019/`.
Each `pilot-JOBID-ARM` contains `manifest.json`, raw per-patch NPZ caches,
`comparison.csv`, `per-patch.csv`, `per-subject.csv`, `nodes.csv`,
`gt-recovery.csv`, `decoded-edges.csv`, `ranking.csv`, PR curves,
`geometry.json`, 3D views and `summary.json`. Successful collection adds
`reproduction.json`, node/edge subtype/site/length summaries,
`subject-bootstrap.json` and `overview.png`.

## Decision gate and remaining work

This round stops at the completed local-refinement pilot and its existing controls,
as requested. No further jobs are submitted. Inspect pilot results and fixed 3D
examples before any later full-validation expansion
(`validation` mode uses all 2,649 validation patches with the same definitions).
Substantial U-edge burden alone does not prove unmatched supervision will work:
oracle pruning must be read alongside unchanged true-edge recall, missing
endpoints, M–M ranking and remaining fragmentation.

Mask-input training, refinement-path ablations, isolated unmatched-negative loss,
matched-negative redesign, and multiscale attention remain unlaunched. P0-E's
full image/label-reliability audit and early/best/final comparison are also pending.
No improved model or completed full-validation attribution is claimed here.

<!-- BEGIN MODERN GRAPH ATTRIBUTION 20261026 -->
## Modern graph/error attribution and long-branch difficulty —26 October2026

**Completed CPU-only analysis; no model inference, training, threshold tuning, test-set evaluation or GPU jobs.** The main comparison reuses the strongest completed, matched SyntheticMRI B0/A-xyz study at its predefined12,000-update endpoint, both fine-tuning seeds364505/92741:128 training-monitor patches/64 subjects and480 validation patches/13 disjoint subjects per model. Both seeds share one source checkpoint. First-seed A-xyz's previously validation-selected9,000 checkpoint is a separate sensitivity analysis, not a replacement chosen from this diagnostic.

IXI adds historical parent, node-head continuation, combined local-image refinement and its **inference-only coordinate component ablation**, on the original256/128 train/validation patches (123/26 subjects). These are not the new fully unfrozen IXI B0/A-xyz models currently owned by another agent. BRAVA is excluded: its ongoing representation study and incomplete SWC labels do not support treating annotation-unmatched structures as confirmed anatomical false positives.

Isolated branch/worktree: `research/graph-error-attribution-20261026` / `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-graph-errors-20261026`. Numerical outputs are under that worktree's `experiments/graph_error_attribution_20261026/`. Original checkpoints/caches and other agents' reports are untouched. Cache archive hashes, checkpoint identities, exact cohort-ID hashes, physical thresholds, train-defined geometry cuts and source-code hashes are retained. No new research report was created.

### Definitions and safeguards

All node/edge decisions use their original0.5 cutoffs. Physical matching remains maximum-cardinality, minimum-distance one-to-one at1/2mm, independent of predicted degree. All raw query pairs and genuine original relation scores are available; there is no score reconstruction from GT. Dense `full_paths`/`path_offsets` with the per-patch `metric` define reference arc length/tortuosity. Curvature is accumulated turning angle after0.5-mm arc resampling, also divided by arc length (rad/mm). GT paths are **offline evaluation references only**, not image/path inputs to a predictor. Quartiles derive solely from the fixed training-monitor GT and are shared across arms/seeds. Candidate counts and prevalence accompany conditional AP; changed oracle query populations do not make their APs directly interchangeable.

MM/MU/UU mean two/one/zero positionally matched endpoints. Duplicate-touch is an overlapping subset, not another additive false-edge class. A GT shortcut joins matched, nonadjacent GT vertices with an annotated path of at least two edges; cross-component errors have no such GT path. Unsupported terminal branches are maximal degree-two chains with an unmatched terminal and no positionally correct GT adjacency along the chain. This is an operational annotation-relative count, not a claim that every such structure lacks a vessel.

Five synthetic-graph tests passed: false-edge partition/cycle accounting, unsupported terminal chains, actual Hungarian versus spatial assignment, the requirement for **two distinct** raw queries to support two endpoints, and confidence-assisted selection preserving coverage without changing scores. A four-patch-per-condition smoke passed. Full tables satisfy FP-partition, FN-budget and Euler identities. Historical142TP/733FP, including668 unmatched-involving and262 duplicate-touch false edges, reproduce **exactly**.

### Graph counts: signed bias and absolute error are different

Validation at2mm. Each `bias /MAE` is a per-patch count. Cycles/components keep isolates; duplicate and unsupported-terminal counts are not removed from predictions.

| Model | Node bias /MAE | Edge bias /MAE | β0 bias /MAE | β1 bias /MAE | Duplicate nodes | Unsupported terminal branches/patch |
|---|---:|---:|---:|---:|---:|---:|
| B0, seed364505 | -0.933 /1.850 | +0.285 /2.440 | +0.306 /0.902 | +1.525 /1.542 | 241 | 1.435 |
| A-xyz, seed364505 | -0.917 /1.846 | +0.279 /2.442 | +0.254 /0.883 | +1.450 /1.471 | 270 | 1.265 |
| B0, seed92741 | -0.917 /1.846 | +0.348 /2.427 | +0.242 /0.871 | +1.506 /1.515 | 247 | 1.452 |
| A-xyz, seed92741 | -0.821 /1.879 | +0.431 /2.510 | +0.273 /0.885 | +1.525 /1.542 | 289 | 1.290 |
| IXI legacy baseline | +2.398 /6.695 | -16.500 /17.422 | +19.117 /19.305 | +0.219 /1.031 | 440 | 2.781 |
| IXI legacy local_image | +2.422 /6.734 | -16.422 /17.375 | +19.086 /19.289 | +0.242 /1.039 | 584 | 2.641 |

SyntheticMRI does **not** globally overpredict node counts: net node bias is approximately−0.82 to−0.93 nodes/patch while false nodes coexist with missed GT anchors. Edge bias is only+0.28 to+0.43, but cycle bias is+1.45 to+1.53. SyntheticMRI terminal-node counts are also **underpredicted** overall (about−3.76 to−3.87/patch), despite1.26–1.45 unsupported terminal branches/patch. Signed counts alone hide these simultaneous omissions/spurious connections. Terminal bias/MAE and all count distributions remain in the per-patch tables. IXI legacy instead has roughly+2.4 nodes but−16.4 edges/patch and severe fragmentation.

### Which false edges dominate?

| Model | FP | MM /MU /UU | Touch spatially unmatched | Touch duplicate | GT shortcuts among MM FP | Touch Hungarian-unassigned |
|---|---:|---:|---:|---:|---:|---:|
| B0, seed364505 | 3226 | 488 /2282 /456 | 84.87% | 16.24% | 325 /488 | 32.55% |
| A-xyz, seed364505 | 2901 | 511 /2053 /337 | 82.39% | 20.20% | 329 /511 | 35.40% |
| B0, seed92741 | 3214 | 465 /2287 /462 | 85.53% | 16.77% | 296 /465 | 32.14% |
| A-xyz, seed92741 | 2954 | 497 /2086 /371 | 83.18% | 21.90% | 313 /497 | 35.38% |
| IXI legacy baseline | 753 | 63 /418 /272 | 91.63% | 29.88% | 42 /63 | 52.86% |
| IXI legacy local_image | 733 | 65 /425 /243 | 91.13% | 35.74% | 41 /65 | 56.07% |

The old91.1% finding remains correct on the same IXI cache, but it is **not a universal fraction**: modern SyntheticMRI gives84.87–85.53% for B0 and82.39–83.18% for A-xyz at2mm. At1mm the corresponding fractions are98.12–98.15% and96.71–96.98%, demonstrating tolerance sensitivity. A-xyz reduces total FP versus B0 in both seeds (3226→2901;3214→2954) while total predicted edge counts barely change (7816→7813;7846→7886). This is not simply pruning to fewer edges. Duplicate-touch explains only about20–22% of A-xyz FP; duplicate suppression alone is not the dominant remedy.

Only about35% of modern A-xyz FP involve a **Hungarian-unassigned** query, despite roughly83% involving a **spatially unmatched** endpoint. Most false edges therefore cannot be attributed solely to never supervising Hungarian-unassigned queries. These are different matching concepts. Among MM false edges,325/488 and296/465 (B0) or329/511 and313/497 (A-xyz) are GT shortcuts; the rest cross annotated components.

### Missing-edge budget: endpoint availability versus relation failure

Rows below partition **all missed GT edges**, not just recovered-endpoint relations. Raw availability means there exist two distinct raw queries within tolerance of the two endpoints, assessed separately for each GT edge. It is an upper bound and does not promise a globally realizable or relation-compatible graph.

| Model | Missing edges | No feasible raw pair | Removed by confidence | Retained matching ambiguity | Both localized, relation absent |
|---|---:|---:|---:|---:|---:|
| B0, seed364505 | 3089 | 41.66% | 46.36% | 4.99% | 6.99% |
| A-xyz, seed364505 | 2767 | 34.30% | 50.92% | 5.46% | 9.32% |
| B0, seed92741 | 3047 | 42.21% | 46.14% | 4.99% | 6.66% |
| A-xyz, seed92741 | 2747 | 33.45% | 52.28% | 5.06% | 9.21% |
| IXI legacy baseline | 2865 | 28.73% | 28.69% | 15.04% | 27.54% |
| IXI legacy local_image | 2835 | 19.47% | 29.91% | 15.77% | 34.85% |

For A-xyz, **90.68–90.79% of false-negative edges lack a recovered endpoint**. Approximately51–52% of missed edges are spatially feasible before confidence filtering,33–34% lack a feasible raw pair, and5% are retained-set correspondence ambiguity; only9.2–9.3% are direct relation misses after both endpoints are recovered. This ranks spatial proposal/selection quality above default-score conditional relation failure for these SyntheticMRI models. Lowering the threshold is not established as a remedy: the same normal outputs already have substantial unmatched-node FP. In legacy IXI, recovered-endpoint relation misses are34.85% of FN, and endpoint recovery alone is insufficient.

### Are long or tortuous branches actually harder?

SyntheticMRI training arc quartiles are4.831/10.216/17.649mm; tortuosity cuts1.00575/1.01445/1.02835. IXI legacy cuts differ (arc1.144/3.384/8.775mm). These reflect different prepared populations and cannot be pooled as the same “long branch.”

| Model | Conditional recall: shortest /longest arc quartile | Long−short [95% subject interval] | Full-GT recall: shortest /longest |
|---|---:|---:|---:|
| B0, seed364505 | 0.9922 /0.9063 | -0.0859 [-0.1053, -0.0675] | 0.4628 /0.6980 |
| A-xyz, seed364505 | 0.9926 /0.8994 | -0.0932 [-0.1112, -0.0761] | 0.4893 /0.7321 |
| B0, seed92741 | 0.9922 /0.9152 | -0.0771 [-0.0936, -0.0612] | 0.4659 /0.6965 |
| A-xyz, seed92741 | 0.9905 /0.8974 | -0.0932 [-0.1151, -0.0712] | 0.4893 /0.7245 |
| IXI legacy baseline | 0.1636 /0.0428 | -0.1209 [-0.1905, -0.0641] | 0.0520 /0.0122 |
| IXI legacy local_image | 0.1562 /0.0380 | -0.1182 [-0.1755, -0.0645] | 0.0601 /0.0138 |

Endpoint-separation quartiles are also saved separately (SyntheticMRI training cuts4.783/9.950/17.100mm). First-seed A-xyz conditional recall is99.26% in the closest GT endpoint-distance quartile versus90.08% in the farthest; full-GT recall is48.94% versus73.53%. Thus the endpoint-distance result agrees with the arc-length distinction, not a blanket “long is worse” pipeline claim.

**Conditional difficulty is not complete-pipeline difficulty.** Modern A-xyz correctly connects about90% of long branches versus99% of short ones once endpoints are recovered. Yet its **full-GT recall is higher for long branches** (about73% versus49%), because short-branch endpoint pairs are recovered much less often (about49% versus81%). It would be incorrect to conclude that long branches dominate total reconstruction failure.

| Model | High−low tortuosity, adjusted for endpoint-distance quartile | High−low curvature, adjusted for arc-length quartile |
|---|---:|---:|
| B0, seed364505 | -0.0263 [-0.0389, -0.0106] | -0.0176 [-0.0321, -0.0002] |
| A-xyz, seed364505 | -0.0288 [-0.0426, -0.0138] | -0.0167 [-0.0300, -0.0052] |
| B0, seed92741 | -0.0256 [-0.0375, -0.0125] | -0.0087 [-0.0215, +0.0043] |
| A-xyz, seed92741 | -0.0278 [-0.0374, -0.0186] | -0.0153 [-0.0280, -0.0042] |
| IXI legacy baseline | -0.1470 [-0.3333, -0.0099] | +0.0297 [-0.1756, +0.3524] |
| IXI legacy local_image | +0.1688 [-0.1667, +0.3466] | +0.0181 [-0.1369, +0.2476] |

These are distance-stratified **associations**, not causal curvature effects. SyntheticMRI retains a smaller2.6–2.9 percentage-point high-tortuosity deficit after distance adjustment and A-xyz retains about1.5–1.7 points for curvature after arc adjustment. B0 curvature estimates span0.9–1.8 points, and one seed’s interval includes zero. IXI curvature/tortuosity adjustment has sparse/empty strata and broad intervals; a distinct curvature effect is not established there. Curvature depends on the fixed0.5-mm numerical sampling; original dense paths, rather than fittedK8 curves, are used.

False-relation distance rates use **all eligible negative pairs**. For A-xyz, matched nonadjacent pairs under5mm are falsely connected in57/119 and60/128 cases (47.9%/46.9%); beyond20mm this is167/58,926 and161/59,169 (0.283%/0.272%). The near-negative rate is much larger, but those pairs are rare: a histogram of selected short edges alone would obscure the denominator. Full1/2-mm fixed-distance tables and GT-geometry quartiles are saved.

### Training versus inference exposure and label confidence

The actual matcher cost is **5 × normalized L1 −2 × object probability**. The original matched-pair criterion keeps positive GT adjacencies, samples/caps only negatives at9999, randomizes endpoint order and applies the configured balancing (SyntheticMRI CE ratio upsampling0.15). There is no positive length cap or length-based rejection. Inference scores all pairs among confidence-retained queries using the cached symmetric orientation logits. The new accounting reconstructs the **potential supervision universe at saved training/validation snapshots**; it does not invent a log of stochastic pairs seen during every historical optimization step.

All four primary SyntheticMRI cases replay stored Hungarian distances exactly over the complete608-patch caches. One patch in the secondary9k A-xyz cache has a distance-replay disagreement (max1.467mm), so its Hungarian exposure interpretation is **excluded from the primary matching conclusion**; its decoded graph/geometry sensitivity results are unaffected. Exact assignment IDs were not stored in the old IXI caches: their configured matcher is reconstructed, without claiming historical bitwise assignment verification.

Operational “unambiguous” means each endpoint is within tolerance of exactly one GT anchor and the two anchors differ. Possible-positive pairs are adjacent GT anchors; topology-negatives are nonadjacent anchors; same-anchor/zero-anchor/multiple-anchor cases are ambiguous for this audit. Alternatives representing the same GT node can still be duplicated queries: potential positives are **query-pair occurrences**, not additional anatomical edges. Hungarian assignment alone never certifies correct localization or anatomical truth.

| Training-monitor snapshot | Matched pair pool | GT-positive /negative labels | Spatially agreeing positive /negative | Ambiguous association pairs | Discordant negative→positive /positive→negative |
|---|---:|---:|---:|---:|---:|
| B0, seed364505 | 30626 | 2132 /28494 | 1254 /15404 | 13906 | 34 /28 |
| A-xyz, seed364505 | 30626 | 2132 /28494 | 1327 /16297 | 12926 | 42 /34 |
| B0, seed92741 | 30626 | 2132 /28494 | 1234 /15261 | 14075 | 27 /29 |
| A-xyz, seed92741 | 30626 | 2132 /28494 | 1347 /16487 | 12722 | 47 /23 |
| IXI legacy baseline | 115540 | 4837 /110703 | 483 /10620 | 104150 | 186 /101 |
| IXI legacy local_image | 115540 | 4837 /110703 | 561 /11614 | 103074 | 195 /96 |

For first-seed A-xyz validation,14,118/99,280 inference candidate pairs involve a Hungarian-unassigned query. Among them253 are potentially positive by unique spatial-anchor association. The matched pair pool also contains169 nominal negatives that are spatially adjacent positives under the positional audit. These are snapshot assignment/representative ambiguities, not proof of incorrect GT supervision; blindly labeling every Hungarian-unassigned query pair negative risks suppressing valid alternatives. Full raw pair populations are much larger than normal inference and are kept separate.

### Counterfactual node selection: scores/coordinates never replaced

Oracles use GT only to select one-to-one candidates within2mm. “Nearest” minimizes spatial error; “confidence” uses a confidence-preferring cost `1−p +1e−5×distance/(1+tolerance)` among maximum-cardinality valid assignments. The small distance term can break near-ties; this is not an optimization of relation accuracy. Neither supplies GT coordinates/edges or optimizes selection using relation targets. The difference tests **representative/query compatibility** as well as coverage, and neither is deployable.

| Model | Normal edge F1 | Delete unmatched | Raw nearest oracle | Raw confidence oracle | Conditional recall normal /nearest /confidence |
|---|---:|---:|---:|---:|---:|
| B0, seed364505 | 0.5924 | 0.7196 | 0.7219 | 0.8157 | 0.9551 /0.7355 /0.9163 |
| A-xyz, seed364505 | 0.6341 | 0.7498 | 0.7200 | 0.8342 | 0.9501 /0.6998 /0.9147 |
| B0, seed92741 | 0.5967 | 0.7251 | 0.7272 | 0.8165 | 0.9580 /0.7439 /0.9179 |
| A-xyz, seed92741 | 0.6337 | 0.7525 | 0.7294 | 0.8376 | 0.9512 /0.7097 /0.9171 |
| IXI legacy baseline | 0.0583 | 0.0711 | 0.0936 | 0.1440 | 0.1243 /0.0777 /0.1247 |
| IXI legacy local_image | 0.0737 | 0.0892 | 0.0999 | 0.1584 | 0.1257 /0.0722 /0.1211 |

At identical maximum raw node coverage (A-xyz92.32%/92.52%), confidence-assisted selection gives edge F1 **0.834/0.838**, versus nearest selection0.720/0.729 and normal0.634/0.634. Geometrically closer alternative queries can have worse learned relations. A poor nearest-oracle score is therefore not by itself proof that extra recovered nodes cannot help, or that the relation head has universally poor ranking. Normal conditional AP is about0.976–0.979 on these SyntheticMRI populations; candidate counts/prevalence for the other oracle APs are saved separately. Legacy IXI confidence-oracle edge F1 remains only0.158 and conditional recall0.121, so its node-selection problem coexists with a large relation/calibration limitation.

Among first/second-seed A-xyz localized direct misses,130/258 and124/253 have an indirect predicted route; only23/258 and25/253 admit a degree-two-only chain without another mapped GT anchor. These are not automatically geometrically correct routes (no new path-shape claim). In legacy IXI the corresponding counts are30/988 and15/988. Contracting extra degree-two chains alone cannot resolve most missing direct connections.

### Quantitative priority ranking and interpretation

1. **SyntheticMRI: spatial proposal/selection/correspondence quality is dominant.** It accounts for90.7–90.8% of A-xyz missed edges and unmatched endpoints touch82.4–83.2% of its false edges. Net node/edge counts and duplicate suppression are insufficient diagnoses. The controlled two-seed **endpoint-local CNN coordinate refinement** has the strongest direct experimental support: it converts many existing predictions to correct correspondences rather than merely producing fewer edges. A deployable confidence/uniqueness improvement remains unproven; the earlier KNN verifier failed and should not be conflated with image-conditioned A-cls, currently under separate investigation.
2. **Residual relation discrimination is concentrated, not uniform.** Long/tortuous edges lose conditional recall, and nearby nonadjacent pairs have high false-positive rates. However direct relation misses after endpoint recovery are only9.2–9.3% of total A-xyz FN; short branches have worse full-pipeline recall. A small path/connection-context diagnostic is justified for that specific residual, with endpoint-only and token controls. The [separate early image-relation evidence](9-ixi-image-aware-relation-learning.md) supports testing a short predefined schedule, not another long fit or an immediate multiscale redesign. These attribution associations alone do not prove path features will help.
3. **Review supervision quality before adding unmatched negatives.** Approximately65% of A-xyz FP already involve two Hungarian-assigned queries; unmatched-involving pools also contain spatially plausible alternatives. Better relation supervision should first preserve positives and audit duplicate/assignment ambiguity and near-nonadjacent versus long-positive contrasts. A fixed-budget matched-versus-unmatched sampling experiment could distinguish these mechanisms; blanket unmatched-negative mining is not supported as the primary remedy.
4. **Legacy IXI needs node and relation recovery together.** The old91.1% spatial-unmatched observation persists, but even GT-assisted confidence selection leaves low relation recovery. This cannot be extrapolated into a result for the ongoing, stronger IXI transfer experiment. Wait for that independently owned batch before selecting an IXI architecture/loss change.

The first seed’s selected9k A-xyz graph sensitivity is consistent (edge F10.6373, spatial-unmatched FP82.60% versus final0.6341/82.39%). No seed/checkpoint is selected from this new attribution. Paired geometry/FP intervals resample whole subjects2,000 times (seed261026), omit training-initialization uncertainty and multiplicity, and describe reused development subjects. Source annotations/extraction uncertainty, candidate-dependent AP prevalence and conditional-selection bias limit causal interpretation. No claim about complete BRAVA anatomy is made.

### Outputs and reproduction

Main tables: `summary.json`, `aggregate.csv`, `geometry_strata.csv`, `geometry_intervals` insideJSON, `distance.csv`, `populations.csv`, `counterfactuals.csv`, `ranking.csv`; geometry-adjusted uncertainty: `adjusted-geometry.json`; confidence oracle/indirect checks: `counterfactual-extensions.json`. Compressed per-case per-patch, per-GT-edge and per-false-edge tables preserve detailed strata. Exact source inputs and checkpoint hashes are in `input-identities.json`; source audit/plan/smoke outputs are retained. All are under the isolated experiment directory above. No Slurm or GPU jobs were submitted.

From the parent `projects` directory, print the saved result summary:

```bash
python3 gnbm-graph-errors-20261026/scripts/show_graph_errors.py
```

<!-- END MODERN GRAPH ATTRIBUTION 20261026 -->

<!-- BEGIN ORACLE REPRESENTATIVES 20261026 -->
## Why confidence-preferred representatives outperform nearest queries — 26 October 2026

**Completed diagnostic, not a new model.** The advantage is caused by changing query representatives while preserving recovered GT vertices. Confidence-preferred representatives are slightly less localized, but their unchanged relation scores discriminate connectivity substantially better. They also agree much more often with the current training-style Hungarian assignment. A small frozen-head intervention supports information beyond the scalar node-classification direction; it does not identify particular semantic latent dimensions.

### Controlled population and provenance

Reused the exact two modern SyntheticMRI A-xyz continuations, seeds 364505 and 92741, fixed step 12,000, original node/edge cutoffs 0.5 and physical 1/2-mm matching. No weights, coordinates, relation logits, targets, thresholds or splits were altered. Each seed has 128 training-monitor patches from **32 subjects**, and all 480 validation patches from 13 disjoint subjects. The source model trained on 25,000 patches; these 128 monitor patches are not the full training population. This corrects any ambiguity between the full training subject count and the actually cached monitor subset. No test data or BRAVA data were used.

One-to-one nearest matching and the existing confidence-preferred assignment both maximize legal cardinality. The latter minimizes `1 - p_object + 1e-5 * distance_mm / (1 + tolerance_mm)` within tolerance; its tiny distance tie-break means it is confidence-preferring, not strict lexicographic confidence maximization. Both use original predicted coordinates and existing pair scores; neither inserts GT nodes/edges.

Checkpoint identities, inherited from the prior checksum audit:

| Seed | Step | Checkpoint SHA-256 |
|---|---:|---|
| 364505 | 12,000 | `a1210fe958dffe0e588d38222cb62bf48b58e9b238fc57c027ec1c7a795277c9` |
| 92741 | 12,000 | `9646b7b266dec3463d94d16cfbcef0332044a9bf0d599c2319a2eff2c93d3e53` |

All numeric outputs and source snapshots are isolated in `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-graph-errors-20261026/experiments/oracle_representatives_20261026`. Cache ZIP hashes were rechecked; Hungarian-distance replay agrees for all primary patches. Native mask axes were checked against SimpleITK. Coordinates are mapped back with `voxel = (physical @ inverse(metric.T)) * native_shape`, including signed metric axes. Foreground tests use the existing synthetic vessel masks **only offline**: nearest-voxel membership, trilinear support, and interpolated physical EDT to foreground voxel centers. The EDT is not an exact distance to a continuous vessel surface.

### Representative choices

At 2 mm both oracles recover **exactly the same GT vertices in every patch**, not merely equal counts. Thus their eligible GT edges, pair identities and prevalence are identical. At 1 mm this also holds throughout validation; one training patch in seed 364505 exchanges one recovered GT vertex, so shared-GT controls are saved separately.

| Seed, validation 2 mm | Changed / recovered representatives | Nearest error | Confidence error | Mean error increase, subject 95% CI | Nearest / confidence node score |
|---|---:|---:|---:|---|---|
| 364505 | 3073/9306 (33.02%) | 0.675 mm | 1.061 mm | 0.387 mm [0.3752, 0.3977] | 0.203 / 0.762 |
| 92741 | 3057/9326 (32.78%) | 0.681 mm | 1.058 mm | 0.377 mm [0.3657, 0.3892] | 0.205 / 0.761 |

About 99.2% of changed confidence choices are farther from their assigned GT; only 28.7–29.6% are farther by over 0.5 mm. Every selected oracle representative still satisfies its physical tolerance. The two alternatives are about 0.92–0.94 mm apart and have about 3.45–3.47 raw candidate queries per GT anchor. These are predominantly nearby, vessel-supported alternatives, not evidence that far off-vessel queries reconstruct better.

| Seed | Foreground: nearest / confidence | Foreground distance: nearest / confidence | Hungarian assigned, any GT | Hungarian assigned to this GT |
|---|---|---|---|---|
| 364505 | 96.23% / 96.94% | 0.156 / 0.104 mm | 6.41% / 95.12% | 3.58% / 90.01% |
| 92741 | 96.01% / 96.93% | 0.158 / 0.106 mm | 5.99% / 95.22% | 3.76% / 89.99% |

Nearest choices are below the normal confidence cutoff in 94.7% of changed cases, versus about 11% of confidence choices. The nearest representative is a geometric canonical winner; the alternative would be called a duplicate under that canonical assignment, despite often carrying stronger relation predictions. There is no intrinsic ground-truth label declaring one nearby query the uniquely valid representative. Both oracle graphs have zero unmatched/duplicate selected nodes **by construction**.

Changed representatives have mean GT degree about 1.48. Approximately 34% of recovered terminal anchors change representatives, versus 29–30% of degree-3 junctions; most changed cases are patch-boundary terminals. Degree, boundary flag, incident-edge count, nearby-anchor ambiguity, exact query IDs and per-query frequencies are saved. Confidence choices use 98/100 distinct query IDs across changed validation vertices, versus 118/119 for nearest choices; this does not establish a universal good-query-ID rule.

**Assignment interpretation:** these are retrospective assignments from the actual normalized-L1/classification Hungarian cost, not a replay of every assignment during optimization. Confidence explicitly enters that cost, so the association is partly built into the matcher. It supports query-role consistency, but cannot alone prove that assignment history caused the relation advantage.

### Complete graph and relation ranking

| Seed | Selection, validation 2 mm | Node recall | Edge TP / FP | Edge F1 | Conditional recall | Conditional AP | Duplicates | β₀ / β₁ signed error per patch |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 364505 | normal | 0.7991 | 4912 / 2901 | 0.6341 | 0.9501 | 0.9776 | 270 | +0.254 / +1.450 |
| 364505 | nearest | 0.9232 | 4686 / 652 | 0.7200 | 0.6998 | 0.9025 | 0 | +3.906 / +0.642 |
| 364505 | confidence | 0.9232 | 6125 / 880 | 0.8342 | 0.9147 | 0.9598 | 0 | +0.802 / +1.010 |
| 92741 | normal | 0.8008 | 4932 / 2954 | 0.6337 | 0.9512 | 0.9780 | 289 | +0.273 / +1.525 |
| 92741 | nearest | 0.9252 | 4776 / 640 | 0.7294 | 0.7097 | 0.9063 | 0 | +3.810 / +0.667 |
| 92741 | confidence | 0.9252 | 6172 / 886 | 0.8376 | 0.9171 | 0.9612 | 0 | +0.796 / +1.073 |

Oracle node precision is 1 by construction; normal precision is 0.8356/0.8334. Raw-query recall equals oracle recall. At 2 mm the identical oracle candidate populations contain 90,994 pairs with 6,696 positives (seed 364505), and 91,439 with 6,730 positives (seed 92741), prevalence about 7.36%. Normal conditional AP uses only its own spatially matched candidates; its higher value is not a fair same-population comparison to the expanded oracle graph. All-retained AP, conditional AP and candidate counts are stored separately.

| Seed | Confidence − nearest edge F1, subject CI | Confidence − nearest same-population AP, subject CI |
|---|---|---|
| 364505 | 0.1143 [0.1069, 0.1219] | 0.0574 [0.0535, 0.0613] |
| 92741 | 0.1082 [0.0997, 0.1160] | 0.0549 [0.0516, 0.0588] |

Confidence selection increases true edges by 1,439/1,396 and false edges by 228/246. It reduces fragmentation strongly, while cycle error rises from about +0.64/+0.67 to +1.01/+1.07. This is an improved precision–recall trade-off with better ranking, not uniform suppression of false connections or evidence of perfect topology.

| Seed / split / tolerance | Raw/oracle node recall | Nearest → confidence edge F1 | Nearest → confidence conditional recall |
|---|---:|---:|---:|
| 364505 / train / 2 mm | 0.9226 | 0.7215 → 0.8342 | 0.7044 → 0.9189 |
| 364505 / train / 1 mm | 0.7055 | 0.5163 → 0.5865 | 0.7056 → 0.8454 |
| 364505 / val / 1 mm | 0.6991 | 0.5018 → 0.5821 | 0.6869 → 0.8480 |
| 92741 / train / 2 mm | 0.9255 | 0.7206 → 0.8356 | 0.7022 → 0.9220 |
| 92741 / train / 1 mm | 0.6979 | 0.5065 → 0.5901 | 0.6999 → 0.8702 |
| 92741 / val / 1 mm | 0.7006 | 0.5084 → 0.5848 | 0.6922 → 0.8466 |

### Same-neighbor and one-query interventions

For each changed GT anchor, the first control holds its neighbor query fixed and uses only neighbors whose representative is identical in both oracles. Positive, nearby nonadjacent (<5 mm), middle-distance, and distant unrelated (≥20 mm) GT pairs are evaluated with identical labels. This isolates the changed endpoint token/query from changes at its neighbors.

| Seed, validation 2 mm | Fixed-neighbor pairs / positives | AP nearest → confidence | AP difference, subject CI | Mean true score | Mean nearby-negative score | Mean distant-negative score |
|---|---:|---:|---|---|---|---|
| 364505 | 39250 / 2746 | 0.8728 → 0.9585 | 0.0857 [0.0781, 0.0931] | 0.532 → 0.870 | 0.494 → 0.547 | 0.003 → 0.005 |
| 92741 | 39464 / 2739 | 0.8734 → 0.9587 | 0.0854 [0.0752, 0.0961] | 0.538 → 0.870 | 0.497 → 0.552 | 0.003 → 0.005 |

True-edge scores rise by about 0.33–0.34, versus 0.05–0.06 for scarce nearby nonadjacent negatives and about 0.002 for distant negatives. The same-anchor positive-minus-near-negative margin improves by +0.1168 [0.0643, 0.1668] and +0.1358 [0.1015, 0.1774], on only 72/73 eligible changed vertices. Nearby-negative acceptance still worsens: 45.0%→58.0% and 53.0%→61.4%. Ranking improvement and remaining near-vessel confusion coexist. A common monotonic probability recalibration cannot explain an AP increase on identical labels.

Training-monitor fixed-neighbor AP also improves: 0.8993→0.9651 and 0.9048→0.9706 at 2 mm. These are optimized-on samples, not independent generalization estimates.

| Seed | Valid single-query swaps | Incident TP change | Incident FP change | All simultaneous swaps: TP, one / both endpoints changed | FP, one / both endpoints changed |
|---|---:|---:|---:|---|---|
| 364505 | 2983 | +1259 | +166 | +1001 / +438 | +175 / +53 |
| 92741 | 2963 | +1248 | +182 | +965 / +431 | +179 / +67 |

Single swaps keep every other nearest representative fixed; alternatives already occupied by another GT vertex are excluded. Incident counts sum separate interventions and must not be mistaken for one jointly realizable graph. Simultaneous decomposition accounts **exactly** for total TP/FP changes. For pairs with both endpoints changed, the four-score interaction `Scc - Scn - Snc + Snn` is saved. About 3.6–3.9% of valid such pairs have a nonadditive threshold-decision interaction. Query-collision cases (89/94 at 2-mm validation) retain factual nearest/both scores but exclude invalid cross combinations.

Assignment-stratified fixed-neighbor results sharpen the explanation. When confidence is assigned to this GT, AP increases 0.8827→0.9697 and 0.8876→0.9702; this accounts for almost all added fixed-neighbor TP. When **neither** representative is assigned to that GT, confidence actually reduces AP: 0.8323→0.7594 and 0.8019→0.7097. High confidence is therefore not a universal certificate of better relation content. These are conditional associations, not an intervention on the training matcher.

### Decoder embedding intervention

Original full-query ZIPs did not cache embeddings. CPU replay failed the predefined parity gate (maximum coordinate differences up to 0.22 mm and relation-probability differences up to 0.0143); those features were excluded. A tiny H100 replay, job **1597409**, completed successfully in **46 seconds**, with a 10-minute wall limit. All 8/8 seed-patch forwards reproduced coordinates, confidences and all 7,140 pair probabilities **exactly** at inference batch 1. There was no training.

This is four unique patches replayed for both seeds: two patches from one training subject and two from one validation subject, deterministically the first changed-representative patches. It supplies 55 alternative-representative comparisons and 529 fixed-neighbor pair occurrences overall; it is a mechanistic illustration, not a population embedding estimate. Full 122×552 decoder tensors (120 object + two relation tokens) are retained.

The frozen relation MLP consumes the two object embeddings and shared relation tokens, averaged over endpoint order; it receives no predicted coordinates or explicit node confidence. For each changed representative, decompose its embedding change into the direction `w_foreground - w_background` of the original linear classifier and the orthogonal remainder. Feed either change alone into the unchanged relation head, keeping neighbor/relation tokens fixed. The classification-direction swap transfers the node log-odds; the orthogonal swap preserves them mathematically. Both are off-manifold latent interventions and do not establish semantic latent axes.

| Seed, illustrative validation | Pairs / positives | AP nearest | AP confidence | AP classifier-direction only | AP orthogonal-change only | True score: nearest / confidence / class-only / orthogonal-only |
|---|---:|---:|---:|---:|---:|---|
| 364505 | 162 / 14 | 0.8118 | 0.9528 | 0.8455 | 0.9565 | 0.516 / 0.897 / 0.548 / 0.883 |
| 92741 | 121 / 11 | 0.9242 | 0.9650 | 0.9153 | 0.9650 | 0.494 / 0.886 / 0.527 / 0.871 |

Validation alternative embeddings have mean cosine similarity 0.871/0.852. The classifier axis accounts for about 8.0%/6.6% of their squared difference, but moving along it reproduces only a small fraction of the true-edge score improvement. The orthogonal change recovers most of the relation advantage. Training illustrations are consistent; one is AP-saturated at 1.0. Neighbor cosine similarities and norms are saved, without interpreting them as a connectivity classifier. This directly supports a relation-relevant **embedding-content** difference on the replay examples, beyond changing the scalar confidence logit. It does not show that a new embedding refinement module is needed.

Applying only the unchanged coordinate MLP to those saved exact tokens also reconstructs Stage-1 coordinates without another image forward. In the small validation illustration, nearest representatives improve from 1.633→0.799 mm and 1.362→0.784 mm after local refinement; confidence representatives improve from 1.757→1.064 mm and 1.605→1.285 mm. Both types benefit geometrically. Nearest representatives already have smaller mean Stage-1 errors here; this does not establish that local refinement created the representative discrepancy. These coordinate diagnostics cover the same small replay sample, not the full population.

### GT-free inference diagnostic and decision

One fixed diagnostic applies confidence-ordered spatial NMS at 1 mm to the normally retained nodes, preserving all remaining scores and coordinates. This modern SyntheticMRI check is separate from the historical unsuccessful IXI NMS result. A top-confidence control keeps exactly the same number of nodes per patch as NMS. No radius/threshold search or GT input is used.

| Seed | Normal / NMS / matched-count edge F1 | NMS minus matched-count, subject CI | Normal → NMS node recall | Normal → NMS duplicates |
|---|---|---|---|---|
| 364505 | 0.6341 / 0.6377 / 0.6375 | 0.0002 [-0.0033, 0.0041] | 0.7991 → 0.7853 | 270 → 126 |
| 92741 | 0.6337 / 0.6372 / 0.6351 | 0.0021 [-0.0013, 0.0054] | 0.8008 → 0.7866 | 289 → 139 |

NMS gives only about +0.0035 Edge F1 over normal, loses about 1.4 percentage points of node recall, and has no established advantage over the matched-count confidence control. It does not recover the oracle gap. A scalar confidence preference is an available inference signal, but oracle association supplies essential GT uniqueness/coverage knowledge that this simple rule does not replace. No oracle is reported as deployable inference.

**Best-supported explanation:** a mixture of representative selection, query-assignment consistency and different relation-relevant token content. Better spatial localization is ruled out as the explanation for this particular oracle gap. Better confidence alone is inadequate: it correlates with the usually supervised query role, but fails in the neither-assigned stratum, and changing its classifier direction alone largely fails to reproduce the relation effect. Current assignment status is not the same as historical supervision, so causality through training assignment remains a hypothesis.

**Next priority: C — relation-aware query selection/uniqueness**, investigated as a controlled GT-free diagnostic before proposing an architecture. Existing queries already contain much stronger connectivity information when the appropriate representatives are chosen. Keep localization improvements, and test whether inference-available relation consistency can select them without inventing extra connections. B, relation-specific embedding/supervision refinement, is the fallback if such selection fails; A, confidence-only correction, is not the leading mechanism identified here. No new architecture, detector loss or training campaign was implemented.

### Uncertainty, artifacts and reproduction

Intervals resample whole subjects with shared draws across compared conditions: 2,000 for graph F1/localization/margins and 500 for exact, tie-aware AP, seed 261026. The two continuations share a source model; they are not independent source-training replications. Validation was reused exploratorily, no multiplicity correction was applied, and the small embedding sample supplies no subject-level generalization claim. Physical foreground support and GT proximity do not make two queries interchangeable for the trained relation head.

Saved artifacts include `summary.json`, `representatives.csv.gz`, `stable-neighbor-pairs.csv.gz`, `single-swaps.csv.gz`, `interactions.csv.gz`, `decomposition.json`, `assignment-strata.json`, `query-identities.csv`, `conditional-ranking.json`, full per-patch graph/candidate tables, margins, exact replay embeddings/interventions and audit files. Source/checkpoint/cache identities and failed CPU parity checks are preserved. Existing experiments and concurrent reports were left untouched.

From the parent `projects` directory, print completed results without inference or training:

```bash
python3 gnbm-graph-errors-20261026/scripts/show_oracle_representatives.py
```

To regenerate CPU tables from the original fixed caches, use the protected research Python in the isolated worktree: `scripts/oracle_representatives.py`, `scripts/oracle_representative_supplements.py`, then `scripts/summarize_oracle_representatives.py`. Exact embedding features are already saved; do not resubmit the replay to print results.

<!-- END ORACLE REPRESENTATIVES 20261026 -->

<!-- BEGIN N08 GT FREE QUERY SELECTION 20261026 -->
## N08 — GT-free relation-aware representative selection — 26 October 2026

**Completed, CPU-only. No consistent benefit over count-matched controls; stop further search.** This follows N06’s GT-assisted oracle analysis and does not turn its 0.834–0.838 Edge F1 into deployable performance. No detector training, relation head, graph loss, architecture, new GPU inference or saved-prediction changes were introduced.

### Fixed data, inference interface and controls

The same two SyntheticMRI A-xyz checkpoints at update 12,000 are used: seeds 364505 and 92741, with the N06 checkpoint/cache checksums preserved in `input-identities.json`. Per seed: 128 training-monitor patches/32 subjects and 480 validation patches/13 disjoint subjects. These monitor patches were used during detector optimization and are not an independent generalization estimate. Test data were not accessed.

Selectors receive **only** predicted physical coordinates, object confidence, original unordered query pairs and their saved symmetric relation probabilities. The caches contain probabilities after averaging the two endpoint-order logits, rather than the original two-component logits; these probabilities retain the binary ranking/decision information required here. No GT coordinate, edge, node count, Hungarian assignment, vessel mask or oracle membership enters grouping, reference construction or query choice. GT is used afterward for training-only setting selection and offline evaluation.

Every selected vertex is an existing query, with its original coordinates/confidence. Every selected pair keeps its original relation score and the 0.5 edge cutoff. Alternative policies replace the node-selection decision; the 0.5 object cutoff is **not reapplied afterward**, which would discard rescued queries. Their node budget comes from normal inference or confidence NMS, never from GT.

The predeclared small candidate set was:

1. **Normal:** original object-confidence cutoff 0.5.
2. **Confidence NMS:** confidence order, 1-mm radius, original retained candidates. This reproduces the N06 control.
3. **Naive global support:** confidence plus uncapped sum of relation probabilities to fixed reference queries; intentionally retained as a hub/self-support risk control.
4. **Guarded global support:** bounded strongest-reference support, a reference-neighbor rank check, and a nonreciprocal-support penalty. References are up to 12 confidence-NMS queries with confidence ≥0.75. References within 2 mm are excluded from support; reference degree normalizes hub contributions. A single fixed scaffold prevents iterative amplification by newly chosen queries, but cannot certify that references are anatomically correct.
5. **Local profile selection:** one representative per original confidence-NMS seed, chosen inside disjoint cells with maximum radius `min(1 mm, 0.49 × nearest-seed separation)`. The score combines confidence with agreement to fixed predicted external neighbors and a strongest-negative profile penalty. These cells keep conservative local identities; they do not identify GT vertices.
6. **Protected NMS:** a predicted connection above 0.5 between nearby queries, or mean external-profile disagreement above 0.25, vetoes suppression. This tests preserving short predicted branches/distinct junctions, but can preserve false duplicate connections as well.

Global candidates require confidence ≥0.1. Each family uses a common weight from {0.25, 1}, chosen using training data only. The criterion is mean two-seed training Edge F1 advantage over the stronger confidence/spatial count control, with at most a one-percentage-point training node-recall loss. Ties prefer the smaller weight. No radius, threshold or reference-set search was performed. The selected weights were naive 0.25, guarded 0.25 and local profile 1.0. The **primary local-profile policy was frozen before validation** because it was closest to neutral; none of the families consistently beat both training controls.

All policies receive both controls at exactly the same predicted node count **per patch**: top original confidence, and confidence NMS truncated/backfilled by original confidence if needed. The pure confidence-NMS count controls for global/local policies are exactly the original NMS sets. No MST, cycle deletion, graph-tree assumption, hard anatomical degree cap or GT edge insertion is used.

### Validation reconstruction at 2 mm

| Seed | Policy | Selected nodes | Node P / R / F1 | Edge P / R / F1 | Edge TP / FP | Duplicates |
|---|---|---:|---|---|---:|---:|
| 364505 | normal | 9640 | 0.8356 / 0.7991 / 0.8169 | 0.6287 / 0.6397 / 0.6341 | 4912 / 2901 | 270 |
| 364505 | confidence_nms | 9312 | 0.8501 / 0.7853 / 0.8164 | 0.6611 / 0.6158 / 0.6377 | 4729 / 2424 | 126 |
| 364505 | naive_global | 9312 | 0.8487 / 0.7840 / 0.8151 | 0.6516 / 0.6244 / 0.6377 | 4795 / 2564 | 126 |
| 364505 | guarded_global | 9312 | 0.8463 / 0.7818 / 0.8128 | 0.6411 / 0.6296 / 0.6353 | 4835 / 2707 | 157 |
| 364505 | local_profile | 9312 | 0.8500 / 0.7852 / 0.8163 | 0.6609 / 0.6160 / 0.6376 | 4730 / 2427 | 127 |
| 364505 | protected_nms | 9631 | 0.8361 / 0.7988 / 0.8170 | 0.6290 / 0.6393 / 0.6341 | 4909 / 2895 | 266 |
| 92741 | normal | 9686 | 0.8334 / 0.8008 / 0.8168 | 0.6254 / 0.6423 / 0.6337 | 4932 / 2954 | 289 |
| 92741 | confidence_nms | 9336 | 0.8493 / 0.7866 / 0.8167 | 0.6595 / 0.6164 / 0.6372 | 4733 / 2444 | 139 |
| 92741 | naive_global | 9336 | 0.8479 / 0.7853 / 0.8154 | 0.6457 / 0.6247 / 0.6350 | 4797 / 2632 | 153 |
| 92741 | guarded_global | 9336 | 0.8452 / 0.7828 / 0.8128 | 0.6393 / 0.6299 / 0.6346 | 4837 / 2729 | 169 |
| 92741 | local_profile | 9336 | 0.8492 / 0.7865 / 0.8166 | 0.6592 / 0.6166 / 0.6372 | 4735 / 2448 | 140 |
| 92741 | protected_nms | 9679 | 0.8339 / 0.8007 / 0.8169 | 0.6259 / 0.6421 / 0.6339 | 4931 / 2947 | 285 |

Raw-query 2-mm coverage is unchanged at 0.9232/0.9252 because the detector and all 120 proposals are unchanged. Filtered/selected recovery, 1-mm metrics and confidence-rejected selections are retained separately. Normal graph metrics and NMS counts reproduce N06 exactly.

### One-millimetre sensitivity

| Seed | Policy | Node P / R / F1 | Edge P / R / F1 | TP / FP |
|---|---|---|---|---:|
| 364505 | normal | 0.5795 / 0.5542 / 0.5665 | 0.3051 / 0.3105 / 0.3078 | 2384 / 5429 |
| 364505 | confidence_nms | 0.5899 / 0.5449 / 0.5665 | 0.3200 / 0.2981 / 0.3087 | 2289 / 4864 |
| 364505 | local_profile | 0.5901 / 0.5451 / 0.5667 | 0.3200 / 0.2982 / 0.3087 | 2290 / 4867 |
| 92741 | normal | 0.5854 / 0.5625 / 0.5737 | 0.3094 / 0.3177 / 0.3135 | 2440 / 5446 |
| 92741 | confidence_nms | 0.5977 / 0.5536 / 0.5748 | 0.3263 / 0.3050 / 0.3153 | 2342 / 4835 |
| 92741 | local_profile | 0.5976 / 0.5535 / 0.5747 | 0.3258 / 0.3047 / 0.3149 | 2340 / 4843 |

The local-profile difference remains mixed across seeds at 1 mm; it does not reveal a consistent stricter-localization benefit.

### Matched-count result and ranking

| Seed | Primary versus matched-count control | Edge F1 difference [subject 95% CI] | TP difference | FP difference |
|---|---|---|---:|---:|
| 364505 | local_profile_count_confidence | +0.000159 [-0.003416, +0.004038] | -24 | -55 |
| 364505 | local_profile_count_spatial | -0.000037 [-0.000395, +0.000301] | +1 | +3 |
| 92741 | local_profile_count_confidence | +0.002071 [-0.001139, +0.005227] | -17 | -85 |
| 92741 | local_profile_count_spatial | +0.000012 [-0.000471, +0.000454] | +2 | +4 |

The local policy changes only 22/26 validation query choices versus NMS, in 22/26 patches out of 480 per seed. It selects four/one originally confidence-rejected queries. It adds only **one/two true edges and three/four false edges** versus the identical-count NMS control. Both graph-F1 intervals include zero, and the gain changes sign across seeds. This is not a usable recovery of the GT-oracle representative advantage. Conservatism limits its coverage; the result rules out this small fixed heuristic set, not every possible GT-free selector.

| Seed | Primary conditional AP | Primary all-retained AP | Shared-GT-anchor AP change versus NMS [subject CI] | Shared-anchor AP change versus normal |
|---|---:|---:|---|---|
| 364505 | 0.9786 | 0.8460 | +0.000161 [+0.000086, +0.000238] | -0.000629 [-0.001150, -0.000080] |
| 92741 | 0.9788 | 0.8422 | -0.000034 [-0.000221, +0.000157] | -0.000472 [-0.000757, -0.000239] |

AP between independently selected populations is confounded by recovered anchors and positive prevalence. The shared-anchor comparison instead fixes the recovered GT vertices and their pair labels in each patch, comparing their existing representative scores. The local-NMS AP difference is tiny and positive in only one seed; versus normal representatives it is slightly negative in both. Candidate counts, positives and prevalence are saved for conditional, all-retained and both shared-anchor populations. No relation scores were recalibrated or reranked by a learned head.

### Connectivity costs and close structures

| Seed | Policy | Signed count error/patch: nodes / edges / β₀ / β₁ | GT shortcuts / near nonadjacent FP | Degree≥6 nodes | Short-edge recall | Close-junction recall |
|---|---|---|---:|---:|---:|
| 364505 | normal | -0.917 / +0.279 / +0.254 / +1.450 | 329 / 51 | 11 | 0.4940 | 0.7196 |
| 364505 | confidence_nms | -1.600 / -1.096 / +0.450 / +0.954 | 289 / 34 | 7 | 0.4281 | 0.6711 |
| 364505 | naive_global | -1.600 / -0.667 / +0.265 / +1.198 | 295 / 33 | 12 | 0.4271 | 0.6711 |
| 364505 | guarded_global | -1.600 / -0.285 / +0.083 / +1.398 | 297 / 32 | 13 | 0.4261 | 0.6711 |
| 364505 | local_profile | -1.600 / -1.087 / +0.442 / +0.954 | 290 / 34 | 7 | 0.4276 | 0.6711 |
| 364505 | protected_nms | -0.935 / +0.260 / +0.254 / +1.450 | 329 / 51 | 11 | 0.4930 | 0.7196 |
| 92741 | normal | -0.821 / +0.431 / +0.273 / +1.525 | 313 / 51 | 9 | 0.4950 | 0.7263 |
| 92741 | confidence_nms | -1.550 / -1.046 / +0.496 / +1.000 | 273 / 32 | 5 | 0.4291 | 0.6623 |
| 92741 | naive_global | -1.550 / -0.521 / +0.260 / +1.290 | 286 / 33 | 9 | 0.4256 | 0.6689 |
| 92741 | guarded_global | -1.550 / -0.235 / +0.090 / +1.404 | 284 / 32 | 8 | 0.4296 | 0.6865 |
| 92741 | local_profile | -1.550 / -1.033 / +0.485 / +1.002 | 274 / 32 | 5 | 0.4286 | 0.6623 |
| 92741 | protected_nms | -0.835 / +0.415 / +0.275 / +1.525 | 312 / 51 | 9 | 0.4950 | 0.7263 |

GT shortcut means a wrong direct edge between localized GT anchors separated by at least two annotated GT graph hops. Nearby nonadjacent FP uses <5-mm GT endpoint separation and also includes disconnected GT components. Short positive edges have <5-mm endpoint separation. A close junction has degree≥3 and another GT anchor within 2 mm. These GT strata are strictly offline and never construct inference groups. Signed and absolute node/edge/component/cycle errors, recovered-anchor gains/losses and all denominators are saved. Degree≥6 is a diagnostic count, not a rule declaring such anatomy invalid.

Guarded support replaces 288/296 query choices versus NMS. It recovers 106/104 extra true edges but adds **283/285 false edges**; duplicates and cycles increase and Edge F1 decreases in both seeds. The bounded/reference-ranked score therefore does not solve self-support or spatial identity ambiguity. The uncapped rule is mixed across seeds and also increases cycles. This is consistent with the anticipated hub/false-support risk; it is not a direct causal proof that every extra false cycle arose from a particular clique.

Confidence NMS itself removes 139/143 recovered GT anchors relative to normal inference. Short-edge recall falls from 0.4940/0.4950 to 0.4281/0.4291, and close-junction recall also declines. Its small F1 gain is primarily a precision trade-off with lost true edges, not improved recovery of connections. Protected NMS retains most of these nearby nodes but also retains nearly all normal false edges/duplicates, returning essentially to normal graph quality. Consequently neither fewer nodes nor fewer cycles is treated as sufficient success.

### Conclusion and stopping decision

**No consistent benefit over count-matched controls; stop further search.** All predeclared comparisons are complete. No extra score variants, validation-selected operating points, GPU inference, trainable selector or architecture were launched. The inference module remains an isolated diagnostic; it was not integrated into the model’s standard decoder.

N06 shows that appropriate existing queries carry useful relation information when GT supplies their spatial identities. N08 shows that **these simple inference-available confidence, distance and relation-support rules do not supply that missing identity reliably**. High internal relation support does not certify correct localization or unique GT-node identity. The strong original conditional relation ranking is preserved, while global support can select internally plausible but spatially wrong representatives.

The best-supported remaining hypothesis is **deeper coupling between spatial node quality, assignment/uniqueness and connectivity supervision**, rather than a demonstrated absence of information in the relation tokens. A future controlled supervision study could test whether spatially equivalent query alternatives receive coherent relation labels/quality calibration, keeping positives and total negative budgets explicit. That remains a hypothesis, not a result of N08. This negative study does not justify blanket unmatched-negative mining, which previously performed poorly, or a new large relation architecture. No deployable selection change is recommended from these results.

### Evidence and reproducibility

Isolated worktree/branch: `gnbm-query-selection-20261026`, `research/gt-free-query-selection-20261026`; outputs: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-query-selection-20261026/experiments/gt_free_query_selection_20261026`. `training-selection.json` records the choice made before validation; `per-patch.csv.gz`, `aggregate.csv`, `per-subject.csv`, `paired-differences.json`, `ranking/`, `selections.jsonl.gz` and `selection-changes.json` retain decisions, all metrics and candidate populations. Node/edge thresholds, physical 1/2-mm matching and graph-count conventions are inherited unchanged. No geometric representation or curve target was modified.

Graph intervals resample whole subjects 2,000 times, AP intervals 500 times with exact tie handling, seed 261026. Validation is reused development evidence and exploratory; two continuations share one pretrained source, and the intervals omit source-training/seed uncertainty and multiplicity correction. Five focused selector tests and a two-patch-per-seed pipeline smoke passed. Source/cache/checkpoint hashes and baseline-parity checks are retained.

Print results from the parent `projects` directory:

```bash
python3 gnbm-query-selection-20261026/scripts/show_query_selection.py
```

CPU reproduction in the isolated worktree, using the protected research Python: `scripts/run_query_selection.py`, then `scripts/summarize_query_selection.py`. No inference or training is needed to display saved results. The available registry is `research/experiments.md`; no separate registry already containing N06/N08 was present in the audited files, so its existing history is preserved and a dated N08 entry is appended there.

<!-- END N08 GT FREE QUERY SELECTION 20261026 -->


<!-- N08 REQUEST RECONCILIATION 20261027 -->
### N08 request reconciliation and single next-direction recommendation — 27 October 2026

The renewed N08 request is covered by the completed study above; it was not rerun. Rechecked the two-seed saved summary, train-only selection record, GT-free selector interface and exact-count/baseline-parity verification, and reran the five focused tests successfully. The canonical [vascular experiment registry](../../../research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md) is now available and records N08 as **DONE — negative/mixed**; this supersedes the historical registry-availability note above. Numeric results, checkpoints and the original completed protocol remain unchanged.

**One recommended next direction: no further selection work on the present evidence.** Keep original A-xyz inference as the reference. Relation-aware selection does not consistently beat identical-count controls in both seeds; broader raw-query support increases false edges/cycles, while conservative local replacements are almost neutral. NMS sacrifices true short branches and nearby junctions. N06 establishes useful alternative-query relation content under GT association, but N08 does not establish a GT-free way to identify those alternatives. A trainable selector, token-refinement architecture or changed relation supervision is therefore not justified by N08 alone. Spatial identity, assignment and connectivity-label coupling remains a hypothesis for separate evidence review. This stopping decision is limited to the tested lightweight rules, not a claim that all GT-free selection is impossible. No new inference, training, GPU job or validation tuning was performed for this reconciliation.
