# Selection after the initial spline experiments

Date: 2026-10-05. **Select V0 control supervision with joint SyntheticMRI fine-tuning as the main approach for the next experiment. Keep V1 sampled-curve supervision as an ablation.** This is a working research decision, not a claim of universal superiority.

## Verification

Reviewed both loss implementations, resolved settings, calibrated weights, decoded overlays, and fresh saved-checkpoint evaluation (job 613399). Both fine-tuning arms start from the same original graph checkpoint and use the same architecture, seed, four training patches, 500 updates and optimizer schedule. V1 inherits V0 settings and changes spline supervision from internal controls to sampled curves. Spline weights are separately calibrated using the same procedure (1.474 / 1.479). V1 is not initialized from trained V0 weights. Neither variant adds test-time target connections in decoded mode.

Fresh inference exactly reproduced the nodes, edges and curves in the viewers. Both fine-tuned variants recover 98 nodes and 79 edges, with node/edge F1 1.00 under the existing box-based protocol. These detection scores do not distinguish shape quality.

Additional verification uses actual decoded controls from the fresh evaluation exports, original complete paths rather than fitted curve targets, and physical coordinates. Cubic splines are sampled at 801 parameter positions. Symmetric mean distance averages both directed distances, using 0.1 mm arc-length sampling and exact distance to each polyline segment. Tortuosity is path length divided by endpoint chord length; report mean absolute error. All metrics below equally weight the 79 branches, not the four patch means. The metric denominator uses each curve's own endpoints. Full paths remain uncertain references, not perfect anatomical truth.

| Approach | Error to fitted spline at corresponding samples (mm) | Symmetric distance to original guide (mm) | Tortuosity absolute error |
| --- | ---: | ---: | ---: |
| V0 joint fine-tuning | 0.1093 | **0.1010** | **0.01013** |
| V1 joint fine-tuning | 0.1311 | 0.1214 | 0.01128 |
| V0 scratch | 0.3811 | 0.3498 | 0.02459 |
| Chord with V0 fine-tuned endpoints | 0.4812 | 0.4731 | 0.02286 |
| Eight-control fitted oracle | — | 0.0242 | 0.00182 |

V0 fine-tuning improves original-guide distance over its same-endpoint chord by approximately 79%, and tortuosity error by approximately 56%. V0 scratch improves guide distance but its aggregate tortuosity error is slightly worse than the chord baseline; successful shape optimization alone does not establish accurate tortuosity.

Reproduction from the representation worktree:

```bash
python scripts/verify_spline_variants.py --run-root experiments/spline_learning/runs/job-612351
```

Full branch-level results: `experiments/spline_learning/runs/job-612351/guide_verification.json`. No new training was performed for this verification.

## Why select V0 now

V0 gives the strongest measured curve and guide agreement, including lower tortuosity error, at this matched update budget. It is straightforward to inspect and reproduce: fixed knots, fixed degree, deterministic endpoint-constrained fitting, and direct internal-control regression. Fixed knots and parameterization are essential to make this coordinate target well-defined.

V1 is a useful geometric alternative because it penalizes curve locations directly. However, the implemented V1 still compares corresponding parameter samples against the same fitted reference; it is not parameterization invariant and does not solve unreliable guide supervision. Control loss can amplify noise in fitted controls, while sampled-curve loss can constrain some combinations of controls weakly. The four-patch results do not establish how either behaves under noisy targets or on unseen subjects. Do not compare their raw spline loss or total loss values to choose a winner: their objectives differ.

The representation and loss comparison are distinct from the initialization comparison. Fine-tuning is currently the practical initialization. Scratch remains necessary as a controlled ablation to test whether conclusions depend on prior graph learning; equal 500-update budgets do not establish its converged performance. Joint representation learning from the beginning has not been compared with fine-tuning at a matched total training budget.

## Evidence needed for a paper

1. Audit subject-level train/validation/test separation, including data used by the original 300-epoch checkpoint. A patch unused for spline fitting is not automatically unseen by the initialized model. If prior checkpoint exposure conflicts with the intended split, create a clean initialization trained only on the permitted training subjects.
2. Train V0 and V1 on the same expanded training set and budgets; use validation subjects for model/loss selection and keep test subjects untouched. Repeat with at least three seeds and report variation. Account for subjects as the independent units rather than treating adjacent branches as independent statistical replicates.
3. Report decoded topology, guide distance, endpoint error, tail geometry errors and tortuosity separately. Include vessel-mask support as complementary image evidence, with uncertainty in guides and masks acknowledged. Evaluate all patches and count missing/extra branches; geometry restricted to recovered pairs must include its coverage.
4. Compare same-endpoint chords, the original dense-node graph approach, and fitted spline oracles. Ablate 6/8/10 controls and fine-tuning versus scratch under explicit budgets. Demonstrate the intended reduction in required object tokens: these first runs still used 120 object queries, so token independence has not been established.
5. Include substantially more tortuous branches (current maximum guide tortuosity is only about 1.197), and report the rate of excluded tangent contacts, loops and parallel branches. The current selected subset does not establish support for those cases.

The defensible current conclusion is that this compact branch representation can be learned on the four selected training patches and captures bends substantially better than a chord with identical predicted endpoints. Generalization, anatomical fidelity and a reduction in object-query requirements remain to be demonstrated. The paper's final loss choice should follow held-out validation evidence; V0 is the selected main candidate now.
