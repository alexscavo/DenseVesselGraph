This is the user-authorized scheduled morning review, not a new research phase.
The user explicitly requested: after both scratch trainings finish, update docs
with the approach, discoveries, numbers and limitations, commit the relevant
changes, and push branch representation to origin. Push authorization persists.

Read /lustre/fswork/projects/rech/vnc/upz25mj/projects/AGENTS.md.
Code repo: /lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph
Research docs: /lustre/fswork/projects/rech/vnc/upz25mj/projects/archive/graph-representation/docs
Run root: /lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012
Jobs: 949063 completed epochs10→150; 949065 continues150→300 (12h limit);
949071 evaluates final and best for both arms on all1952 validation cases.
Morning timer fires 2026-10-13 07:00 Europe/Paris; fallback15:00 if incomplete.
Do not stay alive waiting for jobs: inspect once, report what exists, finish.

1. Check sacct/squeue and outputs. Require real successful completion before
   calling a training/evaluation finished. Inspect missing/failed stages honestly.
   If results are pending, document status, commit/push useful factual updates,
   and end; the fallback handles a later check. Do not launch new GPU experiments.
2. Review both epoch300 summaries/history/provenance, epoch100/200 comparisons,
   and full_validation/{graph_control_final,graph_control_best,
   detached_spline_final,detached_spline_best}/summary.json, comparison.csv/md.
   Verify same data,1952 validation cases, evaluation protocol and saved BN mode.
   Preserve existing checkpoints/results; do not use the test set for selection.
3. Explain the actual setup: random initialization,10000 training patches,batch4,
   eight-control residual B-spline with local image features,300total epochs,
   weights0/1.4739989612791453,graph-only detector gradients with detached geometry
   tokens/endpoints,normal training BN/saved evaluation BN,original750k-update LR
   schedule and restored AdamW. First extension restarts RNG because old pilots
   did not save it; later continuation preserves RNG. Best selection uses128-val
   common_branch_f1 then native_edge_f1, not total loss. Give actual run timings.
4. Report tables for best and final checkpoints: native node/edge F1, common
   branch/curve F1, same-endpoint chord curve F1, topology/count errors, matched
   guide errors, endpoint-neutral curvature gains overall and on curved long
   branches,train-versus-validation behavior,image sensitivity,collapse flags.
   Distinguish representation geometry from graph detection. Single seed and
   uncertain centerline guides limit claims. Poor complete-branch recovery and
   short-branch curvature overfitting matter even if curve F1 improves. Do not
   attribute detector differences causally to detached geometry gradients.
5. Update graph-representation/docs/scratch-extension-300.md and append the
   results/decisions/next justified steps to spline-learning-report.md. Add a
   tracked copy of the ENTIRE graph-representation/docs directory at code repo
   docs/research/spline_learning/, preserving subfolders. The user subsequently
   requested all these docs in the repo. Synchronize the report, journal and all
   associated docs after updating them; the outside research docs are not in a
   Git repository. Keep small final result tables/metadata
   tracked; avoid checkpoints,large predictions,galleries or datasets in Git.
   Existing related links should work in the tracked version or be explained.
   When ALL final results and the completed report are ready, write and track
   experiments/spline_learning/metadata/local-geometry/scratch-extension-final-results.json
   with status="complete", validation_samples=1952,
   training_end_updates={"graph_control":750000,"detached_spline":750000},
   evaluated_arms=["graph_control_final","graph_control_best","detached_spline_final","detached_spline_best"],
   and actual result tables/provenance. Do not mark it complete for pending results.
   This small metadata path is ignored by default: git add -f that exact file.
6. Commit only the report and task-related stable evidence. Other pre-existing
   dirty files concern a separate IXI experiment; do not stage those. Never use
   git add -A,force push,reset,or overwrite remote work. On representation, inspect
   the diff, git diff --check, commit, then git push -u origin representation.
   If push rejects because of permissions/divergence, record the exact problem
   without destructive fixes. Confirm remote head equals the local committed head.
7. End with concise results, conclusions, report paths and commit/push status.
   No new architecture,losses,sweeps or training are authorized by this timer.

Only factual completed results may be reported. If not enough evidence is ready,
say so and stop promptly. Use existing aggregate artifacts; no redundant GPU tests.
