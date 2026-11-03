# Mitigating false loops and off-vessel graph edges

## Observed failure

Real IXI 3D QC shows dense tangles of highlighted cycles and straight graph
edges that cross outside the segmented vessel lumen. These are different
problems and must not be corrected with one blanket loop-deletion heuristic.

## Failure classes

### A. Geometrically invalid chords

The ordered centerline follows foreground voxels, but the straight segment
between two retained graph nodes can leave the vessel. A fixed RDP error bound
does not guarantee lumen containment, especially when the bound exceeds the
local vessel radius or the lumen is non-convex.

Proposed correction:

1. Use local-radius-normalized RDP as an initial simplification.
2. Check every resulting straight chord by sampling it in physical space.
3. If a chord violates the required containment, insert the dense centerline
   point that best separates the invalid interval and repeat.
4. Stop only when every chord meets the containment contract or all dense
   samples have been restored.

This is dataset-agnostic because it enforces a geometric invariant rather than
using a dataset-specific distance threshold.

### B. False topological connections

If the ordered centerline itself follows a cycle inside the segmentation, RDP
cannot repair it. The segmentation may fuse nearby vessels or a 26-neighbour
digital contact may create a connection that is not supported by the raw MRA.

Proposed correction:

1. Contract degree-2 chains and compute a cycle basis.
2. Consider only cycle edges whose removal reduces beta-1 without increasing
   beta-0.
3. Score each candidate bridge using raw-image evidence normalized locally:
   intensity support along its centreline/tube, its radius bottleneck relative
   to adjacent branches, and tangent continuity through its junctions.
4. Remove a bridge only when image support is clearly weaker than both adjacent
   vessel arms. Otherwise retain it as uncertain.
5. Record every removed edge, score component, and before/after topology for QC.

Absolute MRA intensity thresholds are forbidden because scanner/site intensity
scales differ. Scores must be relative to the local connected vessel context.

## Required validation before training

- Manually classify a stratified sample of cycle candidates across Guys, HH,
  IOP, both resolutions, vessel radii, and cycle sizes.
- Report precision/recall for false-bridge detection; prioritize precision so a
  true Circle-of-Willis/anastomotic connection is not removed.
- Verify chord containment, Curve F1, ACD, HD95, beta-0/beta-1, and node budget
  before and after cleaning.
- Keep the uncleaned graph and an auditable cleaning manifest.
- Do not train on cleaned topology until the candidate rule passes this review.

## Current conclusion

Radius-adaptive RDP improves chord containment but does not change beta-1 and is
not sufficient by itself. False-cycle correction requires raw-MRA-supported
topology validation. Training on the current graph targets should remain paused.

