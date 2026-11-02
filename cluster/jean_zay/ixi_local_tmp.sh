#!/usr/bin/env bash
# Source AFTER env.sh: env.sh resets TMPDIR to JOBSCRATCH. Keep package staging
# in JOBSCRATCH, but leave space for multiprocessing's pymp-*/listener-* suffix.
: "${SCRATCH:?SCRATCH must be defined}" "${SLURM_JOB_ID:?SLURM_JOB_ID must be defined}"
export IXI_LOCAL_TMPDIR="${SCRATCH%/}/tmp/icl-${SLURM_JOB_ID}"
if (( ${#IXI_LOCAL_TMPDIR} > 64 )); then
  echo "IXI multiprocessing TMPDIR is still too long: $IXI_LOCAL_TMPDIR" >&2
  return 2 2>/dev/null || exit 2
fi
mkdir -p -- "$IXI_LOCAL_TMPDIR"
export TMPDIR="$IXI_LOCAL_TMPDIR"
