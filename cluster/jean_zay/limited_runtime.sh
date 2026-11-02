#!/usr/bin/env bash
# Source before env.sh, which derives TMPDIR from JOBSCRATCH.
export JOBSCRATCH="${SCRATCH:?}/tmp/lt-${SLURM_JOB_ID:?}"
# Python adds /pymp-XXXXXXXX/listener-XXXXXXXX to this path; Linux sun_path
# permits at most 107 bytes. Leave room for both components.
if (( ${#JOBSCRATCH} > 70 )); then
    echo "Limited-data JOBSCRATCH is too long for multiprocessing sockets: $JOBSCRATCH" >&2
    return 2
fi
export TMPDIR="$JOBSCRATCH"
mkdir -p "$JOBSCRATCH"
