#!/usr/bin/env bash
#SBATCH --job-name=ds-verify-members
#SBATCH --account=vnc@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=experiments/dataset_archive_20261105/logs/%x-%j.out
#SBATCH --error=experiments/dataset_archive_20261105/logs/%x-%j.err
# Byte-compare selected archive members with the (link-followed) source, for members that
# tar --diff could not stat. Usage: sbatch verify_archive_members.slurm <archive> <base-dir> <member-list> <expected-sha256>
# Reads the archive and the source only; deletes only its own temporary extraction under $SCRATCH/tmp.
set -eo pipefail
ARC="$1"; BASE="$2"; LIST="$(readlink -f "$3")"; SHA="$4"
TMPX="$SCRATCH/tmp/verify-$SLURM_JOB_ID"; mkdir -p "$TMPX"
echo "SHA $( [ "$(sha256sum "$ARC" | cut -d' ' -f1)" = "$SHA" ] && echo OK || echo MISMATCH )"
zstd -q -t "$ARC" && echo "ZSTD_TEST OK"
zstd -dc "$ARC" | tar -xf - -C "$TMPX" --verbatim-files-from -T "$LIST"
ok=0; bad=0
while IFS= read -r f; do if cmp -s "$TMPX/$f" "$BASE/$f"; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $f"; fi; done < "$LIST"
rm -rf -- "$TMPX"
echo "MEMBERS_CMP ok=$ok bad=$bad of $(wc -l < "$LIST")"
[ "$bad" = 0 ]
