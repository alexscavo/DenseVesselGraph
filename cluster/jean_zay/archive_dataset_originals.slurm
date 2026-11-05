#!/usr/bin/env bash
#SBATCH --job-name=ds-orig-archive
#SBATCH --account=vnc@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=experiments/dataset_archive_20261105/logs/%x-%j.out
#SBATCH --error=experiments/dataset_archive_20261105/logs/%x-%j.err
# Pack the ORIGINAL files of one dataset into one zstd tar in STORE, then verify it.
# Usage (from the DenseVesselGraph root):
#   sbatch -J ds-<name> cluster/jean_zay/archive_dataset_originals.slurm <name> <base-dir> <members-file>
# <members-file>: one path per line, relative to <base-dir>; directories are archived recursively.
# The source is only read (tar -c, find, du, tar --diff). Symlinks are stored as links (no -h).
# Verification: entry count and regular-file count of the archive equal those of the source,
# and tar --diff finds no content/size/mode/mtime difference against the source. Nothing is deleted.
set -eo pipefail
NAME="$1"; BASE="$2"; MEMBERS="$(readlink -f "$3")"
test -n "$NAME" && test -d "$BASE" && test -s "$MEMBERS"
DEST="$STORE/datasets-originals-20261105"
LOGDIR="$SLURM_SUBMIT_DIR/experiments/dataset_archive_20261105/logs"
OUT="$DEST/$NAME.tar.zst"
PART="$DEST/.$NAME.tar.zst.partial"
mkdir -p "$DEST" "$LOGDIR"
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 1; }
rm -f "$PART"
cd "$BASE"
echo "START $(date -Is) name=$NAME base=$BASE host=$(hostname)"
src_entries=$(while IFS= read -r p; do find "$p"; done < "$MEMBERS" | wc -l)
src_files=$(while IFS= read -r p; do find "$p" -type f; done < "$MEMBERS" | wc -l)
src_links=$(while IFS= read -r p; do find "$p" -type l; done < "$MEMBERS" | wc -l)
src_bytes=$(while IFS= read -r p; do du -sb "$p"; done < "$MEMBERS" | awk -F'\t' '{s+=$1} END {printf "%d", s}')
echo "SOURCE entries=$src_entries files=$src_files links=$src_links bytes=$src_bytes"
tar --verbatim-files-from -T "$MEMBERS" -cf - | zstd -q -T4 -3 -o "$PART"
echo "PACKED $(date -Is) bytes=$(stat -c %s "$PART")"
zstd -q -t "$PART"
zstd -dc "$PART" | tar -tvf - > "$LOGDIR/$NAME.listing.txt"
arc_entries=$(wc -l < "$LOGDIR/$NAME.listing.txt")
arc_files=$(grep -c '^-' "$LOGDIR/$NAME.listing.txt" || true)
arc_links=$(grep -c '^l' "$LOGDIR/$NAME.listing.txt" || true)
diff_status=OK
zstd -dc "$PART" | tar -df - > "$LOGDIR/$NAME.diff.txt" 2>&1 || diff_status=DIFF
sha=$(sha256sum "$PART" | cut -d' ' -f1)
status=OK
[ "$arc_entries" = "$src_entries" ] && [ "$arc_files" = "$src_files" ] && [ "$arc_links" = "$src_links" ] && [ "$diff_status" = OK ] || status=MISMATCH
if [ "$status" = OK ]; then mv "$PART" "$OUT"; else echo "verification failed; partial kept at $PART" >&2; fi
printf 'RESULT\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$src_bytes" "$(stat -c %s "$OUT" 2>/dev/null || stat -c %s "$PART")" \
    "$src_entries" "$arc_entries" "$src_files" "$arc_files" "$src_links" "$arc_links" "$diff_status" "$sha" "$status"
echo "END $(date -Is)"
[ "$status" = OK ]
