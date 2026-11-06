#!/usr/bin/env bash
#SBATCH --job-name=ds-derived-archive
#SBATCH --account=vnc@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=experiments/dataset_archive_20261105/logs/%x-%j.out
#SBATCH --error=experiments/dataset_archive_20261105/logs/%x-%j.err
# Pack one derived training cache (or derived dataset) into one zstd tar in STORE, DEREFERENCING
# symlinks (tar -h) and storing every file as a full copy (--hard-dereference: no tar hard-link
# entries even when two paths reach one inode), then verify. The caller's member list must exclude candidates/ folders, so that
# each patch file is stored once.
# Usage (from the DenseVesselGraph root):
#   sbatch -J dsd-<name> cluster/jean_zay/archive_dataset_derived.slurm <name> <base-dir> <members-file>
# The source is only read. Verification:
#   archive regular files == find -L regular files of the members == (symlinks + regular files);
#   no symlink is left in the archive; zstd -t; tar --diff --dereference against the source
#   (content, size, mode, mtime of the link targets); and 8 random members extracted to
#   $SCRATCH/tmp and compared with cmp. Only that temporary extraction is deleted.
set -eo pipefail
NAME="$1"; BASE="$2"; MEMBERS="$(readlink -f "$3")"
test -n "$NAME" && test -d "$BASE" && test -s "$MEMBERS"
DEST="$STORE/datasets-derived-20261105"
LOGDIR="$SLURM_SUBMIT_DIR/experiments/dataset_archive_20261105/logs"
OUT="$DEST/$NAME.tar.zst"
PART="$DEST/.$NAME.tar.zst.partial"
mkdir -p "$DEST" "$LOGDIR"
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 1; }
rm -f "$PART"
cd "$BASE"
echo "START $(date -Is) name=$NAME base=$BASE host=$(hostname)"
grep -q candidates "$MEMBERS" && { echo "members must not include candidates/" >&2; exit 1; }
src_links=$(while IFS= read -r p; do find "$p" -type l; done < "$MEMBERS" | wc -l)
src_regular=$(while IFS= read -r p; do find "$p" -type f; done < "$MEMBERS" | wc -l)
src_deref_files=$(while IFS= read -r p; do find -L "$p" -type f; done < "$MEMBERS" | wc -l)
src_dangling=$(while IFS= read -r p; do find -L "$p" -type l; done < "$MEMBERS" | wc -l)
src_bytes=$(while IFS= read -r p; do find -L "$p" -type f -printf '%s\n'; done < "$MEMBERS" | awk '{s+=$1} END {printf "%d", s}')
echo "SOURCE links=$src_links regular=$src_regular deref_files=$src_deref_files dangling=$src_dangling deref_bytes=$src_bytes"
[ "$src_dangling" = 0 ] || { echo "dangling links in source; not packing" >&2; exit 1; }
tar --dereference --hard-dereference --verbatim-files-from -T "$MEMBERS" -cf - | zstd -q -T4 -3 -o "$PART"
echo "PACKED $(date -Is) bytes=$(stat -c %s "$PART")"
zstd -q -t "$PART"
zstd -dc "$PART" | tar -tvf - > "$LOGDIR/$NAME.listing.txt"
arc_entries=$(wc -l < "$LOGDIR/$NAME.listing.txt")
arc_files=$(grep -c '^-' "$LOGDIR/$NAME.listing.txt" || true)
arc_links=$(grep -c '^[lh]' "$LOGDIR/$NAME.listing.txt" || true)
diff_status=OK
zstd -dc "$PART" | tar --dereference -df - > "$LOGDIR/$NAME.diff.txt" 2>&1 || diff_status=DIFF
# Spot check: extract 8 random regular members and cmp them with the (link-followed) source.
TMPX="$SCRATCH/tmp/dsd-$NAME-$SLURM_JOB_ID"
mkdir -p "$TMPX"
grep '^-' "$LOGDIR/$NAME.listing.txt" | awk '{ $1=$2=$3=$4=$5=""; sub(/^ +/, ""); print }' | shuf -n 8 > "$TMPX.list"
zstd -dc "$PART" | tar -xf - -C "$TMPX" --verbatim-files-from -T "$TMPX.list"
spot=OK
while IFS= read -r f; do
    if cmp -s "$TMPX/$f" "$BASE/$f"; then echo "SPOT OK $f"; else echo "SPOT FAIL $f"; spot=FAIL; fi
done < "$TMPX.list"
rm -rf -- "$TMPX" "$TMPX.list"
sha=$(sha256sum "$PART" | cut -d' ' -f1)
status=OK
[ "$arc_files" = "$src_deref_files" ] && [ "$arc_files" = "$((src_links + src_regular))" ] && [ "$arc_links" = 0 ] \
    && [ "$diff_status" = OK ] && [ "$spot" = OK ] || status=MISMATCH
if [ "$status" = OK ]; then mv "$PART" "$OUT"; else echo "verification failed; partial kept at $PART" >&2; fi
printf 'RESULT\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$src_bytes" "$(stat -c %s "$OUT" 2>/dev/null || stat -c %s "$PART")" \
    "$src_links" "$src_regular" "$src_deref_files" "$arc_files" "$arc_entries" "$diff_status" "$spot" "$sha" "$status"
echo "END $(date -Is)"
[ "$status" = OK ]
