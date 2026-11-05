#!/usr/bin/env bash
#SBATCH --job-name=ixi-test-archive
#SBATCH --account=vnc@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=experiments/dataset_archive_20261105/logs/%x-%j.out
#SBATCH --error=experiments/dataset_archive_20261105/logs/%x-%j.err
# Add the 21 restored IXI test images (purged from SCRATCH, restored 5 Nov 2026, verified against September patches)
# to the STORE originals backup, whose IXI.tar.zst was packed while they were missing. Read-only on the source.
set -eo pipefail
DEST=$STORE/datasets-originals-20261105; I=/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI
LIST=$SLURM_SUBMIT_DIR/experiments/dataset_archive_20261105/members/IXI_restored_test_images.txt
OUT=$DEST/IXI_test_images_restored_20261105.tar.zst
tar -C $I -cf - -T $LIST | zstd -q -T4 -3 -o $OUT.partial
zstd -t $OUT.partial
n=$(zstd -dc $OUT.partial | tar -tf - | wc -l); want=$(wc -l < $LIST)
zstd -dc $OUT.partial | tar -C $I -df - && echo "tar --diff: no differences"
[ "$n" = "$want" ] || { echo "COUNT MISMATCH $n vs $want"; exit 1; }
mv $OUT.partial $OUT
( cd $DEST && sha256sum $(basename $OUT) >> SHA256SUMS && sha256sum -c --ignore-missing SHA256SUMS | grep $(basename $OUT) )
printf '\n## Addendum 2026-11-05: IXI test images\nIXI.tar.zst was packed while 21 IXI test images were purged from SCRATCH (149/170 in imagesTr). They were restored and verified voxel-for-voxel against 2,178 September test patches, and are archived separately in `IXI_test_images_restored_20261105.tar.zst` (%s files).\n' "$n" >> $DEST/INVENTORY.md
cp $DEST/INVENTORY.md $SLURM_SUBMIT_DIR/experiments/dataset_archive_20261105/INVENTORY.md
echo "ARCHIVED $n files -> $OUT ($(stat -c %s $OUT) bytes)"
