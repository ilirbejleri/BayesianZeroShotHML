#!/bin/bash
#SBATCH --job-name=feat_tiered
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_feat_%j.out
#SBATCH --qos=coc-ice

# ============================================================
#  tiered-ImageNet feature extraction.
#
#  Lustre is very slow at unzipping 779k small files (~2 MB/s).
#  This job copies the zip to node-local $TMPDIR (fast NVMe),
#  unzips there, then runs prepare_dataset.py with a symlink so
#  prepare_dataset.py reads from $TMPDIR instead of Lustre.
#  Final features.npy / labels.npy / etc. land on Lustre as usual.
# ============================================================

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3
conda activate bhzsl

PROJECT="$SLURM_SUBMIT_DIR"
ZIP="$PROJECT/data/tiered_imagenet_raw/tieredimagenet.zip"
RAW_DIR="$PROJECT/data/tiered_imagenet_raw"

if [ ! -f "$ZIP" ]; then
    echo "ERROR: zip not found at $ZIP"
    exit 1
fi

echo "[$(date)] === tiered-ImageNet extract+prep ==="
echo "[$(date)] TMPDIR=$TMPDIR"
df -h "$TMPDIR" | tail -1

echo "[$(date)] Copying zip ($(du -h $ZIP | cut -f1)) to TMPDIR..."
cp "$ZIP" "$TMPDIR/tieredimagenet.zip"

echo "[$(date)] Unzipping in TMPDIR..."
cd "$TMPDIR"
unzip -q tieredimagenet.zip
echo "[$(date)] Unzip done. TMPDIR contents:"
ls -d "$TMPDIR/tiered_imagenet/"{train,val,test}
echo "  train classes: $(ls $TMPDIR/tiered_imagenet/train | wc -l)"
echo "  val classes:   $(ls $TMPDIR/tiered_imagenet/val | wc -l)"
echo "  test classes:  $(ls $TMPDIR/tiered_imagenet/test | wc -l)"

echo "[$(date)] Setting up symlinks under $RAW_DIR ..."
# Remove any pre-existing train/val/test (could be partial) but keep the zip
rm -rf "$RAW_DIR/train" "$RAW_DIR/val" "$RAW_DIR/test"
ln -s "$TMPDIR/tiered_imagenet/train" "$RAW_DIR/train"
ln -s "$TMPDIR/tiered_imagenet/val"   "$RAW_DIR/val"
ln -s "$TMPDIR/tiered_imagenet/test"  "$RAW_DIR/test"

echo "[$(date)] Running prepare_dataset.py..."
cd "$PROJECT"
python3 prepare_dataset.py --dataset tiered_imagenet --device cuda --batch_size 256

echo "[$(date)] Cleaning up dangling symlinks..."
rm -f "$RAW_DIR/train" "$RAW_DIR/val" "$RAW_DIR/test"

echo "[$(date)] === DONE ==="
ls -lh "$PROJECT/data/tiered_imagenet/" 2>/dev/null
