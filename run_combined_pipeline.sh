#!/usr/bin/env bash
# ============================================================================
# Compete1 — Combined Breast Ultrasound Pipeline
#
# three-classify segmentation + preprocessing
#   +
# Compete classification methodology (4-view, 5-fold CV, OOF calibration)
#
# Usage:
#   ./run_combined_pipeline.sh
#   BACKBONE=resnet50 ./run_combined_pipeline.sh
#   N_SPLITS=3 MAX_EPOCHS=10 ./run_combined_pipeline.sh   # quick test
# ============================================================================
set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-8}"
SEG_BATCH_SIZE="${SEG_BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
SEG_EPOCHS="${SEG_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BACKBONE="${BACKBONE:-densenet121}"
N_SPLITS="${N_SPLITS:-5}"
VIEWS="${VIEWS:-full cut_borders border masked}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

cd "$(dirname "$0")"
mkdir -p outputs/{logs,figures}
mkdir -p outputs/models/{segmentation,classification}
mkdir -p outputs/results/{oof,competition}

echo "======================================================================"
echo "  Compete1 Pipeline"
echo "  Backbone: $BACKBONE  |  CV folds: $N_SPLITS  |  Epochs/view: $MAX_EPOCHS"
echo "  Seed: $SEED"
echo "======================================================================"

# ── Step 1: Ultrasound preprocessing ──────────────────────────────────────
echo ""
echo "==== Step 1/8: ultrasound preprocessing ===="
python data/preprocess_ultrasound.py

# ── Step 2: Fine-tune FCBFormer segmentation ─────────────────────────────
echo ""
echo "==== Step 2/8: fine-tune segmentation model ===="
python train_segmentation.py \
  --image-root images/preprocessed/full_image \
  --mask-root images/preprocessed/gt_masks \
  --pretrained-checkpoint checkpoints/pretrained/FCBFormer_checkpoint.pt \
  --output-dir outputs/models/segmentation \
  --batch-size "$SEG_BATCH_SIZE" \
  --max-epochs "$SEG_EPOCHS" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --amp

# ── Step 3: Generate masks ────────────────────────────────────────────────
echo ""
echo "==== Step 3/8: generate masks ===="
python data/generate_masks.py \
  --input-root images/preprocessed/full_image \
  --checkpoint outputs/models/segmentation/best.pt \
  --mask-output-root images/finetuned/masks \
  --save_probs \
  --postprocess \
  --keep_largest

# ── Step 4: Crop lesion regions ───────────────────────────────────────────
echo ""
echo "==== Step 4/8: crop tumor regions ===="
python data/cut_borders.py \
  --image-root images/preprocessed/full_image \
  --mask-root images/finetuned/masks \
  --output-root images/finetuned/cut_borders \
  --postprocess \
  --keep_largest

# ── Step 5: Generate 4-view dataset ───────────────────────────────────────
echo ""
echo "==== Step 5/8: generate 4-view dataset ===="
python data/make_views.py

# ── Step 6: 5-fold CV training per view ──────────────────────────────────
echo ""
echo "==== Step 6/8: 5-fold CV training ===="
for view in $VIEWS; do
  echo ""
  echo "--- Training view: $view ($BACKBONE) ---"
  python train_cls_folds.py \
    --view "$view" \
    --backbone "$BACKBONE" \
    --n-splits "$N_SPLITS" \
    --epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --checkpoint-dir outputs/models/classification \
    --metrics-dir outputs/results/oof
done

# ── Step 7: Calibrate threshold on OOF ────────────────────────────────────
echo ""
echo "==== Step 7/8: calibrate threshold on OOF ===="
python calibrate_threshold.py \
  --backbone "$BACKBONE" \
  --objective youden \
  --output-dir outputs/results/competition \
  --metrics-dir outputs/results/oof

# ── Step 8: Evaluate ensemble on BUSI ─────────────────────────────────────
echo ""
echo "==== Step 8/8: evaluate ensemble on BUSI ===="
python evaluate_ensemble.py \
  --backbone "$BACKBONE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --output-dir outputs/results/competition

echo ""
echo "======================================================================"
echo "  Compete1 pipeline complete!"
echo "  Results: outputs/results/competition/"
echo "======================================================================"
ls -la outputs/results/competition/
