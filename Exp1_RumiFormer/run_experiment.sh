#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Exp1_RumiFormer: Full Pipeline Training + Evaluation
#  Model: RumiFormer (TGAA-SegFormer + VideoMAE + ATF)
#  Dataset: Expanded (14,316 train / 1,402 val / 1,017 test)
# ═══════════════════════════════════════════════════════════════
set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate rumiformer

PROJECT=/home/siu856569517/Taminul/co2_farm
EXP_DIR=$PROJECT/Exp1_RumiFormer
cd $PROJECT

# ── Environment variable overrides ──
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1
export WANDB_DISABLED="true"
export CHECKPOINT_DIR=$EXP_DIR/checkpoints
export LOG_DIR=$EXP_DIR/logs

# Use experiment-local annotations (frozen snapshot)
export ANNOTATIONS_CSV=$EXP_DIR/annotations/annotations.csv
export CLIPS_CSV=$EXP_DIR/annotations/clips.csv

# ── Epoch config ──
export SEG_1A_EPOCHS=8
export SEG_1B_EPOCHS=12
export TEMP_2A_EPOCHS=6
export TEMP_2B_EPOCHS=10
export FUSION_EPOCHS=15
export E2E_EPOCHS=8

mkdir -p $CHECKPOINT_DIR/{segmentation,temporal,fusion,e2e}
mkdir -p $LOG_DIR $EXP_DIR/figures $EXP_DIR/eval_results

echo "═══════════════════════════════════════════════════════════"
echo "  Experiment 1: RumiFormer"
echo "  Epochs: Seg(${SEG_1A_EPOCHS}+${SEG_1B_EPOCHS}) Temp(${TEMP_2A_EPOCHS}+${TEMP_2B_EPOCHS}) Fusion(${FUSION_EPOCHS}) E2E(${E2E_EPOCHS})"
echo "  Output: $EXP_DIR"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════"

# ── STAGE 1: Segmentation ──
echo ""
echo ">>>>> STAGE 1: Segmentation (${SEG_1A_EPOCHS}+${SEG_1B_EPOCHS} epochs) <<<<<"
python src/train/train_segmentation.py 2>&1 | tee $LOG_DIR/stage1_stdout.log
echo "Stage 1 finished at: $(date)"

SEG_CKPT=$(ls -t $CHECKPOINT_DIR/segmentation/*.pt 2>/dev/null | head -1)
echo "Seg checkpoint: $SEG_CKPT"

# ── STAGE 2: Temporal ──
echo ""
echo ">>>>> STAGE 2: Temporal (${TEMP_2A_EPOCHS}+${TEMP_2B_EPOCHS} epochs) <<<<<"
python src/train/train_temporal.py 2>&1 | tee $LOG_DIR/stage2_stdout.log
echo "Stage 2 finished at: $(date)"

TEMP_CKPT=$(ls -t $CHECKPOINT_DIR/temporal/*.pt 2>/dev/null | head -1)
echo "Temporal checkpoint: $TEMP_CKPT"

# ── STAGE 3: Fusion ──
echo ""
echo ">>>>> STAGE 3: Fusion (${FUSION_EPOCHS} epochs) <<<<<"
python src/train/train_fusion.py \
    --seg_checkpoint "$SEG_CKPT" \
    --temporal_checkpoint "$TEMP_CKPT" \
    2>&1 | tee $LOG_DIR/stage3_stdout.log
echo "Stage 3 finished at: $(date)"

FUSION_CKPT=$(ls -t $CHECKPOINT_DIR/fusion/*.pt 2>/dev/null | head -1)
echo "Fusion checkpoint: $FUSION_CKPT"

# ── STAGE 5: E2E (no LLaVA) ──
echo ""
echo ">>>>> STAGE 5: E2E (${E2E_EPOCHS} epochs, seg+cls) <<<<<"
python src/train/train_e2e.py \
    --seg_checkpoint "$SEG_CKPT" \
    --temporal_checkpoint "$TEMP_CKPT" \
    --fusion_checkpoint "$FUSION_CKPT" \
    2>&1 | tee $LOG_DIR/stage5_stdout.log
echo "Stage 5 finished at: $(date)"

# ── EVALUATION ──
echo ""
echo ">>>>> EVALUATION <<<<<"
python src/eval/evaluate.py \
    --seg_checkpoint "$SEG_CKPT" \
    --temporal_checkpoint "$TEMP_CKPT" \
    --output_dir $EXP_DIR/eval_results \
    2>&1 | tee $LOG_DIR/eval_stdout.log

# ── VISUALIZATIONS ──
echo ""
echo ">>>>> VISUALIZATIONS <<<<<"
python src/eval/visualize.py \
    --seg_checkpoint "$SEG_CKPT" \
    --raw_npz $EXP_DIR/eval_results/classification_raw.npz \
    --results_json $EXP_DIR/eval_results/eval_results.json \
    --output_dir $EXP_DIR/figures \
    2>&1 | tee $LOG_DIR/viz_stdout.log

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Experiment 1: RumiFormer — COMPLETE"
echo "  Finished: $(date)"
echo "  Results:  $EXP_DIR/eval_results/"
echo "  Figures:  $EXP_DIR/figures/"
echo "═══════════════════════════════════════════════════════════"
