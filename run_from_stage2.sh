#!/bin/bash
# Resume pipeline from Stage 2 onwards (Stage 1 already complete)
set -e

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate rumiformer

cd /home/siu856569517/Taminul/co2_farm
mkdir -p logs

echo "========================================="
echo "RumiFormer Pipeline — Resuming from Stage 2"
echo "Start: $(date)"
echo "========================================="

# Stage 1 checkpoint
SEG_CKPT=$(ls -t outputs/checkpoints/segmentation/*.pt 2>/dev/null | head -1)
echo "Using seg checkpoint: $SEG_CKPT"

# ---- Stage 2: Temporal ----
echo ""
echo ">>>>> STAGE 2: Temporal Encoder <<<<<"
echo "Started at: $(date)"
python src/train/train_temporal.py 2>&1 | tee logs/temporal_stdout.log
echo "Stage 2 finished at: $(date)"

# ---- Stage 3: ATF Fusion ----
echo ""
echo ">>>>> STAGE 3: ATF Fusion <<<<<"
echo "Started at: $(date)"
python src/train/train_fusion.py --seg_checkpoint "$SEG_CKPT" 2>&1 | tee logs/fusion_stdout.log
echo "Stage 3 finished at: $(date)"

# ---- Stage 4: LLaVA LoRA ----
TEMP_CKPT=$(ls -t outputs/checkpoints/temporal/*.pt 2>/dev/null | head -1)
FUSION_CKPT=$(ls -t outputs/checkpoints/fusion/*.pt 2>/dev/null | head -1)
echo ""
echo ">>>>> STAGE 4: LLaVA LoRA <<<<<"
echo "Started at: $(date)"
python src/train/train_llava.py \
    --seg_checkpoint "$SEG_CKPT" \
    --temporal_checkpoint "$TEMP_CKPT" \
    --fusion_checkpoint "$FUSION_CKPT" 2>&1 | tee logs/llava_stdout.log
echo "Stage 4 finished at: $(date)"

# ---- Stage 5: End-to-End ----
LLAVA_CKPT=$(ls -t outputs/checkpoints/llava/*.pt 2>/dev/null | head -1)
echo ""
echo ">>>>> STAGE 5: End-to-End Fine-tuning <<<<<"
echo "Started at: $(date)"
python src/train/train_e2e.py \
    --seg_checkpoint "$SEG_CKPT" \
    --temporal_checkpoint "$TEMP_CKPT" \
    --fusion_checkpoint "$FUSION_CKPT" \
    --llava_checkpoint "$LLAVA_CKPT" 2>&1 | tee logs/e2e_stdout.log
echo "Stage 5 finished at: $(date)"

echo ""
echo "========================================="
echo "ALL STAGES COMPLETE (Stages 2-5)"
echo "Finished at: $(date)"
echo "========================================="
