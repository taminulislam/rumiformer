#!/bin/bash
#SBATCH --job-name=rumiformer_compare
#SBATCH --output=/home/siu856569517/Taminul/co2_farm/logs/comparison_%j.log
#SBATCH --error=/home/siu856569517/Taminul/co2_farm/logs/comparison_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# ═══════════════════════════════════════════════════════════════
#  Run 3 Model Comparison Experiments Sequentially
#  Models: SegFormer-B2, iFormer, LACTNet
#  Each: Train → Evaluate → Visualize → Next
# ═══════════════════════════════════════════════════════════════

PROJECT=/home/siu856569517/Taminul/co2_farm
cd $PROJECT

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  MODEL COMPARISON PIPELINE                               ║"
echo "║  Models: SegFormer-B2, iFormer, LACTNet                  ║"
echo "║  Started: $(date)                              ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── Experiment 2: SegFormer-B2 (Vanilla) ──
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  [1/3] SegFormer-B2 (Vanilla)                             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
export MODEL_NAME=segformer_b2
export EXP_DIR=$PROJECT/Exp2_SegFormerB2
export SEG_1A_EPOCHS=8
export SEG_1B_EPOCHS=12
bash scripts/run_experiment.sh 2>&1 | tee $EXP_DIR/logs/experiment.log
echo "SegFormer-B2 done at: $(date)"

# ── Experiment 3: iFormer (MobileViT-S) ──
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  [2/3] iFormer (MobileViT-S)                              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
export MODEL_NAME=iformer
export EXP_DIR=$PROJECT/Exp3_iFormer
export SEG_1A_EPOCHS=8
export SEG_1B_EPOCHS=12
bash scripts/run_experiment.sh 2>&1 | tee $EXP_DIR/logs/experiment.log
echo "iFormer done at: $(date)"

# ── Experiment 4: LACTNet ──
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  [3/3] LACTNet (ResNet-18 + Transformer)                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
export MODEL_NAME=lactnet
export EXP_DIR=$PROJECT/Exp4_LACTNet
export SEG_1A_EPOCHS=8
export SEG_1B_EPOCHS=12
bash scripts/run_experiment.sh 2>&1 | tee $EXP_DIR/logs/experiment.log
echo "LACTNet done at: $(date)"

# ── Generate Comparison Table ──
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  GENERATING COMPARISON TABLE                              ║"
echo "╚═══════════════════════════════════════════════════════════╝"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate rumiformer
python scripts/compare_results.py

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE                                  ║"
echo "║  Results: comparison_results.md                            ║"
echo "║  Finished: $(date)                             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
