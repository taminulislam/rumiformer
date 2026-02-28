#!/bin/bash
# Stage 6: DDPM Augmentation Training (single GPU)
set -e

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate rumiformer

cd /home/siu856569517/Taminul/co2_farm

echo "========================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"
echo "========================================="

python src/train/train_ddpm.py --resume 2>&1 | tee logs/ddpm_stdout.log

echo "========================================="
echo "Stage 6 (DDPM) finished at $(date)"
echo "========================================="
