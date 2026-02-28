#!/bin/bash
# GPU & Environment Test (single GPU)
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

# Step 1: Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)')
    x = torch.randn(100, 100, device=f'cuda:{i}')
    print(f'    Tensor test on GPU {i}: OK (sum={x.sum().item():.2f})')
"

echo "========================================="
echo "Step 2: Test model imports"
echo "========================================="

python -c "
import sys
sys.path.insert(0, '.')
from src.models.rumiformer import RumiFormer
from src.models.tgaa import TGAABlock
from src.models.atf import AsymmetricThermalFusion
from src.models.temporal_encoder import TemporalEncoder
from src.models.llava_lora import RumiFormerProjectionMLP
from src.data.dataset import ThermalFrameDataset, ThermalClipDataset
from src.utils.config import SegmentationConfig
from src.utils.trainer import CheckpointManager, ETATracker, setup_ddp, wrap_model_ddp
print('All model imports: OK')
"

echo "========================================="
echo "Step 3: Quick RumiFormer forward pass (GPU 0)"
echo "========================================="

python -c "
import torch, sys
sys.path.insert(0, '.')
from src.models.rumiformer import RumiFormer
model = RumiFormer(num_seg_classes=1, decode_dim=256, use_aux_mask=True).cuda()
overlay = torch.randn(2, 3, 256, 320).cuda()
intensity = torch.randn(2, 1, 256, 320).cuda()
mask = torch.randint(0, 2, (2, 1, 256, 320)).float().cuda()
out = model(overlay, thermal_intensity=intensity, binary_mask=mask)
print(f'seg_logits: {out[\"seg_logits\"].shape}')
print(f'stage4_features: {out[\"stage4_features\"].shape}')
params = sum(p.numel() for p in model.parameters())
print(f'RumiFormer params: {params:,}')
print('Forward pass: OK')
"

echo "========================================="
echo "Step 4: Quick data loading test"
echo "========================================="

python -c "
import sys
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from src.data.dataset import ThermalFrameDataset
ds = ThermalFrameDataset(split='train', img_size=(256, 320), augment=False)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
batch = next(iter(loader))
print(f'overlay: {batch[\"overlay\"].shape}')
print(f'mask: {batch[\"mask\"].shape}')
print(f'class_id: {batch[\"class_id\"]}')
print(f'Dataset size: {len(ds)}')
print('Data loading: OK')
"

echo "========================================="
echo "GPU test completed at $(date)"
echo "========================================="
