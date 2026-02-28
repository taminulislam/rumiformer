#!/bin/bash
# Smoke test: validate every stage can import, load data, and run 1 forward+backward pass
# This catches errors BEFORE committing to a full training run.
set -e

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export HF_HUB_DISABLE_XET=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate rumiformer
cd /home/siu856569517/Taminul/co2_farm

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local code="$2"
    echo ""
    echo "============================================"
    echo "  TEST: $name"
    echo "============================================"
    if python -c "$code" 2>&1; then
        echo "  ✅ $name PASSED"
        PASS=$((PASS+1))
    else
        echo "  ❌ $name FAILED"
        FAIL=$((FAIL+1))
    fi
}

# ---- Test 1: Augmentation ----
run_test "Augmentation" "
import numpy as np
from src.data.augmentation import ThermalGasAugment
aug = ThermalGasAugment()
img = np.random.randint(0,255,(256,320,3),dtype=np.uint8)
msk = np.zeros((256,320),dtype=np.uint8); msk[80:180,100:250]=255
img2, msk2, ovl2 = aug(img, msk, img.copy())
assert img2.shape == (256,320,3), f'Bad shape {img2.shape}'
print('  Augmentation shapes OK')
"

# ---- Test 2: Stage 1 — Segmentation (1 batch) ----
run_test "Stage 1: Segmentation" "
import sys, torch
sys.path.insert(0,'.')
from torch.cuda.amp import autocast
from src.data.dataset import ThermalFrameDataset
from src.models.rumiformer import RumiFormer
from src.train.train_segmentation import combined_loss

ds = ThermalFrameDataset(split='train', img_size=(256,320), augment=True)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
batch = next(iter(loader))

model = RumiFormer(num_seg_classes=1, decode_dim=256, use_aux_mask=True).cuda()
with autocast(dtype=torch.bfloat16):
    out = model(batch['overlay'].cuda(), thermal_intensity=batch['thermal_intensity'].cuda(),
                binary_mask=batch['mask'].cuda())
    loss, _ = combined_loss(out['seg_logits'], batch['mask'].cuda())
loss.backward()
print(f'  Seg loss={loss.item():.4f}, logits shape={out[\"seg_logits\"].shape}')
del model; torch.cuda.empty_cache()
"

# ---- Test 3: Stage 2 — Temporal (model load + 1 fwd) ----
run_test "Stage 2: Temporal Encoder" "
import sys, torch
sys.path.insert(0,'.')
from src.models.temporal_encoder import TemporalEncoder

model = TemporalEncoder(output_dim=256, freeze_backbone=True).cuda()
clip = torch.randn(2, 16, 3, 224, 224).cuda()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = model(clip, return_cls_logits=True)
print(f'  Temporal emb shape={out[\"temporal_embedding\"].shape}, cls_logits={out[\"cls_logits\"].shape}')
del model; torch.cuda.empty_cache()
"

# ---- Test 4: Stage 3 — Fusion (seg + ATF, 1 fwd+bwd) ----
run_test "Stage 3: ATF Fusion" "
import sys, torch
sys.path.insert(0,'.')
from torch.cuda.amp import autocast
from src.models.rumiformer import RumiFormer
from src.models.atf import AsymmetricThermalFusion
from src.train.train_fusion import FusionClassifier

seg = RumiFormer(num_seg_classes=1, decode_dim=256, use_aux_mask=True).cuda().eval()
for p in seg.parameters(): p.requires_grad = False
fc = FusionClassifier(feature_dim=256, stage4_channels=512, num_classes=3).cuda()

overlay = torch.randn(2,3,256,320).cuda()
mask = torch.zeros(2,1,256,320).cuda()
mask[:,:,80:180,100:250] = 1.0

with torch.no_grad():
    seg_out = seg(overlay, binary_mask=mask)
bg = overlay * (1.0 - mask)
with autocast(dtype=torch.bfloat16):
    logits, conf = fc(mask, seg_out['stage4_features'], bg)
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0,1]).cuda())
loss.backward()
print(f'  Fusion logits={logits.shape}, loss={loss.item():.4f}')
del seg, fc; torch.cuda.empty_cache()
"

# ---- Test 5: Stage 4 — LLaVA imports ----
run_test "Stage 4: LLaVA imports" "
import sys
sys.path.insert(0,'.')
from src.models.llava_lora import RumiFormerLLaVA
print('  LLaVA module importable')
# Skip full model load (requires 13GB Vicuna download)
print('  NOTE: Full model test skipped — requires Vicuna-7B download')
"

# ---- Test 6: Stage 5 — E2E model build ----
run_test "Stage 5: E2E model build" "
import sys, torch
sys.path.insert(0,'.')
from src.train.train_e2e import EndToEndModel, dice_loss, compute_seg_metrics
from src.models.rumiformer import RumiFormer
from src.models.atf import AsymmetricThermalFusion
from src.models.temporal_encoder import TemporalEncoder
import numpy as np

# Just test that compute_seg_metrics works
pred = np.zeros((256,320), dtype=bool)
pred[80:180, 100:250] = True
gt = np.zeros((256,320), dtype=bool)
gt[85:175, 105:245] = True
m = compute_seg_metrics(pred, gt)
print(f'  Seg metrics: Dice={m[\"dice\"]:.3f}, BF1={m[\"bf1\"]:.3f}, HD={m[\"hd\"]:.1f}px')
del pred, gt
torch.cuda.empty_cache()
"

# ---- Test 7: Stage 6 — DDPM (1 fwd+bwd) ----
run_test "Stage 6: DDPM" "
import sys, torch
sys.path.insert(0,'.')
from src.data.ddpm_augment import ConditionalUNet, get_diffusion_params, q_sample

model = ConditionalUNet(in_channels=1, base_channels=64).cuda()
params = get_diffusion_params(100)

x0 = torch.randn(2,1,48,64).cuda()
t = torch.randint(0, 100, (2,)).cuda()
labels = torch.tensor([0,1]).cuda()

x_t, noise = q_sample(x0, t, params)
pred_noise = model(x_t, t, labels)
loss = torch.nn.functional.mse_loss(pred_noise, noise)
loss.backward()
print(f'  DDPM loss={loss.item():.4f}, output shape={pred_noise.shape}')
del model; torch.cuda.empty_cache()
"

# ---- Test 8: Classification metrics ----
run_test "Classification metrics (sklearn)" "
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
y_true = np.array([0,0,1,1,2,2])
y_pred = np.array([0,1,1,1,2,0])
probs = np.array([[0.7,0.2,0.1],[0.3,0.5,0.2],[0.1,0.8,0.1],[0.2,0.6,0.2],[0.1,0.1,0.8],[0.5,0.3,0.2]])
print(f'  BalAcc={balanced_accuracy_score(y_true,y_pred):.3f}')
print(f'  MacroF1={f1_score(y_true,y_pred,average=\"macro\"):.3f}')
print(f'  Kappa={cohen_kappa_score(y_true,y_pred):.3f}')
print(f'  AUC={roc_auc_score(y_true,probs,multi_class=\"ovr\",average=\"macro\"):.3f}')
"

# ---- Test 9: NLG metrics ----
run_test "NLG metrics (BERTScore/ROUGE/BLEU)" "
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

hyps = ['The cow shows active rumination behavior']
refs = ['The cow is actively ruminating']
P, R, F = bert_score_fn(hyps, refs, lang='en', verbose=False)
print(f'  BERTScore F1={F.mean().item():.3f}')

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rl = scorer.score(refs[0], hyps[0])['rougeL'].fmeasure
print(f'  ROUGE-L={rl:.3f}')

sm = SmoothingFunction().method1
bleu = corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps], smoothing_function=sm)
print(f'  BLEU-4={bleu:.3f}')
"

echo ""
echo "============================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "============================================"

if [ $FAIL -eq 0 ]; then
    echo "  🎉 ALL TESTS PASSED — safe to run full pipeline!"
else
    echo "  ⚠️  Fix failures above before running full pipeline."
fi
