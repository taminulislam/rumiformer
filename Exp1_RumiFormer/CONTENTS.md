# Exp1_RumiFormer - Complete Training Run

## Storage: 2.0 GB total

## Directory Contents

### 1. annotations/ (4.6 MB)
Frozen dataset snapshot with sequence-based splits:
- `annotations.csv` - Master annotation file (6,005 frames)
- `split_train_val_test.csv` - 18/3/3 sequence assignments
- `clips.csv` - Temporal clip metadata
- `class_mapping.csv` - Class definitions
- `frame_features.csv` - Frame-level gas statistics

### 2. checkpoints/ (1.6 GB)
Trained model weights:

**Segmentation (1.1 GB, 6 files):**
- `segmentation_1a_epoch0000-0001_*.pt` (2 files, frozen backbone)
- `segmentation_1b_epoch0013-0024_*.pt` (3 files, full finetune)
- `segmentation_latest.pt` (symlink to epoch 24)

**Temporal (756 MB, 4 files):**
- `temporal_2b_epoch0017-0019_*.pt` (3 files, VideoMAE)
- `temporal_latest.pt` (symlink to epoch 19)

**Fusion (47 MB, 4 files):**
- `fusion_epoch0012-0014_*.pt` (3 files, ATF fusion)
- `fusion_latest.pt` (symlink to epoch 14)

**E2E (empty):**
- E2E training not yet run

### 3. logs/ (337 KB, 13 files)
Complete training logs:

**Metrics (JSONL format):**
- `segmentation_metrics.jsonl` - Latest seg run (32K)
- `segmentation_metrics_old.jsonl` - Earlier seg run (50K)
- `temporal_metrics.jsonl` - Temporal training (15K)
- `fusion_metrics.jsonl` - Fusion training (68K)
- `e2e_metrics.jsonl` - E2E training (3.5K)

**Training stdout:**
- `stage1_stdout.log` - Segmentation training output (21K)
- `temporal_stdout.log` - Temporal training output (28K)
- `fusion_stdout.log` - Fusion training output (44K)
- `e2e_stdout.log` - E2E training output (12K)

**Evaluation logs:**
- `stage5_eval.log` - E2E evaluation output (35K)
- `eval_output.log` - Test set evaluation (5K)

**Other:**
- `experiment.log` - General experiment log (22K)
- `augmentation.log` - Data augmentation info (1.7K)

### 4. eval_results/ (5.6 KB, 2 files)
Evaluation outputs:
- `classification_raw.npz` - Raw predictions (y_true, y_pred, y_prob)
- `eval_results.json` - Metrics summary (mIoU, mAP, F1, etc.)

### 5. figures/ (4.8 MB, 10 files)
Visualization plots:
- `segmentation_grid.png` (1.7 MB) - Qualitative seg results
- `qualitative_grid.png` (1.0 MB) - Full pipeline samples
- `boundary_quality.png` (772 KB) - Seg boundary analysis
- `training_curves_fusion.png` (535 KB)
- `training_curves_segmentation.png` (391 KB)
- `training_curves_temporal.png` (347 KB)
- `roc_curves.png` (92 KB) - Per-class ROC
- `pr_curves.png` (86 KB) - Precision-Recall
- `confusion_matrix.png` (67 KB)
- `classification_bars.png` (48 KB) - Per-class metrics

### 6. run_experiment.sh (4.5 KB)
Full pipeline training script with environment variable overrides for:
- Checkpoint/log directories
- Annotations CSV paths (uses local frozen snapshot)
- Epoch configurations per stage

### 7. Documentation
- `EXPERIMENT_INFO.md` - Dataset splits, training config, purpose
- `CONTENTS.md` (this file) - File inventory

## Training Summary

**Completed:**
- Stage 1: Segmentation (TGAA-SegFormer) - 8+12 epochs
- Stage 2: Temporal (VideoMAE) - 6+10 epochs
- Stage 3: Fusion (ATF) - 15 epochs
- Stage 5: E2E - 8 epochs
- Evaluation on test set

**Results:**
- Segmentation: See `figures/segmentation_grid.png`
- Classification: See `eval_results/eval_results.json`
- Training curves: See `figures/training_curves_*.png`

## Usage

To reproduce or continue training:
```bash
cd /home/siu856569517/Taminul/co2_farm
./Exp1_RumiFormer/run_experiment.sh
```

All paths are configured via environment variables in the run script.
The frozen annotations ensure reproducibility across comparison experiments.
