# Experiment 1: RumiFormer Baseline

## Overview
Full RumiFormer pipeline with TGAA-SegFormer + VideoMAE + ATF fusion.
This serves as the baseline for comparison with 7-8 alternative models.

## Dataset Split (Sequence-Based)

### Split Strategy
- **Train:** 18 sequences (75%) = 4,157 frames
- **Val:** 3 sequences (12.5%) = 1,059 frames
- **Test:** 3 sequences (12.5%) = 789 frames

### Sequences per Class
| Class | Train Seqs | Val Seqs | Test Seqs | Total Seqs |
|-------|------------|----------|-----------|------------|
| HF (High-Flux) | 11 | 1 | 1 | 13 |
| Control | 4 | 1 | 1 | 6 |
| LF (Low-Flux) | 3 | 1 | 1 | 5 |

### Sequence Assignments
**Train (18 seqs):**
- HF: 483, 484, 486, 490, 491, 493, 495, 496, 497, 498, 510
- Control: 499, 500, 502, 504
- LF: 505, 507, 509

**Val (3 seqs):**
- HF: 488
- Control: 503
- LF: 508

**Test (3 seqs):**
- HF: 492
- Control: 501
- LF: 506

## Training Configuration

### Stage 1: Segmentation (TGAA-SegFormer)
- **1a (frozen backbone):** 8 epochs, lr=6e-5
- **1b (full finetune):** 12 epochs, lr=1e-5
- Batch size: 32
- Loss: BCE (0.5) + Dice (0.5)

### Stage 2: Temporal (VideoMAE)
- **2a (frozen backbone):** 6 epochs
- **2b (full finetune):** 10 epochs
- Clip length: 16 frames
- Temporal stride: 2

### Stage 3: Fusion (ATF)
- **Epochs:** 15
- Fuses segmentation + temporal features
- Loads pretrained Stage 1 + Stage 2 checkpoints

### Stage 5: End-to-End
- **Epochs:** 8
- Joint segmentation + classification fine-tuning
- Loads all pretrained components

## Key Notes
- **Data Leakage Prevention:** Splits are sequence-based (no frames from same cow in train/val/test)
- **Data Scarcity:** Only 3-4 training sequences for Control/LF classes
- **Seed:** 42 (reproducible splits)
- **Environment:** RTX 6000 Ada, conda env `rumiformer`

## File Structure
```
Exp1_RumiFormer/
├── annotations/           # Frozen dataset for this experiment
│   ├── annotations.csv    # Master annotation file
│   ├── split_train_val_test.csv
│   ├── clips.csv
│   ├── class_mapping.csv
│   └── frame_features.csv
├── checkpoints/           # Model checkpoints per stage
│   ├── segmentation/
│   ├── temporal/
│   ├── fusion/
│   └── e2e/
├── logs/                  # Training logs
├── eval_results/          # Evaluation outputs
├── figures/               # Visualizations
└── run_experiment.sh      # Full pipeline script
```

## Running the Experiment
```bash
cd /home/siu856569517/Taminul/co2_farm
./Exp1_RumiFormer/run_experiment.sh
```

## Date Created
2026-02-28

## Purpose
Establish RumiFormer baseline performance for comparison with alternative architectures.
