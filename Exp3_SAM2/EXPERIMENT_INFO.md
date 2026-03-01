# Experiment 3: SAM2 Baseline

## Overview
Full pipeline with **SAM2** segmentation backbone (MobileViTv2-200) + VideoMAE + ATF fusion.
Comparison against Exp1 (RumiFormer). Only the segmentation backbone differs.

## Model Details
| Component | Config |
|---|---|
| Seg backbone | MobileViTv2-200 (`mobilevitv2_200`, pretrained) |
| Seg decode head | MLP decode head (shared, same as RumiFormer) |
| Temporal encoder | VideoMAE-Small (same as Exp1) |
| Fusion | ATF (same as Exp1) |
| `MODEL_NAME` | `sam2` |
| `STAGE4_CHANNELS` | `512` (MobileViTv2-200 last stage) |
| Params | ~18M (seg backbone) |

## Dataset Split (same as Exp1 — frozen)
- Train: 18 sequences / 4,157 frames
- Val:   3 sequences / 1,059 frames
- Test:  3 sequences / 789 frames

## Training Configuration (same epochs as Exp1)
| Stage | Config |
|---|---|
| 1a (frozen backbone) | 8 epochs, lr=6e-5 |
| 1b (full finetune)   | 12 epochs, lr=1e-5 |
| 2a (frozen VideoMAE) | 6 epochs |
| 2b (full VideoMAE)   | 10 epochs |
| 3 (Fusion)           | 15 epochs |
| 5 (E2E)              | 8 epochs |

## Key Notes
- Identical dataset split and seed (42) as Exp1 — results are directly comparable
- `STAGE4_CHANNELS=512` assumed for MobileViTv2-200 last stage — verify with: `print(model.backbone(torch.randn(1,3,256,320))[-1].shape)`
- No TGAA gates — 1a freezes only decode_head, 1b unfreezes all

## Date Created
2026-03-01
