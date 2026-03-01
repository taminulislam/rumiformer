# Experiment 4: SegFormer-B0 Baseline

## Overview
Full pipeline with **SegFormer-B0** segmentation backbone (MiT-B0, smallest variant) + VideoMAE + ATF fusion.
Direct lightweight ablation of RumiFormer — same SegFormer family but B0 (3.7M) vs B2 (25M), and no TGAA.

## Model Details
| Component | Config |
|---|---|
| Seg backbone | MiT-B0 (`nvidia/mit-b0`, pretrained from HuggingFace) |
| Seg decode head | MLP decode head (shared, same as RumiFormer) |
| Temporal encoder | VideoMAE-Small (same as Exp1) |
| Fusion | ATF (same as Exp1) |
| `MODEL_NAME` | `segformer_b0` |
| `STAGE4_CHANNELS` | `256` (MiT-B0 last stage — confirmed from architecture) |
| Params | ~3.7M (seg backbone) — fastest training |

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
- `STAGE4_CHANNELS=256` is definitive — MiT-B0 stage dims are [32, 64, 160, 256]
- Smaller memory footprint: can use `--mem=32G` in sbatch
- No TGAA gates — 1a freezes only decode_head, 1b unfreezes all

## Date Created
2026-03-01
