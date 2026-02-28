"""Stage 1: TGAA-RumiFormer Segmentation Pretraining (4-GPU DDP).

Sub-stage 1a: Freeze backbone, train TGAA gates + decode head (20 epochs, lr=6e-5)
Sub-stage 1b: Unfreeze all, full fine-tune (30 epochs, lr=1e-5)

Loss: 0.5*BCE + 0.5*Dice
Input: thermal overlay (3, 256, 320) + thermal_intensity (1, 256, 320)
Target: binary gas mask (1, 256, 320)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ThermalFrameDataset
from src.models.rumiformer import RumiFormer
from src.utils.config import SegmentationConfig
from src.utils.trainer import (CheckpointManager, ETATracker, MetricsLogger,
                                cleanup_ddp, get_cosine_schedule_with_warmup,
                                get_sampler, is_main_process, print_header,
                                set_seed, setup_ddp, unwrap_model,
                                wrap_model_ddp)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum(dim=(2, 3))
    union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def combined_loss(pred, target, bce_w=0.5, dice_w=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dl = dice_loss(pred, target)
    return bce_w * bce + dice_w * dl, {"bce": bce.item(), "dice": dl.item()}


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    config, epoch, global_step, eta, logger, sampler):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss = 0.0
    for i, batch in enumerate(loader):
        overlay = batch["overlay"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        intensity = batch["thermal_intensity"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16):
            out = model(overlay, thermal_intensity=intensity,
                        binary_mask=mask if config.use_aux_mask else None)
            loss, loss_parts = combined_loss(out["seg_logits"], mask,
                                             config.bce_weight, config.dice_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        total_loss += loss.item()

        if global_step % config.log_every_n_steps == 0 and is_main_process():
            eta_str, elapsed = eta.step(global_step)
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [E{epoch:02d} S{global_step:05d}] "
                  f"loss={loss.item():.4f} bce={loss_parts['bce']:.4f} "
                  f"dice={loss_parts['dice']:.4f} lr={lr:.2e} "
                  f"ETA={eta_str} elapsed={elapsed}")
            logger.log({"train/loss": loss.item(),
                         **{f"train/{k}": v for k, v in loss_parts.items()},
                         "train/lr": lr}, step=global_step)

    avg_loss = total_loss / len(loader)
    return avg_loss, global_step


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for batch in loader:
        overlay = batch["overlay"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        intensity = batch["thermal_intensity"].to(device, non_blocking=True)

        with autocast(dtype=torch.bfloat16):
            raw = unwrap_model(model)
            out = raw(overlay, thermal_intensity=intensity,
                      binary_mask=mask if config.use_aux_mask else None)
            loss, _ = combined_loss(out["seg_logits"], mask,
                                    config.bce_weight, config.dice_weight)

        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
        intersection = (pred * mask).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - intersection
        iou = (intersection / (union + 1e-6)).mean()

        total_loss += loss.item()
        total_iou += iou.item()
        n += 1

    return {"val/loss": total_loss / n, "val/mIoU": total_iou / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, device = setup_ddp()
    config = SegmentationConfig()
    set_seed(config.seed + rank)
    print_header("Stage 1: Segmentation Pretraining", config)
    if is_main_process():
        print(f"World size: {world_size} GPUs")

    # Data
    train_ds = ThermalFrameDataset(split="train", img_size=config.img_size, augment=True)
    val_ds = ThermalFrameDataset(split="val", img_size=config.img_size, augment=False)
    train_sampler = get_sampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=config.num_workers, pin_memory=config.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    if is_main_process():
        print(f"Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    # Model
    model = RumiFormer(num_seg_classes=1, decode_dim=config.decode_dim,
                       use_aux_mask=config.use_aux_mask)
    model = wrap_model_ddp(model, device)
    scaler = GradScaler()
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.stage_name,
                                  keep_last_n=config.keep_last_n_checkpoints)
    logger = MetricsLogger(config.log_dir, config.stage_name,
                           use_wandb=config.use_wandb,
                           wandb_project=config.wandb_project)

    # ---- Sub-stage 1a: freeze backbone, train TGAA gates + decode head ----
    if is_main_process():
        print("\n--- Sub-stage 1a: Freeze backbone ---")
    raw_model = unwrap_model(model)
    for name, param in raw_model.named_parameters():
        if "tgaa" not in name.lower() and "decode_head" not in name and "mask_encoder" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in raw_model.parameters())
    if is_main_process():
        print(f"Trainable: {trainable:,} / {total:,} params")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.substage_1a_lr, weight_decay=config.weight_decay)
    total_steps_1a = config.substage_1a_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps_1a)
    eta = ETATracker(total_steps_1a)

    global_step = 0
    start_epoch = 0

    if args.resume or args.resume_from:
        path = args.resume_from
        info = ckpt_mgr.load(path, model, optimizer, scheduler, scaler) if path else ckpt_mgr.load_latest(model, optimizer, scheduler, scaler)
        if info:
            start_epoch = info["epoch"] + 1
            global_step = info["global_step"]
            eta = ETATracker(total_steps_1a)

    for epoch in range(start_epoch, config.substage_1a_epochs):
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            config, epoch, global_step, eta, logger, train_sampler)
        if is_main_process():
            val_metrics = validate(model, val_loader, device, config)
            print(f"  [Epoch {epoch:02d}] train_loss={avg_loss:.4f} "
                  f"val_loss={val_metrics['val/loss']:.4f} val_mIoU={val_metrics['val/mIoU']:.4f}")
            logger.log(val_metrics, step=global_step)
            if (epoch + 1) % config.save_every_n_epochs == 0:
                ckpt_mgr.save(model, optimizer, scheduler, scaler, epoch,
                              global_step, val_metrics, substage="1a")

    # ---- Sub-stage 1b: unfreeze all, full fine-tune ----
    if is_main_process():
        print("\n--- Sub-stage 1b: Full fine-tune ---")
    for param in model.parameters():
        param.requires_grad = True
    if is_main_process():
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.substage_1b_lr,
                                   weight_decay=config.weight_decay)
    total_steps_1b = config.substage_1b_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps_1b)
    eta = ETATracker(total_steps_1b)

    for epoch in range(config.substage_1b_epochs):
        e = epoch + config.substage_1a_epochs
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            config, e, global_step, eta, logger, train_sampler)
        if is_main_process():
            val_metrics = validate(model, val_loader, device, config)
            print(f"  [Epoch {e:02d}] train_loss={avg_loss:.4f} "
                  f"val_loss={val_metrics['val/loss']:.4f} val_mIoU={val_metrics['val/mIoU']:.4f}")
            logger.log(val_metrics, step=global_step)
            if (epoch + 1) % config.save_every_n_epochs == 0:
                ckpt_mgr.save(model, optimizer, scheduler, scaler, e,
                              global_step, val_metrics, substage="1b")

    logger.finish()
    cleanup_ddp()
    if is_main_process():
        print("\nStage 1 complete.")


if __name__ == "__main__":
    main()
