import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.seg_dataset import CustomSegDataset
from models.segformer_b0 import build_segformer_b0
from utils.logger import AverageMeter, CSVLogger, save_json
from utils.metrics import SegmentationMetrics, batch_metrics_from_logits
from utils.visualize import save_visual_triplet


def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer-B0 for semantic segmentation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_dataloaders(cfg: Dict):
    train_dataset = CustomSegDataset(
        image_dir=cfg["data"]["train_image_dir"],
        mask_dir=cfg["data"]["train_mask_dir"],
        img_suffix=cfg["data"].get("img_suffix", ".jpg"),
        mask_suffix=cfg["data"].get("mask_suffix", ".png"),
        crop_size=cfg["train"]["crop_size"],
        val_size=cfg["data"].get("val_size"),
        scale_range=tuple(cfg["train"].get("scale_range", [0.5, 2.0])),
        hflip_prob=cfg["train"].get("hflip_prob", 0.5),
        training=True,
        color_jitter_prob=cfg["train"].get("color_jitter_prob", 0.2),
    )

    val_dataset = CustomSegDataset(
        image_dir=cfg["data"]["val_image_dir"],
        mask_dir=cfg["data"]["val_mask_dir"],
        img_suffix=cfg["data"].get("img_suffix", ".jpg"),
        mask_suffix=cfg["data"].get("mask_suffix", ".png"),
        crop_size=None,
        val_size=cfg["data"].get("val_size", cfg["train"]["crop_size"]),
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"].get("val_batch_size", 1),
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_optimizer(model: nn.Module, cfg: Dict):
    return AdamW(
        model.parameters(),
        lr=cfg["train"]["base_lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["train"]["weight_decay"],
    )


def build_scheduler(optimizer, total_steps: int):
    def poly_lr(step: int):
        if total_steps <= 0:
            return 1.0
        return max((1 - step / total_steps), 0.0) ** 0.9

    return LambdaLR(optimizer, lr_lambda=poly_lr)


def save_checkpoint(state: Dict, save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(save_path))


def load_checkpoint(model, optimizer, scaler, resume_path: str, device: torch.device):
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_miou = ckpt.get("best_miou", 0.0)
    return start_epoch, best_miou


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    cfg,
    writer=None,
):
    model.train()
    loss_meter = AverageMeter()
    metric_meter = SegmentationMetrics(
        num_classes=cfg["num_classes"],
        ignore_index=cfg["ignore_index"],
        device=device,
    )

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", dynamic_ncols=True)
    global_step_base = epoch * len(loader)

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg["train"].get("amp", False)):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()

        grad_clip = cfg["train"].get("grad_clip", None)
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_meter.update(loss.item(), images.size(0))

        preds = torch.argmax(logits.detach(), dim=1)
        metric_meter.update(preds, masks)

        batch_metric = batch_metrics_from_logits(
            logits.detach(),
            masks,
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
        )

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_metric['pixel_acc']:.4f}",
            "miou": f"{batch_metric['miou']:.4f}",
            "dice": f"{batch_metric['mdice']:.4f}",
            "lr": f"{lr:.2e}",
        })

        if writer is not None:
            global_step = global_step_base + step
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/batch_pixel_acc", batch_metric["pixel_acc"], global_step)
            writer.add_scalar("train/batch_miou", batch_metric["miou"], global_step)
            writer.add_scalar("train/batch_mdice", batch_metric["mdice"], global_step)
            writer.add_scalar("train/lr", lr, global_step)

    metrics = metric_meter.compute()
    result = {
        "loss": loss_meter.avg,
        "pixel_acc": metrics["pixel_acc"],
        "miou": metrics["miou"],
        "mdice": metrics["mdice"],
        "per_class_iou": metrics["per_class_iou"],
    }
    return result


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, cfg, save_dir: Path, writer=None):
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = SegmentationMetrics(
        num_classes=cfg["num_classes"],
        ignore_index=cfg["ignore_index"],
        device=device,
    )

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}", dynamic_ncols=True)
    first_batch_saved = False

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        loss_meter.update(loss.item(), images.size(0))

        preds = torch.argmax(logits, dim=1)
        metric_meter.update(preds, masks)

        batch_metric = batch_metrics_from_logits(
            logits,
            masks,
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
        )
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_metric['pixel_acc']:.4f}",
            "miou": f"{batch_metric['miou']:.4f}",
            "dice": f"{batch_metric['mdice']:.4f}",
        })

        if (not first_batch_saved) and (epoch % cfg["train"].get("save_visuals_every", 1) == 0):
            vis_dir = save_dir / "visuals"
            vis_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(images.size(0), 2)):
                save_visual_triplet(
                    image_tensor=images[i].cpu(),
                    pred_mask=preds[i].cpu(),
                    gt_mask=masks[i].cpu(),
                    save_path=str(vis_dir / f"epoch_{epoch:03d}_sample_{i}.png"),
                    class_names=cfg.get("class_names"),
                )
            first_batch_saved = True

    metrics = metric_meter.compute()
    result = {
        "loss": loss_meter.avg,
        "pixel_acc": metrics["pixel_acc"],
        "miou": metrics["miou"],
        "mdice": metrics["mdice"],
        "per_class_iou": metrics["per_class_iou"],
    }

    if writer is not None:
        writer.add_scalar("val/loss", result["loss"], epoch)
        writer.add_scalar("val/pixel_acc", result["pixel_acc"], epoch)
        writer.add_scalar("val/miou", result["miou"], epoch)
        writer.add_scalar("val/mdice", result["mdice"], epoch)

    return result


def print_epoch_summary(epoch: int, train_result: Dict, val_result: Dict, class_names=None):
    line = (
        f"[Epoch {epoch:03d}] "
        f"Train loss={train_result['loss']:.4f}, acc={train_result['pixel_acc']:.4f}, "
        f"mIoU={train_result['miou']:.4f}, Dice={train_result['mdice']:.4f} | "
        f"Val loss={val_result['loss']:.4f}, acc={val_result['pixel_acc']:.4f}, "
        f"mIoU={val_result['miou']:.4f}, Dice={val_result['mdice']:.4f}"
    )
    print(line)

    if class_names and len(class_names) == len(val_result["per_class_iou"]):
        print("Per-class IoU:")
        for name, iou in zip(class_names, val_result["per_class_iou"]):
            print(f"  - {name}: {iou:.4f}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.get("seed", 42))

    device_name = cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_segformer_b0(
        num_classes=cfg["num_classes"],
        in_chans=cfg["model"].get("in_chans", 3),
        decoder_dim=cfg["model"].get("decoder_dim", 256),
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
    ).to(device)

    if cfg["train"].get("compile_model", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optimizer = build_optimizer(model, cfg)
    total_steps = cfg["train"]["epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", False))

    writer = None
    if cfg["train"].get("use_tensorboard", True):
        writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))

    csv_logger = CSVLogger(
        csv_path=str(save_dir / "metrics.csv"),
        fieldnames=[
            "epoch",
            "train_loss",
            "train_pixel_acc",
            "train_miou",
            "train_mdice",
            "val_loss",
            "val_pixel_acc",
            "val_miou",
            "val_mdice",
            "lr",
            "time_sec",
        ],
    )

    start_epoch = 0
    best_miou = 0.0
    resume_path = cfg.get("resume", "")
    if resume_path:
        start_epoch, best_miou = load_checkpoint(model, optimizer, scaler, resume_path, device)
        print(f"Resumed from {resume_path}, start_epoch={start_epoch}, best_miou={best_miou:.4f}")

    print("=" * 90)
    print("Training configuration")
    print(f"Save dir      : {save_dir}")
    print(f"Device        : {device}")
    print(f"Num classes   : {cfg['num_classes']}")
    print(f"Train samples : {len(train_loader.dataset)}")
    print(f"Val samples   : {len(val_loader.dataset)}")
    print("=" * 90)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        tic = time.time()

        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            cfg=cfg,
            writer=writer,
        )

        val_result = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            cfg=cfg,
            save_dir=save_dir,
            writer=writer,
        )

        print_epoch_summary(epoch, train_result, val_result, class_names=cfg.get("class_names"))

        elapsed = time.time() - tic
        lr = optimizer.param_groups[0]["lr"]
        csv_logger.log({
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_pixel_acc": train_result["pixel_acc"],
            "train_miou": train_result["miou"],
            "train_mdice": train_result["mdice"],
            "val_loss": val_result["loss"],
            "val_pixel_acc": val_result["pixel_acc"],
            "val_miou": val_result["miou"],
            "val_mdice": val_result["mdice"],
            "lr": lr,
            "time_sec": elapsed,
        })

        if writer is not None:
            writer.add_scalar("train/epoch_loss", train_result["loss"], epoch)
            writer.add_scalar("train/epoch_pixel_acc", train_result["pixel_acc"], epoch)
            writer.add_scalar("train/epoch_miou", train_result["miou"], epoch)
            writer.add_scalar("train/epoch_mdice", train_result["mdice"], epoch)

        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_miou": best_miou,
            "config": cfg,
        }

        save_checkpoint(checkpoint_state, save_dir / "last.pth")

        if val_result["miou"] >= best_miou:
            best_miou = val_result["miou"]
            checkpoint_state["best_miou"] = best_miou
            save_checkpoint(checkpoint_state, save_dir / "best_miou.pth")
            save_json(
                {
                    "epoch": epoch,
                    "best_miou": best_miou,
                    "val_result": val_result,
                },
                save_dir / "best_metrics.json",
            )
            print(f"New best checkpoint saved. best_miou={best_miou:.4f}")

    if writer is not None:
        writer.close()

    print("Training finished.")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
