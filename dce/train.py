"""
Training script for the DCE Fusion model.

Features:
    * Configurable backbones and head depth
    * Optional AMP mixed precision
    * Cosine LR scheduling with AdamW
    * Checkpointing (best / last)
    * Basic metrics: loss, accuracy

Usage example:
    python train_dce.py \
        --root /path/to/root \
        --labels /path/to/labels.csv \
        --epochs 50 \
        --batch-size 4 \
        --cnn-backbone convnext_tiny \
        --ssm-backbone swin_tiny_patch4_window7_224

"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from dce_model import DCEFusionModel
from dce_sequence_dataset import DCESequenceDataset, dce_collate_fn, load_label_mapping  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(
    dataset: Dataset,
    val_split: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if val_split < 0 or val_split >= 1:
        raise ValueError("val_split must satisfy 0 <= val_split < 1.")
    generator = torch.Generator().manual_seed(seed)
    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    if val_len == 0:
        train_set = dataset
        val_set = None
    else:
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=dce_collate_fn,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=dce_collate_fn,
        )
        if val_set is not None
        else None
    )
    return train_loader, val_loader


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    with torch.no_grad():
        if num_classes == 1:
            preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
            target = labels.view(-1).long()
        else:
            preds = torch.argmax(logits, dim=-1)
            target = labels.long()
        correct = (preds == target).float().sum().item()
        total = target.numel()
        return correct / max(1, total)


def train_one_epoch(
    model: DCEFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp_enabled: bool,
    num_classes: int,
    log_interval: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    for step, batch in enumerate(loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"]
        if labels is None:
            raise RuntimeError("Dataset must provide labels for supervised training.")
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            logits = outputs["logits"]
            if num_classes == 1:
                loss = loss_fn(logits.view(-1), labels.float())
            else:
                loss = loss_fn(logits, labels.long())

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        acc = compute_accuracy(logits.detach(), labels.detach(), num_classes)
        total_acc += acc * batch_size
        total_samples += batch_size

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"  [Train] step={step:04d} "
                f"loss={loss.item():.4f} "
                f"acc={acc:.4f}"
            )

    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_acc / max(1, total_samples),
    }


@torch.no_grad()
def evaluate(
    model: DCEFusionModel,
    loader: Optional[DataLoader],
    device: torch.device,
    loss_fn: nn.Module,
    num_classes: int,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["label"]
        if labels is None:
            raise RuntimeError("Validation set must provide labels.")
        labels = labels.to(device)

        outputs = model(images)
        logits = outputs["logits"]
        if num_classes == 1:
            loss = loss_fn(logits.view(-1), labels.float())
        else:
            loss = loss_fn(logits, labels.long())

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        acc = compute_accuracy(logits, labels, num_classes)
        total_acc += acc * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_acc / max(1, total_samples),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: DCEFusionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: DCEFusionModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", 0.0)
    return epoch, best_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCE Fusion model.")
    parser.add_argument("--root", required=True, help="Dataset root containing per-case directories.")
    parser.add_argument("--labels", required=True, help="CSV/TSV/JSON label file.")
    parser.add_argument("--cases", default=None, help="Comma-separated case IDs to include.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=2, help="Set to 1 for binary (BCE) classification.")
    parser.add_argument("--cnn-backbone", default="convnext_tiny")
    parser.add_argument("--ssm-backbone", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--temporal-depth", type=int, default=2)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-kernel", type=int, default=3)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--fusion-dropout", type=float, default=0.0)
    parser.add_argument("--fusion-mode", choices=["avg", "concat"], default="avg")
    parser.add_argument("--mil-depth", type=int, default=2)
    parser.add_argument("--mil-grid-size", type=int, default=4)
    parser.add_argument("--mil-dropout", type=float, default=0.1)
    parser.add_argument("--mil-aggregator", choices=["cls", "attn"], default="cls")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--per-case-norm", action="store_true")
    parser.add_argument("--global-norm", action="store_true")
    parser.add_argument("--freeze-backbones", action="store_true")
    parser.add_argument("--pretrained-backbones", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device.lower() else "cpu")
    print(f"[Info] Using device: {device}")

    label_map = load_label_mapping(args.labels)
    if not label_map:
        raise RuntimeError("Label mapping is empty. Check the label file.")

    case_ids = None
    if args.cases:
        case_ids = [cid.strip() for cid in args.cases.split(",") if cid.strip()]

    dataset = DCESequenceDataset(
        root_dir=args.root,
        case_ids=case_ids,
        label_map=label_map,
        augment=not args.no_augment,
        per_case_pre_norm=args.per_case_norm,
        global_normalize=args.global_norm,
        return_dict=True,
    )
    print(dataset.describe())

    train_loader, val_loader = prepare_dataloaders(
        dataset=dataset,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        drop_last=False,
    )

    model = DCEFusionModel(
        cnn_backbone=args.cnn_backbone,
        ssm_backbone=args.ssm_backbone,
        embed_dim=args.embed_dim,
        temporal_depth=args.temporal_depth,
        temporal_dropout=args.temporal_dropout,
        temporal_kernel=args.temporal_kernel,
        fusion_heads=args.fusion_heads,
        fusion_dropout=args.fusion_dropout,
        fusion_mode=args.fusion_mode,
        mil_depth=args.mil_depth,
        mil_grid_size=args.mil_grid_size,
        mil_dropout=args.mil_dropout,
        mil_aggregator=args.mil_aggregator,
        num_classes=args.num_classes,
        pretrained_backbones=args.pretrained_backbones,
    ).to(device)

    if args.freeze_backbones:
        model.freeze_backbones()
        print("[Info] Backbones frozen. Call model.unfreeze_backbones() later for fine-tuning if needed.")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.num_classes == 1:
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            start_epoch, best_metric = load_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            print(f"[Info] Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")
        else:
            print(f"[Warning] Resume path {ckpt_path} not found. Starting fresh.")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            scaler=scaler,
            amp_enabled=args.amp,
            num_classes=args.num_classes,
            log_interval=args.log_interval,
        )
        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            num_classes=args.num_classes,
        )
        scheduler.step()

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        if val_stats is not None:
            history["val_loss"].append(val_stats["loss"])
            history["val_acc"].append(val_stats["acc"])
            metric = val_stats["acc"]
            print(
                f"[Epoch {epoch + 1}] train_loss={train_stats['loss']:.4f} "
                f"train_acc={train_stats['acc']:.4f} "
                f"val_loss={val_stats['loss']:.4f} "
                f"val_acc={val_stats['acc']:.4f}"
            )
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(
                    save_dir / "best.pth",
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_metric=best_metric,
                    args=args,
                )
                print(f"[Info] New best validation accuracy: {best_metric:.4f}. Checkpoint saved.")
        else:
            print(
                f"[Epoch {epoch + 1}] train_loss={train_stats['loss']:.4f} "
                f"train_acc={train_stats['acc']:.4f}"
            )

    # Save final checkpoint and training history.
    save_checkpoint(
        save_dir / "last.pth",
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_metric=best_metric,
        args=args,
    )
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Info] Training completed. Best val acc = {best_metric:.4f}")


if __name__ == "__main__":
    main()