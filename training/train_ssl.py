"""
train_ssl.py
────────────
Phase 1: Self-Supervised Pretraining

Combines three SSL objectives:
  1. DINO self-distillation loss       (student vs teacher, multi-crop)
  2. InfoNCE contrastive loss          (SimCLR-style pairs)
  3. PACL patch-level consistency loss (novel component)

Usage:
    python training/train_ssl.py \\
        --data_dir /path/to/unlabeled_images \\
        --output_dir ./checkpoints/ssl \\
        --epochs 100 \\
        --batch_size 64 \\
        --vit_model vit_base_patch16_224
"""

import os
import sys
import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ── Make sure project root is on path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.backbone  import ViTBackbone
from models.heads     import DINOHead, ProjectionHead, PACLHead
from models.detector  import TeacherModel
from deepfake_ssl.dino         import DINOLoss, EMAScheduler, MultiCropWrapper
from deepfake_ssl.contrastive  import InfoNCELoss
from deepfake_ssl.pacl         import PACLLoss, compute_pacl_features
from data.dataset     import get_ssl_loader
from data.augmentations import FFTTransform
from utils.logger     import setup_logger, MetricTracker
from utils.checkpoint import save_checkpoint, load_checkpoint


# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SSL Pretraining — DINO + InfoNCE + PACL")
    p.add_argument("--data_dir",     required=True,           help="Root dir with unlabelled images")
    p.add_argument("--output_dir",   default="./checkpoints/ssl")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--vit_model",    default="vit_base_patch16_224")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--n_local_crops",type=int,   default=6)
    p.add_argument("--dino_out_dim", type=int,   default=65536)
    p.add_argument("--proj_dim",     type=int,   default=256)
    p.add_argument("--pacl_dim",     type=int,   default=128)
    p.add_argument("--ema_momentum", type=float, default=0.996)
    p.add_argument("--dino_weight",  type=float, default=1.0)
    p.add_argument("--nce_weight",   type=float, default=0.5)
    p.add_argument("--pacl_weight",  type=float, default=0.5)
    p.add_argument("--resume",       default=None,            help="Path to checkpoint to resume")
    p.add_argument("--amp",          action="store_true",     help="Use automatic mixed precision")
    p.add_argument("--log_interval", type=int,   default=50)
    p.add_argument("--save_interval",type=int,   default=10)
    return p.parse_args()


# ──────────────────────────────────────────────
def build_ssl_models(args, device):
    """Build student network + teacher network."""

    # ── Student ──
    student_encoder = ViTBackbone(
        model_name=args.vit_model,
        pretrained=True,
    ).to(device)

    student_dino_head = DINOHead(
        in_dim=student_encoder.embed_dim,
        out_dim=args.dino_out_dim,
    ).to(device)

    student_proj_head = ProjectionHead(
        in_dim=student_encoder.embed_dim,
        out_dim=args.proj_dim,
    ).to(device)

    pacl_head = PACLHead(
        in_dim=student_encoder.embed_dim,
        pacl_dim=args.pacl_dim,
    ).to(device)

    fft_encoder = ViTBackbone(
        model_name=args.vit_model,
        pretrained=True,
    ).to(device)

    # ── Teacher (EMA copy, no gradient) ──
    teacher = TeacherModel(
        vit_model=args.vit_model,
        pretrained=True,
        dino_out_dim=args.dino_out_dim,
    ).to(device)

    # Initialise teacher with student weights
    teacher.encoder.load_state_dict(student_encoder.state_dict())
    teacher.head.load_state_dict(student_dino_head.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # Multi-crop wrapper for student
    student_mc = MultiCropWrapper(student_encoder, student_dino_head)

    return dict(
        student_encoder   = student_encoder,
        student_dino_head = student_dino_head,
        student_proj_head = student_proj_head,
        pacl_head         = pacl_head,
        fft_encoder       = fft_encoder,
        teacher           = teacher,
        student_mc        = student_mc,
    )


# ──────────────────────────────────────────────
def build_optimizer(models: dict, args):
    params = []
    for key, m in models.items():
        if key == "teacher":
            continue   # not directly optimised
        params += list(m.parameters())

    optimizer = AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    return optimizer


# ──────────────────────────────────────────────
def train_one_epoch(
    models,
    loader,
    optimizer,
    scaler,
    dino_loss_fn,
    nce_loss_fn,
    pacl_loss_fn,
    ema_scheduler,
    epoch,
    args,
    logger,
    device,
):
    for m in models.values():
        m.train()
    models["teacher"].eval()   # teacher always in eval

    fft_transform = FFTTransform(size=224)

    tracker = MetricTracker(
        "loss_total", "loss_dino", "loss_nce", "loss_pacl_total",
        "loss_pacl_cont", "loss_pacl_cons",
    )

    pbar = tqdm(loader, desc=f"SSL Epoch {epoch}", leave=False)

    for step, batch in enumerate(pbar):
        crops  = [c.to(device) for c in batch["crops"]]   # list of tensors
        view1  = batch["view1"].to(device)                 # (B,3,224,224)
        view2  = batch["view2"].to(device)
        fft_v  = batch["fft"].to(device)                   # (B,3,224,224)

        B = view1.shape[0]

        # Global crops only for teacher (first 2 crops are global)
        global_crops_only = crops[:2]

        with autocast(enabled=args.amp):

            # ── DINO ──────────────────────────────────
            # Student: all crops (2 global + N local)
            student_all = models["student_mc"](crops)           # (B*(2+N), out_dim)

            # Teacher: global crops only (no gradient)
            with torch.no_grad():
                teacher_global = torch.cat([
                    models["teacher"](g) for g in global_crops_only
                ], dim=0)                                        # (B*2, out_dim)

            loss_dino = dino_loss_fn(student_all, teacher_global, epoch)

            # ── InfoNCE / Contrastive ───────────────
            feat1, _ = models["student_encoder"](view1)         # (B, D)
            feat2, _ = models["student_encoder"](view2)         # (B, D)
            proj1 = F.normalize(models["student_proj_head"](feat1), dim=-1)
            proj2 = F.normalize(models["student_proj_head"](feat2), dim=-1)
            loss_nce = nce_loss_fn(proj1, proj2)

            # ── PACL ────────────────────────────────
            patches_v1, patches_v2, sim_matrix = compute_pacl_features(
                models["student_encoder"],
                models["pacl_head"],
                view1, view2,
            )
            pacl_out   = pacl_loss_fn(patches_v1, patches_v2, sim_matrix, labels=None)
            loss_pacl  = pacl_out["total"]

            # ── Combined Loss ───────────────────────
            loss = (
                args.dino_weight  * loss_dino +
                args.nce_weight   * loss_nce  +
                args.pacl_weight  * loss_pacl
            )

        # ── Backprop ──────────────────────────────
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for m in models.values() if m is not models["teacher"]
                 for p in m.parameters()],
                max_norm=3.0,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for m in models.values() if m is not models["teacher"]
                 for p in m.parameters()],
                max_norm=3.0,
            )
            optimizer.step()

        # ── EMA Teacher Update ───────────────────
        momentum = ema_scheduler.step()
        models["teacher"].update_ema(
            models["student_encoder"],
            models["student_dino_head"],
            momentum,
        )

        # ── Logging ──────────────────────────────
        tracker.update(
            n=B,
            loss_total    = loss.item(),
            loss_dino     = loss_dino.item(),
            loss_nce      = loss_nce.item(),
            loss_pacl_total = loss_pacl.item(),
            loss_pacl_cont  = pacl_out["contrastive"].item(),
            loss_pacl_cons  = pacl_out["consistency"].item(),
        )

        if (step + 1) % args.log_interval == 0:
            pbar.set_postfix({
                "L": f"{tracker.avg('loss_total'):.4f}",
                "DINO": f"{tracker.avg('loss_dino'):.4f}",
                "NCE":  f"{tracker.avg('loss_nce'):.4f}",
                "PACL": f"{tracker.avg('loss_pacl_total'):.4f}",
                "m":    f"{momentum:.4f}",
            })

    return tracker.to_dict()


# ──────────────────────────────────────────────
def main():
    args   = parse_args()
    logger = setup_logger("train_ssl", log_dir=args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── DataLoader ──
    loader = get_ssl_loader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_local_crops=args.n_local_crops,
    )
    steps_per_epoch = len(loader)
    logger.info(f"Training steps per epoch: {steps_per_epoch}")

    # ── Models ──
    models = build_ssl_models(args, device)
    logger.info(f"Student encoder params: {sum(p.numel() for p in models['student_encoder'].parameters()):,}")

    # ── Optimizer & Scheduler ──
    optimizer = build_optimizer(models, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Losses ──
    dino_loss_fn = DINOLoss(
        out_dim=args.dino_out_dim,
        n_local_crops=args.n_local_crops,
        n_epochs=args.epochs,
    ).to(device)

    nce_loss_fn  = InfoNCELoss(temperature=0.07)
    pacl_loss_fn = PACLLoss(
        temperature=0.1,
        alpha=1.0,
        beta=1.0,
        subsample=16,
    )

    # ── EMA Scheduler ──
    ema_scheduler = EMAScheduler(
        base_momentum=args.ema_momentum,
        n_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    scaler = GradScaler(enabled=args.amp)

    # ── Optional Resume ──
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from bundled checkpoint: {args.resume}")
        # Use torch.load directly to unpack the bundled models
        ckpt = torch.load(args.resume, map_location=device)
        model_state = ckpt["model_state"]
        
        # 1. Load sub-models (Student and auxiliary heads)
        for key in ["student_encoder", "student_dino_head", "student_proj_head", "pacl_head", "fft_encoder"]:
            if key in model_state and key in models:
                models[key].load_state_dict(model_state[key])
                logger.debug(f"  ✓ Loaded weights for {key}")

        # 2. Synchronize teachers (DINO requires teacher to start from student)
        if "student_encoder" in models and "teacher_encoder" in models:
            models["teacher_encoder"].load_state_dict(models["student_encoder"].state_dict())
        if "student_dino_head" in models and "teacher_dino_head" in models:
            models["teacher_dino_head"].load_state_dict(models["student_dino_head"].state_dict())
        logger.info("Synchronized teacher models with student weights.")

        # 3. Restore optimizer/scheduler
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed successfully at epoch {start_epoch}")

    # ── Training Loop ──
    logger.info("Starting SSL pretraining …")
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        metrics = train_one_epoch(
            models, loader, optimizer, scaler,
            dino_loss_fn, nce_loss_fn, pacl_loss_fn,
            ema_scheduler, epoch, args, logger, device,
        )
        scheduler.step()

        loss_val = metrics["loss_total"]
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_interval == 0:
            # Bundle encoder + heads into a single state dict
            ssl_state = {
                "student_encoder":   models["student_encoder"].state_dict(),
                "student_dino_head": models["student_dino_head"].state_dict(),
                "student_proj_head": models["student_proj_head"].state_dict(),
                "pacl_head":         models["pacl_head"].state_dict(),
                "fft_encoder":       models["fft_encoder"].state_dict(),
            }
            torch.save({
                "epoch":       epoch,
                "model_state": ssl_state,
                "best_metric": -loss_val,
            }, os.path.join(args.output_dir, f"ssl_epoch_{epoch:03d}.pth"))

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save({
                "epoch":       epoch,
                "model_state": {
                    "student_encoder":   models["student_encoder"].state_dict(),
                    "student_dino_head": models["student_dino_head"].state_dict(),
                    "student_proj_head": models["student_proj_head"].state_dict(),
                    "pacl_head":         models["pacl_head"].state_dict(),
                    "fft_encoder":       models["fft_encoder"].state_dict(),
                },
                "best_metric": -best_loss,
            }, os.path.join(args.output_dir, "best_ssl.pth"))
            logger.info(f"  ✓ New best SSL loss: {best_loss:.4f}")

    logger.info("SSL pretraining complete.")
    logger.info(f"Best checkpoint: {os.path.join(args.output_dir, 'best_ssl.pth')}")


if __name__ == "__main__":
    main()
