"""
train_classifier.py
───────────────────
Phase 2: Supervised Fine-tuning

Loads SSL pretrained weights → trains the full DeepfakeDetector
on labelled real/fake data.

Usage:
    python training/train_classifier.py \\
        --data_dir /path/to/labeled_data \\
        --ssl_checkpoint ./checkpoints/ssl/best_ssl.pth \\
        --output_dir ./checkpoints/classifier \\
        --epochs 50 \\
        --batch_size 32
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # Headless support
from tqdm import tqdm
from typing import List, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.detector  import DeepfakeDetector
from deepfake_ssl.contrastive  import SupConLoss
from deepfake_ssl.pacl         import PACLLoss
from data.dataset     import get_supervised_loaders
from utils.logger     import setup_logger, MetricTracker
from utils.checkpoint import save_checkpoint, load_checkpoint


# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Supervised Fine-tuning — DeepfakeDetector")
    p.add_argument("--data_dir",       nargs='+', required=True, help="One or more data directories (e1, d2, etc.)")
    p.add_argument("--output_dir",     default="./checkpoints/classifier")
    p.add_argument("--ssl_checkpoint", default=None, help="Path to SSL pretrained .pth")
    p.add_argument("--resume",         default=None, help="Path to supervised checkpoint to resume")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--vit_model",      default="vit_base_patch16_224")
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--freeze_layers",  type=int,   default=8, help="Freeze first N ViT blocks")
    p.add_argument("--label_smoothing",type=float, default=0.1)
    p.add_argument("--sup_con_weight", type=float, default=0.2)
    p.add_argument("--pacl_weight",    type=float, default=0.1)
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--patience",       type=int,   default=10, help="Early stopping patience")
    p.add_argument("--image_size",     type=int,   default=224)
    return p.parse_args()


# ──────────────────────────────────────────────
def load_ssl_into_detector(detector: DeepfakeDetector, ssl_path: str, device: str):
    """Transfer SSL pretrained encoder weights into the detector."""
    if not os.path.exists(ssl_path):
        logging.getLogger("train_cls").warning(
            f"SSL checkpoint not found at {ssl_path}. Training from scratch."
        )
        return

    ssl_ckpt = torch.load(ssl_path, map_location=device)
    ssl_state = ssl_ckpt.get("model_state", ssl_ckpt)

    # Transfer encoder weights
    if "student_encoder" in ssl_state:
        missing, unexpected = detector.rgb_encoder.load_state_dict(
            ssl_state["student_encoder"], strict=False
        )
        logging.getLogger("train_cls").info(
            f"RGB encoder loaded from SSL. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        detector.fft_encoder.load_state_dict(
            ssl_state.get("fft_encoder", ssl_state["student_encoder"]),
            strict=False,
        )
        logging.getLogger("train_cls").info("FFT encoder loaded from SSL.")

    # Transfer PACL head
    if "pacl_head" in ssl_state:
        detector.pacl_head.load_state_dict(ssl_state["pacl_head"], strict=False)
        logging.getLogger("train_cls").info("PACL head loaded from SSL.")


# ──────────────────────────────────────────────
def plot_curves(history, output_dir):
    """Plot training and validation curves."""
    epochs = range(len(history["train_loss"]))

    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if history["val_loss"]:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    if history["val_acc"]:
        plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()


# ──────────────────────────────────────────────
def train_epoch(
    model, loader, optimizer, scaler,
    cls_loss_fn, sup_con_loss_fn, pacl_loss_fn,
    args, device,
):
    model.train()
    tracker = MetricTracker(
        "loss", "loss_cls", "loss_supcon", "loss_pacl", "acc"
    )

    all_probs  = []
    all_labels = []

    for rgb, fft, labels in tqdm(loader, desc="  Train", leave=False):
        rgb    = rgb.to(device)
        fft    = fft.to(device)
        labels = labels.to(device)

        B = rgb.shape[0]

        with autocast(enabled=args.amp):
            logit, prob, features = model(rgb, fft, return_features=True)
            logit = logit.squeeze(1)

            # ── Classification loss ──
            y_target = labels.float()
            if args.label_smoothing > 0:
                y_target = y_target * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            
            loss_cls = cls_loss_fn(logit, y_target)

            # ── Supervised Contrastive loss ──
            fused_norm = F.normalize(features["fused"], dim=-1)
            loss_supcon = sup_con_loss_fn(fused_norm, labels)

            # ── PACL supervised ──
            pacl_out = pacl_loss_fn(
                features["patch_proj"],
                features["patch_proj"],   # same-view (no second view in supervised)
                features["sim_matrix"],
                labels,
            )
            loss_pacl = pacl_out["total"]

            loss = (
                loss_cls
                + args.sup_con_weight * loss_supcon
                + args.pacl_weight    * loss_pacl
            )

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        preds = (prob >= 0.5).long()
        acc   = (preds == labels).float().mean().item()

        tracker.update(
            n=B,
            loss=loss.item(), loss_cls=loss_cls.item(),
            loss_supcon=loss_supcon.item(), loss_pacl=loss_pacl.item(),
            acc=acc,
        )
        all_probs.extend(prob.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return tracker.to_dict(), auc


# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs  = []
    all_labels = []
    total_loss = 0.0
    n = 0

    loss_fn = nn.BCEWithLogitsLoss()

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        if len(batch) == 3:
            rgb, fft, labels = batch
        else:
            rgb, labels = batch
            fft = torch.zeros_like(rgb)

        rgb    = rgb.to(device)
        fft    = fft.to(device)
        labels = labels.to(device)
        B      = rgb.shape[0]

        logit, prob = model(rgb, fft)
        logit = logit.squeeze(1)
        loss  = loss_fn(logit, labels.float())

        total_loss += loss.item() * B
        n          += B
        all_probs.extend(prob.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, [p >= 0.5 for p in all_probs])
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return {"loss": avg_loss, "acc": acc, "auc": auc}


# ──────────────────────────────────────────────
def main():
    args   = parse_args()
    logger = setup_logger("train_cls", log_dir=args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──
    loaders = get_supervised_loaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_fft=True,
    )
    logger.info(f"Loaders found: {list(loaders.keys())}")

    # ── Model ──
    model = DeepfakeDetector(
        vit_model=args.vit_model,
        pretrained=True,
        freeze_layers=args.freeze_layers,
        shared_backbone=False,
    ).to(device)

    logger.info(f"Model trainable params: {model.num_trainable_parameters:,}")

    # ── Load SSL weights ──
    if args.ssl_checkpoint:
        load_ssl_into_detector(model, args.ssl_checkpoint, str(device))

    # ── Losses ──
    cls_loss_fn     = nn.BCEWithLogitsLoss()
    sup_con_loss_fn = SupConLoss(temperature=0.07)
    pacl_loss_fn    = PACLLoss(temperature=0.1, alpha=1.0, beta=1.0, subsample=16)

    # ── Optimizer ──
    # Layer-wise LR decay: encoder gets lower LR
    encoder_params   = list(model.rgb_encoder.parameters()) + list(model.fft_encoder.parameters())
    head_params      = (
        list(model.projector.parameters()) +
        list(model.dino_head.parameters()) +
        list(model.pacl_head.parameters()) +
        list(model.fusion.parameters()) +
        list(model.classifier.parameters())
    )
    optimizer = AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},
        {"params": head_params,    "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler    = GradScaler(enabled=args.amp)

    # ── Optional Resume ──
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=str(device)
        )
        start_epoch = ckpt.get("epoch", 0) + 1

    # ── Training Loop ──
    # ── History ──
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_auc     = 0.0
    patience_cnt = 0

    logger.info(f"Datasets: {args.data_dir}")
    logger.info("Starting supervised fine-tuning …")

    for epoch in range(start_epoch, args.epochs):
        # Train
        if "train" in loaders:
            train_metrics, train_auc = train_epoch(
                model, loaders["train"], optimizer, scaler,
                cls_loss_fn, sup_con_loss_fn, pacl_loss_fn,
                args, device,
            )
        else:
            train_metrics, train_auc = {}, 0.0

        # Validate
        val_metrics = {}
        if "val" in loaders:
            val_metrics = evaluate(model, loaders["val"], device)

        scheduler.step()

        # Update history
        history["train_loss"].append(train_metrics.get("loss", 0))
        history["train_acc"].append(train_metrics.get("acc", 0))
        if val_metrics:
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])

        # Logging
        log_str = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss: {train_metrics.get('loss', 0):.4f} "
            f"acc: {train_metrics.get('acc', 0):.3f} "
            f"AUC: {train_auc:.3f}"
        )
        if val_metrics:
            log_str += (
                f" | Val loss: {val_metrics['loss']:.4f} "
                f"acc: {val_metrics['acc']:.3f} "
                f"AUC: {val_metrics['auc']:.3f}"
            )
        logger.info(log_str)

        # Save best
        current_auc = val_metrics.get("auc", train_auc)
        if current_auc > best_auc:
            best_auc = current_auc
            patience_cnt = 0
            save_checkpoint(
                os.path.join(args.output_dir, "best_model.pth"),
                model, optimizer, scheduler,
                epoch=epoch, best_metric=best_auc,
            )
            logger.info(f"  ✓ New best AUC: {best_auc:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

        # Periodic save
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth"),
                model, epoch=epoch,
            )

    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")
    logger.info(f"Best model: {os.path.join(args.output_dir, 'best_model.pth')}")

    # ── Final Testing phase ──
    test_key = next((k for k in loaders if k.lower() == "test"), None)
    if test_key:
        logger.info(f"Starting final evaluation on '{test_key}' set…")
        
        # Load best model for testing (Only if we actually trained something)
        if start_epoch < args.epochs:
            best_ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            if os.path.exists(best_ckpt_path):
                load_checkpoint(best_ckpt_path, model, device=str(device))
        else:
            logger.info("Evaluation-only mode: using current weights (no reload).")
        
        test_metrics = evaluate(model, loaders[test_key], device)
        logger.info(
            f"FINAL TEST RESULTS | "
            f"Loss: {test_metrics['loss']:.4f} | "
            f"Acc: {test_metrics['acc']:.3f} | "
            f"AUC: {test_metrics['auc']:.3f}"
        )

    # ── Plotting ──
    plot_curves(history, args.output_dir)
    logger.info(f"Training curves saved to {args.output_dir}")


if __name__ == "__main__":
    main()
