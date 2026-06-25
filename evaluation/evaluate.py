"""
evaluation/evaluate.py
──────────────────────
Full evaluation suite for the DeepfakeDetector.

Metrics:
  - Binary accuracy
  - AUC-ROC
  - Equal Error Rate (EER)
  - Precision / Recall / F1
  - Per-dataset breakdown (if multiple subsets present)

Usage:
    python evaluation/evaluate.py \\
        --data_dir /path/to/test_data \\
        --checkpoint ./checkpoints/classifier/best_model.pth \\
        --output_dir ./eval_results
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.detector  import DeepfakeDetector
from data.dataset     import get_supervised_loaders
from utils.checkpoint import load_checkpoint
from utils.logger     import setup_logger


# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DeepfakeDetector Evaluation")
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output_dir",  default="./eval_results")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--vit_model",   default="vit_base_patch16_224")
    p.add_argument("--split",       default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold",   type=float, default=0.5)
    return p.parse_args()


# ──────────────────────────────────────────────
def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple:
    """
    Compute Equal Error Rate (EER) — where FAR == FRR.

    Returns:
        eer       : float
        threshold : operating threshold at EER
    """
    fnr = 1.0 - tpr
    # Find the threshold where |FPR - FNR| is minimised
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return float(eer), float(thresholds[eer_idx])


# ──────────────────────────────────────────────
def compute_youden_j(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple:
    """
    Compute Youden's J statistic: J = Sensitivity + Specificity - 1
    (Equivalent to TPR - FPR).

    Returns:
        threshold : optimal threshold according to Youden's J
        j_score   : the J statistic value
    """
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(j_scores[best_idx])


# ──────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device) -> dict:
    """Run model on the full dataset and collect predictions."""
    model.eval()
    all_probs  = []
    all_labels = []
    all_paths  = []

    for batch in tqdm(loader, desc="Evaluating"):
        if len(batch) == 3:
            rgb, fft, labels = batch
        else:
            rgb, labels = batch
            fft = torch.zeros_like(rgb)

        rgb    = rgb.to(device)
        fft    = fft.to(device)

        _, probs = model(rgb, fft)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    return {
        "probs":  np.array(all_probs),
        "labels": np.array(all_labels),
    }


# ──────────────────────────────────────────────
def compute_metrics(probs, labels, threshold=0.5) -> dict:
    """Compute a comprehensive set of metrics."""
    preds = (probs >= threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(labels, probs)
    eer, eer_threshold        = compute_eer(fpr, tpr, roc_thresholds)
    youden_thresh, youden_j  = compute_youden_j(fpr, tpr, roc_thresholds)
    precision, recall, _      = precision_recall_curve(labels, probs)

    return {
        "accuracy":           float(accuracy_score(labels, preds)),
        "auc":                float(roc_auc_score(labels, probs)),
        "f1":                 float(f1_score(labels, preds)),
        "eer":                eer,
        "eer_threshold":      eer_threshold,
        "youden_threshold":   youden_thresh,
        "youden_j":           youden_j,
        "n_samples":          int(len(labels)),
        "n_real":             int((labels == 0).sum()),
        "n_fake":             int((labels == 1).sum()),
        # For plot data
        "_fpr":       fpr,
        "_tpr":       tpr,
        "_precision": precision,
        "_recall":    recall,
        "_roc_thresholds": roc_thresholds,
    }


# ──────────────────────────────────────────────
def plot_roc_curve(fpr, tpr, auc, youden_idx, youden_thresh, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2ecc71", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    # Mark Youden's Optimal Point
    ax.scatter(
        fpr[youden_idx], tpr[youden_idx],
        color="red", s=100, edgecolors="black", zorder=5,
        label=f"Youden's Optimal (T={youden_thresh:.3f})"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Deepfake Detector")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] ROC curve saved → {save_path}")


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Confusion matrix saved → {save_path}")


def plot_score_distribution(probs, labels, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    real_scores = probs[labels == 0]
    fake_scores = probs[labels == 1]
    ax.hist(real_scores, bins=50, alpha=0.6, color="#3498db", label="Real", density=True)
    ax.hist(fake_scores, bins=50, alpha=0.6, color="#e74c3c", label="Fake", density=True)
    ax.axvline(0.5, color="black", linestyle="--", label="Threshold=0.5")
    ax.set_xlabel("Fake Probability Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Score distribution saved → {save_path}")


# ──────────────────────────────────────────────
def main():
    args   = parse_args()
    logger = setup_logger("evaluate", log_dir=args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──
    model = DeepfakeDetector(
        vit_model=args.vit_model,
        pretrained=False,
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=str(device), strict=False)
    model.eval()
    logger.info(f"Model loaded from {args.checkpoint}")

    # ── Data ──
    loaders = get_supervised_loaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fft=True,
    )

    if args.split not in loaders:
        raise RuntimeError(f"Split '{args.split}' not found. Available: {list(loaders.keys())}")

    loader = loaders[args.split]

    # ── Inference ──
    results = run_inference(model, loader, device)
    probs   = results["probs"]
    labels  = results["labels"]
    preds   = (probs >= args.threshold).astype(int)

    # ── Metrics ──
    metrics = compute_metrics(probs, labels, threshold=args.threshold)

    logger.info("=" * 50)
    logger.info(f"  Split     : {args.split}")
    logger.info(f"  Samples   : {metrics['n_samples']} ({metrics['n_real']} real, {metrics['n_fake']} fake)")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  AUC-ROC   : {metrics['auc']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"  EER       : {metrics['eer']:.4f}  (threshold={metrics['eer_threshold']:.4f})")
    logger.info(f"  Youden's J: {metrics['youden_j']:.4f}  (threshold={metrics['youden_threshold']:.4f})")
    logger.info("=" * 50)
    logger.info("\n" + classification_report(labels, preds, target_names=["Real", "Fake"]))

    # ── Save JSON results ──
    export_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    json_path = os.path.join(args.output_dir, f"metrics_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(export_metrics, f, indent=2)
    logger.info(f"Metrics saved → {json_path}")

    # ── Plots ──
    # Find index of best Youden threshold for plotting
    youden_idx = np.argmin(np.abs(metrics["_roc_thresholds"] - metrics["youden_threshold"]))

    plot_roc_curve(
        metrics["_fpr"], metrics["_tpr"], metrics["auc"],
        youden_idx, metrics["youden_threshold"],
        os.path.join(args.output_dir, "roc_curve.png"),
    )
    plot_confusion_matrix(
        labels, preds,
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    plot_score_distribution(
        probs, labels,
        os.path.join(args.output_dir, "score_distribution.png"),
    )

    logger.info(f"\nAll results saved to: {args.output_dir}")
    return export_metrics


if __name__ == "__main__":
    main()
