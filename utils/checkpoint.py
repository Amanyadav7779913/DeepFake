"""
utils/checkpoint.py
───────────────────
Save and load model checkpoints with full training state.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def save_checkpoint(
    path:       str,
    model:      torch.nn.Module,
    optimizer:  Optional[torch.optim.Optimizer] = None,
    scheduler:  Optional[Any]                   = None,
    epoch:      int                             = 0,
    best_metric: float                          = 0.0,
    extra:      Optional[Dict]                  = None,
):
    """
    Save a training checkpoint.

    Args:
        path        : file path (.pth)
        model       : model to save (state_dict)
        optimizer   : optional optimizer state
        scheduler   : optional LR scheduler state
        epoch       : current epoch
        best_metric : best validation metric so far
        extra       : any additional items to store
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    state = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "best_metric": best_metric,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if extra is not None:
        state.update(extra)

    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}  (epoch {epoch}, metric {best_metric:.4f})")


def load_checkpoint(
    path:     str,
    model:    torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any]                   = None,
    device:    str                             = "cpu",
    strict:    bool                            = True,
) -> Dict:
    """
    Load a checkpoint into model (and optionally optimizer/scheduler).

    Returns the full checkpoint dict for access to epoch, best_metric, etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Handle DataParallel / DDP wrapped models
    state_dict = checkpoint["model_state"]
    new_state  = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")   # strip DDP prefix
        new_state[new_key] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    epoch  = checkpoint.get("epoch",       0)
    metric = checkpoint.get("best_metric", 0.0)
    logger.info(f"Loaded checkpoint from {path}  (epoch {epoch}, metric {metric:.4f})")

    return checkpoint


def load_ssl_weights(
    path:   str,
    model:  torch.nn.Module,
    device: str = "cpu",
):
    """
    Load only the backbone weights from an SSL pretrained checkpoint
    into a classifier model. Ignores SSL-specific heads.
    """
    checkpoint = torch.load(path, map_location=device)
    ssl_state  = checkpoint.get("model_state", checkpoint)

    # Filter: only transfer encoder weights
    model_state = model.state_dict()
    transferred = {}
    skipped     = []

    for k, v in ssl_state.items():
        # Strip DDP prefix
        clean_k = k.replace("module.", "")
        if clean_k in model_state and model_state[clean_k].shape == v.shape:
            transferred[clean_k] = v
        else:
            skipped.append(clean_k)

    model_state.update(transferred)
    model.load_state_dict(model_state, strict=False)

    logger.info(
        f"SSL weights loaded: {len(transferred)} params transferred, "
        f"{len(skipped)} skipped."
    )
    return len(transferred)
