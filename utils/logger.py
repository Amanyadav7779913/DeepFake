"""
utils/logger.py
───────────────
Logging and TensorBoard setup.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name:      str              = "deepfake",
    log_dir:   Optional[str]   = None,
    log_level: int              = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger with console + optional file output.

    Args:
        name      : logger name
        log_dir   : if set, also write to {log_dir}/run_{timestamp}.log
        log_level : logging.INFO / DEBUG / WARNING

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()   # avoid duplicate handlers on re-init

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"run_{ts}.log"))
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """Tracks running average of a scalar metric."""

    def __init__(self, name: str = ""):
        self.name  = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricTracker:
    """Tracks multiple AverageMeters simultaneously."""

    def __init__(self, *names: str):
        self.meters = {name: AverageMeter(name) for name in names}

    def update(self, n: int = 1, **kwargs):
        for k, v in kwargs.items():
            if k in self.meters:
                self.meters[k].update(float(v), n)

    def reset(self):
        for m in self.meters.values():
            m.reset()

    def avg(self, name: str) -> float:
        return self.meters[name].avg

    def summary(self) -> str:
        return " | ".join(str(m) for m in self.meters.values())

    def to_dict(self) -> dict:
        return {k: m.avg for k, m in self.meters.items()}
