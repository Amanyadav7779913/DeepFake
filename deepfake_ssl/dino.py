"""
dino.py
───────
DINO Self-Distillation components:

  - DINOLoss        : loss with centering & temperature sharpening
  - EMAScheduler    : momentum schedule for teacher EMA
  - MultiCropWrapper: passes multiple crop sizes through same encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


# ─────────────────────────────────────────────
# DINO Loss
# ─────────────────────────────────────────────
class DINOLoss(nn.Module):
    """
    DINO cross-entropy loss between student and teacher predictions.

    Key components:
      1. Temperature sharpening  — student uses high temp, teacher uses low temp
      2. Centering               — teacher output is centred with EMA mean
         (prevents mode collapse without requiring explicit contrastive negatives)
      3. Cross-entropy over all student views vs teacher global views

    Args:
        out_dim           : number of DINO prototypes
        n_local_crops     : number of local crops (M in the paper)
        warmup_teacher_temp      : initial teacher temperature
        teacher_temp             : final teacher temperature (after warmup)
        warmup_teacher_temp_epochs: epochs for teacher temp warmup
        student_temp      : student softmax temperature
        center_momentum   : EMA momentum for centering
        n_epochs          : total training epochs (for scheduler)
    """

    def __init__(
        self,
        out_dim:                   int   = 65536,
        n_local_crops:             int   = 6,
        warmup_teacher_temp:       float = 0.04,
        teacher_temp:              float = 0.04,
        warmup_teacher_temp_epochs: int  = 30,
        student_temp:              float = 0.1,
        center_momentum:           float = 0.9,
        n_epochs:                  int   = 100,
    ):
        super().__init__()
        self.student_temp   = student_temp
        self.center_momentum = center_momentum
        self.n_local_crops  = n_local_crops

        # Register center as a buffer (not a parameter — updated via EMA)
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Teacher temperature schedule: linear warmup
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(0, n_epochs - warmup_teacher_temp_epochs)) * teacher_temp,
        ])

    def forward(
        self,
        student_output: torch.Tensor,   # (B*(2+M), out_dim)
        teacher_output: torch.Tensor,   # (B*2, out_dim)
        epoch:          int,
    ) -> torch.Tensor:
        """
        Compute DINO loss.

        Student processes ALL crops (2 global + M local).
        Teacher processes only the 2 GLOBAL crops.

        Both tensors should be raw logits (before softmax).
        """
        teacher_temp = self.teacher_temp_schedule[
            min(epoch, len(self.teacher_temp_schedule) - 1)
        ]

        # ── Teacher: centre + sharpen ──
        teacher_centered = teacher_output - self.center
        teacher_probs    = F.softmax(teacher_centered / teacher_temp, dim=-1).detach()

        # ── Student: sharpen ──
        # Split into per-crop chunks
        student_chunks = student_output.chunk(2 + self.n_local_crops)

        # Teacher chunks (only 2 global crops)
        teacher_chunks = teacher_probs.chunk(2)

        total_loss = 0.0
        n_loss_terms = 0

        for t_idx, t_chunk in enumerate(teacher_chunks):
            for s_idx, s_chunk in enumerate(student_chunks):
                if s_idx == t_idx:
                    continue   # skip same-view pairs

                s_probs = F.softmax(s_chunk / self.student_temp, dim=-1)

                # Cross-entropy: H(teacher, student) = -sum(teacher * log(student))
                loss = torch.sum(-t_chunk * torch.log(s_probs + 1e-8), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # ── Update center ──
        self._update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor):
        """
        EMA update of the centering vector.
        center = m * center + (1 - m) * batch_mean
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center  = (
            self.center * self.center_momentum
            + batch_center * (1.0 - self.center_momentum)
        )


# ─────────────────────────────────────────────
# EMA Momentum Scheduler
# ─────────────────────────────────────────────
class EMAScheduler:
    """
    Cosine schedule for teacher EMA momentum.
    Starts at base_momentum, increases to 1.0 over training.

    Following the DINO paper: momentum_t = 1 - (1 - base) * (cos(pi*t/T) + 1) / 2
    """

    def __init__(
        self,
        base_momentum: float = 0.996,
        n_epochs:      int   = 100,
        steps_per_epoch: int = 1000,
    ):
        total_steps = n_epochs * steps_per_epoch
        self.schedule = np.array([
            1 - (1 - base_momentum) * (np.cos(np.pi * i / total_steps) + 1) / 2
            for i in range(total_steps + 1)
        ])
        self._step = 0

    def step(self) -> float:
        m = self.schedule[min(self._step, len(self.schedule) - 1)]
        self._step += 1
        return float(m)

    @property
    def current_momentum(self) -> float:
        return float(self.schedule[min(self._step, len(self.schedule) - 1)])


# ─────────────────────────────────────────────
# Multi-Crop Wrapper
# ─────────────────────────────────────────────
class MultiCropWrapper(nn.Module):
    """
    Efficient multi-crop forward pass.

    Groups crops of the same resolution and processes them together
    in a single forward pass to maximise GPU utilisation.

    Args:
        backbone: ViTBackbone (or similar)
        head    : DINOHead (or similar projection head)
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            crops: list of tensors, each (B, 3, H, W)
                   May have different spatial sizes.

        Returns:
            output: (B * num_crops, out_dim) — all crops concatenated
        """
        # Group crops by spatial size for batched processing
        unique_sizes = {}
        for i, crop in enumerate(crops):
            size = crop.shape[-2:]
            if size not in unique_sizes:
                unique_sizes[size] = []
            unique_sizes[size].append((i, crop))

        # Process each resolution group together
        outputs = {}
        for size, idx_crops in unique_sizes.items():
            indices = [ic[0] for ic in idx_crops]
            batch   = torch.cat([ic[1] for ic in idx_crops], dim=0)

            cls_token, _ = self.backbone(batch)
            out = self.head(cls_token)

            # Split back by original batch size
            B = idx_crops[0][1].shape[0]
            chunks = out.chunk(len(idx_crops))
            for i, (orig_idx, chunk) in enumerate(zip(indices, chunks)):
                outputs[orig_idx] = chunk

        # Reconstruct in original order and concatenate
        result = torch.cat([outputs[i] for i in range(len(crops))], dim=0)
        return result
