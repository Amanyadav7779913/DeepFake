"""
contrastive.py
──────────────
Contrastive learning losses for self-supervised pretraining:

  - InfoNCELoss      : NT-Xent / SimCLR-style loss
  - MoCo-style queue support (optional large negative pool)
  - SupConLoss       : Supervised contrastive loss for fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# InfoNCE / NT-Xent Loss
# ─────────────────────────────────────────────
class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise-Contrastive Estimation) loss.
    Also known as NT-Xent (Normalized Temperature-scaled Cross Entropy)
    as used in SimCLR.

    Given a batch of 2N embeddings (N original + N augmented):
      - Positive pair: (z_i, z_j) from the same image
      - Negative pairs: all other 2N-2 embeddings in the batch

    Loss = -log [ exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]

    Args:
        temperature : softmax temperature τ (default: 0.07)
        reduction   : "mean" | "sum"
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction   = reduction

    def forward(
        self,
        z1: torch.Tensor,   # (B, D)  — first augmented view
        z2: torch.Tensor,   # (B, D)  — second augmented view
    ) -> torch.Tensor:
        """
        Args:
            z1, z2 : L2-normalised embeddings of shape (B, D)

        Returns:
            scalar loss
        """
        assert z1.shape == z2.shape, "z1 and z2 must have same shape"
        B = z1.shape[0]
        device = z1.device

        # L2 normalise (in case not already done)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Concatenate: (2B, D)
        z = torch.cat([z1, z2], dim=0)

        # Pairwise cosine similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity on diagonal
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_self, float("-inf"))

        # Labels: positive of sample i is sample i+B (and vice versa)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0,     B, device=device),
        ])  # (2B,)

        loss = F.cross_entropy(sim, labels, reduction=self.reduction)
        return loss


# ─────────────────────────────────────────────
# MoCo-Style Memory Queue
# ─────────────────────────────────────────────
class MoCoQueue(nn.Module):
    """
    MoCo-style memory queue for maintaining a large pool of negatives.
    Stores key embeddings as a FIFO queue.

    Args:
        queue_size : number of negative samples in queue (default: 65536)
        feat_dim   : embedding dimension
    """

    def __init__(self, queue_size: int = 65536, feat_dim: int = 256):
        super().__init__()
        self.queue_size = queue_size
        self.feat_dim   = feat_dim

        # Initialise queue with random unit vectors
        self.register_buffer("queue", F.normalize(
            torch.randn(feat_dim, queue_size), dim=0
        ))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue_dequeue(self, keys: torch.Tensor):
        """
        Add new keys to queue, removing oldest entries.

        Args:
            keys : (B, D) L2-normalised embeddings
        """
        B = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Handle wrap-around
        if ptr + B > self.queue_size:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].t()
            self.queue[:, :B - remaining] = keys[remaining:].t()
        else:
            self.queue[:, ptr:ptr + B] = keys.t()

        self.queue_ptr[0] = (ptr + B) % self.queue_size

    def get_negatives(self) -> torch.Tensor:
        """Return current queue: (feat_dim, queue_size)."""
        return self.queue.clone().detach()


class MoCoLoss(nn.Module):
    """
    MoCo-v2 style contrastive loss using a momentum queue.

    Args:
        temperature : InfoNCE temperature
        queue_size  : negative queue size
        feat_dim    : embedding dimension
    """

    def __init__(
        self,
        temperature: float = 0.07,
        queue_size:  int   = 65536,
        feat_dim:    int   = 256,
    ):
        super().__init__()
        self.temperature = temperature
        self.queue = MoCoQueue(queue_size=queue_size, feat_dim=feat_dim)

    def forward(
        self,
        q: torch.Tensor,    # (B, D) query features (student/online)
        k: torch.Tensor,    # (B, D) key features (teacher/momentum)
    ) -> torch.Tensor:
        """
        Args:
            q : (B, D) L2-normalised query embeddings
            k : (B, D) L2-normalised key embeddings (no gradient)

        Returns:
            scalar InfoNCE loss
        """
        B = q.shape[0]
        device = q.device

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Positive logits: (B, 1)
        pos_logits = torch.sum(q * k, dim=-1, keepdim=True) / self.temperature

        # Negative logits: (B, queue_size)
        neg_logits = torch.mm(q, self.queue.get_negatives()) / self.temperature

        # Concatenate: (B, 1 + queue_size)
        logits = torch.cat([pos_logits, neg_logits], dim=1)

        # Labels: positives at index 0
        labels = torch.zeros(B, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)

        # Update queue with new keys
        self.queue.enqueue_dequeue(k.detach())

        return loss


# ─────────────────────────────────────────────
# Supervised Contrastive Loss
# ─────────────────────────────────────────────
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Useful for fine-tuning where labels are available.

    Positives = same-class samples; negatives = different-class samples.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,   # (B, D) L2-normalised
        labels:   torch.Tensor,   # (B,) integer class labels
    ) -> torch.Tensor:
        B      = features.shape[0]
        device = features.device

        features = F.normalize(features, dim=-1)

        # Pairwise similarity: (B, B)
        sim = torch.mm(features, features.t()) / self.temperature

        # Build positive mask: same label, different sample
        labels_eq   = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        self_mask   = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask    = labels_eq & self_mask

        # Log-sum-exp denominator over all non-self pairs
        sim_exp     = torch.exp(sim) * self_mask.float()
        log_denom   = torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-8)

        # SupCon loss
        loss_per_sample = -(sim - log_denom) * pos_mask.float()
        n_pos = pos_mask.float().sum(dim=1)
        n_pos = torch.clamp(n_pos, min=1.0)
        loss  = (loss_per_sample.sum(dim=1) / n_pos).mean()

        return loss
