"""
pacl.py
───────
Patch-Level Artifact Consistency Learning (PACL)

A novel SSL signal for deepfake detection, based on the observation that:
  - Real images: patch representations are spatially consistent
  - Fake images: patch representations show inconsistencies (GAN artefacts,
    blending seams, frequency anomalies)

Two complementary losses:

  1. PatchContrastiveLoss
       Real patches from the same image cluster together;
       patches from different images (especially real vs fake) are pushed apart.

  2. PatchConsistencyLoss
       Real images: maximise mean pairwise patch similarity
       Fake images: penalise artificially uniform patch similarity
       (some GANs produce unrealistically smooth patches → consistency alone
        isn't enough; we need to contrast real vs fake patterns)

Combined: PACLLoss = α * patch_contrastive + β * patch_consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Patch Contrastive Loss
# ─────────────────────────────────────────────
class PatchContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss over patch tokens.

    Positive pair  : patch i from view_1 ↔ patch i from view_2
                     (same spatial location, same image, different augmentation)
    Negative pairs : all other patches across the batch

    Args:
        temperature : softmax temperature
        n_patches   : number of ViT patch tokens (e.g. 196 for 224×224)
        subsample   : sample a random subset of patches per image for speed
    """

    def __init__(
        self,
        temperature: float = 0.1,
        subsample:   int   = 16,   # use 16 randomly sampled patches per image
    ):
        super().__init__()
        self.temperature = temperature
        self.subsample   = subsample

    def forward(
        self,
        patches_v1: torch.Tensor,   # (B, N, D) from view 1
        patches_v2: torch.Tensor,   # (B, N, D) from view 2
    ) -> torch.Tensor:
        """
        Args:
            patches_v1, patches_v2 : (B, N, D)  L2-normalised patch features

        Returns:
            scalar loss
        """
        B, N, D = patches_v1.shape
        device  = patches_v1.device

        # Optionally subsample patches (keeps memory tractable)
        if self.subsample and self.subsample < N:
            idx = torch.randperm(N, device=device)[:self.subsample]
            patches_v1 = patches_v1[:, idx, :]   # (B, K, D)
            patches_v2 = patches_v2[:, idx, :]
            K = self.subsample
        else:
            K = N

        # Normalise
        p1 = F.normalize(patches_v1, dim=-1)  # (B, K, D)
        p2 = F.normalize(patches_v2, dim=-1)

        total_loss = 0.0
        count = 0

        # Per-image contrastive loss (avoid O(B²·K²) complexity)
        for b in range(B):
            q = p1[b]  # (K, D) — anchors
            k = p2[b]  # (K, D) — positives

            # (K, K) similarity matrix
            sim = torch.mm(q, k.t()) / self.temperature

            # Positives: diagonal; negatives: off-diagonal
            labels = torch.arange(K, device=device)
            loss_b = F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
            total_loss += loss_b
            count += 1

        return total_loss / count


# ─────────────────────────────────────────────
# 2. Patch Consistency Loss
# ─────────────────────────────────────────────
class PatchConsistencyLoss(nn.Module):
    """
    Encourages real images to have highly consistent patch representations
    and penalises fake images for having either too-inconsistent OR
    artificially-too-uniform patch statistics.

    For real images  (label=0): maximise mean off-diagonal similarity → consistency
    For fake images  (label=1): minimise mean off-diagonal similarity → inconsistency

    Loss formulation:
      real_loss = 1 - mean(sim_matrix_off_diagonal)
      fake_loss = mean(sim_matrix_off_diagonal)         [push towards 0]

    Note: when labels are unavailable (SSL), we use a self-supervised proxy —
    applying strong vs weak augmentation and treating strongly augmented views
    as "fake-like" (inconsistent).

    Args:
        margin : minimum desired gap between real and fake consistency scores
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        sim_matrix: torch.Tensor,   # (B, N, N) pairwise cosine similarity
        labels: torch.Tensor,       # (B,) 0=real, 1=fake  OR None for SSL proxy
    ) -> torch.Tensor:
        """
        Args:
            sim_matrix : (B, N, N) from PACLHead
            labels     : (B,) or None

        Returns:
            scalar loss
        """
        B, N, _ = sim_matrix.shape
        device   = sim_matrix.device

        # Off-diagonal mask
        mask     = ~torch.eye(N, dtype=torch.bool, device=device)  # (N, N)
        off_diag = sim_matrix[:, mask].view(B, N * (N - 1))        # (B, N*(N-1))
        mean_sim = off_diag.mean(dim=-1)                            # (B,)

        if labels is None:
            # SSL proxy: just maximise consistency (acts as a regulariser)
            loss = (1.0 - mean_sim).mean()
            return loss

        real_mask = (labels == 0).float()
        fake_mask = (labels == 1).float()

        n_real = real_mask.sum().clamp(min=1)
        n_fake = fake_mask.sum().clamp(min=1)

        # Real images: push consistency towards 1
        real_loss = ((1.0 - mean_sim) * real_mask).sum() / n_real

        # Fake images: push consistency towards 0 (or at least < real)
        fake_loss = (mean_sim * fake_mask).sum() / n_fake

        # Hinge: enforce margin between real and fake consistency
        real_mean = (mean_sim * real_mask).sum() / n_real
        fake_mean = (mean_sim * fake_mask).sum() / n_fake
        margin_loss = F.relu(fake_mean - real_mean + self.margin)

        return real_loss + fake_loss + margin_loss


# ─────────────────────────────────────────────
# 3. Combined PACL Loss
# ─────────────────────────────────────────────
class PACLLoss(nn.Module):
    """
    Combined Patch-Level Artifact Consistency Learning loss.

    Wraps PatchContrastiveLoss + PatchConsistencyLoss with
    configurable weights.

    Args:
        temperature       : contrastive temperature
        consistency_margin: margin for consistency loss
        alpha             : weight for patch contrastive loss
        beta              : weight for patch consistency loss
        subsample         : number of patches to subsample per image
    """

    def __init__(
        self,
        temperature:        float = 0.1,
        consistency_margin: float = 0.2,
        alpha:              float = 1.0,
        beta:               float = 1.0,
        subsample:          int   = 16,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.contrastive = PatchContrastiveLoss(
            temperature=temperature,
            subsample=subsample,
        )
        self.consistency = PatchConsistencyLoss(
            margin=consistency_margin,
        )

    def forward(
        self,
        patches_v1:  torch.Tensor,           # (B, N, D)
        patches_v2:  torch.Tensor,           # (B, N, D)
        sim_matrix:  torch.Tensor,           # (B, N, N)
        labels:      torch.Tensor = None,    # (B,) optional
    ) -> dict:
        """
        Returns:
            dict with keys: "total", "contrastive", "consistency"
        """
        contrastive_loss = self.contrastive(patches_v1, patches_v2)
        consistency_loss = self.consistency(sim_matrix, labels)

        total = self.alpha * contrastive_loss + self.beta * consistency_loss

        return {
            "total":       total,
            "contrastive": contrastive_loss,
            "consistency": consistency_loss,
        }


# ─────────────────────────────────────────────
# Utility: compute PACL features from two crop views
# ─────────────────────────────────────────────
def compute_pacl_features(backbone, pacl_head, view1, view2):
    """
    Convenience function to extract patch features for PACL.

    Args:
        backbone  : ViTBackbone
        pacl_head : PACLHead
        view1, view2 : (B, 3, H, W)

    Returns:
        patches_v1 : (B, N, pacl_dim)
        patches_v2 : (B, N, pacl_dim)
        sim_matrix : (B, N, N)  from view1
    """
    _, patches1 = backbone(view1)
    _, patches2 = backbone(view2)

    proj1, sim_matrix, _ = pacl_head(patches1)
    proj2, _,          _ = pacl_head(patches2)

    return proj1, proj2, sim_matrix
