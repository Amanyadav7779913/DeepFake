"""
heads.py
────────
Neural network heads mounted on top of the ViT backbone:

  1. ProjectionHead   — MLP projection for SSL (DINO / contrastive)
  2. PACLHead         — Patch-Level Artifact Consistency Learning head
  3. ClassifierHead   — Final binary real/fake classifier
  4. DINOHead         — DINO-specific projection + normalisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Generic MLP Projection Head
# ─────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    3-layer MLP projection head used for contrastive learning.
    Architecture: Linear → BN → ReLU → Linear → BN → ReLU → Linear

    Args:
        in_dim    : input feature dimension (e.g. 768 for ViT-B)
        hidden_dim: hidden layer dimension
        out_dim   : output projection dimension (e.g. 128 for SimCLR)
        use_bn    : use BatchNorm after each linear layer
    """

    def __init__(
        self,
        in_dim:     int = 768,
        hidden_dim: int = 2048,
        out_dim:    int = 256,
        use_bn:     bool = True,
        n_layers:   int = 3,
    ):
        super().__init__()
        layers = []
        current_dim = in_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 2. DINO Head
# ─────────────────────────────────────────────
class DINOHead(nn.Module):
    """
    DINO projection head with:
      - MLP projector
      - L2 normalisation
      - Prototypical output layer (weight-normalised)

    Args:
        in_dim        : backbone output dimension
        out_dim       : number of DINO prototypes (default: 65536)
        bottleneck_dim: MLP bottleneck before prototype layer
    """

    def __init__(
        self,
        in_dim:         int = 768,
        out_dim:        int = 65536,
        hidden_dim:     int = 2048,
        bottleneck_dim: int = 256,
        n_layers:       int = 3,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        layers = []
        current = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(current, hidden_dim), nn.GELU()]
            current = hidden_dim
        layers += [nn.Linear(current, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# ─────────────────────────────────────────────
# 3. PACL Head — Patch-Level Artifact Consistency Learning
# ─────────────────────────────────────────────
class PACLHead(nn.Module):
    """
    Patch-Level Artifact Consistency Learning (PACL) head.

    Takes patch tokens from ViT → projects them to a lower-dimensional
    space where consistency / inconsistency can be measured.

    The PACL hypothesis:
      - Real images:  patch embeddings are spatially consistent
                      (smooth texture, coherent lighting → high cosine sim)
      - Fake images:  patch embeddings show local inconsistencies
                      (GAN seams, blending artefacts → low / noisy cosine sim)

    Outputs:
        patch_proj   : (B, N, pacl_dim)   projected patch features
        sim_matrix   : (B, N, N)          pairwise cosine similarity
        consistency  : (B,)               mean patch consistency score
    """

    def __init__(
        self,
        in_dim:   int = 768,
        pacl_dim: int = 128,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, pacl_dim),
        )

    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens : (B, N, D)  raw patch features from ViT

        Returns:
            patch_proj   : (B, N, pacl_dim)
            sim_matrix   : (B, N, N)
            consistency  : (B,)
        """
        patch_proj = self.projector(patch_tokens)          # (B, N, pacl_dim)
        patch_norm = F.normalize(patch_proj, dim=-1, p=2)  # (B, N, pacl_dim)

        # Pairwise cosine similarity matrix
        sim_matrix = torch.bmm(patch_norm, patch_norm.transpose(1, 2))  # (B, N, N)

        # Consistency score: mean off-diagonal similarity
        B, N, _ = sim_matrix.shape
        mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[:, mask].view(B, N * (N - 1))
        consistency = off_diag.mean(dim=-1)  # (B,)

        return patch_proj, sim_matrix, consistency


# ─────────────────────────────────────────────
# 4. Binary Classifier Head
# ─────────────────────────────────────────────
class ClassifierHead(nn.Module):
    """
    Final classification head.
    Input: fused RGB + FFT features (CLS tokens concatenated → 2*D)
    Output: logit for fake probability

    Args:
        in_dim    : fused input dimension (e.g. 2 * 768 = 1536)
        hidden_dim: hidden FC dimension
        dropout   : dropout rate
    """

    def __init__(
        self,
        in_dim:     int = 1536,
        hidden_dim: int = 512,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logit : (B, 1) — apply sigmoid externally for probability
        """
        return self.net(x)


# ─────────────────────────────────────────────
# 5. Fusion Module
# ─────────────────────────────────────────────
class FeatureFusion(nn.Module):
    """
    Fuses RGB and FFT CLS-token features via learned attention weights.

    Supports two modes:
      - "concat"    : simple concatenation  → Linear reduction
      - "attention" : cross-attention style gating
    """

    def __init__(
        self,
        feat_dim: int = 768,
        mode:     str = "concat",
    ):
        super().__init__()
        self.mode = mode

        if mode == "concat":
            self.reducer = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.GELU(),
                nn.LayerNorm(feat_dim),
            )
        elif mode == "attention":
            self.gate = nn.Sequential(
                nn.Linear(feat_dim * 2, 2),
                nn.Softmax(dim=-1),
            )
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

        self.out_dim = feat_dim if mode == "attention" else feat_dim

    def forward(
        self,
        rgb_feat: torch.Tensor,
        fft_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rgb_feat : (B, D)
            fft_feat : (B, D)
        Returns:
            fused    : (B, D)
        """
        if self.mode == "concat":
            return self.reducer(torch.cat([rgb_feat, fft_feat], dim=-1))
        else:  # attention
            weights = self.gate(torch.cat([rgb_feat, fft_feat], dim=-1))  # (B, 2)
            fused = weights[:, 0:1] * rgb_feat + weights[:, 1:2] * fft_feat
            return fused
