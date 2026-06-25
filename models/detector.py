"""
detector.py
───────────
Complete DeepfakeDetector model.

Architecture:
  Input (any res.)
    ├── RGB Stream  → ViTBackbone → CLS token + Patch tokens
    └── FFT Stream  → ViTBackbone → CLS token + Patch tokens
                                            │
                              ┌─────────────┴────────────┐
                         PACL Head               Feature Fusion
                              │                       │
                    Patch consistency           Classifier Head
                       signal (SSL)              (real / fake)

Two operating modes:
  - SSL mode    : returns SSL-relevant features (projections, patches, etc.)
  - Inference   : returns logit + confidence

Two backbone strategies:
  - shared      : RGB and FFT share the same ViT weights (parameter efficient)
  - independent : separate ViT encoders per stream (more expressive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from models.backbone import ViTBackbone
from models.heads import (
    ProjectionHead,
    DINOHead,
    PACLHead,
    ClassifierHead,
    FeatureFusion,
)


class DeepfakeDetector(nn.Module):
    """
    Full dual-stream deepfake detector.

    Args:
        vit_model        : timm ViT model name
        pretrained       : use ImageNet-pretrained ViT
        shared_backbone  : share ViT weights between RGB and FFT streams
        proj_dim         : contrastive projection dimension
        dino_out_dim     : DINO prototype dimension
        pacl_dim         : PACL patch projection dimension
        fusion_mode      : "concat" | "attention"
        freeze_layers    : freeze first N ViT blocks
        dropout          : classifier dropout
    """

    def __init__(
        self,
        vit_model:       str   = "vit_base_patch16_224",
        pretrained:      bool  = True,
        shared_backbone: bool  = False,
        proj_dim:        int   = 256,
        dino_out_dim:    int   = 65536,
        pacl_dim:        int   = 128,
        fusion_mode:     str   = "concat",
        freeze_layers:   int   = 0,
        dropout:         float = 0.3,
    ):
        super().__init__()

        # ── Backbone(s) ──────────────────────────
        self.rgb_encoder = ViTBackbone(
            model_name=vit_model,
            pretrained=pretrained,
            freeze_layers=freeze_layers,
        )
        embed_dim = self.rgb_encoder.embed_dim  # 768 for ViT-B

        if shared_backbone:
            self.fft_encoder = self.rgb_encoder   # shared weights
        else:
            self.fft_encoder = ViTBackbone(
                model_name=vit_model,
                pretrained=pretrained,
                freeze_layers=freeze_layers,
            )

        # ── SSL Heads ────────────────────────────
        # Contrastive (SimCLR-style)
        self.projector = ProjectionHead(
            in_dim=embed_dim,
            hidden_dim=2048,
            out_dim=proj_dim,
        )

        # DINO
        self.dino_head = DINOHead(
            in_dim=embed_dim,
            out_dim=dino_out_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
        )

        # PACL (applied to RGB patches by default, optionally both)
        self.pacl_head = PACLHead(
            in_dim=embed_dim,
            pacl_dim=pacl_dim,
        )

        # ── Fusion + Classifier ──────────────────
        self.fusion = FeatureFusion(feat_dim=embed_dim, mode=fusion_mode)

        # After concat fusion: 2 * embed_dim → embed_dim (via FeatureFusion concat mode)
        # After attention fusion: embed_dim
        fused_dim = embed_dim  # FeatureFusion always returns embed_dim

        self.classifier = ClassifierHead(
            in_dim=fused_dim,
            hidden_dim=512,
            dropout=dropout,
        )

        # Store config
        self.embed_dim  = embed_dim
        self.proj_dim   = proj_dim
        self.pacl_dim   = pacl_dim

    # ─────────────────────────────────────────────
    def forward_features(self, rgb: torch.Tensor, fft: torch.Tensor):
        """
        Shared feature extraction for both SSL and classification.

        Returns:
            rgb_cls    : (B, D)
            rgb_patches: (B, N, D)
            fft_cls    : (B, D)
            fft_patches: (B, N, D)
            fused      : (B, D)
        """
        rgb_cls, rgb_patches = self.rgb_encoder(rgb)
        fft_cls, fft_patches = self.fft_encoder(fft)
        fused = self.fusion(rgb_cls, fft_cls)
        return rgb_cls, rgb_patches, fft_cls, fft_patches, fused

    # ─────────────────────────────────────────────
    def forward_ssl(self, rgb: torch.Tensor, fft: torch.Tensor) -> Dict:
        """
        SSL-mode forward pass.
        Returns all intermediate representations needed for DINO / InfoNCE / PACL losses.

        Args:
            rgb : (B, 3, H, W)
            fft : (B, 3, H, W)

        Returns dict with keys:
            cls_proj     : (B, proj_dim)   for InfoNCE
            dino_logits  : (B, dino_out)   for DINO loss
            patch_proj   : (B, N, pacl_dim)
            sim_matrix   : (B, N, N)
            consistency  : (B,)
            fused        : (B, D)
        """
        rgb_cls, rgb_patches, fft_cls, fft_patches, fused = \
            self.forward_features(rgb, fft)

        # Contrastive projection (on RGB CLS)
        cls_proj = F.normalize(self.projector(rgb_cls), dim=-1)

        # DINO projection (on RGB CLS)
        dino_logits = self.dino_head(rgb_cls)

        # PACL (on RGB patches)
        patch_proj, sim_matrix, consistency = self.pacl_head(rgb_patches)

        return {
            "cls_proj":    cls_proj,
            "dino_logits": dino_logits,
            "patch_proj":  patch_proj,
            "sim_matrix":  sim_matrix,
            "consistency": consistency,
            "fused":       fused,
            "rgb_cls":     rgb_cls,
            "fft_cls":     fft_cls,
        }

    # ─────────────────────────────────────────────
    def forward(
        self,
        rgb: torch.Tensor,
        fft: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple:
        """
        Classification-mode forward pass.

        Args:
            rgb             : (B, 3, 224, 224)
            fft             : (B, 3, 224, 224)
            return_features : also return intermediate features

        Returns:
            logit           : (B, 1)
            prob            : (B,)   sigmoid probability of FAKE
            [features dict if return_features=True]
        """
        rgb_cls, rgb_patches, fft_cls, fft_patches, fused = \
            self.forward_features(rgb, fft)

        logit = self.classifier(fused)          # (B, 1)
        prob  = torch.sigmoid(logit).squeeze(1) # (B,)

        if return_features:
            patch_proj, sim_matrix, consistency = self.pacl_head(rgb_patches)
            return logit, prob, {
                "rgb_cls":    rgb_cls,
                "fft_cls":    fft_cls,
                "fused":      fused,
                "patch_proj": patch_proj,
                "sim_matrix": sim_matrix,
                "consistency": consistency,
            }

        return logit, prob

    # ─────────────────────────────────────────────
    def predict(
        self,
        rgb: torch.Tensor,
        fft: torch.Tensor,
    ) -> Tuple[str, float]:
        """
        High-level inference. Returns human-readable label + confidence.

        Args:
            rgb : (1, 3, 224, 224)
            fft : (1, 3, 224, 224)

        Returns:
            label      : "Real" | "Fake"
            confidence : float in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            _, prob = self.forward(rgb, fft)
        p = prob.item()
        label      = "Fake" if p >= 0.5 else "Real"
        confidence = p if p >= 0.5 else 1.0 - p
        return label, confidence

    # ─────────────────────────────────────────────
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Teacher Model for DINO EMA
# ─────────────────────────────────────────────
class TeacherModel(nn.Module):
    """
    EMA (exponential moving average) teacher for DINO.
    Wraps ViTBackbone + DINOHead; parameters are never directly trained.
    Updated via copy_from_student().
    """

    def __init__(
        self,
        vit_model:    str  = "vit_base_patch16_224",
        pretrained:   bool = True,
        dino_out_dim: int  = 65536,
    ):
        super().__init__()
        self.encoder = ViTBackbone(
            model_name=vit_model,
            pretrained=pretrained,
        )
        embed_dim = self.encoder.embed_dim
        self.head = DINOHead(
            in_dim=embed_dim,
            out_dim=dino_out_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
        )

        # No gradients for teacher
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns DINO logits for global crops."""
        cls_token, _ = self.encoder(x)
        return self.head(cls_token)

    @torch.no_grad()
    def update_ema(self, student_encoder, student_head, momentum: float):
        """
        EMA update:  teacher = momentum * teacher + (1 - momentum) * student
        """
        for t_param, s_param in zip(
            self.encoder.parameters(), student_encoder.parameters()
        ):
            t_param.data.mul_(momentum).add_(s_param.data * (1.0 - momentum))

        for t_param, s_param in zip(
            self.head.parameters(), student_head.parameters()
        ):
            t_param.data.mul_(momentum).add_(s_param.data * (1.0 - momentum))
