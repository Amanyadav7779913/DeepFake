"""
backbone.py
───────────
ViT feature extractor built on top of `timm`.

Key design decisions:
  - Classification head is removed; raw features are returned.
  - Both CLS token and patch tokens are exposed.
  - Supports any `timm` ViT variant via `model_name`.
  - Handles the 96×96 local crops by interpolating positional embeddings.
"""

import math
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer


class ViTBackbone(nn.Module):
    """
    Wraps a timm ViT model and exposes:
      - cls_token  : (B, D)        global representation
      - patch_tokens: (B, N, D)    per-patch representations

    Args:
        model_name : timm model identifier (default: vit_base_patch16_224)
        pretrained : load ImageNet-21k weights
        freeze_layers: number of transformer blocks to freeze (0 = none)
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_layers: int = 0,
    ):
        super().__init__()

        # Load ViT without classification head
        self.vit: VisionTransformer = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # removes the head, returns CLS features
            dynamic_img_size=True,  # Enables support for varying input resolutions
        )

        self.embed_dim   = self.vit.embed_dim         # e.g. 768 for base
        self.patch_size  = self.vit.patch_embed.patch_size[0]  # e.g. 16
        self.num_patches = self.vit.patch_embed.num_patches    # e.g. 196

        # Optionally freeze early blocks
        if freeze_layers > 0:
            self._freeze_blocks(freeze_layers)

    # ─────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 3, H, W) — any resolution (interpolation handled below)

        Returns:
            cls_token    : (B, D)
            patch_tokens : (B, N, D)
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.vit.patch_embed(x)   # (B, N, D), (B, D, H, W) or (B, H, W, D)
        if x.ndim == 4:
            if x.shape[-1] == self.embed_dim:
                x = x.flatten(1, 2)           # (B, H, W, D) -> (B, N, D)
            else:
                x = x.flatten(2).transpose(1, 2)  # (B, D, H, W) -> (B, N, D)

        # Prepend CLS token
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)               # (B, N+1, D)

        # Positional embedding — interpolate if resolution differs from training
        x = x + self._interpolate_pos_embed(x, H, W)

        x = self.vit.pos_drop(x)

        # Transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)   # (B, N+1, D)

        cls_token    = x[:, 0]       # (B, D)
        patch_tokens = x[:, 1:]      # (B, N, D)

        return cls_token, patch_tokens

    # ─────────────────────────────────────────
    def _interpolate_pos_embed(
        self, x: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Interpolates positional embeddings to match (H, W) resolution.
        Handles the case where local crops (96×96) produce fewer patches
        than the 224×224 training resolution.
        """
        pos_embed = self.vit.pos_embed   # (1, 1+N_orig, D)
        N_orig    = pos_embed.shape[1] - 1  # patches at training resolution

        # Compute number of patches for current input
        ph = H // self.patch_size
        pw = W // self.patch_size
        N_new = ph * pw

        if N_orig == N_new:
            return pos_embed  # no interpolation needed

        # Separate CLS and patch position embeddings
        cls_pos   = pos_embed[:, :1, :]    # (1, 1, D)
        patch_pos = pos_embed[:, 1:, :]    # (1, N_orig, D)

        # Reshape to spatial grid
        h_orig = w_orig = int(math.sqrt(N_orig))
        patch_pos = patch_pos.reshape(1, h_orig, w_orig, -1).permute(0, 3, 1, 2)
        # (1, D, h_orig, w_orig)

        # Bilinear interpolation
        patch_pos = nn.functional.interpolate(
            patch_pos,
            size=(ph, pw),
            mode="bilinear",
            align_corners=False,
        )  # (1, D, ph, pw)

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, N_new, -1)
        # (1, N_new, D)

        return torch.cat([cls_pos, patch_pos], dim=1)  # (1, 1+N_new, D)

    # ─────────────────────────────────────────
    def _freeze_blocks(self, n: int):
        """Freeze the first n transformer blocks and the patch embedding."""
        # Freeze patch embedding
        for p in self.vit.patch_embed.parameters():
            p.requires_grad = False
        # Freeze first n blocks
        for i, blk in enumerate(self.vit.blocks):
            if i < n:
                for p in blk.parameters():
                    p.requires_grad = False
        print(f"[ViTBackbone] Frozen patch_embed + first {n} blocks.")

    # ─────────────────────────────────────────
    def get_intermediate_layers(self, x: torch.Tensor, n: int = 4):
        """
        Return the last n intermediate layer outputs (for DINO-style use).

        Returns:
            List of (B, N+1, D) tensors (length n)
        """
        B, C, H, W = x.shape
        x = self.vit.patch_embed(x)
        if x.ndim == 4:
            if x.shape[-1] == self.embed_dim:
                x = x.flatten(1, 2)
            else:
                x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self._interpolate_pos_embed(x, H, W)
        x = self.vit.pos_drop(x)

        outputs = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i >= len(self.vit.blocks) - n:
                outputs.append(self.vit.norm(x))

        return outputs   # list of (B, N+1, D)
