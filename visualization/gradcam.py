"""
visualization/gradcam.py
────────────────────────
Attention Map visualisation for ViT-based DeepfakeDetector.

Implements:
  1. AttentionRollout — DINO-style self-attention visualisation
  2. PatchHeatmap     — PACL similarity matrix visualised as heatmap
  3. VisualisationPipeline — all-in-one overlay generator
"""

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional


# ──────────────────────────────────────────────
# 1. Attention Rollout (DINO-style)
# ──────────────────────────────────────────────
class AttentionRollout:
    """
    Visualise ViT self-attention via attention rollout.
    Shows which patches the CLS token attends to.
    """

    def __init__(self, model, discard_ratio: float = 0.9):
        self.model         = model
        self.discard_ratio = discard_ratio

    @torch.no_grad()
    def generate(self, rgb: torch.Tensor) -> np.ndarray:
        """
        Args:
            rgb : (1, 3, H, W)

        Returns:
            rollout : (H, W) numpy array in [0, 1]
        """
        H, W  = rgb.shape[2], rgb.shape[3]
        attns = []

        # Step the ViT manually so we can capture per-layer attention weights
        vit      = self.model.rgb_encoder.vit
        backbone = self.model.rgb_encoder

        x = vit.patch_embed(rgb)   # (B, N, D) or (B, H', W', D) depending on timm version
        B = x.shape[0]
        embed_dim = backbone.embed_dim

        # Normalise to 3D: (B, N, D)
        if x.ndim == 4:
            if x.shape[-1] == embed_dim:
                x = x.flatten(1, 2)               # (B, H', W', D) → (B, N, D)
            else:
                x = x.flatten(2).transpose(1, 2)  # (B, D, H', W') → (B, N, D)

        cls = vit.cls_token.expand(B, -1, -1)     # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)           # (B, N+1, D)
        x   = x + backbone._interpolate_pos_embed(x, H, W)
        x   = vit.pos_drop(x)

        for blk in vit.blocks:
            norm_x    = blk.norm1(x)
            qkv       = blk.attn.qkv(norm_x)      # (B, N+1, 3*D)
            B2, N, _  = qkv.shape
            num_heads = blk.attn.num_heads
            head_dim  = blk.attn.head_dim
            qkv  = qkv.reshape(B2, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)               # each (B, heads, N, head_dim)
            attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn = attn.softmax(dim=-1)            # (B, heads, N, N)
            attns.append(attn.mean(dim=1).cpu())   # avg over heads → (B, N, N)
            x = blk(x)

        # Rollout — work entirely in 3D (B, N, N)
        N_tokens = attns[0].shape[-1]
        result   = torch.eye(N_tokens).unsqueeze(0).expand(B, -1, -1).clone()  # (B, N, N)
        for attn in attns:
            flat      = attn.reshape(B, -1)
            threshold = torch.quantile(flat, self.discard_ratio, dim=-1,
                                       keepdim=True).unsqueeze(-1)  # (B, 1, 1)
            attn_th   = attn.clone()
            attn_th[attn_th < threshold] = 0

            I      = torch.eye(N_tokens).unsqueeze(0)   # (1, N, N)
            a      = (attn_th + I) / 2
            a      = a / a.sum(dim=-1, keepdim=True)
            result = torch.bmm(a, result)               # (B, N, N)

        # CLS-row → per-patch attention weights
        mask = result[0, 0, 1:].numpy()               # (N,)
        N    = mask.shape[0]
        h = w = int(N ** 0.5)
        mask = mask[:h*w].reshape(h, w)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        img = Image.fromarray((mask * 255).astype(np.uint8))
        img = img.resize((W, H), Image.BILINEAR)
        return np.array(img).astype(np.float32) / 255.0


# ──────────────────────────────────────────────
# 2. PACL Heatmap
# ──────────────────────────────────────────────
def pacl_consistency_heatmap(
    sim_matrix: torch.Tensor,   # (1, N, N)
    image_size: int = 224,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Convert PACL similarity matrix to a spatial consistency heatmap.
    Each patch shows its mean similarity to all other patches.

    Returns:
        heatmap : (H, W) numpy array in [0, 1]
    """
    sim = sim_matrix.squeeze(0).cpu()     # (N, N)
    N   = sim.shape[0]
    h = w = int(N ** 0.5)

    # Mean off-diagonal similarity per patch
    mask      = ~torch.eye(N, dtype=torch.bool)
    per_patch = (sim * mask.float()).sum(dim=-1) / (N - 1)   # (N,)
    per_patch = per_patch[:h*w].reshape(h, w).numpy()
    per_patch = (per_patch - per_patch.min()) / (per_patch.max() - per_patch.min() + 1e-8)

    img = Image.fromarray((per_patch * 255).astype(np.uint8))
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


# ──────────────────────────────────────────────
# 3. Overlay Utility
# ──────────────────────────────────────────────
def overlay_heatmap(
    image:    np.ndarray,    # (H, W, 3) uint8
    heatmap:  np.ndarray,    # (H, W) float [0, 1]
    alpha:    float = 0.5,
    colormap: str   = "hot",
) -> np.ndarray:
    """Overlay a heatmap on an image with transparency."""
    cmap   = cm.get_cmap(colormap)
    hm_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * hm_rgb + (1 - alpha) * image).astype(np.uint8)


# ──────────────────────────────────────────────
# 4. Full Visualisation Pipeline
# ──────────────────────────────────────────────
class VisualisationPipeline:
    """
    3-panel visualisation: Attention Rollout + PACL heatmap.

    Usage:
        pipeline = VisualisationPipeline(model)
        fig = pipeline.visualise("image.jpg", rgb_tensor, fft_tensor)
        fig.savefig("visualisation.png")
    """

    def __init__(self, model, device=None):
        self.model   = model
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rollout = AttentionRollout(model)

    def visualise(
        self,
        img_path:   str,
        rgb_tensor: torch.Tensor,
        fft_tensor: torch.Tensor,
        label:      str   = "Unknown",
        confidence: float = 0.0,
        save_path:  Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate a 3-panel visualisation figure.

        Panels:
          1. Original image
          2. Attention Rollout
          3. PACL Consistency Map
        """
        # Original image
        orig = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))

        # Attention Rollout
        try:
            rollout_map  = self.rollout.generate(rgb_tensor.to(self.device))
            attn_overlay = overlay_heatmap(orig, rollout_map, alpha=0.5, colormap="hot")
        except Exception as e:
            print(f"[Visualisation] AttentionRollout failed: {e}")
            attn_overlay = orig.copy()

        # PACL consistency
        with torch.no_grad():
            _, _, features = self.model(
                rgb_tensor.to(self.device),
                fft_tensor.to(self.device),
                return_features=True,
            )
        pacl_map     = pacl_consistency_heatmap(
            features["sim_matrix"],
            image_size=224,
            patch_size=self.model.rgb_encoder.patch_size,
        )
        pacl_overlay = overlay_heatmap(orig, pacl_map, alpha=0.6, colormap="RdYlGn")

        # Figure — 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"Deepfake Detection: {label} (Confidence: {confidence:.2f})",
            fontsize=14, fontweight="bold",
            color="#e74c3c" if label == "Fake" else "#27ae60",
        )

        panels = [
            (orig,         "Original Image"),
            (attn_overlay, "Attention Rollout"),
            (pacl_overlay, "PACL Consistency"),
        ]

        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Visualisation] Saved → {save_path}")

        return fig
