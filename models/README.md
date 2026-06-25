# models/

## Contents

| File | Description |
|------|-------------|
| `backbone.py` | `ViTBackbone` — timm ViT wrapper, exposes CLS + patch tokens, handles positional embedding interpolation |
| `heads.py` | `ProjectionHead`, `DINOHead`, `PACLHead`, `ClassifierHead`, `FeatureFusion` |
| `detector.py` | `DeepfakeDetector` — full dual-stream model; `TeacherModel` for DINO EMA |

## Architecture

```
RGB (B,3,H,W) ──► ViTBackbone ──► CLS (B,D)  ──► FeatureFusion ──► ClassifierHead ──► logit
                              └──► Patches (B,N,D) ──► PACLHead ──► sim_matrix

FFT (B,3,H,W) ──► ViTBackbone ──► CLS (B,D)  ──┘
```

## Dimensions (ViT-Base)

| Tensor | Shape |
|--------|-------|
| embed_dim (D) | 768 |
| patch_tokens (N) | 196 (224×224 input) |
| proj_dim | 256 |
| pacl_dim | 128 |
| dino_out_dim | 65536 |

## Key Design Notes

### Positional Embedding Interpolation
Local crops (96×96) produce 36 patches vs 196 for global crops (224×224).
`ViTBackbone._interpolate_pos_embed()` handles this with bilinear interpolation —
essential for DINO multi-crop to work correctly.

### Shared vs Independent Backbone
- `shared_backbone=True`: RGB and FFT streams share weights — 2× fewer params, faster training
- `shared_backbone=False`: Independent ViTs — more capacity, better for final fine-tuning
