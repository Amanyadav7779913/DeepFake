# ssl/

Self-supervised learning components for Phase 1 pretraining.

## Contents

| File | Description |
|------|-------------|
| `dino.py` | `DINOLoss` (centering + sharpening), `EMAScheduler`, `MultiCropWrapper` |
| `contrastive.py` | `InfoNCELoss`, `MoCoLoss` + `MoCoQueue`, `SupConLoss` |
| `pacl.py` | `PatchContrastiveLoss`, `PatchConsistencyLoss`, `PACLLoss` (novel component) |

## DINO

DINO (Self-**Di**stillation with **No** labels) trains a student network to match the output of a slowly-evolving teacher (EMA copy of student).

### Key details
- **Centering**: The teacher output is offset by an EMA batch mean. This prevents all outputs from collapsing to a single prototype without needing explicit negatives.
- **Sharpening**: Teacher uses low temperature (τ_t=0.04); student uses higher temperature (τ_s=0.1).
- **Multi-crop**: Student sees all 2+N crops; teacher only sees 2 global crops.

### EMA momentum schedule
```
m_t = 1 - (1 - m_base) * (cos(π * t / T) + 1) / 2
```
Momentum starts at `m_base=0.996` and approaches `1.0` by end of training.

## Contrastive (InfoNCE)

InfoNCE loss on the projection head outputs:
- Positive pair: two augmented views of the same image
- Negatives: all other images in the batch
- Temperature: τ = 0.07

Optional MoCo queue extends the effective negative count to 65536 without requiring larger batch sizes.

## PACL (Novel Component)

**Patch-Level Artifact Consistency Learning** is a novel SSL objective designed specifically for deepfake detection.

### Hypothesis
| Image Type | Patch Consistency | Explanation |
|------------|-------------------|-------------|
| Real | High | Natural texture / lighting flows consistently across patches |
| GAN-generated | Low | Synthesis artefacts create patch-level inconsistencies |
| Blended (FF++) | Mixed | Real regions consistent; blended region breaks consistency |

### Two Losses
1. **Patch Contrastive**: patch_i from view_1 should match patch_i from view_2 (same spatial location, same image). Acts as a spatially-grounded contrastive signal.
2. **Patch Consistency**: Real images → high mean patch similarity. Fake images → low mean patch similarity. Margin loss enforces separation.

### Why it helps generalise to unseen GANs
New GAN architectures will still produce spatial artefacts. PACL doesn't rely on any specific artefact pattern — it learns a general notion of spatial coherence vs incoherence.
