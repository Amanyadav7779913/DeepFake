# training/

## Scripts

| Script | Purpose |
|--------|---------|
| `train_ssl.py` | Phase 1: SSL pretraining (DINO + InfoNCE + PACL) |
| `train_classifier.py` | Phase 2: Supervised fine-tuning |

## Phase 1: SSL Pretraining

```bash
python training/train_ssl.py \
    --data_dir /path/to/unlabeled \
    --output_dir ./checkpoints/ssl \
    --epochs 100 \
    --batch_size 64 \
    --amp
```

### What happens during SSL:
1. For each batch, 3 types of views are generated:
   - **Multi-crop** (2 global + 6 local) for DINO
   - **Contrastive pair** (view1, view2) for InfoNCE  
   - **FFT view** for frequency domain signal
2. Student processes all crops; teacher only sees global crops
3. Three losses are combined: `L = DINO + 0.5*NCE + 0.5*PACL`
4. Teacher is updated via EMA (momentum scheduler)
5. Best checkpoint saved based on combined loss

## Phase 2: Fine-tuning

```bash
python training/train_classifier.py \
    --data_dir /path/to/labeled_data \
    --ssl_checkpoint ./checkpoints/ssl/best_ssl.pth \
    --output_dir ./checkpoints/classifier \
    --epochs 50
```

### Training strategy:
- Encoder LR = 0.1 × head LR (differential LR)
- First 8 ViT blocks frozen at start
- Cosine annealing with warm restarts
- Early stopping on validation AUC
- Combined loss: BCE + 0.2*SupCon + 0.1*PACL

## Recommended Hardware

| Config | GPU | Batch Size | Time/Epoch |
|--------|-----|-----------|-----------|
| SSL | A100 80GB | 64 | ~30 min |
| SSL | V100 32GB | 32 | ~60 min |
| Fine-tune | A100 80GB | 32 | ~10 min |
| Fine-tune | V100 32GB | 16 | ~15 min |
