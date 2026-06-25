# 🔍 DeepFake Detector — ViT + SSL (DINO + Contrastive + PACL)

A production-ready, modular deepfake detection system using Vision Transformers and Self-Supervised Learning.

---

## Architecture Overview

```
Input Image (any resolution)
        │
        ├─── RGB Branch ──────────────────┐
        │    └── ViT Encoder              │
        │         ├── CLS Token           │
        │         └── Patch Tokens        │
        │                                 ├──► Feature Fusion ──► Classifier ──► Real/Fake + Confidence
        ├─── FFT Branch ──────────────────┤
             └── ViT Encoder              │
                  ├── CLS Token           │
                  └── Patch Tokens ───────┘
                                          │
                              PACL Head ──┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **ViT Backbone** | `vit_base_patch16_224` from `timm` — classification head removed |
| **DINO SSL** | Student-Teacher with EMA, multi-crop (2 global + 6 local), centering & sharpening |
| **Contrastive** | SimCLR/MoCo-style InfoNCE loss on augmented pairs |
| **PACL** | Patch-Level Artifact Consistency Learning — novel component |
| **Dual Stream** | RGB + FFT frequency domain, both through ViT encoder |
| **Classifier** | Fused features → FC layers → sigmoid output |

---

## Project Structure

```
deepfake_detector/
├── data/                    # Dataset loaders and augmentation pipelines
│   ├── dataset.py           # DeepfakeDataset, SSLDataset
│   ├── augmentations.py     # Multi-crop, FFT transforms
│   └── README.md
├── models/                  # Neural network architectures
│   ├── backbone.py          # ViT feature extractor wrapper
│   ├── detector.py          # Full DeepfakeDetector model
│   ├── heads.py             # Projection, PACL, classification heads
│   └── README.md
├── deepfake_ssl/            # Self-supervised learning components
│   ├── dino.py              # DINO loss, EMA teacher update
│   ├── contrastive.py       # InfoNCE / NT-Xent loss
│   ├── pacl.py              # Patch-Level Artifact Consistency Learning
│   └── README.md
├── training/                # Training scripts and utilities
│   ├── train_ssl.py         # Phase 1: SSL pretraining
│   ├── train_classifier.py  # Phase 2: Supervised fine-tuning
│   └── README.md
├── inference/               # Inference utilities
│   ├── engine.py            # InferenceEngine class
│   └── README.md
├── evaluation/              # Evaluation and benchmarking
│   ├── evaluate.py          # Metrics: AUC, accuracy, EER
│   └── README.md
├── visualization/           # Grad-CAM and attention maps
│   ├── gradcam.py           # Grad-CAM for ViT
│   └── README.md
├── utils/                   # Shared utilities
│   ├── logger.py            # Logging setup
│   ├── checkpoint.py        # Save/load checkpoints
│   └── README.md
├── configs/                 # YAML configuration files
│   ├── ssl_config.yaml      # SSL pretraining config
│   └── finetune_config.yaml # Fine-tuning config
├── predict.py               # Main inference entry point
├── requirements.txt
└── README.md                # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Inference (pretrained or fine-tuned checkpoint)

```bash
python predict.py --img test.jpg
# Output: Fake (Confidence: 0.93)

# With visualization
python predict.py --img test.jpg --visualize

# Batch inference
python predict.py --img_dir ./test_images/ --batch_size 16

# Custom checkpoint
python predict.py --img test.jpg --checkpoint ./checkpoints/best_model.pth
```

### 3. Train SSL (Phase 1)

```bash
python training/train_ssl.py \
    --data_dir /path/to/unlabeled_data \
    --output_dir ./checkpoints/ssl \
    --epochs 100 \
    --batch_size 64
```

### 4. Fine-tune Classifier (Phase 2)

```bash
python training/train_classifier.py \
    --data_dir /path/to/labeled_data \
    --ssl_checkpoint ./checkpoints/ssl/best_ssl.pth \
    --output_dir ./checkpoints/classifier \
    --epochs 50
```

### 5. Evaluate

```bash
python evaluation/evaluate.py \
    --data_dir /path/to/test_data \
    --checkpoint ./checkpoints/classifier/best_model.pth
```

---

## Expected Dataset Structure

```
data_root/
├── train/
│   ├── real/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── fake/
│       ├── img_001.jpg
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Recommended Datasets:**
- **Real**: FFHQ, CelebA-HQ
- **Fake**: StyleGAN2/3, FaceForensics++, CelebDF

---

## Novel Component: PACL

**Patch-Level Artifact Consistency Learning** is a novel SSL signal specifically designed for deepfake detection:

- Real images tend to have **spatially consistent** patch relationships — texture, lighting, and structure flow naturally across patches.
- Fake images contain **inconsistent patches** — GAN artifacts, blending seams, and frequency anomalies break local consistency.

PACL extracts ViT patch tokens, computes a pairwise similarity matrix, and applies:
1. **Patch Contrastive Loss**: Pulls patch representations of the same real image together, pushes fake-image patches apart
2. **Patch Consistency Loss**: Maximizes within-image patch similarity for real images; penalizes artificially high consistency in fake images

---

## SSL Training Details

| Hyperparameter | Value |
|---------------|-------|
| Student temp | 0.1 |
| Teacher temp | 0.04 |
| EMA momentum | 0.996 → 1.0 |
| Global crop scale | 0.4–1.0 |
| Local crop scale | 0.05–0.4 |
| Num local crops | 6 |
| InfoNCE temperature | 0.07 |
| PACL weight | 0.5 |

---

## Performance Targets

| Dataset | AUC | Accuracy |
|---------|-----|----------|
| FaceForensics++ | >0.98 | >95% |
| CelebDF | >0.95 | >92% |
| Cross-dataset | >0.88 | >85% |

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA ≥ 11.7 (optional but recommended)
- timm ≥ 0.9
- See `requirements.txt` for full list
