# data/

## Contents

| File | Purpose |
|------|---------|
| `dataset.py` | `DeepfakeDataset`, `SSLDataset`, DataLoader factories |
| `augmentations.py` | Multi-crop, FFT/DCT, contrastive, train/val transforms |

## Dataset Structure

```
data_root/
├── train/
│   ├── real/   ← FFHQ, CelebA-HQ frames
│   └── fake/   ← StyleGAN2/3, FF++, CelebDF
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## SSL Data

For SSL pretraining, you can use the raw **unlabelled** mix — no folder structure required:
```
ssl_data/
├── real_images/
└── fake_images/    ← or just all images flat
```

## Augmentation Pipeline

### DINO Multi-Crop
- 2 global crops: 224×224, scale [0.4, 1.0]
- 6 local crops:   96×96,  scale [0.05, 0.4]
- ColorJitter, GaussianBlur, RandomGrayscale, RandomSolarize

### FFT Transform
- Per-channel 2D FFT → log magnitude spectrum
- Normalised to [0,1], then ImageNet-normalised
- Same spatial size as RGB input

### Supervised (train)
- RandomResizedCrop, HorizontalFlip, ColorJitter, Grayscale

### Supervised (val/test)
- Resize → CenterCrop → Normalise (deterministic)
