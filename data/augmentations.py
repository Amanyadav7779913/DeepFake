"""
augmentations.py
────────────────
Augmentation pipelines for:
  - DINO-style multi-crop (2 global + N local crops)
  - FFT / DCT frequency-domain transforms
  - Standard supervised augmentations
"""

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import cv2


# ─────────────────────────────────────────────
# Normalization constants (ImageNet)
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# FFT Frequency Transform
# ─────────────────────────────────────────────
class FFTTransform:
    """
    Converts an RGB image to its 2-D FFT magnitude spectrum.
    Each channel is FFT'd independently, log-scaled, then
    stacked back into a 3-channel tensor of the same spatial
    size as the input.  Values are normalised to [0,1].
    """

    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Resize to target size first
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

        channels = []
        for c in range(arr.shape[2]):
            fft = np.fft.fft2(arr[:, :, c])
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.log1p(np.abs(fft_shifted))   # log scaling
            # Normalise per-channel to [0,1]
            mag_min, mag_max = magnitude.min(), magnitude.max()
            if mag_max > mag_min:
                magnitude = (magnitude - mag_min) / (mag_max - mag_min)
            channels.append(magnitude)

        fft_img = np.stack(channels, axis=-1).astype(np.float32)  # (H, W, 3)
        tensor = torch.from_numpy(fft_img).permute(2, 0, 1)       # (3, H, W)
        # Normalise with ImageNet stats for compatibility
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor


class DCTTransform:
    """
    Discrete Cosine Transform (DCT) frequency representation.
    Uses OpenCV's DCT on grayscale luminance, replicated to 3 channels.
    """

    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.size, self.size), Image.BILINEAR)
        gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
        dct = cv2.dct(gray)
        dct_log = np.log1p(np.abs(dct))
        dct_norm = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-8)
        tensor = torch.from_numpy(dct_norm).unsqueeze(0).repeat(3, 1, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std


# ─────────────────────────────────────────────
# DINO Multi-Crop Augmentation
# ─────────────────────────────────────────────
class MultiCropAugmentation:
    """
    DINO-style multi-crop strategy:
      - 2 global crops at 224×224  (scale 0.4–1.0)
      - N local  crops at  96×96   (scale 0.05–0.4)

    Returns a list of tensors: [global_1, global_2, local_1, ..., local_N]
    """

    def __init__(
        self,
        global_size: int = 224,
        local_size:  int = 96,
        n_local_crops: int = 6,
        global_scale: tuple = (0.4, 1.0),
        local_scale:  tuple = (0.05, 0.4),
    ):
        self.n_local_crops = n_local_crops

        color_jitter = T.ColorJitter(0.4, 0.4, 0.2, 0.1)

        self.global_transform = T.Compose([
            T.RandomResizedCrop(global_size, scale=global_scale, interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=local_scale, interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image):
        crops = []
        # 2 global crops
        crops.append(self.global_transform(img))
        crops.append(self.global_transform(img))
        # N local crops
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(img))
        return crops


# ─────────────────────────────────────────────
# SimCLR / Contrastive Pair Augmentation
# ─────────────────────────────────────────────
class ContrastivePairAugmentation:
    """
    Generates two strongly-augmented views of the same image
    for SimCLR/MoCo-style contrastive learning.
    """

    def __init__(self, size: int = 224):
        color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.RandomSolarize(threshold=128, p=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image):
        return self.transform(img), self.transform(img)


# ─────────────────────────────────────────────
# Supervised Fine-tuning Augmentation
# ─────────────────────────────────────────────
class TrainTransform:
    """Standard supervised training augmentation pipeline."""

    def __init__(self, size: int = 224):
        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class ValTransform:
    """Deterministic validation/test transform."""

    def __init__(self, size: int = 224):
        self.transform = T.Compose([
            T.Resize(int(size * 1.143), interpolation=Image.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


# ─────────────────────────────────────────────
# Universal Preprocessing (for inference)
# ─────────────────────────────────────────────
def preprocess_image(img: Image.Image, size: int = 224) -> torch.Tensor:
    """
    Preprocess any-resolution PIL image to a normalised tensor.
    Handles grayscale → RGB conversion automatically.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    transform = ValTransform(size=size)
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def preprocess_fft(img: Image.Image, size: int = 224) -> torch.Tensor:
    """FFT-domain preprocessing for inference."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    fft_transform = FFTTransform(size=size)
    return fft_transform(img).unsqueeze(0)  # (1, 3, H, W)
