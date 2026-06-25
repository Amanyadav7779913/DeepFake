"""
dataset.py
──────────
PyTorch Dataset classes:
  - DeepfakeDataset   : Supervised binary classification (real/fake)
  - SSLDataset        : Unlabelled dataset for SSL pretraining
  - SSLContrastiveDataset : Returns multi-crop + FFT views for combined SSL
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import numpy as np

from data.augmentations import (
    MultiCropAugmentation,
    ContrastivePairAugmentation,
    FFTTransform,
    TrainTransform,
    ValTransform,
    preprocess_fft,
)


# ─────────────────────────────────────────────
# Supported image extensions
# ─────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _find_images(root: str) -> List[str]:
    """Recursively find all images under root."""
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(paths))


# ─────────────────────────────────────────────
# Supervised Dataset
# ─────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    Binary classification dataset.

    Expected structure:
        root/
          real/
            *.jpg …
          fake/
            *.jpg …

    Labels: real → 0, fake → 1
    """

    def __init__(
        self,
        root: str,
        split: str = "train",           # "train" | "val" | "test"
        transform: Optional[Callable] = None,
        fft_transform: Optional[Callable] = None,
        image_size: int = 224,
        use_fft: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.root        = Path(root) / split
        self.use_fft     = use_fft
        self.image_size  = image_size

        if transform is None:
            if split == "train":
                transform = TrainTransform(size=image_size)
            else:
                transform = ValTransform(size=image_size)
        self.transform = transform

        if fft_transform is None:
            self.fft_transform = FFTTransform(size=image_size)
        else:
            self.fft_transform = fft_transform

        # Gather samples (case-insensitive check)
        real_imgs = []
        for d in ["real", "Real"]:
            rd = self.root / d
            if rd.exists():
                real_imgs.extend(_find_images(str(rd)))
        
        fake_imgs = []
        for d in ["fake", "Fake"]:
            fd = self.root / d
            if fd.exists():
                fake_imgs.extend(_find_images(str(fd)))

        self.samples: List[Tuple[str, int]] = (
            [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
        )

        if max_samples is not None:
            # Stratified subsample
            real_s = [(p, l) for p, l in self.samples if l == 0][:max_samples // 2]
            fake_s = [(p, l) for p, l in self.samples if l == 1][:max_samples // 2]
            self.samples = real_s + fake_s

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}. "
                               f"Expected subdirs: real/, fake/")

        print(f"[DeepfakeDataset] {split}: "
              f"{sum(l==0 for _,l in self.samples)} real, "
              f"{sum(l==1 for _,l in self.samples)} fake images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        rgb_tensor = self.transform(img)

        if self.use_fft:
            fft_tensor = self.fft_transform(img)
            return rgb_tensor, fft_tensor, torch.tensor(label, dtype=torch.long)

        return rgb_tensor, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────
# SSL Dataset (unlabelled)
# ─────────────────────────────────────────────
class SSLDataset(Dataset):
    """
    Unlabelled dataset for self-supervised pretraining.
    Returns DINO multi-crop views + FFT views.

    root/ may have any sub-structure; all images are found recursively.
    """

    def __init__(
        self,
        root: str,
        image_size: int = 224,
        n_local_crops: int = 6,
        max_samples: Optional[int] = None,
    ):
        self.image_paths = _find_images(root)
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under {root}")

        self.multi_crop = MultiCropAugmentation(
            global_size=image_size,
            local_size=96,
            n_local_crops=n_local_crops,
        )
        self.contrastive_aug = ContrastivePairAugmentation(size=image_size)
        self.fft_transform   = FFTTransform(size=image_size)

        print(f"[SSLDataset] {len(self.image_paths)} images found in {root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img  = Image.open(path).convert("RGB")

        # Multi-crop views for DINO
        crops = self.multi_crop(img)             # list of tensors

        # Contrastive pair for InfoNCE
        view1, view2 = self.contrastive_aug(img)

        # FFT view of original (centre-cropped)
        fft_view = self.fft_transform(img)       # (3, H, W)

        return {
            "crops":   crops,      # list[Tensor]  lengths: 2+N_local
            "view1":   view1,      # Tensor (3, 224, 224)
            "view2":   view2,      # Tensor (3, 224, 224)
            "fft":     fft_view,   # Tensor (3, 224, 224)
            "path":    path,
        }


# ─────────────────────────────────────────────
# Custom collate for SSL (list-of-crops)
# ─────────────────────────────────────────────
def ssl_collate_fn(batch):
    """
    Collates a batch from SSLDataset.
    Stacks each crop position across the batch separately.
    """
    n_crops = len(batch[0]["crops"])

    # Stack crops: crops[i] → (B, 3, H, W)
    crops_batched = [
        torch.stack([item["crops"][i] for item in batch]) for i in range(n_crops)
    ]
    view1 = torch.stack([item["view1"] for item in batch])
    view2 = torch.stack([item["view2"] for item in batch])
    fft   = torch.stack([item["fft"]   for item in batch])
    paths = [item["path"] for item in batch]

    return {
        "crops": crops_batched,
        "view1": view1,
        "view2": view2,
        "fft":   fft,
        "paths": paths,
    }


# ─────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────
def get_supervised_loaders(
    root: Union[str, List[str]],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    use_fft: bool = True,
    pin_memory: bool = True,
):
    """Return train, val, test DataLoaders for supervised training."""
    if isinstance(root, str):
        roots = [root]
    else:
        roots = root

    loaders = {}
    for split_key in ["train", "val", "test"]:
        datasets = []
        for r in roots:
            # Handle variations in folder names (lowercase/capitalized)
            possible_splits = [split_key, split_key.capitalize()]
            if split_key == "val":
                possible_splits += ["validation", "Validation"]
            
            actual_split = None
            for p_split in possible_splits:
                if (Path(r) / p_split).exists():
                    actual_split = p_split
                    break
            
            if actual_split is None:
                print(f"[Warning] Split '{split_key}' not found in {r}, skipping.")
                continue

            split_path = Path(r) / actual_split

            ds = DeepfakeDataset(
                root=r,
                split=actual_split,
                image_size=image_size,
                use_fft=use_fft,
            )
            datasets.append(ds)

        if not datasets:
            continue

        combined_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        loaders[split_key] = DataLoader(
            combined_ds,
            batch_size=batch_size,
            shuffle=(split_key == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_key == "train" and len(combined_ds) > batch_size),
        )
    return loaders


def get_ssl_loader(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    n_local_crops: int = 6,
    image_size: int = 224,
    pin_memory: bool = True,
):
    """Return a DataLoader for SSL pretraining."""
    ds = SSLDataset(
        root=root,
        image_size=image_size,
        n_local_crops=n_local_crops,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ssl_collate_fn,
        drop_last=True,
    )
