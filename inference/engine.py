"""
inference/engine.py
───────────────────
InferenceEngine — production-ready wrapper around DeepfakeDetector.

Features:
  - Handles any input type: file path, PIL image, numpy array, torch tensor
  - Automatic resize + normalisation
  - Single image or batch inference
  - Optional Grad-CAM visualisation
  - GPU acceleration when available
"""

import os
from typing import Union, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from models.detector import DeepfakeDetector
from data.augmentations import preprocess_image, preprocess_fft
from utils.checkpoint import load_checkpoint


# ──────────────────────────────────────────────
class InferenceEngine:
    """
    High-level inference wrapper for DeepfakeDetector.

    Usage:
        engine = InferenceEngine.from_checkpoint("./checkpoints/best_model.pth")
        label, conf = engine.predict("path/to/image.jpg")
        print(f"{label} (Confidence: {conf:.2f})")
    """

    def __init__(
        self,
        model:      DeepfakeDetector,
        device:     Optional[torch.device] = None,
        image_size: int = 224,
        threshold:  float = 0.5,
    ):
        self.model      = model
        self.image_size = image_size
        self.threshold  = threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

    # ── Factory ──────────────────────────────
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        vit_model:       str   = "vit_base_patch16_224",
        image_size:      int   = 224,
        threshold:       float = 0.5,
        device:          Optional[torch.device] = None,
    ) -> "InferenceEngine":
        """
        Build an InferenceEngine from a saved .pth checkpoint.

        Args:
            checkpoint_path : path to the .pth file
            vit_model       : timm model name (must match training config)
            image_size      : inference image size
            threshold       : decision threshold (default 0.5)
            device          : CPU or CUDA device

        Returns:
            InferenceEngine instance ready for prediction
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DeepfakeDetector(
            vit_model=vit_model,
            pretrained=False,  # weights loaded from checkpoint
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            load_checkpoint(
                path=checkpoint_path,
                model=model,
                device=str(device),
                strict=False,
            )
            print(f"[InferenceEngine] Loaded weights from {checkpoint_path}")
        else:
            print(f"[InferenceEngine] WARNING: checkpoint not found at {checkpoint_path}. "
                  f"Using randomly initialised model.")

        return cls(model=model, device=device, image_size=image_size, threshold=threshold)

    # ── Image Loading ────────────────────────
    def _load_image(self, img_input) -> Image.Image:
        """Accept path, PIL image, or numpy array → PIL RGB image."""
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Image not found: {img_input}")
            img = Image.open(img_input).convert("RGB")

        elif isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")

        elif isinstance(img_input, np.ndarray):
            if img_input.dtype != np.uint8:
                img_input = (img_input * 255).clip(0, 255).astype(np.uint8)
            if img_input.ndim == 2:
                img = Image.fromarray(img_input, mode="L").convert("RGB")
            else:
                img = Image.fromarray(img_input).convert("RGB")

        elif isinstance(img_input, torch.Tensor):
            # Expect (3, H, W) or (H, W, 3)
            arr = img_input.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(img_input)}")

        return img

    # ── Preprocessing ────────────────────────
    def _preprocess(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (rgb_tensor, fft_tensor), both (1, 3, H, W) on device."""
        rgb = preprocess_image(img, size=self.image_size).to(self.device)
        fft = preprocess_fft(img,   size=self.image_size).to(self.device)
        return rgb, fft

    # ── Single Prediction ────────────────────
    @torch.no_grad()
    def predict(
        self,
        img_input,
        return_features: bool = False,
    ) -> dict:
        """
        Predict whether an image is real or fake.

        Returns:
            dict with:
                - label      : "Real" | "Fake"
                - confidence : float in [0, 1]
                - prob       : raw probability of being "Fake"
                - [features] : dict — if return_features=True
        """
        img      = self._load_image(img_input)
        rgb, fft = self._preprocess(img)

        if return_features:
            logit, prob, features = self.model(rgb, fft, return_features=True)
        else:
            logit, prob = self.model(rgb, fft)
            features    = None

        p          = prob.item()
        label      = "Fake" if p >= self.threshold else "Real"
        confidence = p if p >= self.threshold else 1.0 - p

        res = {
            "label":      label,
            "confidence": confidence,
            "prob":       p
        }
        if return_features:
            res["features"] = features
        return res

    # ── Batch Prediction ────────────────────
    @torch.no_grad()
    def predict_batch(
        self,
        img_inputs: List,
        batch_size: int = 16,
    ) -> List[Tuple[str, float]]:
        """
        Run inference on a list of images.

        Args:
            img_inputs : list of (path | PIL | array | tensor)
            batch_size : processing batch size

        Returns:
            List of (label, confidence) tuples
        """
        results = []

        for start in range(0, len(img_inputs), batch_size):
            batch_inputs = img_inputs[start : start + batch_size]
            rgb_batch = []
            fft_batch = []

            for inp in batch_inputs:
                img = self._load_image(inp)
                rgb, fft = self._preprocess(img)
                rgb_batch.append(rgb)
                fft_batch.append(fft)

            rgb_tensor = torch.cat(rgb_batch, dim=0)  # (B, 3, H, W)
            fft_tensor = torch.cat(fft_batch, dim=0)

            _, probs = self.model(rgb_tensor, fft_tensor)

            for p in probs.cpu().numpy():
                label      = "Fake" if p >= self.threshold else "Real"
                confidence = float(p) if p >= self.threshold else float(1.0 - p)
                results.append((label, confidence))

        return results

    # ── Directory Prediction ─────────────────
    def predict_directory(
        self,
        dir_path:   str,
        batch_size: int = 16,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> List[dict]:
        """
        Run inference on all images in a directory.

        Returns:
            List of dicts: {"path": ..., "label": ..., "confidence": ...}
        """
        from pathlib import Path
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(dir_path).glob(f"**/*{ext}"))
            image_paths.extend(Path(dir_path).glob(f"**/*{ext.upper()}"))
        image_paths = sorted(set(str(p) for p in image_paths))

        if not image_paths:
            print(f"[InferenceEngine] No images found in {dir_path}")
            return []

        print(f"[InferenceEngine] Processing {len(image_paths)} images …")
        preds = self.predict_batch(image_paths, batch_size=batch_size)

        return [
            {"path": p, "label": label, "confidence": conf}
            for p, (label, conf) in zip(image_paths, preds)
        ]
