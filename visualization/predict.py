#!/usr/bin/env python3
"""
predict.py
──────────
🎯 Main inference entry point for the DeepFake Detector.

Accepts any image (any resolution) and outputs:
  - Prediction: Real or Fake
  - Confidence score (0–1)

Usage examples:
──────────────
  # Single image (quick)
  python predict.py --img test.jpg

  # With Grad-CAM visualisation
  python predict.py --img test.jpg --visualize

  # Batch: directory of images
  python predict.py --img_dir ./test_images/

  # Batch with CSV export
  python predict.py --img_dir ./test_images/ --output results.csv

  # Custom checkpoint
  python predict.py --img test.jpg --checkpoint ./checkpoints/classifier/best_model.pth

  # Change decision threshold
  python predict.py --img test.jpg --threshold 0.6

  # CPU-only mode
  python predict.py --img test.jpg --device cpu
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional

# ── Ensure project root is importable ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


# ──────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="DeepFake Detector — ViT + SSL (DINO + Contrastive + PACL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--img",     type=str, help="Path to a single image file")
    inp.add_argument("--img_dir", type=str, help="Directory of images for batch inference")

    # Model
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth). If not provided, uses default location.",
    )
    p.add_argument(
        "--vit_model",
        type=str,
        default="vit_base_patch16_224",
        help="timm ViT model name (must match training config)",
    )

    # Inference
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold. Scores >= threshold → Fake. (default: 0.5)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for directory inference (default: 8)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'cpu', 'cuda:0' etc. Auto-detected if not set.",
    )

    # Output
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="(Batch mode) Save results as CSV to this path",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Generate Grad-CAM + Attention visualisation (single image mode)",
    )
    p.add_argument(
        "--vis_output",
        type=str,
        default=None,
        help="Path to save visualisation image (default: <img>_vis.png)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON (single image mode)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except the final prediction line",
    )

    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

BANNER = """
╔══════════════════════════════════════════════════╗
║     🔍  DeepFake Detector  — ViT + SSL           ║
║     DINO · Contrastive · PACL                    ║
╚══════════════════════════════════════════════════╝
"""

FAKE_COLOR  = "\033[91m"   # red
REAL_COLOR  = "\033[92m"   # green
BOLD        = "\033[1m"
RESET       = "\033[0m"
DIM         = "\033[2m"


def _find_images(directory: str) -> List[str]:
    """Recursively find all supported image files in a directory."""
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(Path(directory).glob(f"**/*{ext}"))
        paths.extend(Path(directory).glob(f"**/*{ext.upper()}"))
    return sorted(set(str(p) for p in paths))


def _resolve_checkpoint(checkpoint_arg: Optional[str]) -> Optional[str]:
    """
    Resolve checkpoint path:
      1. Use explicitly provided path
      2. Search default locations
      3. Return None if not found (model will use random init + warn)
    """
    if checkpoint_arg and os.path.exists(checkpoint_arg):
        return checkpoint_arg

    # Default search locations
    default_paths = [
        "./checkpoints/classifier/best_model.pth",
        "./checkpoints/best_model.pth",
        "best_model.pth",
        str(PROJECT_ROOT / "checkpoints" / "classifier" / "best_model.pth"),
    ]
    for path in default_paths:
        if os.path.exists(path):
            return path

    return None


def _format_bar(confidence: float, width: int = 30) -> str:
    """Render a confidence bar in the terminal."""
    filled = int(confidence * width)
    empty  = width - filled
    return "█" * filled + "░" * empty


# ──────────────────────────────────────────────
# Single Image Prediction
# ──────────────────────────────────────────────
def predict_single(args, engine) -> dict:
    """Run prediction on a single image and print/return the result."""
    img_path = args.img

    if not os.path.exists(img_path):
        print(f"❌  Image not found: {img_path}")
        sys.exit(1)

    if not args.quiet:
        print(f"{DIM}  Processing: {img_path}{RESET}")

    t0 = time.perf_counter()
    res = engine.predict(img_path)
    label, confidence = res["label"], res["confidence"]
    elapsed = time.perf_counter() - t0

    # ── Compose result ──
    result = {
        "path":       img_path,
        "label":      label,
        "confidence": round(confidence, 4),
        "score_fake": round(res["prob"], 4),
        "elapsed_ms": round(elapsed * 1000, 1),
    }

    # ── Print ──
    if args.json:
        import json
        print(json.dumps(result, indent=2))
    else:
        color = FAKE_COLOR if label == "Fake" else REAL_COLOR
        bar   = _format_bar(confidence)
        print(
            f"\n{BOLD}{color}  {label}{RESET}"
            f"  (Confidence: {BOLD}{confidence:.2f}{RESET})"
            f"  [{color}{bar}{RESET}]"
        )
        if not args.quiet:
            print(f"{DIM}  Inference time: {elapsed*1000:.1f} ms{RESET}\n")

    # ── Visualisation ──
    if args.visualize:
        _run_visualisation(args, engine, img_path, label, confidence)

    return result


def _get_tensors(img_path: str, engine):
    """Get (rgb, fft) tensors for a single image."""
    from PIL import Image
    from data.augmentations import preprocess_image, preprocess_fft
    img = Image.open(img_path).convert("RGB")
    rgb = preprocess_image(img, size=engine.image_size)
    fft = preprocess_fft(img,   size=engine.image_size)
    return rgb, fft


def _run_visualisation(args, engine, img_path: str, label: str, confidence: float):
    """Generate and save Grad-CAM + attention visualisation."""
    try:
        from visualization.gradcam import VisualisationPipeline
        from data.augmentations    import preprocess_image, preprocess_fft
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        rgb = preprocess_image(img, size=engine.image_size).to(engine.device)
        fft = preprocess_fft(img,   size=engine.image_size).to(engine.device)

        vis_path = args.vis_output
        if vis_path is None:
            stem    = Path(img_path).stem
            vis_dir = Path(img_path).parent
            vis_path = str(vis_dir / f"{stem}_vis.png")

        pipeline = VisualisationPipeline(engine.model, device=engine.device)
        fig = pipeline.visualise(
            img_path=img_path,
            rgb_tensor=rgb,
            fft_tensor=fft,
            label=label,
            confidence=confidence,
            save_path=vis_path,
        )
        print(f"  📊 Visualisation saved → {vis_path}")

    except Exception as e:
        print(f"  ⚠️  Visualisation failed: {e}")


# ──────────────────────────────────────────────
# Batch Prediction
# ──────────────────────────────────────────────
def predict_batch(args, engine) -> List[dict]:
    """Run batch inference on a directory."""
    image_paths = _find_images(args.img_dir)

    if not image_paths:
        print(f"❌  No images found in: {args.img_dir}")
        sys.exit(1)

    if not args.quiet:
        print(f"  Found {len(image_paths)} images in {args.img_dir}")

    t0 = time.perf_counter()
    predictions = engine.predict_batch(image_paths, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0

    results = []
    n_fake  = 0
    n_real  = 0

    for path, (label, conf) in zip(image_paths, predictions):
        results.append({
            "path":       path,
            "label":      label,
            "confidence": round(conf, 4),
        })
        if label == "Fake":
            n_fake += 1
        else:
            n_real += 1

    # ── Summary ──
    if not args.quiet:
        print(f"\n{'─'*50}")
        print(f"  Total images  : {len(results)}")
        print(f"  {REAL_COLOR}Real{RESET}          : {n_real}")
        print(f"  {FAKE_COLOR}Fake{RESET}          : {n_fake}")
        print(f"  Throughput    : {len(results)/elapsed:.1f} imgs/s")
        print(f"{'─'*50}")

        # Show top results
        print("\n  Top predictions:")
        for r in results[:10]:
            color = FAKE_COLOR if r["label"] == "Fake" else REAL_COLOR
            print(
                f"  {color}{r['label']:4s}{RESET}"
                f"  {r['confidence']:.3f}"
                f"  {DIM}{Path(r['path']).name}{RESET}"
            )
        if len(results) > 10:
            print(f"  {DIM}... and {len(results)-10} more{RESET}")

    # ── CSV export ──
    if args.output:
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"\n  💾 Results saved → {args.output}")
        except ImportError:
            import csv
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "label", "confidence"])
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  💾 Results saved → {args.output}")

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Banner ──
    if not args.quiet:
        print(BANNER)

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.quiet:
        print(f"  Device      : {device}")

    # ── Checkpoint ──
    checkpoint = _resolve_checkpoint(args.checkpoint)
    if checkpoint:
        if not args.quiet:
            print(f"  Checkpoint  : {checkpoint}")
    else:
        print(
            f"  ⚠️  No checkpoint found. Using randomly initialised weights.\n"
            f"     Run training first, or provide --checkpoint path.\n"
        )

    # ── Build engine ──
    from inference.engine import InferenceEngine
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint or "",
        vit_model=args.vit_model,
        threshold=args.threshold,
        device=device,
    )

    if not args.quiet:
        print(f"  Threshold   : {args.threshold}")
        print(f"  ViT model   : {args.vit_model}\n")

    # ── Run inference ──
    if args.img:
        predict_single(args, engine)
    else:
        predict_batch(args, engine)


if __name__ == "__main__":
    main()
