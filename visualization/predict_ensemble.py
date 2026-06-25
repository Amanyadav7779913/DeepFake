#!/usr/bin/env python3
"""
predict_ensemble.py
───────────────────
Run inference using an ensemble of multiple DeepFake Detector models.

Usage (from root or visualization folder):
  python visualization/predict_ensemble.py --img test.jpg --checkpoints path/to/m1.pth ...
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List

# ── Robust project root discovery ──
# This allows the script to find 'inference' and 'models' whether it's run from root or visualization/
current_dir = Path(__file__).resolve().parent
root_dir = current_dir
while root_dir != root_dir.parent:
    if (root_dir / "inference").exists() and (root_dir / "models").exists():
        break
    root_dir = root_dir.parent

sys.path.insert(0, str(root_dir))

import torch
try:
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from inference.ensemble import EnsembleInferenceEngine

def parse_args():
    p = argparse.ArgumentParser(description="Ensemble DeepFake Detector")
    
    # Input
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--img",     type=str, help="Path to a single image file")
    inp.add_argument("--img_dir", type=str, help="Directory of images for batch inference")

    # Models
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of paths to model checkpoints (.pth)"
    )
    p.add_argument(
        "--vit_model",
        type=str,
        default="vit_base_patch16_224",
    )

    # Inference
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--device",     type=str,   default=None)

    # Output
    p.add_argument("--output",     type=str,   default=None, help="Save results as CSV")
    
    return p.parse_args()

def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*50)
    print(" 🛡️  Ensemble DeepFake Detector")
    print("="*50)
    print(f" Models    : {len(args.checkpoints)}")
    print(f" Device    : {device}")
    
    # ── Load Ensemble ──
    try:
        ensemble = EnsembleInferenceEngine.from_checkpoints(
            checkpoint_paths=args.checkpoints,
            vit_model=args.vit_model,
            device=device,
            threshold=args.threshold
        )
    except Exception as e:
        print(f"❌ Error loading ensemble: {e}")
        return

    # ── Single Image ──
    if args.img:
        print(f"\n Processing: {args.img}")
        t0 = time.perf_counter()
        res = ensemble.predict(args.img)
        elapsed = time.perf_counter() - t0
        
        color = "\033[91m" if res["label"] == "Fake" else "\033[92m"
        reset = "\033[0m"
        
        print(f"\n Prediction: {color}{res['label']}{reset}")
        print(f" Confidence: {res['confidence']:.4f}")
        print(f" Avg Prob  : {res['prob']:.4f}")
        print(f" Time      : {elapsed*1000:.1f}ms")
        
        print("\n Individual Model Scores:")
        for i, p in enumerate(res["all_probs"]):
            print(f"   Model {i+1}: {p:.4f}")

    # ── Directory ──
    elif args.img_dir:
        from predict import _find_images
        image_paths = _find_images(args.img_dir)
        
        if not image_paths:
            print(f"❌ No images found in {args.img_dir}")
            return
            
        print(f"\n Found {len(image_paths)} images. Running ensemble...")
        
        results = []
        y_true = []
        y_pred = []
        y_prob = []
        
        t0 = time.perf_counter()
        for path in image_paths:
            res = ensemble.predict(path)
            res["path"] = path
            results.append(res)
            
            # ── Detect Ground Truth ──
            # Look for "fake" or "real" in the path (standard for deepfake datasets)
            norm_path = path.lower().replace("\\", "/")
            label_true = None
            if "/fake/" in norm_path or "/fake_" in norm_path or "fake/" in norm_path:
                label_true = 1
            elif "/real/" in norm_path or "/real_" in norm_path or "real/" in norm_path:
                label_true = 0
            
            if label_true is not None:
                y_true.append(label_true)
                y_pred.append(1 if res["label"] == "Fake" else 0)
                y_prob.append(res["prob"])
        
        elapsed = time.perf_counter() - t0
        n_fake = sum(1 for r in results if r["label"] == "Fake")
        
        print(f"\n{'━'*50}")
        print(f" 📊  ENSEMBLE SUMMARY")
        print(f"{'━'*50}")
        print(f"  Total Images : {len(results)}")
        print(f"  Real         : {len(results) - n_fake}")
        print(f"  Fake         : {n_fake}")
        print(f"  Avg Speed    : {len(results)/elapsed:.1f} imgs/s")
        
        if y_true and HAS_SKLEARN:
            acc = accuracy_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.0
            
            print(f"{'─'*50}")
            print(f"  ✅ ACCURACY  : {acc*100:.2f}%")
            print(f"  📈 AUC SCORE : {auc:.4f}")
            
            if len(set(y_true)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                print(f"  🎯 Precision : {precision:.4f}")
                print(f"  🔄 Recall    : {recall:.4f}")
        
        print(f"{'━'*50}")

        if args.output:
            summary = {
                "total_images": len(results),
                "real":         len(results) - n_fake,
                "fake":         n_fake,
                "avg_speed":    round(len(results)/elapsed, 2),
            }
            if y_true and HAS_SKLEARN:
                summary["accuracy"]  = round(acc, 4)
                summary["auc_score"] = round(auc, 4)
                if len(set(y_true)) > 1:
                    summary["precision"] = round(precision, 4)
                    summary["recall"]    = round(recall, 4)

            # Save as JSON or plain text
            output_path = Path(args.output)
            if output_path.suffix == ".json":
                import json
                with open(output_path, "w") as f:
                    json.dump(summary, f, indent=4)
            else:
                # Save as a clean text report
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("━"*50 + "\n")
                    f.write(" 📊  ENSEMBLE EVALUATION REPORT\n")
                    f.write("━"*50 + "\n")
                    for k, v in summary.items():
                        f.write(f" {k.replace('_', ' ').title():<15} : {v}\n")
                    f.write("━"*50 + "\n")
            
            print(f"\n 💾 Saved summary report to {args.output}")

if __name__ == "__main__":
    main()
