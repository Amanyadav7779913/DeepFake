"""
inference/ensemble.py
──────────────────────
EnsembleInferenceEngine — Combines multiple InferenceEngines for more robust predictions.
Uses Soft-Voting (probability averaging).
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional
from .engine import InferenceEngine

class EnsembleInferenceEngine:
    """
    Manages multiple InferenceEngine instances and aggregates their predictions.
    """

    def __init__(
        self,
        engines:   List[InferenceEngine],
        threshold: float = 0.5,
    ):
        self.engines    = engines
        self.threshold  = threshold
        self.device     = engines[0].device if engines else torch.device("cpu")

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[str],
        vit_model:        str   = "vit_base_patch16_224",
        image_size:       int   = 224,
        threshold:        float = 0.5,
        device:           Optional[torch.device] = None,
    ) -> "EnsembleInferenceEngine":
        """Build an ensemble from a list of checkpoint files."""
        engines = []
        for path in checkpoint_paths:
            if not os.path.exists(path):
                print(f"[Ensemble] Warning: Checkpoint not found at {path}. Skipping.")
                continue
            
            eng = InferenceEngine.from_checkpoint(
                checkpoint_path=path,
                vit_model=vit_model,
                image_size=image_size,
                threshold=threshold,
                device=device
            )
            engines.append(eng)
        
        if not engines:
            raise RuntimeError("No valid checkpoints found. Cannot create ensemble.")
        
        return cls(engines=engines, threshold=threshold)

    @torch.no_grad()
    def predict(self, img_input) -> dict:
        """
        Run inference through all models and average the results.
        """
        probs = []
        for engine in self.engines:
            res = engine.predict(img_input)
            probs.append(res["prob"])
        
        avg_prob = float(np.mean(probs))
        
        label      = "Fake" if avg_prob >= self.threshold else "Real"
        confidence = avg_prob if avg_prob >= self.threshold else 1.0 - avg_prob

        return {
            "label":      label,
            "confidence": confidence,
            "prob":       avg_prob,
            "all_probs":  probs  # useful for debugging
        }

    @torch.no_grad()
    def predict_batch(
        self,
        img_inputs: List,
        batch_size: int = 16,
    ) -> List[dict]:
        """
        Run batch inference through all models and average.
        """
        # Dictionary to store accumulated probabilities per image
        all_model_probs = [[] for _ in range(len(img_inputs))]

        for engine in self.engines:
            preds = engine.predict_batch(img_inputs, batch_size=batch_size)
            # engine.predict_batch currently returns (label, confidence)
            # We should probably update it to return probs too, or use raw model output.
            # To keep it simple for now, we'll re-implement the averaging at the logit level if needed,
            # but for now let's just use the single predict loop if batch is not optimised.
            pass

        # For simplicity and correctness in this first version, we'll iterate
        # (Optimisation can be added later by batching across all models)
        results = []
        for inp in img_inputs:
            results.append(self.predict(inp))
        
        return results
