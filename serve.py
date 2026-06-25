#!/usr/bin/env python3
"""
serve.py
────────
🚀  FastAPI inference server for the DeepFake Detector.

Endpoints:
  POST /predict          — JSON response with scores only
  POST /predict/visualize — JSON response with scores + base64 visualisation PNG

Usage:
  uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

Environment variables (optional):
  CHECKPOINT   : path to .pth checkpoint  (default: ./checkpoints/classifier_d3/best_model.pth)
  VIT_MODEL    : timm ViT name            (default: vit_base_patch16_224)
  THRESHOLD    : decision threshold       (default: 0.5)
  DEVICE       : cuda / cpu              (default: auto)
"""

from __future__ import annotations

import io
import os
import sys
import time
import base64
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# ── Ensure project root is importable ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deepfake-api")


# ══════════════════════════════════════════════════════════════════════════════
# App-level state (loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════════
class AppState:
    engine     = None
    vis_pipeline = None


state = AppState()


# ── Startup / Shutdown ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts."""
    checkpoint = os.environ.get(
        "CHECKPOINT",
        str(PROJECT_ROOT / "checkpoints" / "classifier_d3" / "best_model.pth"),
    )
    vit_model = os.environ.get("VIT_MODEL", "vit_base_patch16_224")
    threshold = float(os.environ.get("THRESHOLD", "0.5"))
    device_str = os.environ.get("DEVICE", None)

    device = (
        torch.device(device_str)
        if device_str
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    log.info("━" * 50)
    log.info("🔍  DeepFake Detector API — ViT + SSL")
    log.info("━" * 50)
    log.info(f"  Device     : {device}")
    log.info(f"  Checkpoint : {checkpoint}")
    log.info(f"  ViT model  : {vit_model}")
    log.info(f"  Threshold  : {threshold}")

    from inference.engine import InferenceEngine
    from visualization.gradcam import VisualisationPipeline

    state.engine = InferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint,
        vit_model=vit_model,
        threshold=threshold,
        device=device,
    )
    state.vis_pipeline = VisualisationPipeline(
        model=state.engine.model,
        device=device,
    )

    log.info("✅  Model ready — server accepting requests")
    log.info("━" * 50)

    yield  # ── server is running ──

    log.info("🛑  Shutting down — releasing resources")
    state.engine     = None
    state.vis_pipeline = None


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="DeepFake Detector API",
    description=(
        "ViT + SSL (DINO · Contrastive · PACL) deepfake detection.\n\n"
        "Upload an image to get a **Real / Fake** prediction with confidence "
        "scores and an optional Grad-CAM + PACL visualisation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow any frontend origin ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Response Schemas
# ══════════════════════════════════════════════════════════════════════════════
class PredictionResult(BaseModel):
    label: str             = Field(...,  example="Real",  description="Real or Fake")
    confidence: float      = Field(...,  example=0.965,   description="Confidence of the predicted label [0-1]")
    fake_prob: float       = Field(...,  example=0.035,   description="Raw probability of being Fake [0-1]")
    real_prob: float       = Field(...,  example=0.965,   description="Raw probability of being Real [0-1]")
    elapsed_ms: float      = Field(...,  example=141.7,   description="Inference time in milliseconds")
    filename: str          = Field(...,  example="face.png")


class VisualizeResult(PredictionResult):
    visualization_b64: str = Field(
        ...,
        description=(
            "Base64-encoded PNG of the 3-panel visualisation "
            "(Original | Attention Rollout | PACL Consistency)."
        ),
    )
    visualization_mime: str = Field(default="image/png")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _read_upload(upload: UploadFile) -> Image.Image:
    """Read an UploadFile and return a PIL RGB image."""
    raw = upload.file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")
    return img


def _run_inference(img: Image.Image, filename: str) -> dict:
    """Run InferenceEngine.predict on a PIL image and return enriched result."""
    t0     = time.perf_counter()
    result = state.engine.predict(img)
    elapsed = (time.perf_counter() - t0) * 1000  # ms

    prob       = result["prob"]
    label      = result["label"]
    confidence = result["confidence"]

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "fake_prob":  round(prob, 4),
        "real_prob":  round(1.0 - prob, 4),
        "elapsed_ms": round(elapsed, 1),
        "filename":   filename,
    }


def _generate_visualization(img: Image.Image, label: str, confidence: float) -> str:
    """
    Run VisualisationPipeline and return the result as a base64-encoded PNG string.
    Uses a temporary file so we don't clutter the working directory.
    """
    from data.augmentations import preprocess_image, preprocess_fft

    device = state.engine.device

    rgb = preprocess_image(img, size=state.engine.image_size).to(device)
    fft = preprocess_fft(img,   size=state.engine.image_size).to(device)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        img.save(tmp_img.name)
        tmp_img_path = tmp_img.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_vis:
        vis_path = tmp_vis.name

    try:
        state.vis_pipeline.visualise(
            img_path   = tmp_img_path,
            rgb_tensor = rgb,
            fft_tensor = fft,
            label      = label,
            confidence = confidence,
            save_path  = vis_path,
        )
        with open(vis_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    finally:
        Path(tmp_img_path).unlink(missing_ok=True)
        Path(vis_path).unlink(missing_ok=True)

    return b64


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """Health check — confirms the API is alive."""
    return {
        "status":  "ok",
        "service": "DeepFake Detector API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check including model readiness."""
    model_ready = state.engine is not None
    return {
        "status":      "ok" if model_ready else "loading",
        "model_ready": model_ready,
        "device":      str(state.engine.device) if model_ready else None,
        "threshold":   state.engine.threshold   if model_ready else None,
    }


# ── /predict ─────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Predict Real / Fake",
    tags=["Inference"],
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WebP, BMP …)"),
):
    """
    Upload an image and receive a **Real / Fake** prediction with confidence scores.

    Returns:
    - `label`      — "Real" or "Fake"
    - `confidence` — confidence of the predicted label (0–1)
    - `fake_prob`  — raw probability of being Fake
    - `real_prob`  — raw probability of being Real
    - `elapsed_ms` — server-side inference time
    """
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    img    = _read_upload(file)
    result = _run_inference(img, filename=file.filename or "upload")
    log.info(
        f"[predict] {result['filename']} → {result['label']} "
        f"({result['confidence']*100:.1f}%)  {result['elapsed_ms']} ms"
    )
    return result


# ── /predict/visualize ────────────────────────────────────────────────────
@app.post(
    "/predict/visualize",
    response_model=VisualizeResult,
    summary="Predict + Visualisation",
    tags=["Inference"],
)
async def predict_visualize(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WebP, BMP …)"),
):
    """
    Upload an image and receive a **Real / Fake** prediction **plus** a
    base64-encoded 3-panel visualisation PNG:

    | Panel | Description |
    |-------|-------------|
    | 1     | Original image (resized to 224×224) |
    | 2     | Attention Rollout — which patches the ViT focuses on |
    | 3     | PACL Consistency Map — inter-patch similarity heatmap |

    Decode the `visualization_b64` field in your frontend with:
    ```js
    const src = `data:image/png;base64,${response.visualization_b64}`;
    ```
    """
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    img    = _read_upload(file)
    result = _run_inference(img, filename=file.filename or "upload")

    try:
        b64 = _generate_visualization(img, result["label"], result["confidence"])
    except Exception as exc:
        log.warning(f"[visualize] Visualisation failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference succeeded but visualisation failed: {exc}",
        )

    result["visualization_b64"]  = b64
    result["visualization_mime"] = "image/png"

    log.info(
        f"[visualize] {result['filename']} → {result['label']} "
        f"({result['confidence']*100:.1f}%)  {result['elapsed_ms']} ms  "
        f"[vis OK — {len(b64)//1024} KB]"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Dev entrypoint
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # set True for dev; False keeps GPU model in memory
        log_level="info",
    )
