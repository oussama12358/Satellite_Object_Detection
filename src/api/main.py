"""
SatDet FastAPI Service
========================
Production REST API for satellite object detection inference.

Endpoints:
    GET  /health                 — Service health check
    GET  /info                   — Model information
    POST /predict                — Single image detection
    POST /predict_batch          — Multi-image batch detection
    GET  /docs                   — Swagger UI (auto-generated)

Design:
    - Async file I/O to avoid blocking the event loop on uploads
    - In-memory model loading at startup (no per-request cold start)
    - Optional SAHI inference via query parameter
    - Structured logging with request IDs for traceability
"""

import os
import time
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn
from loguru import logger

from src.api.schemas import (
    DetectionResponse, BatchDetectionResponse,
    HealthResponse, ModelInfoResponse, Detection, BoundingBox
)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "models/weights/best.pt")
MODEL_VERSION = os.getenv("MODEL_VERSION", "yolov8s-satdet-v1")
DEVICE = os.getenv("DEVICE", "cpu")        # "cuda:0" in production
DEFAULT_CONF = float(os.getenv("CONF_THRESHOLD", "0.25"))
DEFAULT_IOU = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_MB", "50"))

app = FastAPI(
    title="SatDet — Satellite Object Detection API",
    description=(
        "Production inference API for detecting aircraft, ships, vehicles, "
        "and infrastructure in satellite/aerial imagery using YOLOv8."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────────
_model = None
_sahi_predictor = None


def as_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or array-like value to a NumPy array."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


@app.on_event("startup")
async def load_model():
    """Load model once at startup — not per request."""
    global _model, _sahi_predictor
    if _model is not None:
        logger.info("Model already loaded; skipping startup load.")
        return

    try:
        from ultralytics import YOLO
        weights_path = Path(MODEL_WEIGHTS)

        if not weights_path.exists():
            logger.warning(
                f"Weights not found at {weights_path}. "
                "Attempting to load pretrained yolov8s.pt for demo..."
            )
            _model = YOLO("yolov8s.pt")
        else:
            _model = YOLO(str(weights_path))

        logger.success(f"Model loaded: {MODEL_WEIGHTS} | device={DEVICE}")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError(f"Cannot start service: {e}")


def get_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to OpenCV BGR image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    return img


def run_inference(
    img: np.ndarray,
    conf: float,
    iou: float,
    use_sahi: bool,
    tile_size: int,
    overlap: float,
) -> dict:
    """Unified inference dispatch — standard or SAHI."""
    model = get_model()

    if use_sahi:
        from src.inference.sahi_predictor import SAHIPredictor
        predictor = SAHIPredictor(
            MODEL_WEIGHTS, conf=conf, iou=iou,
            tile_size=tile_size, overlap=overlap,
            device=DEVICE,
        )
        return predictor.predict(img)
    else:
        t0 = time.time()
        results = model.predict(img, conf=conf, iou=iou, device=DEVICE, verbose=False)
        ms = (time.time() - t0) * 1000

        boxes, scores, labels = [], [], []
        for r in results:
            if r.boxes:
                boxes.extend(as_numpy(r.boxes.xyxy).tolist())
                scores.extend(as_numpy(r.boxes.conf).tolist())
                labels.extend(as_numpy(r.boxes.cls).astype(int).tolist())

        return {
            "boxes": np.array(boxes).reshape(-1, 4) if boxes else np.array([]).reshape(0, 4),
            "scores": np.array(scores),
            "labels": np.array(labels),
            "class_names": model.names,
            "inference_ms": ms,
            "num_tiles": 1,
        }


def format_detections(result: dict, img_h: int, img_w: int) -> List[Detection]:
    """Convert raw result dict to Detection schema list."""
    detections = []
    class_names = result.get("class_names", {})

    boxes = result.get("boxes", np.array([]))
    scores = result.get("scores", np.array([]))
    labels = result.get("labels", np.array([]))

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(float, box)
        detections.append(Detection(
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=float(score),
            class_id=int(label),
            class_name=class_names.get(int(label), f"class_{int(label)}"),
        ))
    return detections


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect browser visits to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health probe."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        device=DEVICE,
        model_version=MODEL_VERSION,
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata."""
    model = get_model()
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        architecture="YOLOv8s",
        num_classes=len(model.names),
        class_names={int(k): v for k, v in model.names.items()},
        input_size=640,
        supports_sahi=True,
    )


@app.post("/predict", response_model=DetectionResponse)
async def predict(
    file: UploadFile = File(..., description="Satellite image (JPEG/PNG/TIFF)"),
    conf: float = Query(DEFAULT_CONF, ge=0.01, le=1.0, description="Confidence threshold"),
    iou: float = Query(DEFAULT_IOU, ge=0.1, le=1.0, description="NMS IoU threshold"),
    use_sahi: bool = Query(False, description="Use SAHI sliced inference (for large images)"),
    tile_size: int = Query(640, ge=320, le=1280, description="SAHI tile size in pixels"),
    overlap: float = Query(0.2, ge=0.0, le=0.5, description="SAHI tile overlap fraction"),
):
    """
    Detect objects in a single satellite image.

    Returns bounding boxes, class labels, and confidence scores.
    Set `use_sahi=true` for images larger than 1280×1280 pixels.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/tiff", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {MAX_IMAGE_SIZE_MB} MB)"
        )

    img = bytes_to_cv2(image_bytes)
    img_h, img_w = img.shape[:2]

    result = run_inference(img, conf, iou, use_sahi, tile_size, overlap)
    detections = format_detections(result, img_h, img_w)

    return DetectionResponse(
        filename=file.filename or "unknown",
        image_width=img_w,
        image_height=img_h,
        num_detections=len(detections),
        detections=detections,
        inference_ms=result.get("inference_ms", 0.0),
        model_version=MODEL_VERSION,
        use_sahi=use_sahi,
    )


@app.post("/predict_batch", response_model=BatchDetectionResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple satellite images"),
    conf: float = Query(DEFAULT_CONF, ge=0.01, le=1.0),
    iou: float = Query(DEFAULT_IOU, ge=0.1, le=1.0),
    use_sahi: bool = Query(False),
):
    """
    Batch detection for multiple satellite images.

    Processes up to 20 images per request.
    """
    if len(files) > 20:
        raise HTTPException(
            status_code=400, detail="Maximum 20 images per batch request"
        )

    results = []
    for file in files:
        try:
            image_bytes = await file.read()
            img = bytes_to_cv2(image_bytes)
            img_h, img_w = img.shape[:2]
            result = run_inference(img, conf, iou, use_sahi, 640, 0.2)
            detections = format_detections(result, img_h, img_w)

            results.append(DetectionResponse(
                filename=file.filename or "unknown",
                image_width=img_w,
                image_height=img_h,
                num_detections=len(detections),
                detections=detections,
                inference_ms=result.get("inference_ms", 0.0),
                model_version=MODEL_VERSION,
                use_sahi=use_sahi,
            ))
        except Exception as e:
            logger.error(f"Failed on {file.filename}: {e}")
            results.append(DetectionResponse(
                filename=file.filename or "unknown",
                image_width=0, image_height=0,
                num_detections=0, detections=[],
                inference_ms=0.0, model_version=MODEL_VERSION,
            ))

    return BatchDetectionResponse(
        total_images=len(files),
        total_detections=sum(r.num_detections for r in results),
        results=results,
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
