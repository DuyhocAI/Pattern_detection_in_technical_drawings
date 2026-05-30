"""FastAPI backend for BOM Pattern Detection web UI."""
import io
import csv
import base64
import time
import sys
import os
import traceback

# Fix OpenMP conflict on Windows/Anaconda
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline import PatternDetectionPipeline

app = FastAPI(title="BOM Pattern Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Cached pipeline instance
_pipeline: Optional[PatternDetectionPipeline] = None


def get_pipeline(config: dict = None) -> PatternDetectionPipeline:
    global _pipeline
    if _pipeline is None:
        print("[Server] Loading pipeline...")
        _pipeline = PatternDetectionPipeline(config=config)
        print("[Server] Pipeline ready.")
    return _pipeline


def upload_to_numpy(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def numpy_to_b64(img: np.ndarray) -> str:
    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode="L")
    else:
        pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


def _coerce_bool(val) -> bool:
    """Form values arrive as strings; coerce 'true'/'1'/'on' to bool."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "on", "yes")


@app.post("/api/detect")
async def detect(
    pattern: UploadFile = File(...),
    drawing: UploadFile = File(...),
    mode: str = Form("auto"),
    ncc_threshold: float = Form(0.55),
    cosine_threshold: float = Form(0.84),
    final_nms_iou: float = Form(0.4),
    use_vlm: str = Form("false"),
):
    try:
        t_start = time.time()

        pattern_bytes = await pattern.read()
        drawing_bytes = await drawing.read()

        pattern_np = upload_to_numpy(pattern_bytes)
        drawing_np = upload_to_numpy(drawing_bytes)

        pipeline = get_pipeline()
        pipeline.update_thresholds(
            ncc_threshold=ncc_threshold,
            cosine_threshold=cosine_threshold,
            final_nms_iou=final_nms_iou,
        )
        # VLM Stage-3 is toggled at runtime (model lazy-loads on first enable).
        pipeline.use_vlm = _coerce_bool(use_vlm)

        if mode == "auto":
            result = pipeline.detect_auto(pattern_np, drawing_np, return_visualization=True)
        else:
            result = pipeline.detect(pattern_np, drawing_np, return_visualization=True)

        elapsed = round(time.time() - t_start, 2)

        viz_b64 = None
        if "visualization" in result and result["visualization"] is not None:
            viz_b64 = numpy_to_b64(result["visualization"])

        # Also return the original drawing as b64 for display
        drawing_b64 = numpy_to_b64(drawing_np)

        return JSONResponse({
            "success": True,
            "total_detections": result["total_detections"],
            "detections": result["detections"],
            "elapsed": elapsed,
            "visualization": viz_b64,
            "drawing_preview": drawing_b64,
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def _detections_to_csv(detections: list) -> str:
    """Convert detection list to CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "x", "y", "width", "height", "confidence", "ncc_score", "dino_score", "scale", "angle"])
    for i, d in enumerate(detections, 1):
        b = d["bbox"]
        writer.writerow([
            i,
            b["x"], b["y"], b["w"], b["h"],
            round(d.get("confidence", 0), 4),
            round(d.get("ncc_score", 0), 4),
            round(d.get("dino_score", 0), 4),
            round(d.get("scale", 1.0), 4),
            round(d.get("angle", 0), 1),
        ])
    return buf.getvalue()


@app.post("/api/detect/csv")
async def detect_csv(
    pattern: UploadFile = File(...),
    drawing: UploadFile = File(...),
    mode: str = Form("auto"),
    ncc_threshold: float = Form(0.55),
    cosine_threshold: float = Form(0.84),
    final_nms_iou: float = Form(0.4),
    use_vlm: str = Form("false"),
):
    """Run detection and return results as a downloadable CSV file."""
    try:
        pattern_bytes = await pattern.read()
        drawing_bytes = await drawing.read()
        pattern_np = upload_to_numpy(pattern_bytes)
        drawing_np = upload_to_numpy(drawing_bytes)

        pipeline = get_pipeline()
        pipeline.update_thresholds(
            ncc_threshold=ncc_threshold,
            cosine_threshold=cosine_threshold,
            final_nms_iou=final_nms_iou,
        )
        pipeline.use_vlm = _coerce_bool(use_vlm)
        if mode == "auto":
            result = pipeline.detect_auto(pattern_np, drawing_np, return_visualization=False)
        else:
            result = pipeline.detect(pattern_np, drawing_np, return_visualization=False)

        csv_str = _detections_to_csv(result["detections"])
        return StreamingResponse(
            io.BytesIO(csv_str.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=detections.csv"},
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}


if __name__ == "__main__":
    # Pre-load pipeline before serving
    get_pipeline()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
