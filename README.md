---
title: BOM Pattern Detection
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Zero-Shot Pattern Detection for Engineering BOM Drawings

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![HuggingFace Demo](https://img.shields.io/badge/🤗-Demo-yellow)](https://huggingface.co/spaces/PLACEHOLDER)

## Overview

This project solves the problem of finding all occurrences of a given pattern (e.g., a component symbol) inside large engineering BOM drawings — with **zero training data**. The pipeline combines classical NCC template matching for fast candidate proposal with DINOv2 ViT-S/14 self-supervised features for accurate zero-shot verification. No fine-tuning or labeled data is required: any pattern can be detected at inference time.

## Pipeline Architecture

```
[Pattern Image] ──┐
                  ├──► Stage 0: Preprocess ──► Stage 1: NCC Multi-scale
[Drawing Image] ──┘         (binarize,              (propose 30–200
                              denoise)                candidates, CPU)
                                                          │
                                                          ▼
                                              Stage 2: DINOv2 Verify
                                              (cosine similarity filter,
                                               zero-shot, CPU/GPU)
                                                          │
                                                          ▼
                                              Stage 4: Post-process
                                              (IoU-NMS, format JSON,
                                               draw bounding boxes)
                                                          │
                                                          ▼
                                              [BBoxes + Score JSON]
```

| Stage | Module | Purpose | Typical Time |
|-------|--------|---------|-------------|
| 0 | Preprocessor | Adaptive binarize, denoise | < 0.5s |
| 1 | NCCMatcher | Multi-scale + rotation candidate proposal | 5–15s CPU |
| 2 | DINOVerifier | Zero-shot cosine similarity filter | 5–15s CPU |
| 4 | Postprocessor | Final NMS, JSON output, visualization | < 0.1s |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/pattern-detection-bom.git
cd pattern-detection-bom
pip install -r requirements.txt
```

DINOv2 weights are downloaded automatically at first run via `torch.hub` — no manual download needed.

## Quick Start

### Python API

```python
from src.pipeline import run_detection

result = run_detection("pattern.png", "drawing.png")
print(result)
```

### Web UI (local)

```bash
python app/app.py
```

Then open `http://localhost:8000` in your browser. The dashboard lets you upload a pattern and drawing, adjust detection thresholds via sliders, and view annotated results.

### CLI Tuning

```bash
python scripts/tune_thresholds.py --examples_dir examples --quick
```

## Output Format

```json
{
  "detections": [
    {
      "bbox": {"x": 142, "y": 310, "w": 64, "h": 64},
      "confidence": 0.83,
      "ncc_score": 0.7812,
      "dino_score": 0.8754,
      "scale": 1.0,
      "angle": 0.0
    }
  ],
  "total_detections": 3,
  "image_size": {"width": 2480, "height": 3508}
}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scales` | `[0.85..1.15]` | Scale range for template resizing in Stage 1 |
| `angles` | `[-10..10]` | Rotation sweep in Stage 1 (degrees) |
| `ncc_threshold` | `0.55` | Minimum NCC score to propose a candidate (low = high recall) |
| `nms_iou_threshold` | `0.3` | IoU threshold for NMS in Stage 1 |
| `dino_model` | `dinov2_vits14` | DINOv2 variant: `vits14` (fast) or `vitb14` (accurate) |
| `cosine_threshold` | `0.84` | Minimum cosine similarity to accept a detection |
| `final_nms_iou` | `0.4` | Final IoU threshold after Stage 2 |

Override via config dict:

```python
from src.pipeline import PatternDetectionPipeline

pipeline = PatternDetectionPipeline(config={
    "ncc_threshold": 0.60,
    "cosine_threshold": 0.80,
    "dino_model": "dinov2_vitb14",
})
result = pipeline.detect("pattern.png", "drawing.png")
```

## Approach & Design Choices

### Why DINOv2 instead of a fine-tuned model?

DINOv2 is trained self-supervised on 142M images and produces patch-level features that capture geometric structure rather than texture or color. Engineering drawings (line art on white background) are a significant domain shift from natural photos, yet DINOv2 generalizes well because its features encode shape primitives. Crucially, **no fine-tuning means truly zero-shot operation** — the model works for any new pattern without collecting labeled data.

### Why hybrid NCC + DINOv2?

- **NCC alone** is fast but brittle: it fails under lighting variation, noise, and even small domain differences.
- **DINOv2 alone** (brute-force sliding window) over a large drawing would be prohibitively slow on CPU.
- The hybrid approach uses NCC for high-recall candidate generation (~30–200 boxes), then DINOv2 as an intelligent filter. This gives the accuracy of a neural verifier at a fraction of the cost.

### Trade-offs

- NCC threshold is intentionally low (0.55) to maximize recall at Stage 1; false positives are cleaned up by DINOv2.
- ViT-S/14 (21M params) is chosen over ViT-B/14 for speed on CPU deployments; swap to `vitb14` for higher accuracy.
- The pipeline runs on CPU in ~20–40s for typical A3 drawings; GPU reduces Stage 2 to 2–5s.

## Limitations & Future Work

- **Small templates (< 28px):** DINOv2 Stage 2 is skipped for very small patterns; only NCC is used.
- **Heavy rotation (> 15°):** The sweep is limited to ±10° by default; large rotations require wider sweep at extra cost.
- **Identical-looking components:** Two structurally similar but semantically different symbols may confuse the cosine verifier.
- **Throughput:** Not optimized for batch processing of many drawings; a production system would benefit from ONNX export and TensorRT.
- **Optional Stage 3 (LightGlue):** Geometric keypoint verification for handling significant rotation/scale variation is planned but not yet integrated.

## License

MIT License — see [LICENSE](LICENSE).
