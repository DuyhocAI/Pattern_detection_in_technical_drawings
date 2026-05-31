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

## Overview

Find every occurrence of a given component symbol inside large engineering BOM drawings — with **zero training data**. The pipeline combines three stages: classical NCC template matching (fast candidate proposal) → DINOv2 ViT-S/14 self-supervised verification → optional Qwen2-VL-2B semantic filter. No fine-tuning or labelled data needed; any pattern works at inference time.

## Pipeline Architecture

```
[Pattern Image] ──┐
                  ├──► Stage 0: Preprocess  ──► Stage 1: NCC Multi-scale
[Drawing Image] ──┘      (binarize, denoise)      (multi-angle candidate
                                                   proposal on CPU)
                                                        │
                                                        ▼
                                            Stage 2: DINOv2 Verify
                                            (cosine similarity, zero-shot,
                                             orientation-invariant crops)
                                                        │
                                                        ▼
                                          Stage 2b: Structural Filters
                                          (wire-leads, Chamfer, NMS,
                                           confidence-gap pruning)
                                                        │
                                             [use_vlm=True only]
                                                        │
                                                        ▼
                                            Stage 3: VLM Filter
                                            (Qwen2-VL-2B open-classification,
                                             borderline candidates only)
                                                        │
                                                        ▼
                                            [BBoxes + Score JSON]
```

| Stage | Module | Purpose | Typical time (GPU) |
|-------|--------|---------|-------------------|
| 0 | `Preprocessor` | Adaptive binarize, denoise | < 0.5 s |
| 1 | `NCCMatcher` | Multi-scale + rotation candidate proposal | 15–60 s |
| 2 | `DINOVerifier` | Zero-shot cosine similarity filter | 2–10 s |
| 2b | `Postprocessor` | Wire-lead / Chamfer / NMS / gap filter | < 0.5 s |
| 3 | `VLMVerifier` | Qwen2-VL-2B semantic cross-check (optional) | ~0.4 s/crop |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/pattern-detection-bom.git
cd pattern-detection-bom
pip install -r requirements.txt
```

DINOv2 weights download automatically via `torch.hub` on first run. VLM weights (`Qwen/Qwen2-VL-2B-Instruct`, ~5 GB) download automatically from HuggingFace on first VLM-enabled run.

## Quick Start

### Python API

```python
from src.pipeline import PatternDetectionPipeline

pipe = PatternDetectionPipeline()
result = pipe.detect_auto("pattern.png", "drawing.png")
print(result["total_detections"])
for d in result["detections"]:
    print(d["bbox"], d["confidence"])
```

### Web UI (local)

```bash
python app/web/server.py
# open http://localhost:8000
```

Upload a pattern and drawing, click **Detect**, and view annotated results. The **Use VLM** checkbox activates the optional Stage-3 semantic filter.

## Output Format

```json
{
  "detections": [
    {
      "bbox": {"x": 142, "y": 310, "w": 64, "h": 32},
      "confidence": 0.91,
      "ncc_score": 0.72,
      "dino_score": 0.88,
      "scale": 1.05,
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
| `ncc_threshold` | `0.55` (strict) / `0.47` (relaxed) | NCC score gate for candidate proposal |
| `cosine_threshold` | `0.84` | DINOv2 cosine similarity gate |
| `use_vlm` | `False` | Enable Stage-3 VLM semantic filter |
| `vlm_model` | `Qwen/Qwen2-VL-2B-Instruct` | VLM model ID (HuggingFace) |
| `vlm_keep_min_conf` | `0.75` | Candidates above this are trusted; VLM is not called |
| `vlm_reject_only` | `True` | Blacklist mode: drop only candidates the VLM confidently identifies as a different component |
| `vlm_recall_boost` | follows `use_vlm` | Widen the scale sweep and relax Chamfer gates when VLM is on |
| `vlm_symbol_name` | `None` | Optional hint for VLM classification prompt |

```python
pipe = PatternDetectionPipeline(config={
    "use_vlm": True,
    "vlm_symbol_name": "a resistor (zigzag or plain rectangle)",
    "vlm_keep_min_conf": 0.78,
})
result = pipe.detect_auto("resistor_template.png", "schematic.png")
```

## Approach & Design Choices

### Stage 1 — NCC Template Matching

Multi-scale, multi-angle normalized cross-correlation proposes 30–300 candidate regions per drawing. The threshold is kept intentionally low (high recall); false positives are cleaned up downstream. Two passes run in sequence: a strict pass (NCC ≥ 0.55) and a relaxed pass (NCC ≥ 0.47). For complex templates a third rotated pass (0° and 90°) is added automatically.

### Stage 2 — DINOv2 Zero-Shot Verification

DINOv2 ViT-S/14 produces orientation-invariant patch features by normalizing crops to a canonical pose before embedding. Candidates are kept only if their cosine similarity to the template embedding exceeds `cosine_threshold`. DINOv2 generalizes well to engineering line-art despite being trained on natural images because it captures shape primitives rather than texture.

The hybrid NCC → DINOv2 design is key: NCC is fast on CPU but brittle; DINOv2 is accurate but a dense sliding-window would be prohibitively slow. Combining them gives neural-level accuracy at a fraction of the cost.

### Stage 2b — Structural Filters

A suite of geometry-based post-processors reject common false positives that NCC and DINOv2 cannot separate:

- **Wire-lead filter** — a real component has visible wire leads protruding from its bbox.
- **Chamfer distance filter** — rejects crops whose edge silhouette diverges too far from the template.
- **Neighborhood complexity filter** — genuine components sit in a locally sparse region; text/grid regions are excluded.
- **Junction-dot and rect-integrity filters** — reject L-junctions, connector dots, and partial matches.
- **Confidence-gap filter** — drops a trailing cluster of low-confidence candidates when a clear gap exists above them.

### Stage 3 — VLM Semantic Filter (optional)

`VLMVerifier` uses Qwen2-VL-2B in an open-classification mode: each borderline candidate crop is labelled into a closed 10-class component vocabulary. Candidates the VLM calls a concretely different component (inductor, capacitor, diode, op-amp, crystal, logic-gate) are dropped.

**Why blacklist instead of whitelist?** The 2B model defaults ambiguous line-art to "transistor" — a strict whitelist wrongly discards genuine resistors it mislabels. The blacklist (`vlm_reject_only=True`) preserves them while still removing clearly wrong labels.

**Why open-classification?** Yes/no confirmation gives 100% agreement bias with small VLMs. Forced open-classification breaks the bias.

High-confidence candidates (≥ `vlm_keep_min_conf`) are never sent to the VLM — they are trusted unconditionally. This shields correct detections from the 2B model's occasional mislabelling.

## Limitations

- **VLM labels are individually noisy** at 2B scale. The filter is effective at the population level (cuts ~20–30% of borderline FPs) but may miss individual errors.
- **Heavy rotation (> 15°):** The standard sweep covers ±10°; complex templates add 0° and 90° passes. Arbitrary angles require a wider sweep at extra cost.
- **Throughput:** Single-drawing processing. A production system would benefit from ONNX export.

## License

MIT License — see [LICENSE](LICENSE).
