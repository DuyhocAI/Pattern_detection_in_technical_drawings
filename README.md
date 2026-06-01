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

![Python](https://img.shields.io/badge/python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red)
![DINOv2](https://img.shields.io/badge/DINOv2-ViT--S%2F14-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

Find every occurrence of a given component symbol inside large engineering BOM drawings — with **zero training data**. This is a Sotatek AI/Computer Vision assessment project that implements a complete **zero-shot pattern detection pipeline** combining three intelligence layers:

1. **Classical NCC multi-scale matching** — fast CPU-based candidate proposal
2. **DINOv2 ViT-S/14 zero-shot verification** — self-supervised semantic filtering
3. **Optional Qwen2-VL-2B VLM semantic classifier** — borderline confidence refinement

**No fine-tuning or labelled data needed.** Any pattern template works at inference time.

---

## 📊 Results (Test Drawing 4 + Zigzag Resistor Template)

| Metric | Value |
|--------|-------|
| **GT Boxes Detected** | 22/24 (91.7% recall) |
| **False Positives** | 0 |
| **Total Detections** | 22 |
| **Runtime (GPU RTX 3060)** | ~25 seconds |
| **Unit Tests** | 10/10 passing ✓ |

---

## 🏗️ Pipeline Architecture

```
┌──────────────┐         ┌──────────────┐
│ Pattern IMG  │         │ Drawing IMG  │
└──────┬───────┘         └───────┬──────┘
       │                         │
       └────────────┬────────────┘
                    │
            ┌───────▼────────┐
            │  Stage 0:      │
            │  Preprocess    │
            │  (binarize,    │
            │   denoise)     │
            └────────┬───────┘
                     │ < 0.5 s
         ┌───────────▼────────────┐
         │ Stage 1: NCC Matching   │
         │ (multi-scale, ±10°,90°)│
         │ Candidate Proposal      │
         │                         │
         │ CPU: 30-60s | GPU: 15s  │
         └───────────┬────────────┘
                     │ ~200-400 candidates
         ┌───────────▼────────────┐
         │ Stage 2: DINOv2 Verify  │
         │ (cosine similarity)     │
         │ Zero-shot (no ft)       │
         │                         │
         │ GPU: 2-10s              │
         └───────────┬────────────┘
                     │ ~50-100 candidates
         ┌───────────▼───────────┐
         │ Stage 2b: Filters      │
         │ • Wire-leads           │
         │ • Chamfer distance     │
         │ • Structural checks    │
         │ • NMS                  │
         │ • Gap filter           │
         │                        │
         │ < 1 s                  │
         └───────────┬───────────┘
                     │ ~20-30 candidates
            [if use_vlm=True]
         ┌───────────▼───────────┐
         │ Stage 3: VLM Filter    │
         │ (Qwen2-VL-2B)          │
         │ Open-classification    │
         │ Borderline only        │
         │                        │
         │ ~0.4 s/crop            │
         └───────────┬───────────┘
                     │
            ┌────────▼─────────┐
            │ Output:          │
            │ BBoxes + Scores  │
            │ JSON             │
            └──────────────────┘
```

---

## 📦 Installation

### Requirements
- Python 3.11+
- CUDA 11.8+ (optional, for GPU acceleration)
- 4GB+ RAM (8GB+ recommended for VLM)

### Setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/pattern-detection-bom.git
cd pattern-detection-bom

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Model Weights** download automatically on first run:
- **DINOv2 ViT-S/14** (~86 MB) — via `torch.hub`
- **Qwen2-VL-2B** (~4.5 GB, optional) — via HuggingFace Hub (lazy-loaded when `use_vlm=True`)

---

## 🚀 Quick Start

### Python API

```python
from src.pipeline import PatternDetectionPipeline

# Initialize (GPU auto-detected)
pipe = PatternDetectionPipeline()

# Run detection
result = pipe.detect_auto(
    pattern_path="template.png",
    drawing_path="schematic.png",
    return_visualization=True
)

# Access results
print(f"Found {result['total_detections']} instances")
for det in result['detections']:
    bbox = det['bbox']
    print(f"  ({bbox['x']}, {bbox['y']}) {bbox['w']}×{bbox['h']} | "
          f"conf={det['confidence']:.3f} | dino={det['dino_score']:.3f}")
```

### Web UI (FastAPI)

```bash
# Start server
python app/web/server.py

# Open browser
# http://localhost:8000
```

**UI Features:**
- Drag-and-drop image upload
- Auto-tune mode + manual threshold sliders
- Real-time visualization with bounding boxes
- Download results (PNG + CSV)
- **About page** with system specification link
- **VLM toggle** for borderline confidence filtering

### Docker (HuggingFace Spaces)

```bash
docker build -t bom-detector .
docker run -p 7860:7860 bom-detector
```

---

## 📋 Output Format

```json
{
  "success": true,
  "total_detections": 3,
  "elapsed": 24.52,
  "detections": [
    {
      "bbox": {
        "x": 142,
        "y": 310,
        "w": 64,
        "h": 32
      },
      "confidence": 0.905,
      "ncc_score": 0.823,
      "dino_score": 0.897,
      "scale": 1.05,
      "angle": 0.0
    }
  ],
  "visualization": "base64_png_string"
}
```

---

## ⚙️ Configuration

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ncc_threshold` | `0.55` / `0.47` | NCC gate (strict/relaxed) |
| `cosine_threshold` | `0.84` | DINOv2 cosine similarity minimum |
| `final_nms_iou` | `0.40` | Final NMS IoU threshold |
| `use_vlm` | `False` | Enable Qwen2-VL stage |
| `vlm_keep_min_conf` | `0.75` | Skip VLM for confidence ≥ this |
| `vlm_reject_only` | `True` | Blacklist mode (recommended) |
| `vlm_recall_boost` | `auto` | Relax gates when VLM on |

### Python Config Example

```python
config = {
    "cosine_threshold": 0.84,
    "final_nms_iou": 0.40,
    "use_vlm": True,
    "vlm_symbol_name": "a zigzag resistor",
    "vlm_keep_min_conf": 0.78,
    "vlm_reject_only": True,
}

pipe = PatternDetectionPipeline(config=config)
result = pipe.detect_auto("pattern.png", "drawing.png")
```

---

## 🔍 Design Choices

### Why NCC + DINOv2?

- **NCC alone** is fast but brittle (sensitive to scale/noise)
- **DINOv2 dense sliding-window** is accurate but prohibitively slow
- **Hybrid approach** = fast proposal (NCC) + accurate verification (DINOv2)

### Why DINOv2 over CLIP?

- **CLIP image-to-image** offers no better FP separation than DINOv2 on line-art
- **CLIP text-guided** breaks zero-shot requirement (need to know class names)
- **DINOv2 self-supervised** generalizes well to technical drawings despite natural-image training

### Why VLM open-classification instead of yes/no?

- **Yes/no prompting** causes 100% agreement bias with small VLMs
- **Open-classification** forces the model to choose from a fixed vocabulary
- **Blacklist mode** (`vlm_reject_only=True`) avoids over-aggressive whitelisting

---

## 📚 Stages in Detail

### Stage 1: NCC Multi-scale Matching
- Scales: 0.70×–1.80× (adaptive per template)
- Angles: ±10° (standard) + 0°/90° (complex templates)
- NCC threshold: 0.55 (strict) / 0.47 (relaxed)
- Outputs: 30–400 candidates per drawing

### Stage 2: DINOv2 Zero-Shot Verification
- Model: `dinov2_vits14` (21M params, 384-D embeddings)
- Approach: cosine similarity on center-crop + full-crop (max)
- Derotation: 90° candidates derotated to horizontal for comparison
- Outputs: ~50–100 verified candidates

### Stage 2b: Structural Filters
- **Wire-leads**: Real components have visible leads (probe left/right/top/bottom)
- **Chamfer distance**: Edge alignment between template and crop (max 5.0)
- **Neighborhood complexity**: Component must sit in sparse region
- **Junction dots / Rect integrity**: Reject partial matches, L-corners
- **Confidence gap**: Drop low-confidence tail if bimodal distribution detected
- **Final NMS**: Containment-aware suppression (IoU threshold 0.4)

### Stage 3: VLM Semantic Filter (Optional)
- Model: `Qwen/Qwen2-VL-2B-Instruct` (~5GB, lazy-loaded)
- Classes: resistor, inductor, capacitor, diode, crystal, transistor, op-amp, logic-gate, wire-junction, other
- Mode: **Blacklist** (reject only known FPs, trust unknowns)
- Target: Borderline candidates only (confidence < 0.75)

---

## 📖 Documentation

**For detailed system specification, see:**
- [System Specification (HTML)](design_spec/system_spec.html) — complete architecture, algorithms, experimental results
- [Model Survey](design_spec/model_survey.md) — DINOv2 vs CLIP, VLM yes/no vs open-classification experiments

**Web UI About page** links to full specification (click **About** → **Mở đặc tả hệ thống**).

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_pipeline.py::test_detect_auto -v

# Test coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Results:** 10/10 passing ✓
- Simple template detection
- Complex template detection  
- Output format validation
- Config override tests
- VLM toggle tests

---

## 📁 Project Structure

```
pattern-detection-bom/
├── src/
│   ├── pipeline.py              # Main orchestrator
│   ├── preprocessor.py          # Image binarization, denoise
│   ├── ncc_matcher.py           # NCC multi-scale matching
│   ├── dino_verifier.py         # DINOv2 zero-shot verification
│   ├── postprocessor.py         # Structural filters, NMS
│   ├── vlm_verifier.py          # Qwen2-VL semantic filter (optional)
│   └── dino_dense_matcher.py    # DINOv2 dense sliding window (fallback)
│
├── app/web/
│   ├── server.py                # FastAPI server
│   ├── index.html               # SPA frontend
│   └── static/
│       ├── css/style.css        # UI styling
│       └── js/app.js            # Frontend logic (page nav, detection)
│
├── design_spec/
│   ├── system_spec.html         # Complete system documentation (13 sections)
│   ├── model_survey.md          # Model comparison experiments
│   └── design_decisions.md      # Architecture rationale
│
├── tests/
│   ├── test_pipeline.py         # 10 unit tests
│   └── conftest.py              # Pytest fixtures
│
├── Dockerfile                   # HuggingFace Spaces deployment
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 🚦 Performance

| Metric | Value |
|--------|-------|
| **Speed (GPU RTX 3060 12GB)** | 20–30 seconds per A3 drawing |
| **Speed (CPU i7-10700)** | 60–120 seconds per A3 drawing |
| **Memory (GPU)** | 4.5 GB (DINOv2) + 4.5 GB (optional VLM) |
| **Memory (CPU)** | 2 GB |
| **Recall (test drawing 4)** | 91.7% (22/24 GT boxes) |
| **False Positives** | 0 (after DINOv2 + structural filters) |

---

## ⚠️ Known Limitations

1. **Boundary resistors** — Symbols at image edges may be suppressed by NMS
2. **Resistor in framing box** — When a symbol sits inside a dense bounding frame, DINOv2 score collides with FP diodes (irreducible without VLM)
3. **Heavy rotation (> 15°)** — Only ±10° and 90° supported
4. **VLM labels** — Qwen2-VL-2B individually noisy at borderline (effective at population level)
5. **Single-draw processing** — No batch inference yet

---

## 🔧 Development

### Running with custom config

```python
from src.pipeline import PatternDetectionPipeline

pipe = PatternDetectionPipeline(config={
    "cosine_threshold": 0.88,  # stricter DINOv2
    "use_vlm": True,
    "vlm_symbol_name": "a resistor",
})
```

### Adding a new structural filter

Edit `src/postprocessor.py`, add method to `Postprocessor` class following the pattern of `filter_wire_leads()`.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sotatek AI/CV Assessment — Duy Hoang**  
Assessment Duration: 96 hours  
Submission: June 1, 2026

---

## 🙏 Acknowledgments

- **DINOv2** — Facebook AI ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193))
- **Qwen2-VL** — Alibaba Qwen Team
- **FastAPI** — Modern web framework for Python
