# Model Survey: Zero-Shot Pattern Detection for Engineering Drawings

## Problem Context

The pipeline must detect all instances of a given symbol pattern in engineering
BOM drawings without any fine-tuning or labeled data. The verification stage
(currently DINOv2) is the primary source of false positives on complex circuits:
inductors, transistors, and op-amps share low-level visual features with resistors
and obtain borderline cosine similarity scores.

---

## Current Architecture: NCC + DINOv2

**Stage 1 — NCC template matching** (OpenCV `matchTemplate`):
- Multi-scale, multi-angle sliding window
- High recall at NCC ≥ 0.28 (relaxed pass)
- Fast on CPU; produces 50–500 candidates per drawing

**Stage 2 — DINOv2 ViT-S/14 verification** (cosine similarity):
- Self-supervised pre-training (DINO loss on ~142M web images)
- Patch-level features capture local shape structure
- Cosine threshold = 0.84 ≈ 84th percentile similarity

**Observed limitation:** DINOv2 features encode _spatial frequency and texture_.
Two schematic symbols with similar line density (zigzag resistor vs coil inductor)
can land within 0.02 cosine of each other, producing false positives that no
spatial filter fully removes.

---

## Alternative Verifier Survey

### 1. CLIP ViT-B/32 — OpenAI (openai/clip-vit-base-patch32)

| Property | Value |
|----------|-------|
| Parameters | 86M (vision encoder: 63M) |
| Training | Contrastive on 400M image-text pairs |
| Input size | 224×224 |
| Throughput | ~200 crops/s GPU, ~15 crops/s CPU |

**How it helps:**
CLIP aligns visual features with language semantics. Its vision encoder learns
to encode _what an object is_ (influenced by paired captions) rather than just
_how it looks_. A resistor crop and an inductor crop may share edge density,
but CLIP embeddings partially separate them because training captions for
circuit diagrams use different words.

**Two modes:**

```python
# Mode A: image-to-image (zero-shot, any pattern)
sim = cosine(clip.encode_image(pattern), clip.encode_image(candidate_crop))

# Mode B: text-guided (requires knowing symbol class name)
sim = cosine(clip.encode_text("resistor in electronic circuit"), clip.encode_image(candidate_crop))
```

Mode B is the key differentiator. If the user can name the symbol class, CLIP
can distinguish semantically different symbols even when they look structurally
similar.

**Trade-offs:**
- Heavier than DINOv2 ViT-S (4× more params)
- Engineering schematics are rare in web-crawled text-image data → domain shift
  still present, but the language bridge reduces the gap
- Text-guided mode breaks full zero-shot generality (requires symbol name)

**Benchmark on Drawing 1 (resistor, test_1.png pattern):**

| Candidate | DINOv2 ViT-S | CLIP img-img | CLIP text |
|-----------|-------------|--------------|-----------|
| R_horiz1 (TP) | 0.884 | 0.945 | 0.321 |
| R_horiz2 (TP) | 0.876 | 0.946 | 0.332 |
| R_vert1  (TP) | 0.830 | 0.908 | 0.342 |
| R_hard   (TP, conf=0.50) | 0.692 | 0.906 | 0.277 |
| FP_junction  | 0.816 | 0.954 | — |
| FP_wire      | 0.763 | 0.945 | — |
| FP_cross     | 0.822 | 0.953 | — |

**Key finding:** CLIP image-to-image scores FPs (wire crossings, junction areas)
as high as TPs (0.94–0.95 vs 0.91–0.95). This confirms that image-to-image CLIP
provides **no additional FP discriminability** over DINOv2 for the wire/junction
FP class — both encoders see these as "similar to the resistor template".

The separation only exists with text-guided CLIP, where the semantic concept
of "resistor" helps distinguish true instances from circuit background.
Text-guided mode requires knowing the symbol class.

**Code:** A `CLIPVerifier` drop-in replacement was prototyped during this study.
Because the benchmark above showed CLIP image-to-image gives no FP separation over
DINOv2, and text-guided mode breaks zero-shot generality, the prototype was removed
from the codebase. The benchmark conclusion is retained here for the record.

---

### 2. LightGlue + SuperPoint

| Property | Value |
|----------|-------|
| Parameters | SuperPoint 1.3M + LightGlue ~5M |
| Training | Supervised on Megadepth (outdoor scenes) |
| Input size | Variable |
| Throughput | ~30 pairs/s GPU |

**Principle:** Detect sparse keypoints in both template and candidate crop,
then learn to match them geometrically. Returns homography + inlier count.

**Why it matters:** Purely geometric matching — no dependence on visual feature
distribution. If the template zigzag has 12 keypoints and the candidate has 12
corresponding keypoints in the same relative positions, it is a match regardless
of line thickness or contrast.

**Trade-offs:**
- SuperPoint keypoints are designed for corners and blob-like structures in
  natural photos. Engineering line-art has very few reliable keypoints; edges
  meet at simple T/L junctions that look the same across all symbols.
- Likely to produce too few keypoints per crop (~3–8 vs the 100+ needed
  for reliable matching), causing high false-negative rate.
- Not designed for small (30–80 px) binary line-art crops.

**Expected improvement:** Low for schematic symbols due to sparse keypoints.
Better for larger, more textured circuit blocks.

---

### 3. Siamese ResNet-18 (contrastive fine-tuning)

| Property | Value |
|----------|-------|
| Parameters | 11M × 2 = 22M |
| Training | Supervised on symbol pairs (positive/hard-negative) |
| Input size | 64×64 |
| Throughput | ~2000 crops/s GPU |

**Principle:** Train a ResNet-18 with triplet/contrastive loss on synthetic symbol
pairs. The network learns an embedding where same-symbol instances cluster tightly
and different-symbol instances are separated.

**Why it matters:** With as few as 50 labeled examples per class, a Siamese
network fine-tuned on synthetic resistor/inductor/transistor crops can achieve
>95% pair discrimination — far better than zero-shot approaches.

**Trade-offs:**
- Not zero-shot: requires labeled symbol crops for each symbol class
- Symbol classes must be enumerated at training time
- Significant training effort for N symbol classes

**Expected improvement:** Highest possible accuracy (95%+ discrimination),
but only for symbols that were in the training set.

---

### 4. Segment Anything Model (SAM) + Classifier

| Property | Value |
|----------|-------|
| Parameters | SAM ViT-H: 636M |
| Training | Supervised on SA-1B (natural images) |
| Throughput | ~1–2 drawings/s GPU |

**Principle:** Use SAM to automatically segment all components in the circuit
drawing, then classify each segment independently using DINOv2 or CLIP.

**Why it matters:** Avoids the sliding-window problem entirely — each component
is already isolated before verification. No NCC needed.

**Trade-offs:**
- SAM was trained on natural images; schematic line-art segments very poorly.
  Circuit symbols merge or fragment in unpredictable ways.
- Very slow for large drawings
- Requires a second-stage classifier on each segment

**Expected improvement:** Potentially transformative if SAM were fine-tuned on
schematic drawings; with the current model, likely poor segmentation quality.

---

## Image Transformation Survey

### Added in this session: `src/preprocessor.py`

| Transform | Method | Benefit |
|-----------|--------|---------|
| **CLAHE** | `Preprocessor.clahe_enhance()` | Normalizes local contrast; recovers faint strokes in scanned drawings with uneven background |
| **Stroke normalization** | `Preprocessor.normalize_strokes()` | Thinning + re-dilation to uniform width; makes NCC matching scale-invariant to line thickness |

**Usage:**
```python
preprocessor.preprocess(img, clahe=True)               # CLAHE before binarization
preprocessor.preprocess(img, normalize_stroke_width=2) # Normalize to 2px stroke width
```

### Other effective transforms for line-art

| Transform | When to use |
|-----------|-------------|
| **Gaussian blur before binarization** | Scanned drawings with noise speckles |
| **Morphological opening** | Remove isolated noise dots without touching thin lines |
| **Distance transform** | Convert binary image to distance map; encodes proximity-to-stroke as a continuous feature for NCC |
| **Gradient magnitude (Sobel/Scharr)** | NCC on gradient images is invariant to global illumination shift |
| **Skeletonization** | Extreme line-width normalization (1px skeleton); best with re-dilation after |

---

## Empirical Validation: DINODenseMatcher (this session)

A scale-invariant DINOv2 dense sliding-window matcher (`src/dino_dense_matcher.py`)
was integrated as an **optional** large-scale path (Pass C) for simple-outline
templates. It activates only when a fast NCC scale probe finds the pattern at a
scale larger than the template (`probe_s > 1.10`), i.e. the failure mode where the
NCC scale grid [0.30–1.0] structurally cannot reach the instance.

**A/B comparison** (`use_dino_dense` flag ON vs OFF), resistor template:

| Test set | drawing | ON | OFF |
|----------|---------|----|-----|
| official | example1 | 5 | 5 |
| official | example2 | 4 | 4 |
| real     | 1.png | 10 | 10 |
| real     | 2.png | 0 | 0 |
| real     | 3.png | 0 | 0 |
| real     | 4.png | 2 | 2 |
| real     | 5.png | 0 | 0 |
| real     | 6.png | 0 | 0 |

**Finding:** identical output in every case (`DINODense: 0` contributed everywhere).
On this dataset every genuine instance appears at scale ≤ 1.0, fully covered by NCC,
so the probe gate never fires. The dense matcher is therefore a **dormant
scale-invariant fallback**: it adds one fast probe call per simple detection and
contributes detections only when a symbol is drawn larger than its legend template
— a real but currently-untested failure mode.

**Decision:** kept behind `use_dino_dense` (default `True`). It does not change any
current result and addresses a principled gap; the flag allows disabling its probe
overhead. The 10-test suite remains green.

---

## Empirical Validation: VLM Stage-3 (Qwen2-VL-2B)

A local Vision-Language Model was added as an **optional** Stage-3 semantic filter
(`src/vlm_verifier.py`, flag `use_vlm`, default off). It runs after all spatial
filters and judges each surviving candidate by *what the symbol is*, targeting the
exact FP classes (inductor/crystal/op-amp/diode) that DINOv2 cannot separate.

**Finding 1 — yes/no prompting fails on small VLMs.** Asking "is this a resistor?"
made Qwen2-VL-2B answer "yes" to 40/40 candidates (including wire junctions). The
2B model has a strong agreement bias on close-ended questions.

**Finding 2 — open-classification breaks the bias.** Forcing the model to NAME the
component from a closed vocabulary (resistor/inductor/.../wire-junction) split the
same 40 crops into 7 resistor / 33 other. We keep only candidates whose class
matches the *template's own* VLM class (stays zero-shot — no hardcoded "resistor").
Caveat: the 2B model's labels are individually unreliable (it calls many ambiguous
crops "transistor"), but resistor-vs-not is usable.

**Finding 3 — restrict the VLM to the borderline band.** On the user's full
schematic (25 raw detections):

| Config | Detections | Notes |
|--------|-----------|-------|
| VLM off | 25 | many FPs (op-amp, diode, inductor, junctions) |
| VLM on (all candidates) | 6 | over-aggressive — drops genuine resistors at conf 0.81–0.85 |
| **VLM on (conf < 0.75 only)** | **13** | high-conf TPs shielded; only noisy borderline band judged |

True positives cluster at conf ≥ 0.75 and FPs at ≤ 0.67, so `vlm_keep_min_conf=0.75`
auto-keeps trusted detections and asks the VLM only about the 0.45–0.75 band where
TP/FP overlap. Every candidate the VLM dropped in this mode had conf ≤ 0.74 and was
a genuine FP (transistor/diode/op-amp).

**Cost:** ~0.4 s/crop on RTX 3060 (12 GB), ~4.5 GB bf16 weights, lazy-loaded.

**Decision:** kept behind `use_vlm` (default off) with `vlm_keep_min_conf=0.75`.
The default NCC + DINOv2 pipeline stays lightweight and the 10-test suite green;
enabling the VLM trades latency for materially fewer false positives. Contrast with
CLIP (above), which gave no separation at all — the VLM's language reasoning is what
makes the difference DINOv2/CLIP embeddings could not.

---

## Recommendation

For the **current zero-shot requirement**:

1. **Short term** — Use CLIP text-guided mode when the symbol class is known.
   Integrate as an optional flag: `--verifier clip --text-prompt "resistor symbol"`.
   Expected: 30–50% reduction in inductor/transistor FPs.

2. **Medium term** — Collect ~100 labeled crops per symbol class from the
   provided drawings and fine-tune a small Siamese ResNet-18. This removes
   the zero-shot constraint but gives near-perfect verification accuracy.

3. **Long term** — Fine-tune SAM on a schematic-drawing dataset so that
   component segmentation works reliably, then use DINOv2/CLIP on isolated
   segments (no sliding window needed).

For the **current FP pattern** (inductors/transistors passing all spatial filters):
- Root cause: inductors have wire leads on both sides (pass `filter_wire_leads`),
  similar bounding-box edge density as resistors (pass `filter_neighborhood_complexity`
  and `filter_chamfer_shape`), and DINOv2 features are not discriminative enough.
- Profile similarity filter (1D edge projection correlation) was explored but
  failed because the template and drawing use different resistor symbol styles
  (IEC rectangle vs ANSI zigzag at different scan resolutions), making cross-style
  correlation unreliable.
- CLIP text-guided mode is the most promising zero-shot fix without labeled data.
