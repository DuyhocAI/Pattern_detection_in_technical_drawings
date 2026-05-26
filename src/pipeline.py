import time
import cv2
import numpy as np
from typing import Union, Optional

from .preprocessor import Preprocessor
from .ncc_matcher import NCCMatcher
from .dino_verifier import DINOVerifier
from .postprocessor import Postprocessor


class PatternDetectionPipeline:
    """Orchestrator combining all detection stages."""

    def __init__(self, config: dict = None):
        """Initialize pipeline with optional config overrides.

        Args:
            config: Optional dict with keys:
                scales, angles, ncc_threshold, nms_iou_threshold,
                dino_model, cosine_threshold, device, final_nms_iou
        """
        cfg = config or {}

        self.preprocessor = Preprocessor()
        # dilate_pattern > 0: thicken template strokes for style-mismatched drawings
        self.dilate_pattern = cfg.get("dilate_pattern", 0)

        self.ncc_matcher = NCCMatcher(
            scales=cfg.get("scales"),
            angles=cfg.get("angles"),
            ncc_threshold=cfg.get("ncc_threshold", 0.55),
            nms_iou_threshold=cfg.get("nms_iou_threshold", 0.3),
        )

        self.dino_verifier = DINOVerifier(
            model_name=cfg.get("dino_model", "dinov2_vits14"),
            device=cfg.get("device"),
            cosine_threshold=cfg.get("cosine_threshold", 0.84),
        )

        self.postprocessor = Postprocessor()
        self.final_nms_iou = cfg.get("final_nms_iou", 0.4)

        print(f"[Pipeline] Device: {self.dino_verifier.device}")
        print("[Pipeline] All stages initialized.")

    def detect_auto(
        self,
        pattern_input: Union[str, np.ndarray],
        drawing_input: Union[str, np.ndarray],
        return_visualization: bool = True,
    ) -> dict:
        """Auto-tuning detect: runs two NCC passes (strict + relaxed), merges all
        candidates, then verifies the merged set with a single DINOv2 pass.

        This ensures legend symbols AND main-circuit components are both found,
        even when they differ in scale or drawing style.

        Strategy:
          Pass 1 — strict (ncc=0.55, dilate=0): catches clean/legend copies
          Pass 2 — relaxed (ncc=0.28, dilate=5): catches style-mismatched + larger components

        Args:
            pattern_input: Pattern image path or numpy array.
            drawing_input: Drawing image path or numpy array.
            return_visualization: Whether to include annotated image.

        Returns:
            Detection result dict with all found instances.
        """
        try:
            t0 = time.time()

            pattern_data = self.preprocessor.preprocess(pattern_input)
            drawing_data = self.preprocessor.preprocess(drawing_input)
            pattern_proc = pattern_data["processed"]
            drawing_proc = drawing_data["processed"]
            t1 = time.time()
            print(f"[Pipeline] Auto-detect preprocess: {t1 - t0:.2f}s")

            # Detect whether the template is a "plain outline" shape (e.g. bare rectangle).
            # Criterion: low Canny edge density AND almost no dark pixels in the interior
            # of the symbol bounding box.  Complex symbols (bridge rectifier, fuse) have
            # internal strokes that push interior_fill above ~5%, so they are NOT classified
            # as simple and get the full standard pipeline.
            _edges = cv2.Canny(pattern_proc.astype(np.uint8), 50, 150)
            _edge_density = float(np.count_nonzero(_edges)) / _edges.size
            _dark = pattern_proc < 128
            _rows_any = np.any(_dark, axis=1)
            _cols_any = np.any(_dark, axis=0)
            _interior_fill = 0.0
            _tmpl_ar = 1.0
            if _rows_any.any() and _cols_any.any():
                _rmin, _rmax = np.where(_rows_any)[0][[0, -1]]
                _cmin, _cmax = np.where(_cols_any)[0][[0, -1]]
                _tmpl_h = _rmax - _rmin + 1
                _tmpl_w = _cmax - _cmin + 1
                _tmpl_ar = _tmpl_w / max(1, _tmpl_h)
                _mh = max(1, int(_tmpl_h * 0.20))
                _mw = max(1, int(_tmpl_w * 0.20))
                _inner = _dark[_rmin + _mh : _rmax - _mh, _cmin + _mw : _cmax - _mw]
                _interior_fill = float(np.sum(_inner)) / max(1, _inner.size)
            _is_simple = _edge_density < 0.05 and _interior_fill < 0.02
            print(
                f"[Pipeline] Template: edge={_edge_density:.4f} interior_fill={_interior_fill:.4f} "
                f"AR={_tmpl_ar:.2f} -> {'SIMPLE (outline only)' if _is_simple else 'complex'}"
            )

            # Pass 1: strict — undilated template, high NCC threshold
            self.ncc_matcher.ncc_threshold = 0.55
            candidates_strict = self.ncc_matcher.match(drawing_proc, pattern_proc)
            print(f"[Pipeline] Pass 1 (strict): {len(candidates_strict)} candidates")

            # Pass 2: relaxed — dilated template.
            # For simple-outline templates raise threshold: structural FPs (frame/table)
            # match poorly at non-native scales, while real components match at 0.50+.
            self.ncc_matcher.ncc_threshold = 0.50 if _is_simple else 0.28
            pattern_dilated = self.preprocessor.dilate_strokes(pattern_proc, kernel_size=5)
            candidates_relaxed = self.ncc_matcher.match(drawing_proc, pattern_dilated)
            print(f"[Pipeline] Pass 2 (relaxed): {len(candidates_relaxed)} candidates")

            all_candidates = candidates_strict + candidates_relaxed
            t2 = time.time()
            print(f"[Pipeline] NCC total: {t2 - t1:.2f}s — {len(all_candidates)} combined candidates")

            # DINOv2 on standard-scale candidates (skip if none found)
            verified = self.dino_verifier.verify_candidates(
                drawing_proc, pattern_proc, all_candidates
            ) if all_candidates else []
            t3 = time.time()
            print(f"[Pipeline] DINOv2 (standard): {t3 - t2:.2f}s — {len(verified)} verified")

            _saved_scales  = self.ncc_matcher.scales
            _saved_ncc     = self.ncc_matcher.ncc_threshold
            _saved_angles  = self.ncc_matcher.angles
            _saved_dino    = self.dino_verifier.cosine_threshold

            if _is_simple:
                # Simple (outline-only) templates: single combined micro pass.
                # Scales [0.45–1.05] cover circuit components typically 45–105% of the
                # legend symbol size; very small scales (< 0.45) generated too many FPs
                # on thin line segments.  0° + 90° rotation sweep catches vertical components.
                self.ncc_matcher.scales = [0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 1.0]
                self.ncc_matcher.ncc_threshold = 0.42
                self.ncc_matcher.angles = [-10, -5, 0, 5, 10, 80, 85, 90, 95, 100]
                cands_s = self.ncc_matcher.match(drawing_proc, pattern_proc)
                ncc_s_count = len(cands_s)
                # Chamfer pre-filter replaces DINOv2 for simple templates: DINOv2 is
                # unreliable for binary line-art and rejects correctly-detected vertical
                # components (R2/R4/R6/R8 score < 0.84 despite being real resistors).
                if cands_s:
                    # Strict chamfer pre-filter (≤3.0) — catches most components.
                    # Borderline range (3.0–3.7): run DINOv2 to rescue high-confidence
                    # real components whose drawing region has adjacent noise edges
                    # pushing their Chamfer score just above 3.0 (dino≥0.86 required).
                    _cands_with_ch = self.postprocessor.filter_chamfer_shape(
                        cands_s, drawing_proc, pattern_proc, max_chamfer=3.7
                    )
                    cands_strict     = [c for c in _cands_with_ch if c.get("chamfer_dist", 0) <= 3.0]
                    cands_borderline = [c for c in _cands_with_ch if c.get("chamfer_dist", 0) > 3.0]
                    if cands_borderline:
                        _bl_saved_thr = self.dino_verifier.cosine_threshold
                        self.dino_verifier.cosine_threshold = 0.0
                        bl_scored = self.dino_verifier.verify_candidates(
                            drawing_proc, pattern_proc, cands_borderline, derotate=False
                        )
                        self.dino_verifier.cosine_threshold = _bl_saved_thr
                        cands_borderline = [c for c in bl_scored if c.get("dino_score", 0) >= 0.88]
                    else:
                        cands_borderline = []
                    for c in cands_strict:
                        c.setdefault("dino_score", 0.0)
                        c.setdefault("confidence", c.get("ncc_score", 0.5))
                    for c in cands_borderline:
                        c.setdefault("confidence", (c.get("ncc_score", 0.5) + c.get("dino_score", 0)) / 2)
                    cands_s = cands_strict + cands_borderline
                before_s = len(cands_s)
                cands_s = self.postprocessor.filter_title_block(cands_s, drawing_proc)
                print(f"[Pipeline] Simple micro pass: {ncc_s_count} NCC -> {before_s} chamfer -> {len(cands_s)} (zone filter)")
                verified = verified + cands_s
            else:
                # Complex templates: separate near-0° (3a) and near-90° (3b) micro passes.
                # Standard passes 1+2 only sweep ±10° around 0°, so pass 3b is the only
                # path for 90°-rotated components (e.g. bridge rectifiers mounted vertically).
                # Scales cover [0.70–1.35] so rotated components at their natural drawing
                # size are found (pass 2 relaxed scales stop at 0.85 minimum).
                # Same relaxed NCC threshold as pass 2 — BR components score ~0.28–0.40
                # even at their correct scale/rotation, so 0.45 misses them entirely.
                _complex_scales = [0.70, 0.85, 1.0, 1.1, 1.2, 1.35]
                self.ncc_matcher.ncc_threshold = 0.28

                # Sub-pass 3a: near-0° at smaller/wider scales than standard pass
                self.ncc_matcher.scales = _complex_scales
                self.ncc_matcher.angles = [-10, -5, 0, 5, 10]
                self.dino_verifier.cosine_threshold = 0.82
                cands_3a = self.ncc_matcher.match(drawing_proc, pattern_proc)
                verified_3a = self.dino_verifier.verify_candidates(
                    drawing_proc, pattern_proc, cands_3a, derotate=True
                ) if cands_3a else []
                print(f"[Pipeline] Pass 3a (micro 0°): {len(cands_3a)} cands -> {len(verified_3a)} verified")

                # Sub-pass 3b: near-90° — same scale range, slightly stricter DINOv2
                self.ncc_matcher.scales = _complex_scales
                self.ncc_matcher.angles = [80, 85, 90, 95, 100]
                self.dino_verifier.cosine_threshold = 0.83
                cands_3b = self.ncc_matcher.match(drawing_proc, pattern_proc)
                verified_3b = self.dino_verifier.verify_candidates(
                    drawing_proc, pattern_proc, cands_3b, derotate=True
                ) if cands_3b else []
                print(f"[Pipeline] Pass 3b (micro 90°): {len(cands_3b)} cands -> {len(verified_3b)} verified")

                verified = verified + verified_3a + verified_3b

            self.ncc_matcher.scales  = _saved_scales
            self.ncc_matcher.ncc_threshold = _saved_ncc
            self.ncc_matcher.angles  = _saved_angles
            self.dino_verifier.cosine_threshold = _saved_dino
            t3 = time.time()
            print(f"[Pipeline] All passes: {t3 - t2:.2f}s — {len(verified)} total verified")

            # Simple-template post-filters: isolation + aspect-ratio + neighborhood.
            # Isolation: circuit components sit in white space; BOM/title-block cells have
            # solid grid lines directly adjacent to their long sides.
            # Aspect ratio: keep candidates whose bbox AR is within 2× of the template AR
            # *or* its reciprocal — the reciprocal check allows 90°-rotated components
            # (e.g. vertical resistors) whose bbox AR is ~1/_tmpl_ar.
            # Neighborhood complexity: reject candidates whose surrounding ring has too
            # many Canny edges — this eliminates false positives inside complex symbols
            # (e.g. bridge-rectifier bodies) which have many adjacent edges.
            # Notes-area exemption: the bottom 20% of the drawing contains notes/legend
            # symbols which have annotation text nearby — skip isolation and neighborhood
            # checks for those so legitimate legend symbols are not filtered out.
            # Top-margin exclusion: discard detections whose top edge is within the outer
            # coordinate-margin strip (border grid cells look like plain rectangles).
            if _is_simple and verified:
                before = len(verified)
                _drw_h, _drw_w = drawing_proc.shape[:2]
                _notes_y = int(_drw_h * 0.80)   # below this → notes/legend area
                _top_margin = max(30, int(_drw_h * 0.04))  # top coordinate strip

                # Reject border-grid cells in the top margin
                verified = [c for c in verified if c["y"] >= _top_margin]

                # Split into circuit area and notes/legend area
                _circuit = [c for c in verified if c["y"] < _notes_y]
                _notes   = [c for c in verified if c["y"] >= _notes_y]

                # Isolation: reject candidates adjacent to grid lines (circuit area only)
                _circuit = self.postprocessor.filter_isolated(_circuit, drawing_proc)

                # Aspect ratio: accept both normal (horizontal) and 90°-rotated (vertical) AR
                _ar_inv = 1.0 / max(0.01, _tmpl_ar)
                def _ar_ok(c):
                    ar = c["w"] / max(1, c["h"])
                    return (
                        (_tmpl_ar / 2.0 <= ar <= _tmpl_ar * 2.0)
                        or (_ar_inv / 2.0 <= ar <= _ar_inv * 2.0)
                    )
                _circuit = [c for c in _circuit if _ar_ok(c)]
                _notes   = [c for c in _notes   if _ar_ok(c)]

                # Wire-lead check: real resistors have straight wire leads on both
                # connecting sides; false positives embedded in complex symbols do not.
                # Circuit area: require one strong lead (>=6px) + one non-zero lead (>=1px)
                # — handles resistors connected directly to a power rail where only
                # 1-2px of wire is visible on the rail-side.
                # Notes/legend area: skip wire-lead filter (legend symbols may have no
                # protruding leads). Restrict to left half of drawing (legends are always
                # on the left; BOM/title-block cells are on the right). Keep top-2 by score.
                _circuit = self.postprocessor.filter_wire_leads(_circuit, drawing_proc)
                _circuit = self.postprocessor.filter_wire_passthrough(_circuit, drawing_proc)
                _circuit = self.postprocessor.filter_neighborhood_complexity(
                    _circuit, drawing_proc, expand_ratio=0.5, max_edge_density=0.022
                )
                _circuit = self.postprocessor.filter_junction_dots(_circuit, drawing_proc)
                _circuit = self.postprocessor.filter_rect_integrity(_circuit, drawing_proc)
                # DINOv2-verified borderline candidates (dino ≥ 0.86) bypass the
                # final Chamfer check — they've already been semantically confirmed.
                _dino_ok   = [c for c in _circuit if c.get("dino_score", 0) >= 0.88]
                _needs_ch  = [c for c in _circuit if c.get("dino_score", 0) < 0.86]
                _needs_ch  = self.postprocessor.filter_chamfer_shape(_needs_ch, drawing_proc, pattern_proc)
                _circuit   = _needs_ch + _dino_ok
                _bottom_margin = max(30, int(_drw_h * 0.04))
                _notes = [c for c in _notes
                          if c["y"] + c["h"] <= _drw_h - _bottom_margin]
                _notes = self.postprocessor.filter_rect_borders(_notes, drawing_proc)
                _notes = sorted(_notes, key=lambda c: c.get("dino_score", 0), reverse=True)[:1]

                verified = _circuit + _notes
                print(
                    f"[Pipeline] Simple-template filters: {before} -> {len(verified)} "
                    f"(top-margin + isolation + AR + wire-leads + passthrough | notes_kept={len(_notes)})"
                )

            # Title-block zone filter: remove candidates inside the BOM / right-frame
            # area for all templates (complex templates skip the simple-filter block
            # that previously applied this only to simple micro-pass candidates).
            before_tb = len(verified)
            verified = self.postprocessor.filter_title_block(verified, drawing_proc)
            if len(verified) != before_tb:
                print(f"[Pipeline] Title-block filter: {before_tb} -> {len(verified)}")

            # Final NMS + format
            # Simple templates: use tight IoU (0.25) to merge offset-position duplicates
            # of the same component, and keep the best-fit bbox (not union) to avoid
            # over-expanding the box when a lower-confidence offset duplicate gets merged in.
            # Complex templates: use default IoU (0.40) with union bbox expansion.
            if _is_simple:
                verified = self.postprocessor.final_nms(
                    verified, iou_threshold=0.25, use_union_bbox=False
                )
            else:
                verified = self.postprocessor.final_nms(
                    verified, iou_threshold=self.final_nms_iou
                )
            result = self.postprocessor.format_output(verified, drawing_proc.shape)
            t4 = time.time()
            print(f"[Pipeline] Auto-detect total: {t4 - t0:.2f}s — {result['total_detections']} detections")

            if return_visualization:
                # Draw on original (non-binarized) image for clearer output
                result["visualization"] = self.postprocessor.draw_boxes(
                    drawing_data["original"], result["detections"]
                )

            return result

        except Exception as e:
            raise RuntimeError(f"Pipeline auto-detect failed: {e}") from e

    def detect(
        self,
        pattern_input: Union[str, np.ndarray],
        drawing_input: Union[str, np.ndarray],
        return_visualization: bool = True,
    ) -> dict:
        """Run full detection pipeline.

        Args:
            pattern_input: Pattern image path or numpy array.
            drawing_input: Drawing image path or numpy array.
            return_visualization: Whether to include annotated image in output.

        Returns:
            Dict from Postprocessor.format_output(), plus optional "visualization" key.

        Raises:
            RuntimeError: If any stage fails.
        """
        try:
            t0 = time.time()

            # Stage 0: Preprocess
            pattern_data = self.preprocessor.preprocess(pattern_input)
            drawing_data = self.preprocessor.preprocess(drawing_input)
            t1 = time.time()
            print(f"[Pipeline] Stage 0 (Preprocess): {t1 - t0:.2f}s")

            pattern_proc = pattern_data["processed"]   # original — used for DINOv2
            drawing_proc = drawing_data["processed"]

            # Optionally dilate pattern strokes for NCC to handle style mismatch
            if self.dilate_pattern > 0:
                pattern_for_ncc = self.preprocessor.dilate_strokes(
                    pattern_proc, kernel_size=self.dilate_pattern
                )
            else:
                pattern_for_ncc = pattern_proc

            # Stage 1: NCC matching (uses dilated pattern if configured)
            candidates = self.ncc_matcher.match(drawing_proc, pattern_for_ncc)
            t2 = time.time()
            print(f"[Pipeline] Stage 1 (NCC): {t2 - t1:.2f}s — {len(candidates)} candidates")

            if not candidates:
                print("[Pipeline] No candidates from Stage 1, returning empty result.")
                result = self.postprocessor.format_output([], drawing_proc.shape)
                if return_visualization:
                    result["visualization"] = self.postprocessor.draw_boxes(
                        drawing_data["original"], []
                    )
                return result

            # Stage 2: DINOv2 verification
            candidates = self.dino_verifier.verify_candidates(drawing_proc, pattern_proc, candidates)
            t3 = time.time()
            print(f"[Pipeline] Stage 2 (DINOv2): {t3 - t2:.2f}s — {len(candidates)} verified")

            # Stage 3: Final NMS + format
            candidates = self.postprocessor.final_nms(candidates, iou_threshold=self.final_nms_iou)
            result = self.postprocessor.format_output(candidates, drawing_proc.shape)
            t4 = time.time()
            print(f"[Pipeline] Stage 3 (Post): {t4 - t3:.2f}s")
            print(f"[Pipeline] Total: {t4 - t0:.2f}s — {result['total_detections']} detections")

            if return_visualization:
                # Draw on original (non-binarized) image for clearer output
                result["visualization"] = self.postprocessor.draw_boxes(
                    drawing_data["original"], result["detections"]
                )

            return result

        except Exception as e:
            raise RuntimeError(f"Pipeline detection failed: {e}") from e

    def update_thresholds(
        self,
        ncc_threshold: Optional[float] = None,
        cosine_threshold: Optional[float] = None,
        final_nms_iou: Optional[float] = None,
    ):
        """Update detection thresholds at runtime (for UI sliders).

        Args:
            ncc_threshold: New NCC threshold for Stage 1.
            cosine_threshold: New cosine similarity threshold for Stage 2.
        """
        if ncc_threshold is not None:
            self.ncc_matcher.ncc_threshold = ncc_threshold
        if cosine_threshold is not None:
            self.dino_verifier.cosine_threshold = cosine_threshold
        if final_nms_iou is not None:
            self.final_nms_iou = final_nms_iou


def run_detection(pattern_path: str, drawing_path: str, auto: bool = True, **kwargs) -> dict:
    """Convenience function: create pipeline, run detection, return result.

    Args:
        pattern_path: Path to pattern image.
        drawing_path: Path to drawing image.
        auto: If True (default), use detect_auto() which self-tunes thresholds.
        **kwargs: Config overrides passed to PatternDetectionPipeline.

    Returns:
        Detection result dict.
    """
    pipeline = PatternDetectionPipeline(config=kwargs if kwargs else None)
    if auto:
        return pipeline.detect_auto(pattern_path, drawing_path)
    return pipeline.detect(pattern_path, drawing_path)
