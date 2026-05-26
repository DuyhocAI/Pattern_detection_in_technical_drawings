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
                    cands_s = self.postprocessor.filter_chamfer_shape(
                        cands_s, drawing_proc, pattern_proc, max_chamfer=3.0
                    )
                    for c in cands_s:
                        c.setdefault("dino_score", 0.0)
                        c.setdefault("confidence", c.get("ncc_score", 0.5))
                before_s = len(cands_s)
                cands_s = self.postprocessor.filter_title_block(cands_s, drawing_proc)
                print(f"[Pipeline] Simple micro pass: {ncc_s_count} NCC -> {before_s} chamfer -> {len(cands_s)} (zone filter)")
                verified = verified + cands_s
            else:
                # Complex templates: separate near-0° (3a) and near-90° (3b) micro passes.
                # Standard passes 1+2 only sweep ±10° around 0°, so pass 3b is the only
                # path for 90°-rotated components (e.g. bridge rectifiers mounted vertically).
                #
                # When passes 1+2 found zero NCC candidates, the template's natural matching
                # scale lies outside [0.85-1.15].  A quick scale probe (13 scales across
                # [0.25–2.5]) finds the best-NCC scale, then the micro passes sweep a ±40%
                # window around it.  This avoids the previous approach of sweeping all scales
                # from 0.30 to 1.35, which produced thousands of FPs at small scales for
                # large-scale templates (e.g. zigzag resistors at scale ~1.5).
                #
                # When passes 1+2 already found candidates, the standard scale range suffices
                # (bridge rectifiers at 70–135% template size).
                _no_std_candidates = len(all_candidates) == 0
                _is_gate_like = 0.25 <= _tmpl_ar <= 2.5
                # Filled-elongated: non-gate-like template whose interior is largely filled
                # by strokes (e.g. ANSI zigzag resistor: AR~4.8, fill~0.33).
                # Distinguishes "filled" patterns (zigzag, inductor coils) from "hollow"
                # patterns (IEC rectangle: AR~5.3, fill~0.17) that look similar in AR but
                # have a mostly-empty interior.  Only filled-elongated templates need the
                # Chamfer + size filter and the skip-3a optimization.
                _is_filled_elongated = not _is_gate_like and _interior_fill > 0.20
                # Use adaptive scale probe when:
                # - passes 1+2 found nothing (template at unusual scale), OR
                # - template is filled-elongated (may have FPs at wrong scales from std passes)
                _use_adaptive = _no_std_candidates or _is_filled_elongated
                if _use_adaptive:
                    # Quick scale probe: find natural matching scale
                    _probe_scales = [0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.85, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
                    _best_probe_s, _best_probe_ncc = 1.0, 0.0
                    _ph, _pw = pattern_proc.shape[:2]
                    _drwH, _drwW = drawing_proc.shape[:2]
                    for _ps in _probe_scales:
                        _ptw = int(_pw * _ps); _pth = int(_ph * _ps)
                        if _ptw < 10 or _pth < 10 or _drwH < _pth or _drwW < _ptw:
                            continue
                        _pt_s = cv2.resize(pattern_proc, (_ptw, _pth), interpolation=cv2.INTER_AREA)
                        _pres = cv2.matchTemplate(drawing_proc, _pt_s, cv2.TM_CCOEFF_NORMED)
                        _, _pncc, _, _ = cv2.minMaxLoc(_pres)
                        if _pncc > _best_probe_ncc:
                            _best_probe_ncc = _pncc
                            _best_probe_s = _ps
                    print(f"[Pipeline] Scale probe: best={_best_probe_s:.2f} ncc={_best_probe_ncc:.3f}")
                    # Adaptive scale range: ±40% around the best probe scale
                    _fracs = [0.75, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.25, 1.40]
                    _complex_scales = sorted(set(
                        round(_best_probe_s * f, 2)
                        for f in _fracs if 0.20 <= _best_probe_s * f <= 3.0
                    ))
                else:
                    _complex_scales = [0.70, 0.85, 1.0, 1.1, 1.2, 1.35]
                    _best_probe_s = 1.0
                self.ncc_matcher.ncc_threshold = 0.28

                # Sub-pass 3a: near-0° at smaller/wider scales than standard pass.
                # Skip for non-gate-like templates when standard passes already found
                # candidates: the default scale list [0.85-2.0] already covers 0°
                # rotations at all relevant scales, and adding new intermediate scales
                # only introduces extra FPs from non-resistor components.
                # Skip pass 3a only for filled-elongated templates when standard passes
                # already found candidates: the default scale list [0.85-2.0] covers 0°
                # at all relevant scales, so new intermediate scales just add FPs.
                # Hollow-elongated (IEC) and gate-like templates still run pass 3a.
                _skip_3a = _is_filled_elongated and not _no_std_candidates
                if not _skip_3a:
                    self.ncc_matcher.scales = _complex_scales
                    self.ncc_matcher.angles = [-10, -5, 0, 5, 10]
                    self.dino_verifier.cosine_threshold = 0.82
                    cands_3a = self.ncc_matcher.match(drawing_proc, pattern_proc)
                    verified_3a = self.dino_verifier.verify_candidates(
                        drawing_proc, pattern_proc, cands_3a, derotate=True
                    ) if cands_3a else []
                    print(f"[Pipeline] Pass 3a (micro 0°): {len(cands_3a)} cands -> {len(verified_3a)} verified")
                else:
                    verified_3a = []
                    print("[Pipeline] Pass 3a skipped (filled-elongated + std candidates found)")

                # Sub-pass 3b: near-90° — same scale range, slightly stricter DINOv2
                self.ncc_matcher.scales = _complex_scales
                self.ncc_matcher.angles = [80, 85, 90, 95, 100]
                self.dino_verifier.cosine_threshold = 0.83
                cands_3b = self.ncc_matcher.match(drawing_proc, pattern_proc)
                verified_3b = self.dino_verifier.verify_candidates(
                    drawing_proc, pattern_proc, cands_3b, derotate=True
                ) if cands_3b else []
                print(f"[Pipeline] Pass 3b (micro 90°): {len(cands_3b)} cands -> {len(verified_3b)} verified")

                all_complex = verified + verified_3a + verified_3b

                # Post-filters for complex micro-pass results
                if _is_gate_like:
                    # Gate-like templates (AR 0.25-2.5): output-bubble discrimination.
                    # Only applied when the adaptive scale path was used (template at
                    # unusual scale, e.g. XOR gates much smaller than legend copy).
                    # Bridge rectifiers use standard scales and skip this filter.
                    if _no_std_candidates:
                        before_bubble = len(all_complex)
                        all_complex = self.postprocessor.filter_output_bubble(
                            all_complex, drawing_proc, pattern_proc
                        )
                        if len(all_complex) != before_bubble:
                            print(f"[Pipeline] Bubble filter: {before_bubble} -> {len(all_complex)}")
                elif _is_filled_elongated:
                    # Filled-elongated templates (high interior fill, AR > 2.5): Chamfer + AR.
                    # Standard passes produce many FP candidates for these shapes
                    # (e.g. inductors matching zigzag NCC at 0.28); Chamfer distance
                    # discriminates by edge structure (angular zigzag vs curved coils).
                    # Hollow-elongated templates (IEC rectangles, low fill) skip this
                    # block and rely on DINOv2 alone.
                    before_cf = len(all_complex)
                    all_complex = self.postprocessor.filter_chamfer_shape(
                        all_complex, drawing_proc, pattern_proc, max_chamfer=3.0
                    )
                    print(f"[Pipeline] Chamfer filter: {before_cf} -> {len(all_complex)}")
                    # AR filter: keep horizontal and vertical (90°-rotated) orientations
                    _full_pat_ar = pattern_proc.shape[1] / max(1, pattern_proc.shape[0])
                    _ar_inv = 1.0 / max(0.01, _full_pat_ar)
                    ar_tol = 0.60
                    before_ar = len(all_complex)
                    all_complex = [
                        c for c in all_complex
                        if (_full_pat_ar * (1 - ar_tol) <= c["w"] / max(1, c["h"]) <= _full_pat_ar * (1 + ar_tol))
                        or (_ar_inv * (1 - ar_tol) <= c["w"] / max(1, c["h"]) <= _ar_inv * (1 + ar_tol))
                    ]
                    if len(all_complex) != before_ar:
                        print(f"[Pipeline] AR filter: {before_ar} -> {len(all_complex)}")
                    # Size filter: reject candidates at wrong scale.
                    # When scale probe found the natural matching scale, reject candidates
                    # whose longest side is < 65% of the expected long side at probe scale.
                    # This removes matches found at too-small NCC scales (e.g. scale 0.85
                    # when the true scale is 1.50), which pass NCC+DINOv2 but are FPs.
                    if _best_probe_s > 1.0:
                        _min_long = int(max(_pw, _ph) * _best_probe_s * 0.75)
                        before_sz = len(all_complex)
                        all_complex = [
                            c for c in all_complex
                            if max(c["w"], c["h"]) >= _min_long
                        ]
                        if len(all_complex) != before_sz:
                            print(f"[Pipeline] Size filter: {before_sz} -> {len(all_complex)} (min_long={_min_long})")
                verified = all_complex

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
                _circuit = self.postprocessor.filter_chamfer_shape(_circuit, drawing_proc, pattern_proc)
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
