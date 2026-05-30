import time
import cv2
import numpy as np
from typing import Union, Optional

from .preprocessor import Preprocessor
from .ncc_matcher import NCCMatcher
from .dino_verifier import DINOVerifier
from .dino_dense_matcher import DINODenseMatcher
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

        # DINODenseMatcher: scale-invariant DINO sliding window. Used as an
        # OPTIONAL large-scale path for simple templates (Pass C in detect_auto),
        # complementing NCC which only covers scales 0.30-1.0. Toggle via config
        # `use_dino_dense` (default True — preserves the validated 10/10 behaviour).
        self.use_dino_dense = cfg.get("use_dino_dense", True)
        self.dino_dense = DINODenseMatcher(
            dino_verifier=self.dino_verifier,
            sim_threshold=cfg.get("dense_sim_threshold", 0.78),
            stride_ratio=0.40,
            batch_size=32,
        )

        # Stage 3 (optional): Vision-Language Model semantic filter.
        # Rejects false positives that survive NCC + DINOv2 because they share
        # low-level structure with the target symbol (inductor/crystal/op-amp vs
        # resistor). Uses open-classification (not yes/no) to avoid small-VLM
        # agreement bias. Lazy-loaded: the ~4.5 GB model is only fetched on first
        # use, so use_vlm=False keeps the pipeline lightweight (and tests green).
        self.use_vlm = cfg.get("use_vlm", False)
        self.vlm_model_name = cfg.get("vlm_model", "Qwen/Qwen2-VL-2B-Instruct")
        self.vlm_symbol_name = cfg.get("vlm_symbol_name")  # optional class hint
        # Only borderline candidates are sent to the VLM. High-confidence detections
        # (>= this) are trusted and kept WITHOUT asking — the 2B model mislabels some
        # genuine high-conf resistors as "transistor", so shielding them avoids
        # false rejections while still letting the VLM prune the noisy borderline band.
        self.vlm_keep_min_conf = cfg.get("vlm_keep_min_conf", 0.75)
        self._vlm = None  # lazy VLMVerifier instance

        print(f"[Pipeline] Device: {self.dino_verifier.device}")
        print(f"[Pipeline] VLM Stage-3: {'ENABLED' if self.use_vlm else 'disabled'}")
        print("[Pipeline] All stages initialized.")

    def _get_vlm(self):
        """Lazily construct the VLMVerifier (defers the heavy import + model load)."""
        if self._vlm is None:
            from .vlm_verifier import VLMVerifier
            self._vlm = VLMVerifier(
                model_name=self.vlm_model_name,
                device=self.dino_verifier.device.type
                if hasattr(self.dino_verifier.device, "type")
                else None,
                symbol_name=self.vlm_symbol_name,
            )
        return self._vlm

    def _template_upscale_factor(
        self,
        pattern_proc: np.ndarray,
        trigger_px: int = 55,
        target_px: int = 130,
        max_factor: float = 4.0,
    ) -> float:
        """Return the upscale factor for a tiny template (1.0 = no upscale).

        Only GENUINELY tiny templates are upscaled. A normal-sized template (the
        bridge rectifier at 70px, resistor at 70px) returns 1.0 -- upscaling those
        shifts the probe scale and breaks their tuned detection path.

        Args:
            pattern_proc: Preprocessed (binarised) template image.
            trigger_px: Only upscale if the symbol's larger side is below this.
            target_px: Upscale tiny templates so their larger side reaches this.
            max_factor: Maximum upscale factor (prevents extreme blur).
        """
        dark = pattern_proc < 128
        rows_any = np.any(dark, axis=1)
        cols_any = np.any(dark, axis=0)
        if not (rows_any.any() and cols_any.any()):
            return 1.0
        rmin, rmax = np.where(rows_any)[0][[0, -1]]
        cmin, cmax = np.where(cols_any)[0][[0, -1]]
        larger = max(int(rmax - rmin + 1), int(cmax - cmin + 1))
        if larger >= trigger_px:
            return 1.0
        return min(max_factor, target_px / max(1, larger))

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
          Pass 1 -- strict (ncc=0.55, dilate=0): catches clean/legend copies
          Pass 2 -- relaxed (ncc=0.28, dilate=5): catches style-mismatched + larger components

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

            # Auto-upscale tiny templates for richer zero-shot features.
            #
            # A template provides the feature query. When its symbol content is very
            # small (e.g. a 26x39 XNOR crop), both NCC and DINOv2 receive too few
            # pixels of detail; matching at the necessary upscale factor blurs the
            # symbol and produces many false positives.
            #
            # Measured impact (XNOR 43x55 template on CLC-003): upscaling cut false
            # positives from 17 -> 6 and raised TP confidence 0.56 -> 0.72-0.86.
            #
            # IMPORTANT: upscale the RAW GRAYSCALE then re-binarise. Upscaling an
            # already-binarised low-res template produces blocky edges and FPs;
            # upscaling the grayscale first preserves smooth symbol detail.
            _factor = self._template_upscale_factor(pattern_data["processed"])
            if _factor > 1.05:
                _orig = pattern_data["original"]
                _up = cv2.resize(
                    _orig,
                    (int(_orig.shape[1] * _factor), int(_orig.shape[0] * _factor)),
                    interpolation=cv2.INTER_CUBIC,
                )
                pattern_data = self.preprocessor.preprocess(_up)
                print(f"[Pipeline] Template upscaled {_factor:.1f}x (raw grayscale, then re-binarised)")

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

            # For complex templates: run scale probe early to decide if standard passes
            # can be skipped.  When probe_s > 1.40 those passes produce candidates at
            # wrong scale that get discarded anyway -- skipping saves ~215 s per run.
            _ph_p, _pw_p = pattern_proc.shape[:2]
            _drwH_p, _drwW_p = drawing_proc.shape[:2]
            _pre_probe_s, _pre_probe_ncc = 1.0, 0.0
            _skip_std_passes = False
            if not _is_simple:
                for _ps in [0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.85, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
                    _ptw_p = int(_pw_p * _ps); _pth_p = int(_ph_p * _ps)
                    if _ptw_p < 10 or _pth_p < 10 or _drwH_p < _pth_p or _drwW_p < _ptw_p:
                        continue
                    _pt_s_p = cv2.resize(pattern_proc, (_ptw_p, _pth_p), interpolation=cv2.INTER_AREA)
                    _pres_p = cv2.matchTemplate(drawing_proc, _pt_s_p, cv2.TM_CCOEFF_NORMED)
                    _, _pncc_p, _, _ = cv2.minMaxLoc(_pres_p)
                    if _pncc_p > _pre_probe_ncc:
                        _pre_probe_ncc = _pncc_p
                        _pre_probe_s = _ps
                _skip_std_passes = _pre_probe_s > 1.40
                print(f"[Pipeline] Scale probe: best={_pre_probe_s:.2f} ncc={_pre_probe_ncc:.3f}"
                      + (" -- skipping std passes" if _skip_std_passes else ""))

            if not _skip_std_passes:
                # Pass 1: strict -- undilated template, high NCC threshold
                self.ncc_matcher.ncc_threshold = 0.55
                candidates_strict = self.ncc_matcher.match(drawing_proc, pattern_proc)
                print(f"[Pipeline] Pass 1 (strict): {len(candidates_strict)} candidates")

                # Pass 2: relaxed -- dilated template.
                # For simple-outline templates raise threshold: structural FPs (frame/table)
                # match poorly at non-native scales, while real components match at 0.50+.
                self.ncc_matcher.ncc_threshold = 0.50 if _is_simple else 0.28
                pattern_dilated = self.preprocessor.dilate_strokes(pattern_proc, kernel_size=5)
                candidates_relaxed = self.ncc_matcher.match(drawing_proc, pattern_dilated)
                print(f"[Pipeline] Pass 2 (relaxed): {len(candidates_relaxed)} candidates")

                all_candidates = candidates_strict + candidates_relaxed
                t2 = time.time()
                print(f"[Pipeline] NCC total: {t2 - t1:.2f}s -- {len(all_candidates)} combined candidates")

                # DINOv2 on standard-scale candidates (skip if none found)
                verified = self.dino_verifier.verify_candidates(
                    drawing_proc, pattern_proc, all_candidates
                ) if all_candidates else []
                t3 = time.time()
                print(f"[Pipeline] DINOv2 (standard): {t3 - t2:.2f}s -- {len(verified)} verified")
            else:
                # Standard passes skipped: candidates at standard scales [0.70–1.35]
                # would be discarded in the probe-focused path anyway.
                all_candidates = []
                verified = []
                t2 = time.time()
                t3 = t2

            _saved_scales  = self.ncc_matcher.scales
            _saved_ncc     = self.ncc_matcher.ncc_threshold
            _saved_angles  = self.ncc_matcher.angles
            _saved_dino    = self.dino_verifier.cosine_threshold

            if _is_simple:
                # --- Pass A: NCC (primary, covers scale 0.30-1.0) ---
                _SIMPLE_SCALES = [0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 1.0]
                self.ncc_matcher.scales = _SIMPLE_SCALES
                self.ncc_matcher.ncc_threshold = 0.42
                self.ncc_matcher.angles = [-10, -5, 0, 5, 10, 80, 85, 90, 95, 100]
                cands_s = self.ncc_matcher.match(drawing_proc, pattern_proc)
                ncc_s_count = len(cands_s)
                if cands_s:
                    cands_s = self.postprocessor.filter_chamfer_shape(
                        cands_s, drawing_proc, pattern_proc, max_chamfer=3.0
                    )
                    for c in cands_s:
                        c.setdefault("dino_score", 0.0)
                        c.setdefault("confidence", c.get("ncc_score", 0.5))
                before_s = len(cands_s)
                cands_s = self.postprocessor.filter_title_block(cands_s, drawing_proc)
                print(f"[Pipeline] NCC pass A: {ncc_s_count} -> {before_s} chamfer -> {len(cands_s)}")

                # --- Pass B: 90°-rotated drawing for vertical instances ---
                cands_rot90 = []
                if abs(_tmpl_ar - 1.0) > 0.25:
                    _drw_orig_H, _drw_orig_W = drawing_proc.shape[:2]
                    drawing_rot90 = cv2.rotate(drawing_proc, cv2.ROTATE_90_CLOCKWISE)
                    self.ncc_matcher.scales = _SIMPLE_SCALES
                    self.ncc_matcher.ncc_threshold = 0.42
                    self.ncc_matcher.angles = [-10, -5, 0, 5, 10]
                    raw_rot = self.ncc_matcher.match(drawing_rot90, pattern_proc)
                    if raw_rot:
                        raw_rot = self.postprocessor.filter_chamfer_shape(
                            raw_rot, drawing_rot90, pattern_proc, max_chamfer=3.0
                        )
                        for c in raw_rot:
                            rx, ry, rw, rh = c["x"], c["y"], c["w"], c["h"]
                            c["x"] = ry; c["y"] = _drw_orig_H - rx - rw
                            c["w"] = rh; c["h"] = rw; c["angle"] = 90
                            c.setdefault("dino_score", 0.0)
                            c.setdefault("confidence", c.get("ncc_score", 0.5))
                        raw_rot = self.postprocessor.filter_title_block(raw_rot, drawing_proc)
                        cands_rot90 = [c for c in raw_rot if c.get("confidence", 0) >= 0.58]

                # --- Pass C: DINODense for LARGE-SCALE instances (probe_s > 1.1) ---
                # Activates when NCC's scale range [0.30-1.0] misses instances because
                # they are LARGER than the template. Only the probe test runs fast (NCC
                # on one scale); the dense pass only runs when needed.
                cands_dense = []
                _probe_s, _probe_ncc = self.dino_dense._probe_scale(drawing_proc, pattern_proc)
                if self.use_dino_dense and _probe_s > 1.10 and _probe_ncc >= 0.35:
                    print(f"[Pipeline] DINODense activated (probe_s={_probe_s:.2f}, ncc={_probe_ncc:.3f})")
                    _dense_angles = [0, 90] if abs(_tmpl_ar - 1.0) > 0.20 else [0]
                    cands_dense = self.dino_dense.match(
                        drawing_proc, pattern_proc, angles=_dense_angles
                    )
                    for c in cands_dense:
                        c["from_dino_dense"] = True  # tag: skip NCC-era struct filters
                    cands_dense = self.postprocessor.filter_title_block(cands_dense, drawing_proc)
                    print(f"[Pipeline] DINODense: {len(cands_dense)} large-scale candidates")

                verified = verified + cands_s + cands_rot90 + cands_dense
            else:
                # General complex template path: scale probe -> adaptive search.
                # No hardcoded shape classifiers; decisions are driven by probe results.
                _ph, _pw = pattern_proc.shape[:2]
                _drwH, _drwW = drawing_proc.shape[:2]
                _no_std_candidates = len(all_candidates) == 0

                # Scale probe was already run before passes 1+2 -- reuse the result.
                _best_probe_s = _pre_probe_s
                _best_probe_ncc = _pre_probe_ncc

                # Decide whether to use probe-focused scales or standard complex scales.
                #
                # Probe-focused (±20% around probe) when:
                # - No standard NCC candidates at all -> template is at a very unusual scale
                # - Probe found scale > 1.40 -> template appears larger in drawing than legend
                #   (e.g. zigzag resistors at 1.5x); standard scales [0.70–1.35] would miss them
                #
                # Standard complex scales when:
                # - Probe scale <= 1.40 with standard candidates -> probe may catch a false maximum
                #   at small scales (e.g. scale 0.25 for a template whose real instances are at
                #   0.85–1.15); standard sweep is more reliable in this regime
                _use_probe_focused = _no_std_candidates or _best_probe_s > 1.40

                if _use_probe_focused:
                    _fracs = [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]
                    _micro_scales = sorted(set(
                        round(_best_probe_s * f, 2)
                        for f in _fracs if 0.20 <= _best_probe_s * f <= 3.0
                    ))
                    # Standard-pass candidates are at wrong scale; drop them.
                    # (When no_std_candidates=True, verified is empty anyway.)
                    verified_filtered = []
                    if verified:
                        print(f"[Pipeline] Std candidates dropped (probe scale {_best_probe_s:.2f} > standard range)")
                    # Stricter DINOv2 for probe-focused micro passes to compensate for
                    # the broader search area and reduce FPs from similarly-shaped components.
                    _micro_dino = min(0.88, max(0.82, _best_probe_ncc * 1.04))
                    # No 3b bump for probe-focused path: 90°-rotated candidates have naturally
                    # lower DINOv2 scores; an extra +0.01 would cut real vertical detections.
                    _micro_dino_3b = _micro_dino
                    # Tight NMS for probe-focused path: each candidate bbox is already at
                    # the right scale; union-expansion would create oversized merged boxes.
                    _complex_use_union = False
                else:
                    _micro_scales = [0.70, 0.85, 1.0, 1.1, 1.2, 1.35]
                    # Standard-pass candidates are at the right scale -- pass all through.
                    # Filtering here reduces chain-suppression in the final NMS and causes
                    # over-counting; the micro passes + NMS handle deduplication.
                    verified_filtered = verified
                    # Moderate DINOv2 for standard-range micro passes.
                    _micro_dino = 0.82
                    # Slight 3b bump: reduces FPs from 90°-rotated non-components that
                    # happen to pass the 0.82 threshold (e.g. extra bridge rectifier FP).
                    _micro_dino_3b = min(_micro_dino + 0.01, 0.88)
                    # Use best-fit bbox (no union expand) for the standard path.
                    # Union expansion caused oversized boxes when a FP at one scale
                    # and a TP at another overlapped and merged into a huge union box.
                    _complex_use_union = False

                self.ncc_matcher.ncc_threshold = 0.28

                # Pass 3a: near-0°
                self.ncc_matcher.scales = _micro_scales
                self.ncc_matcher.angles = [-10, -5, 0, 5, 10]
                self.dino_verifier.cosine_threshold = _micro_dino
                cands_3a = self.ncc_matcher.match(drawing_proc, pattern_proc)
                verified_3a = self.dino_verifier.verify_candidates(
                    drawing_proc, pattern_proc, cands_3a, derotate=True
                ) if cands_3a else []
                print(f"[Pipeline] Pass 3a (micro 0°): {len(cands_3a)} cands -> {len(verified_3a)} verified")

                # Pass 3b: near-90°
                self.ncc_matcher.scales = _micro_scales
                self.ncc_matcher.angles = [80, 85, 90, 95, 100]
                self.dino_verifier.cosine_threshold = _micro_dino_3b
                cands_3b = self.ncc_matcher.match(drawing_proc, pattern_proc)
                verified_3b = self.dino_verifier.verify_candidates(
                    drawing_proc, pattern_proc, cands_3b, derotate=True
                ) if cands_3b else []
                print(f"[Pipeline] Pass 3b (micro 90°): {len(cands_3b)} cands -> {len(verified_3b)} verified")

                all_complex = verified_filtered + verified_3a + verified_3b

                # Chamfer shape filter: structural edge-alignment quality check.
                # Applied only on the probe-focused path where DINOv2 alone is
                # insufficient -- components at unusual scales (>1.40x) attract FPs
                # from visually similar but structurally different symbols.
                # Standard-path templates (BR, IEC) have complex internal edge
                # structure; Chamfer at 3.0 wrongly rejects real detections there.
                if _use_probe_focused and all_complex:
                    before_ch = len(all_complex)
                    # Threshold 5.0: real zigzag TPs score 0.5–4.1; the single confirmed
                    # FP at scale boundary scored 6.4 -- this cleanly removes it.
                    # Standard path skips Chamfer: IEC has some candidates with
                    # Chamfer 9–10 due to bbox distortion from the dilated-template
                    # pass; filtering them drops real detections.
                    all_complex = self.postprocessor.filter_chamfer_shape(
                        all_complex, drawing_proc, pattern_proc, max_chamfer=5.0
                    )
                    if len(all_complex) != before_ch:
                        print(f"[Pipeline] Chamfer filter: {before_ch} -> {len(all_complex)}")

                    # Tighter Chamfer for horizontal candidates: horizontal TPs max at
                    # ~3.3 (measured); vertical TPs can reach ~4.1 due to rotation/resize.
                    # Candidates with angle ≈ 0° and Chamfer 4.0–5.0 are structural FPs
                    # (inductors, transistors) that the global 5.0 threshold misses.
                    before_hch = len(all_complex)
                    all_complex = [
                        c for c in all_complex
                        if abs(c.get("angle", 0)) >= 45
                        or c.get("chamfer_dist", 0) <= 4.0
                    ]
                    if len(all_complex) != before_hch:
                        print(f"[Pipeline] Chamfer H filter: {before_hch} -> {len(all_complex)}")

                # Output-bubble filter: only for gate-like templates at unusual scales.
                # XOR/XNOR output bubbles match the gate body; filter them out.
                # Standard-scale templates (including bridge rectifiers) skip this.
                _is_gate_like = 0.25 <= _tmpl_ar <= 2.5
                if _is_gate_like and _no_std_candidates:
                    before_bubble = len(all_complex)
                    all_complex = self.postprocessor.filter_output_bubble(
                        all_complex, drawing_proc, pattern_proc
                    )
                    if len(all_complex) != before_bubble:
                        print(f"[Pipeline] Bubble filter: {before_bubble} -> {len(all_complex)}")

                verified = all_complex

            self.ncc_matcher.scales  = _saved_scales
            self.ncc_matcher.ncc_threshold = _saved_ncc
            self.ncc_matcher.angles  = _saved_angles
            self.dino_verifier.cosine_threshold = _saved_dino
            t3 = time.time()
            print(f"[Pipeline] All passes: {t3 - t2:.2f}s -- {len(verified)} total verified")

            # Simple-template post-filters: isolation + aspect-ratio + neighborhood.
            # Isolation: circuit components sit in white space; BOM/title-block cells have
            # solid grid lines directly adjacent to their long sides.
            # Aspect ratio: keep candidates whose bbox AR is within 2x of the template AR
            # *or* its reciprocal -- the reciprocal check allows 90°-rotated components
            # (e.g. vertical resistors) whose bbox AR is ~1/_tmpl_ar.
            # Neighborhood complexity: reject candidates whose surrounding ring has too
            # many Canny edges -- this eliminates false positives inside complex symbols
            # (e.g. bridge-rectifier bodies) which have many adjacent edges.
            # Notes-area exemption: the bottom 20% of the drawing contains notes/legend
            # symbols which have annotation text nearby -- skip isolation and neighborhood
            # checks for those so legitimate legend symbols are not filtered out.
            # Top-margin exclusion: discard detections whose top edge is within the outer
            # coordinate-margin strip (border grid cells look like plain rectangles).
            if _is_simple and verified:
                before = len(verified)
                _drw_h, _drw_w = drawing_proc.shape[:2]
                _notes_y = int(_drw_h * 0.80)   # below this -> notes/legend area
                _top_margin = max(30, int(_drw_h * 0.04))  # top coordinate strip

                # Reject border-grid cells in the top margin
                verified = [c for c in verified if c["y"] >= _top_margin]

                # DINODense candidates are already DINOv2-verified (sim >= threshold).
                # They bypass the NCC-era structural filters (wire-leads, chamfer, etc.)
                # which assume tight bbox alignment. DINOv2 score IS the structural check.
                _dense_cands  = [c for c in verified if c.get("from_dino_dense")]
                _ncc_cands    = [c for c in verified if not c.get("from_dino_dense")]

                # Split NCC candidates into circuit area and notes/legend area
                _circuit = [c for c in _ncc_cands if c["y"] < _notes_y]
                _notes   = [c for c in _ncc_cands if c["y"] >= _notes_y]

                # Dense candidates: circuit zone only (no structural filters)
                _dense_circuit = [c for c in _dense_cands if c["y"] < _notes_y]

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
                # -- handles resistors connected directly to a power rail where only
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

                # Orientation-aware minimum confidence.
                #
                # The template has aspect ratio _tmpl_ar (width / height).
                # Candidates in the same orientation as the template ("native") are
                # matched directly by NCC and should score high -- low-confidence
                # native candidates are almost certainly FPs (inductors, transistors,
                # op-amps that partially match the template bbox at low NCC).
                #
                # Candidates in the perpendicular orientation ("rotated") are matched
                # after a 90° rotation, which inherently lowers the NCC score even
                # for genuine resistors; they deserve a much more lenient threshold.
                #
                # Calibration from drawing 1 (test_1.png AR≈2.5, horizontal template):
                #   Native (horizontal) TPs:  conf 0.77, 0.78, 0.78  (all >= 0.70)
                #   Rotated (vertical) TPs:   conf 0.50–0.78          (all >= 0.45)
                #   Observed FPs in complex drawings: conf 0.58–0.67 (all horizontal)
                # -> threshold 0.70 for native / 0.45 for rotated removes FPs while
                #   keeping all drawing-1 TPs.
                _tmpl_native_wide = _tmpl_ar >= 1.0  # template is wider-than-tall
                before_oc = len(_circuit)
                def _is_native_orient(c):
                    cand_wide = c["w"] >= c["h"]
                    return cand_wide == _tmpl_native_wide

                # Pass-B (rotated-image) candidates are marked angle=90 and have already
                # been filtered at conf>=0.58 before structural filters; they represent
                # native-orientation matches so use a moderate threshold here.
                # Pass-A vertical candidates (not from rotated-image) still matched via
                # template rotation and score lower -- they need conf>=0.50 instead of 0.45.
                _circuit = [
                    c for c in _circuit
                    if (_is_native_orient(c) and c.get("confidence", 0) >= 0.70)
                    or (not _is_native_orient(c) and c.get("confidence", 0) >= 0.45)
                ]
                if len(_circuit) != before_oc:
                    print(
                        f"[Pipeline] Orient-conf filter: {before_oc} -> {len(_circuit)} "
                        f"(native>=0.70 | rotated>=0.45)"
                    )

                _bottom_margin = max(30, int(_drw_h * 0.04))
                _notes = [c for c in _notes
                          if c["y"] + c["h"] <= _drw_h - _bottom_margin]
                _notes = self.postprocessor.filter_rect_borders(_notes, drawing_proc)
                _notes = sorted(_notes, key=lambda c: c.get("dino_score", 0), reverse=True)[:1]

                # DINOv2 Self-Supervised Prototype Filter
                #
                # Instead of comparing borderline candidates to the TEMPLATE (which
                # may be a different drawing style), build a prototype from the
                # HIGH-CONFIDENCE detections in THIS DRAWING. These are confirmed
                # instances of the target symbol in the actual drawing style and scale.
                #
                # Algorithm:
                #   1. Extract DINOv2 embeddings for all high-conf TPs (conf >= 0.72)
                #      -- orientation-normalised so horizontal and vertical instances
                #      of the same symbol produce comparable embeddings.
                #   2. Prototype = mean unit-normalised embedding.
                #   3. For each borderline candidate: compute cosine(candidate, prototype).
                #   4. Reject if similarity < min_sim.
                #
                # Why this works:
                #   -- Resistors (any orientation) -> similar DINOv2 embedding -> high sim
                #   -- Inductors/transistors/op-amps -> different embedding -> low sim
                #   -- Prototype adapts to the specific drawing style automatically
                _proto_threshold = 0.72   # high-conf TPs used for prototype
                _proto_min_sim   = 0.82   # borderline candidates below this are rejected
                _hc = [c for c in _circuit if c.get("confidence", 0) >= _proto_threshold]
                _bl = [c for c in _circuit if c.get("confidence", 0) < _proto_threshold]

                if len(_hc) >= 3 and _bl:
                    dh, dw = drawing_proc.shape[:2]
                    hc_crops = [
                        self.dino_verifier._crop_with_padding(drawing_proc, c, dh, dw)
                        for c in _hc
                    ]
                    hc_embeds = self.dino_verifier.embed_crops_normalized(hc_crops)
                    prototype = hc_embeds.mean(axis=0)
                    _pnorm = float(np.linalg.norm(prototype))
                    if _pnorm > 1e-6:
                        prototype = prototype / _pnorm
                        bl_crops = [
                            self.dino_verifier._crop_with_padding(drawing_proc, c, dh, dw)
                            for c in _bl
                        ]
                        bl_embeds = self.dino_verifier.embed_crops_normalized(bl_crops)
                        bl_sims = bl_embeds @ prototype    # (M,) cosine similarities
                        _accepted = [c for c, s in zip(_bl, bl_sims.tolist())
                                     if s >= _proto_min_sim]
                        _rejected = [c for c, s in zip(_bl, bl_sims.tolist())
                                     if s < _proto_min_sim]
                        if _rejected:
                            print(
                                f"[Pipeline] DINO-proto: {len(_bl)} border -> "
                                f"{len(_accepted)} kept, {len(_rejected)} rejected "
                                f"(sim>={_proto_min_sim})"
                            )
                        _circuit = _hc + _accepted
                    else:
                        _circuit = _hc + _bl   # fallback
                else:
                    _circuit = _hc + _bl   # not enough high-conf to build prototype

                # Merge: NCC circuit + DINODense circuit (skipped structural filters)
                # DINODense candidates already have high DINOv2 similarity >= threshold
                verified = _circuit + _dense_circuit + _notes
                print(
                    f"[Pipeline] Simple-template filters: {before} -> {len(verified)} "
                    f"(NCC:{len(_circuit)} | DINODense:{len(_dense_circuit)} | notes:{len(_notes)})"
                )

            # Title-block zone filter: remove candidates inside the BOM / right-frame
            # area for all templates (complex templates skip the simple-filter block
            # that previously applied this only to simple micro-pass candidates).
            before_tb = len(verified)
            verified = self.postprocessor.filter_title_block(verified, drawing_proc)
            if len(verified) != before_tb:
                print(f"[Pipeline] Title-block filter: {before_tb} -> {len(verified)}")

            # Adaptive confidence-gap filter: detect bimodal confidence distribution
            # and remove the low-confidence cluster (structural FPs: inductors,
            # transistors, op-amps that barely pass DINOv2 with low NCC).
            # Only applied to complex templates -- simple templates use dedicated
            # structural filters (wire-leads, chamfer, DINO-prototype) that are more
            # precise; the gap filter risks removing genuine low-confidence TPs there.
            if not _is_simple:
                before_gap = len(verified)
                verified = self.postprocessor.filter_confidence_gap(verified)
                if len(verified) != before_gap:
                    print(f"[Pipeline] Confidence gap filter: {before_gap} -> {len(verified)}")

            # Stage 3 (optional): VLM semantic filter.
            # Runs AFTER all spatial/structural filters and BEFORE final NMS so the
            # VLM only judges a small, already-pruned candidate set (fast: ~0.4s/crop).
            # Zero-shot: the template's own VLM class defines the target, so no
            # symbol name is hardcoded. Removes inductor/crystal/op-amp FPs that
            # share low-level structure with the target and pass DINOv2.
            if self.use_vlm and verified:
                before_vlm = len(verified)
                try:
                    vlm = self._get_vlm()
                    verified = vlm.filter_by_template_class(
                        drawing_proc, pattern_proc, verified,
                        keep_min_conf=self.vlm_keep_min_conf, verbose=True
                    )
                    print(f"[Pipeline] VLM Stage-3 filter: {before_vlm} -> {len(verified)}")
                except Exception as vlm_err:
                    # The VLM is an optional precision booster. If it fails (most
                    # commonly CUDA OOM on a 12 GB card already holding DINOv2 +
                    # candidate tensors), degrade gracefully: keep the NCC+DINOv2
                    # results rather than failing the whole request.
                    is_oom = "out of memory" in str(vlm_err).lower()
                    print(f"[Pipeline] VLM Stage-3 SKIPPED ({'OOM' if is_oom else type(vlm_err).__name__}): "
                          f"{vlm_err}")
                    self._vlm = None  # drop the half-loaded model
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            # Final NMS + format
            # Simple templates: tight IoU (0.25), no union expand (keep best-fit bbox).
            # Complex templates:
            #   - probe-focused path: tight IoU (0.25) to suppress FP clusters in dense
            #     circuit regions where multiple off-scale detections land on the same symbol
            #   - standard path (_complex_use_union=True): union expand for chain suppression
            if _is_simple:
                verified = self.postprocessor.final_nms(
                    verified, iou_threshold=0.25, use_union_bbox=False
                )
            else:
                verified = self.postprocessor.final_nms(
                    verified, iou_threshold=self.final_nms_iou,
                    use_union_bbox=_complex_use_union
                )
            result = self.postprocessor.format_output(verified, drawing_proc.shape)
            t4 = time.time()
            print(f"[Pipeline] Auto-detect total: {t4 - t0:.2f}s -- {result['total_detections']} detections")

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

            pattern_proc = pattern_data["processed"]   # original -- used for DINOv2
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
            print(f"[Pipeline] Stage 1 (NCC): {t2 - t1:.2f}s -- {len(candidates)} candidates")

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
            print(f"[Pipeline] Stage 2 (DINOv2): {t3 - t2:.2f}s -- {len(candidates)} verified")

            # Stage 3: Final NMS + format
            candidates = self.postprocessor.final_nms(candidates, iou_threshold=self.final_nms_iou)
            result = self.postprocessor.format_output(candidates, drawing_proc.shape)
            t4 = time.time()
            print(f"[Pipeline] Stage 3 (Post): {t4 - t3:.2f}s")
            print(f"[Pipeline] Total: {t4 - t0:.2f}s -- {result['total_detections']} detections")

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
