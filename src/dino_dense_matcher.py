"""Dense DINO-based template matching — replaces NCC for zero-shot detection.

Why NCC fails at scale:
  NCC resizes the template to a fixed scale list (e.g. 0.30-1.0x). Patterns
  larger than the template are missed, and large-scale candidates create big
  bounding boxes that suppress real detections via NMS containment.

This module replaces NCC with a DINOv2-based dense sliding window:

  1. SCALE PROBE (fast NCC probe at coarse scales) — finds the scale at which
     the pattern actually appears in the drawing. This is O(scales * W * H)
     with a single cv2.matchTemplate call per scale — very fast.

  2. DENSE SLIDING WINDOW at the probed scale. Each window is cropped from the
     drawing at the correct absolute pixel size and embedded by DINOv2. Windows
     with cosine similarity above a threshold become candidates.
     Batched GPU inference keeps this fast (~1-3s for a full drawing).

  3. NMS on candidates from all angles (0° + 90°, via drawing rotation).

Key properties vs NCC:
  * Truly scale-invariant: probe finds any scale 0.15x-5.0x template size.
  * Style-invariant: DINOv2 features abstract over drawing style (IEC vs ANSI).
  * No large-bbox NMS clash: all windows at a given scale are the same size.
  * Rotation via drawing flip: vertical patterns found cleanly without template
    rotation artifacts.
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Coarse scale probe grid: wide range, logarithmically spaced
_PROBE_SCALES = [
    0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65,
    0.80, 1.0, 1.20, 1.50, 1.80, 2.20, 2.80, 3.50,
]


class DINODenseMatcher:
    """Scale-invariant template matcher using DINOv2 dense sliding window.

    Drop-in replacement for NCCMatcher in the simple-template pipeline path.

    Args:
        dino_verifier: Initialised DINOVerifier instance (model already loaded).
        nms_iou_threshold: IoU threshold for NMS on candidates.
        sim_threshold: Minimum cosine similarity to accept a window as candidate.
        stride_ratio: Sliding window stride as fraction of window size (0.0-1.0).
            Smaller = finer coverage but slower. 0.40 is a good default.
        batch_size: Number of crops per DINOv2 forward pass. 32 works well on GPU.
        probe_ncc_min: Minimum probe NCC to trust the probed scale. If the best
            probe NCC is below this (flat/empty drawing region), fall back to
            scale=1.0 so at least something is searched.
    """

    def __init__(
        self,
        dino_verifier,
        nms_iou_threshold: float = 0.30,
        sim_threshold: float = 0.84,
        stride_ratio: float = 0.60,
        batch_size: int = 32,
        probe_ncc_min: float = 0.30,
    ):
        self.dino = dino_verifier
        self.nms_iou_threshold = nms_iou_threshold
        self.sim_threshold = sim_threshold
        self.stride_ratio = stride_ratio
        self.batch_size = batch_size
        self.probe_ncc_min = probe_ncc_min

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def match(
        self,
        drawing: np.ndarray,
        template: np.ndarray,
        angles: Optional[List[int]] = None,
    ) -> List[dict]:
        """Find all occurrences of `template` in `drawing`.

        Args:
            drawing: Preprocessed (binarised) drawing, grayscale uint8.
            template: Preprocessed template image, grayscale uint8.
            angles: List of angles to search. Currently supports [0, 90].
                    Angle 0  -> horizontal search on original drawing.
                    Angle 90 -> search on 90°-rotated drawing (finds vertical).
                    Defaults to [0, 90] for non-square templates, [0] for square.

        Returns:
            List of candidate dicts with keys:
              x, y, w, h  — bounding box in ORIGINAL drawing coordinates
              ncc_score   — cosine similarity (named ncc_score for API compat.)
              dino_score  — same value
              confidence  — same value
              scale       — detected scale relative to template
              angle       — 0 or 90
        """
        t0 = time.time()
        ph, pw = template.shape[:2]
        tmpl_ar = pw / max(1, ph)

        if angles is None:
            angles = [0, 90] if abs(tmpl_ar - 1.0) > 0.20 else [0]

        # Template embedding (once, reused for all scales and angles)
        tmpl_embed = self._template_embed(template)   # (D,) unit vector

        all_cands: List[dict] = []

        for angle in angles:
            if angle == 0:
                search_img = drawing
                orig_H, orig_W = drawing.shape[:2]
            else:
                # 90° CW rotation: vertical instances become horizontal
                search_img = cv2.rotate(drawing, cv2.ROTATE_90_CLOCKWISE)
                orig_H, orig_W = drawing.shape[:2]

            cands_a = self._match_one_orientation(
                search_img, template, tmpl_embed, angle, orig_H, orig_W
            )
            all_cands.extend(cands_a)

        # NMS across all candidates
        all_cands = self._apply_nms(all_cands)

        t1 = time.time()
        print(
            f"[DINODense] {len(all_cands)} candidates "
            f"({len(angles)} orientations) in {t1-t0:.2f}s"
        )
        return all_cands

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _template_embed(self, template: np.ndarray) -> np.ndarray:
        """Return unit-normalised DINOv2 embedding for the template."""
        emb = self.dino.embed_crops_batch([template], batch_size=1)  # (1, D)
        return emb[0]  # (D,)

    def _probe_scale(
        self, drawing: np.ndarray, template: np.ndarray
    ) -> Tuple[float, float]:
        """Fast NCC scale probe. Returns (best_scale, best_ncc)."""
        ph, pw = template.shape[:2]
        dh, dw = drawing.shape[:2]
        best_s, best_ncc = 1.0, 0.0
        for s in _PROBE_SCALES:
            tw, th = int(pw * s), int(ph * s)
            if tw < 8 or th < 8 or dw < tw or dh < th:
                continue
            t_scaled = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(drawing, t_scaled, cv2.TM_CCOEFF_NORMED)
            _, ncc, _, _ = cv2.minMaxLoc(res)
            if ncc > best_ncc:
                best_ncc, best_s = ncc, s
        return best_s, best_ncc

    def _match_one_orientation(
        self,
        search_img: np.ndarray,
        template: np.ndarray,
        tmpl_embed: np.ndarray,
        angle: int,
        orig_H: int,
        orig_W: int,
    ) -> List[dict]:
        """Dense DINO match for one orientation (0° or 90°)."""
        ph, pw = template.shape[:2]
        sh, sw = search_img.shape[:2]

        # Scale probe
        best_s, best_ncc = self._probe_scale(search_img, template)
        if best_ncc < self.probe_ncc_min:
            best_s = 1.0   # fallback: search at template native scale

        # Search at +-20% around best scale (3 sub-steps for speed)
        search_scales = sorted({
            round(best_s * f, 2)
            for f in [0.85, 1.0, 1.15]
            if 0.10 <= best_s * f <= 6.0
        })

        print(
            f"[DINODense] angle={angle} probe: scale={best_s:.2f} "
            f"ncc={best_ncc:.3f} -> search {search_scales}"
        )

        # Edge map for fast pre-filter (skip blank regions)
        edges = cv2.Canny(search_img, 30, 100)

        candidates: List[dict] = []

        for s in search_scales:
            win_w = int(pw * s)
            win_h = int(ph * s)
            if win_w < 10 or win_h < 10 or sw < win_w or sh < win_h:
                continue

            stride_x = max(4, int(win_w * self.stride_ratio))
            stride_y = max(4, int(win_h * self.stride_ratio))

            # Build window positions, skip low-edge-density regions
            positions = []
            min_edge_px = max(3, int(win_w * win_h * 0.005))
            for y in range(0, sh - win_h + 1, stride_y):
                for x in range(0, sw - win_w + 1, stride_x):
                    if int(np.count_nonzero(edges[y:y+win_h, x:x+win_w])) >= min_edge_px:
                        positions.append((x, y))

            if not positions:
                continue

            # Crop windows and batch-embed
            crops = [
                search_img[y:y+win_h, x:x+win_w]
                for x, y in positions
            ]
            embeds = self.dino.embed_crops_batch(crops, batch_size=self.batch_size)
            sims = embeds @ tmpl_embed   # cosine similarities (already unit-normed)

            # Collect candidates above threshold
            for (x_s, y_s), sim in zip(positions, sims.tolist()):
                if sim < self.sim_threshold:
                    continue

                if angle == 0:
                    # Original coords
                    cx, cy, cw, ch = x_s, y_s, win_w, win_h
                else:
                    # 90° CW rotation inverse:
                    # (rx, ry, rw, rh) in rotated -> (ry, origH-rx-rw, rh, rw)
                    cx = y_s
                    cy = orig_H - x_s - win_w
                    cw = win_h
                    ch = win_w

                # Clamp to original drawing bounds
                cx = max(0, min(orig_W - 1, cx))
                cy = max(0, min(orig_H - 1, cy))
                cw = min(cw, orig_W - cx)
                ch = min(ch, orig_H - cy)
                if cw < 4 or ch < 4:
                    continue

                candidates.append({
                    "x": cx, "y": cy, "w": cw, "h": ch,
                    "ncc_score": round(float(sim), 4),
                    "dino_score": round(float(sim), 4),
                    "confidence": round(float(sim), 4),
                    "scale": float(s),
                    "angle": angle,
                })

        return candidates

    def _apply_nms(self, candidates: List[dict]) -> List[dict]:
        """Simple IoU-based NMS; keep highest-confidence in overlapping groups."""
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
        keep = []
        suppressed = [False] * len(candidates)

        for i, c in enumerate(candidates):
            if suppressed[i]:
                continue
            keep.append(c)
            ax1, ay1 = c["x"], c["y"]
            ax2, ay2 = ax1 + c["w"], ay1 + c["h"]
            for j in range(i + 1, len(candidates)):
                if suppressed[j]:
                    continue
                b = candidates[j]
                bx1, by1 = b["x"], b["y"]
                bx2, by2 = bx1 + b["w"], by1 + b["h"]
                ix = max(0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0, min(ay2, by2) - max(ay1, by1))
                inter = ix * iy
                if inter == 0:
                    continue
                union = c["w"]*c["h"] + b["w"]*b["h"] - inter
                min_a = min(c["w"]*c["h"], b["w"]*b["h"])
                overlap = max(inter/union if union>0 else 0,
                              inter/min_a if min_a>0 else 0)
                if overlap >= self.nms_iou_threshold:
                    suppressed[j] = True

        return keep
