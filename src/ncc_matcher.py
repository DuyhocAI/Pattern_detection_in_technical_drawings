import cv2
import numpy as np
from typing import List


class NCCMatcher:
    """Stage 1: NCC multi-scale template matching for candidate proposal."""

    def __init__(
        self,
        scales: List[float] = None,
        angles: List[float] = None,
        ncc_threshold: float = 0.55,
        nms_iou_threshold: float = 0.3,
    ):
        self.scales = scales or [0.85, 0.95, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0]
        self.angles = angles or [-10, -5, 0, 5, 10]
        self.ncc_threshold = ncc_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.min_template_px = 18

    def rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
        """Rotate template around its center, filling border with white.

        Args:
            template: Grayscale template image.
            angle: Rotation angle in degrees.

        Returns:
            Rotated template image.
        """
        if angle == 0:
            return template
        h, w = template.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # For near-90° rotations, expand the canvas to (h, w) so the rotated
        # template fits completely without clipping.  Without this, a 90°-rotated
        # horizontal rectangle is cropped to the ORIGINAL (w×h) canvas, hiding
        # the wire leads and producing a bbox with wrong aspect ratio.
        if 75 <= abs(angle % 180) <= 105:
            out_w, out_h = h, w
            # Shift so the rotated content is centred in the new canvas
            M[0, 2] += (out_w - w) / 2
            M[1, 2] += (out_h - h) / 2
        else:
            out_w, out_h = w, h

        rotated = cv2.warpAffine(
            template, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return rotated

    def match(self, drawing: np.ndarray, template: np.ndarray) -> List[dict]:
        """Run NCC multi-scale/rotation template matching.

        Args:
            drawing: Grayscale drawing image to search in.
            template: Grayscale template pattern to search for.

        Returns:
            List of candidate dicts with keys: x, y, w, h, ncc_score, scale, angle.
        """
        dh, dw = drawing.shape[:2]
        th, tw = template.shape[:2]
        all_candidates = []

        for scale in self.scales:
            scaled_w = max(1, int(tw * scale))
            scaled_h = max(1, int(th * scale))

            if scaled_w >= dw or scaled_h >= dh:
                continue
            if scaled_w < self.min_template_px or scaled_h < self.min_template_px:
                continue

            scaled_tmpl = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

            for angle in self.angles:
                rotated = self.rotate_template(scaled_tmpl, angle)
                rh, rw = rotated.shape[:2]

                if rw >= dw or rh >= dh:
                    continue

                result = cv2.matchTemplate(drawing, rotated, cv2.TM_CCOEFF_NORMED)
                locs = np.where(result >= self.ncc_threshold)

                for pt_y, pt_x in zip(*locs):
                    # Clamp bounding box within drawing boundaries
                    x = int(np.clip(pt_x, 0, dw - rw))
                    y = int(np.clip(pt_y, 0, dh - rh))
                    all_candidates.append({
                        "x": x,
                        "y": y,
                        "w": rw,
                        "h": rh,
                        "ncc_score": float(result[pt_y, pt_x]),
                        "scale": scale,
                        "angle": angle,
                    })

        print(f"[NCCMatcher] Candidates before NMS: {len(all_candidates)}")
        filtered = self._apply_nms(all_candidates)
        print(f"[NCCMatcher] Candidates after NMS: {len(filtered)}")
        return filtered

    def _apply_nms(self, candidates: List[dict]) -> List[dict]:
        """IoU-based NMS, keeping highest-score box when overlap > threshold.

        Args:
            candidates: List of candidate dicts.

        Returns:
            Filtered list after NMS.
        """
        if not candidates:
            return []

        # Sort by score descending
        candidates = sorted(candidates, key=lambda c: c["ncc_score"], reverse=True)

        kept = []
        suppressed = [False] * len(candidates)

        for i, cand in enumerate(candidates):
            if suppressed[i]:
                continue
            kept.append(cand)
            for j in range(i + 1, len(candidates)):
                if suppressed[j]:
                    continue
                # Never suppress across orientation groups: a horizontal match
                # (angle near 0°) should not eliminate a vertical match (angle
                # near 90°) at the same location — they represent different
                # shape hypotheses and the structural filters decide later.
                a_vert = 70 <= abs(cand.get("angle", 0)) <= 110
                b_vert = 70 <= abs(candidates[j].get("angle", 0)) <= 110
                if a_vert != b_vert:
                    continue
                if self._iou(cand, candidates[j]) > self.nms_iou_threshold:
                    suppressed[j] = True

        return kept

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        """Compute IoU between two bounding boxes."""
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["w"], by1 + b["h"]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = a["w"] * a["h"]
        area_b = b["w"] * b["h"]
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area
