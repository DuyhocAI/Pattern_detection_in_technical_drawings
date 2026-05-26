import cv2
import numpy as np
from typing import List, Tuple, Callable


class Postprocessor:
    """Stage 4: Final NMS, output formatting, and visualization."""

    def final_nms(
        self,
        candidates: List[dict],
        iou_threshold: float = 0.4,
        use_union_bbox: bool = True,
    ) -> List[dict]:
        """Cluster-and-merge NMS: group overlapping boxes and keep one per cluster.

        When use_union_bbox=True (default): the output bbox is the union of all
        boxes in the cluster — good for complex templates where multi-scale
        detections should together define the component boundary.
        When use_union_bbox=False: the output bbox is taken from the highest-
        confidence candidate — better for simple templates where an offset
        duplicate should not expand the bbox beyond the best-fit box.

        Args:
            candidates: Candidate dicts with "confidence" key.
            iou_threshold: Overlap threshold for grouping (max of IoU and containment).
            use_union_bbox: Whether to expand bbox to union of cluster.

        Returns:
            List of merged detections, one per cluster.
        """
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
        n = len(candidates)
        cluster_id = list(range(n))

        def find(i):
            while cluster_id[i] != i:
                cluster_id[i] = cluster_id[cluster_id[i]]
                i = cluster_id[i]
            return i

        def union(i, j):
            cluster_id[find(i)] = find(j)

        for i in range(n):
            for j in range(i + 1, n):
                if self._overlap_ratio(candidates[i], candidates[j]) > iou_threshold:
                    union(i, j)

        clusters: dict = {}
        for i, cand in enumerate(candidates):
            root = find(i)
            clusters.setdefault(root, []).append(cand)

        merged = []
        for cluster in clusters.values():
            best = max(cluster, key=lambda c: c["confidence"])
            merged_cand = dict(best)
            if use_union_bbox:
                merged_cand["x"] = min(c["x"] for c in cluster)
                merged_cand["y"] = min(c["y"] for c in cluster)
                merged_cand["w"] = max(c["x"] + c["w"] for c in cluster) - merged_cand["x"]
                merged_cand["h"] = max(c["y"] + c["h"] for c in cluster) - merged_cand["y"]
            merged.append(merged_cand)

        return sorted(merged, key=lambda c: c["confidence"], reverse=True)

    def format_output(self, candidates: List[dict], image_shape: Tuple) -> dict:
        """Format final detections into structured output dict.

        Args:
            candidates: Filtered detection candidates.
            image_shape: Shape tuple (H, W) or (H, W, C) of the drawing image.

        Returns:
            Structured detection dict.
        """
        h, w = image_shape[:2]
        detections = []
        for cand in candidates:
            bw, bh = int(cand["w"]), int(cand["h"])
            if bw < 20 or bh < 20:
                continue
            # Skip degenerate boxes: aspect ratio outside [0.33, 3.0]
            aspect = bw / bh if bh > 0 else 0
            if aspect < 0.33 or aspect > 3.0:
                continue
            detections.append({
                "bbox": {
                    "x": int(cand["x"]),
                    "y": int(cand["y"]),
                    "w": bw,
                    "h": bh,
                },
                "confidence": round(float(cand.get("confidence", 0.0)), 2),
                "ncc_score": round(float(cand.get("ncc_score", 0.0)), 4),
                "dino_score": round(float(cand.get("dino_score", 0.0)), 4),
                "scale": round(float(cand.get("scale", 1.0)), 4),
                "angle": round(float(cand.get("angle", 0.0)), 4),
            })

        return {
            "detections": detections,
            "total_detections": len(detections),
            "image_size": {"width": int(w), "height": int(h)},
        }

    def filter_grid_clusters(
        self,
        candidates: List[dict],
        row_tol: int = 15,
        col_tol: int = 15,
        min_grid_size: int = 3,
    ) -> List[dict]:
        """Remove candidates that belong to a table row or frame line.

        When min_grid_size or more detections share the same row (y) or column (x)
        within tolerance, they are structural artifacts (BOM rows, border segments)
        rather than isolated circuit components.
        """
        if len(candidates) < min_grid_size:
            return candidates

        n = len(candidates)
        cy = [c["y"] + c["h"] // 2 for c in candidates]
        cx = [c["x"] + c["w"] // 2 for c in candidates]
        keep = [True] * n

        for i in range(n):
            row_count = sum(1 for j in range(n) if abs(cy[j] - cy[i]) <= row_tol)
            if row_count >= min_grid_size:
                keep[i] = False
                continue
            col_count = sum(1 for j in range(n) if abs(cx[j] - cx[i]) <= col_tol)
            if col_count >= min_grid_size:
                keep[i] = False

        return [c for c, k in zip(candidates, keep) if k]

    def find_table_exclusion_zones(self, drawing: np.ndarray) -> dict:
        """Detect frame border and BOM/title-block area boundaries.

        Returns:
            dict with:
              title_block_x: leftmost x of the right-frame vertical lines
                             (candidates whose right edge exceeds this are frame FPs)
              bom_start_x:   leftmost x where horizontal-line density jumps to
                             table-area levels (candidates with center beyond this
                             are BOM/title-block FPs)
        """
        H, W = drawing.shape[:2]
        binary = (drawing < 128).astype(np.uint8)

        # 1. Frame border: find rightmost long vertical line (>=40% of height)
        v_min_len = max(50, int(H * 0.40))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_min_len))
        v_line_img = cv2.erode(binary, v_kernel)
        cols_with_vlines = np.where(v_line_img.any(axis=0))[0]
        title_block_x = W
        if len(cols_with_vlines) > 0:
            right_cols = cols_with_vlines[cols_with_vlines > int(W * 0.75)]
            if len(right_cols) > 0:
                title_block_x = int(right_cols.min())

        # 2. BOM/title-block area: scan column strips for horizontal-line density jump.
        # Circuit wires produce ~10-50 rows with density > 15%; table rows produce 60+.
        # Only run this scan when a real right-frame was detected (title_block_x < W).
        # If no vertical frame lines exist (title_block_x == W), the drawing has no
        # structured BOM table, so skip the scan entirely to avoid false positives in
        # complex circuit areas.
        strip_w = 30
        bom_threshold = 60
        bom_start_x = title_block_x  # default to frame border (safe: = W if no frame)
        if title_block_x < W:
            for x0 in range(int(W * 0.80), W - strip_w, strip_w):
                strip = binary[:, x0: x0 + strip_w]
                row_density = strip.mean(axis=1)
                n_lines = int(np.sum(row_density > 0.15))
                if n_lines >= bom_threshold:
                    bom_start_x = x0
                    break

        return {"title_block_x": title_block_x, "bom_start_x": bom_start_x}

    def filter_title_block(
        self, candidates: List[dict], drawing: np.ndarray
    ) -> List[dict]:
        """Remove candidates inside the outer frame border or BOM/title-block zone.

        Two criteria:
          1. Right-edge: bbox reaches into the outer frame column
             (catches border-corner FPs whose x+w overlaps the frame line).
          2. BOM zone: candidate centre is to the right of the structured
             table area (BOM rows, company info, drawing number cells).
        """
        zones = self.find_table_exclusion_zones(drawing)
        tx = zones["title_block_x"]   # outer right frame x
        bom_x = zones["bom_start_x"]  # BOM/title-block left boundary
        margin = 10  # small margin inward from frame line

        result = []
        for c in candidates:
            right_edge = c["x"] + c["w"]
            center_x = c["x"] + c["w"] // 2
            # Exclude if bbox right edge reaches into the outer frame
            if right_edge > tx - margin:
                continue
            # Exclude if candidate centre falls inside BOM/title-block area
            if center_x >= bom_x:
                continue
            result.append(c)
        return result

    def filter_isolated(
        self,
        candidates: List[dict],
        drawing: np.ndarray,
        probe: int = 10,
        dark_threshold: float = 0.25,
    ) -> List[dict]:
        """Remove candidates whose long sides are bordered by adjacent grid lines.

        Circuit components sit in open white space; BOM/title-block cells have
        solid lines directly above and below (or left/right for vertical).
        Rejects any candidate where the strip just outside a long side has
        dark-pixel ratio > dark_threshold.
        """
        H, W = drawing.shape[:2]
        result = []
        for c in candidates:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            margin = max(2, min(w, h) // 6)

            if w >= h:  # horizontal → check top and bottom
                strips = [
                    drawing[max(0, y - probe) : y, x + margin : x + w - margin],
                    drawing[y + h : min(H, y + h + probe), x + margin : x + w - margin],
                ]
            else:  # vertical → check left and right
                strips = [
                    drawing[y + margin : y + h - margin, max(0, x - probe) : x],
                    drawing[y + margin : y + h - margin, x + w : min(W, x + w + probe)],
                ]

            isolated = True
            for s in strips:
                if s.size == 0:
                    continue
                if float(np.sum(s < 128)) / s.size > dark_threshold:
                    isolated = False
                    break

            if isolated:
                result.append(c)

        return result

    def draw_boxes(self, drawing: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw color-coded bounding boxes on the drawing image.

        Boxes are green (conf >= 0.70), amber (>= 0.55), or red (< 0.55).
        Labels show detection index and confidence score.

        Args:
            drawing: Grayscale, RGB, or RGBA image.
            detections: List of detection dicts from format_output.

        Returns:
            RGB numpy array (H, W, 3) with annotations.
        """
        output = drawing.copy()

        # Normalise to 3-channel BGR
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        elif output.shape[2] == 4:
            output = cv2.cvtColor(output, cv2.COLOR_RGBA2BGR)
        else:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        img_h, img_w = output.shape[:2]
        # Scale thickness and font to image size
        thickness = max(2, int(max(img_h, img_w) / 600))
        font_scale = max(0.4, min(0.75, max(img_h, img_w) / 2000))

        def _conf_color(conf: float) -> Tuple:
            if conf >= 0.70:
                return (40, 200, 60)    # green
            elif conf >= 0.55:
                return (30, 160, 245)   # amber-blue
            else:
                return (40, 40, 220)    # red

        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            conf = float(det.get("confidence", 0))
            color = _conf_color(conf)

            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness=thickness)

            label = f"#{idx + 1}  {conf:.2f}"
            (lw, lh), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            tag_y = max(y - 4, lh + 6)
            # Filled label background
            cv2.rectangle(
                output,
                (x, tag_y - lh - 4),
                (x + lw + 8, tag_y + bl),
                color, -1,
            )
            cv2.putText(
                output, label,
                (x + 4, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    def filter_wire_leads(
        self,
        candidates: List[dict],
        drawing: np.ndarray,
        probe: int = 24,
        min_run: int = 2,
        min_run_weak: int = 0,
        dino_bypass_threshold: float = 0.88,
    ) -> List[dict]:
        """Keep only candidates that have wire leads on their connecting sides.

        Two acceptance paths:

        1. High-confidence bypass: candidates whose DINOv2 score exceeds
           dino_bypass_threshold are accepted unconditionally — DINOv2 at
           that level is a strong semantic match, making it almost certain
           the region IS the component and not a wire artifact.

        2. Wire-lead scan: scan ALL rows across the full bbox height (not
           just the centre ±1 row).  This finds leads even when the wire
           connects at the top or bottom edge of the bbox rather than the
           centre, which was the root cause of missed detections.  Acceptance
           requires at least one side to have a run ≥ min_run.
        """
        H, W = drawing.shape[:2]
        binary = (drawing < 128).astype(np.uint8)

        def _max_run(arr: np.ndarray) -> int:
            if not arr.any():
                return 0
            best = cur = 0
            for v in arr:
                if v:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        result = []
        for c in candidates:
            # Path 1: high-confidence DINOv2 bypass
            if c.get("dino_score", 0.0) >= dino_bypass_threshold:
                result.append(c)
                continue

            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            angle = c.get("angle", 0)
            is_vertical = 70 <= abs(angle) <= 110

            passed = False
            for shrink_frac in [0.0, 0.12, 0.25, 0.38]:
                sx = int(w * shrink_frac)
                sy = int(h * shrink_frac)
                bx = x + sx
                by = y + sy
                bw = w - 2 * sx
                bh = h - 2 * sy
                if bw < 8 or bh < 4:
                    continue

                left_best = right_best = 0
                if not is_vertical:
                    # Scan ALL rows across bbox height (not just centre ±1).
                    # Wires can connect at any point along the short edge.
                    for row in range(max(0, by), min(H, by + bh)):
                        lslice = binary[row, max(0, bx - probe) : bx]
                        rslice = binary[row, bx + bw : min(W, bx + bw + probe)]
                        left_best  = max(left_best,  _max_run(lslice))
                        right_best = max(right_best, _max_run(rslice))
                else:
                    # Scan ALL columns across bbox width.
                    for col in range(max(0, bx), min(W, bx + bw)):
                        tslice = binary[max(0, by - probe) : by, col]
                        bslice = binary[by + bh : min(H, by + bh + probe), col]
                        left_best  = max(left_best,  _max_run(tslice))
                        right_best = max(right_best, _max_run(bslice))

                strong = max(left_best, right_best)
                weak   = min(left_best, right_best)
                if strong >= min_run and weak >= min_run_weak:
                    passed = True
                    break

            if passed:
                result.append(c)

        return result

    def filter_wire_passthrough(
        self,
        candidates: List[dict],
        drawing: np.ndarray,
        passthrough_threshold: float = 0.60,
    ) -> List[dict]:
        """Remove candidates where a straight wire runs through the bbox body.

        A component body (resistor rectangle) has a white interior — the
        wire enters one terminal, the body contains the symbol, and the wire
        exits the other terminal.  A wire segment or T-junction has a
        continuous dark line running from one side straight through to the
        other without any interior gap.

        For horizontal candidates: if any row in the centre third of the
        bbox height contains a dark run spanning ≥ passthrough_threshold of
        the inner width, the candidate is rejected as a wire pass-through.
        For vertical: same logic on centre columns.
        """
        H, W = drawing.shape[:2]
        binary = (drawing < 128).astype(np.uint8)

        def _max_run(arr: np.ndarray) -> int:
            if not arr.any():
                return 0
            best = cur = 0
            for v in arr:
                if v:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        result = []
        for c in candidates:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            angle = c.get("angle", 0)
            is_vertical = 70 <= abs(angle) <= 110
            margin = max(2, min(w, h) // 6)

            passthrough = False
            if not is_vertical:
                col_from = x + margin
                col_to   = x + w - margin
                inner_w  = col_to - col_from
                row_from = y + h // 3
                row_to   = y + 2 * h // 3
                # Check 1: horizontal dark run through the center third of height.
                if inner_w >= 4 and row_from < row_to:
                    for row in range(max(0, row_from), min(H, row_to)):
                        inner = binary[row, max(0, col_from) : min(W, col_to)]
                        if inner.size == 0:
                            continue
                        if _max_run(inner) / inner.size >= passthrough_threshold:
                            passthrough = True
                            break
                # Check 2: continuous vertical dark run through the CENTRE half of the
                # inner columns (avoids the box-border vertical lines which fall within
                # margin distance of the bbox edge but outside the centre half).
                # T-junctions have wires running through the body; a plain rectangle has
                # only a white interior, so no column has a long continuous dark run.
                if not passthrough and h > 0:
                    v_col_from = x + w // 4
                    v_col_to   = x + 3 * w // 4
                    if v_col_from < v_col_to:
                        for col in range(max(0, v_col_from), min(W, v_col_to)):
                            col_slice = binary[max(0, y) : min(H, y + h), col]
                            if col_slice.size > 0 and _max_run(col_slice) / col_slice.size >= passthrough_threshold:
                                passthrough = True
                                break
            else:
                row_from = y + margin
                row_to   = y + h - margin
                inner_h  = row_to - row_from
                col_from = x + w // 3
                col_to   = x + 2 * w // 3
                if inner_h >= 4 and col_from < col_to:
                    for col in range(max(0, col_from), min(W, col_to)):
                        inner = binary[max(0, row_from) : min(H, row_to), col]
                        if inner.size == 0:
                            continue
                        if _max_run(inner) / inner.size >= passthrough_threshold:
                            passthrough = True
                            break

            if not passthrough:
                result.append(c)

        return result

    def filter_rect_borders(
        self,
        candidates: List[dict],
        drawing: np.ndarray,
        border_run_ratio: float = 0.55,
    ) -> List[dict]:
        """Keep only candidates that show symmetric rectangular borders (top + bottom).

        Real schematic component symbols (resistors, capacitors) have a rectangular
        outline with a visible horizontal border at ~20-35% and ~65-80% of the bbox
        height.  Wire T-junctions and corners have only ONE horizontal line (either
        near the top or bottom, asymmetric), so this filter removes them.

        Designed for use on notes/legend-area candidates where the wire-lead filter
        is intentionally not applied.
        """
        H, W = drawing.shape[:2]
        binary = (drawing < 128).astype(np.uint8)

        def _max_run(arr: np.ndarray) -> int:
            if not arr.any():
                return 0
            best = cur = 0
            for v in arr:
                if v:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        def _zone_has_border(x, y_start, y_end, w, threshold):
            for row in range(max(0, y_start), min(H, y_end + 1)):
                line = binary[row, max(0, x) : min(W, x + w)]
                if line.size > 0 and _max_run(line) / w >= threshold:
                    return True
            return False

        result = []
        for c in candidates:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            angle = c.get("angle", 0)
            is_vertical = 70 <= abs(angle) <= 110

            if not is_vertical:
                top_start = y + int(h * 0.15)
                top_end   = y + int(h * 0.35)
                bot_start = y + int(h * 0.65)
                bot_end   = y + int(h * 0.85)
                has_top = _zone_has_border(x, top_start, top_end, w, border_run_ratio)
                has_bot = _zone_has_border(x, bot_start, bot_end, w, border_run_ratio)
                if has_top and has_bot:
                    result.append(c)
            else:
                left_start = x + int(w * 0.15)
                left_end   = x + int(w * 0.35)
                right_start = x + int(w * 0.65)
                right_end   = x + int(w * 0.85)

                def _col_zone_has_border(col_s, col_e, threshold):
                    for col in range(max(0, col_s), min(W, col_e + 1)):
                        line = binary[max(0, y) : min(H, y + h), col]
                        if line.size > 0 and _max_run(line) / h >= threshold:
                            return True
                    return False

                has_left  = _col_zone_has_border(left_start, left_end, border_run_ratio)
                has_right = _col_zone_has_border(right_start, right_end, border_run_ratio)
                if has_left and has_right:
                    result.append(c)

        return result

    def filter_junction_dots(
        self,
        candidates: List[dict],
        drawing: np.ndarray,
        bbox_margin: int = 3,
        min_blob_area: int = 15,
        max_blob_ar: float = 2.5,
        min_blob_fill: float = 0.40,
    ) -> List[dict]:
        """Reject candidates that contain a junction dot inside their bounding box.

        Circuit junction nodes carry a small filled circle at wire crossings.
        When NCC matches a junction-node region, that dot appears as a compact
        dark blob inside the detected bbox.  Real component symbols (resistors)
        have only thin-line structure (rectangle outline, wire leads) — no
        compact filled blobs.

        Detection: find connected components inside the bbox (after skipping the
        border-line strip) and check for any component that is:
          - large enough to be a dot  (area ≥ min_blob_area)
          - roughly equidimensional   (aspect ratio ≤ max_blob_ar)
          - densely filled            (area / bounding-rect ≥ min_blob_fill)

        Args:
            bbox_margin:   Pixels to skip from each edge of the bbox before
                           looking for blobs (avoids the component border lines).
            min_blob_area: Minimum connected-component area in pixels.
            max_blob_ar:   Maximum width/height ratio (or inverse) of the blob
                           bounding rect; keeps roughly circular shapes only.
            min_blob_fill: Minimum fill ratio (area / bounding-rect area);
                           rejects elongated or sparse shapes.
        """
        H, W = drawing.shape[:2]
        binary = (drawing < 128).astype(np.uint8)

        result = []
        for c in candidates:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            m = bbox_margin
            y0 = max(0, y + m); y1_c = min(H, y + h - m)
            x0 = max(0, x + m); x1_c = min(W, x + w - m)
            if y1_c <= y0 or x1_c <= x0:
                result.append(c)
                continue

            crop = binary[y0:y1_c, x0:x1_c]
            n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                crop, connectivity=8
            )

            has_dot = False
            for i in range(1, n_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                cw   = int(stats[i, cv2.CC_STAT_WIDTH])
                ch   = int(stats[i, cv2.CC_STAT_HEIGHT])
                if area < min_blob_area or cw < 1 or ch < 1:
                    continue
                ar   = cw / ch
                fill = area / (cw * ch)
                if (1.0 / max_blob_ar) <= ar <= max_blob_ar and fill >= min_blob_fill:
                    has_dot = True
                    break

            if not has_dot:
                result.append(c)

        return result

    def filter_rect_integrity(
        self,
        candidates: List[dict],
        drawing: np.ndarray,
        border_run_ratio: float = 0.50,
        top_zone: tuple = (0.10, 0.40),
        bot_zone: tuple = (0.60, 0.92),
        side_run_min: int = 2,
        dino_bypass_threshold: float = 0.89,
    ) -> List[dict]:
        """Reject candidates that lack the basic rectangular structure of a component.

        A well-formed component symbol (resistor) has symmetric rectangular borders:
        both a top horizontal line and a bottom horizontal line.  Two degenerate
        artefact patterns are rejected:

        1. *Only-top artifact*: a single horizontal border in the top zone with NO
           matching border in the bottom zone.  This catches L-junction FPs whose
           top-line matches the template's top border but whose bottom is absent.

        2. *Empty-bus artifact*: a single horizontal border in the bottom zone with
           ZERO dark side-line content in the rows above it.  A real component
           (even one that is partially cropped by the bbox) still shows its left and
           right vertical side lines (≥ side_run_min dark pixels in a row).  A bare
           bus-wire section has nothing above the horizontal line.

        Candidates with two visible borders, or with one border and visible side
        lines above, are kept.  Vertical candidates (rotated ~90°) are skipped.

        High-confidence DINOv2 bypass: candidates with dino_score ≥
        dino_bypass_threshold pass unconditionally — at that similarity level the
        semantic match overrides the structural heuristic.

        Args:
            border_run_ratio:       Minimum fraction of bbox width for a row to count
                                    as a "border" (significant horizontal dark run).
            top_zone:               (lo, hi) fractions of bbox height for the top zone.
            bot_zone:               (lo, hi) fractions of bbox height for the bottom zone.
            side_run_min:           Minimum dark-run length in a row above the detected
                                    bottom border for the component side-lines to be
                                    considered present.
            dino_bypass_threshold:  DINOv2 cosine score above which the structural
                                    check is skipped entirely.
        """
        H, W = drawing.shape[:2]
        binary = (drawing < 128).astype(np.uint8)

        def _max_run(arr: np.ndarray) -> int:
            if not arr.any():
                return 0
            best = cur = 0
            for v in arr:
                if v:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        result = []
        for c in candidates:
            if c.get("dino_score", 0.0) >= dino_bypass_threshold:
                result.append(c)
                continue

            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            angle = c.get("angle", 0)
            if 70 <= abs(angle) <= 110:
                result.append(c)
                continue

            top_lo = y + int(h * top_zone[0])
            top_hi = y + int(h * top_zone[1])
            bot_lo = y + int(h * bot_zone[0])
            bot_hi = y + int(h * bot_zone[1])

            has_top = False
            has_bot = False
            for row in range(max(0, top_lo), min(H, top_hi + 1)):
                line = binary[row, max(0, x) : min(W, x + w)]
                if line.size > 0 and _max_run(line) / w >= border_run_ratio:
                    has_top = True
                    break
            for row in range(max(0, bot_lo), min(H, bot_hi + 1)):
                line = binary[row, max(0, x) : min(W, x + w)]
                if line.size > 0 and _max_run(line) / w >= border_run_ratio:
                    has_bot = True
                    break

            if has_top and has_bot:
                result.append(c)
                continue

            if has_top and not has_bot:
                # Only-top artifact: top border with no matching bottom → reject
                continue

            if has_bot and not has_top:
                # Bottom border only — check for side lines above it.
                # Find the topmost row of the bottom border group.
                border_row = bot_lo
                for row in range(max(0, bot_lo), min(H, bot_hi + 1)):
                    line = binary[row, max(0, x) : min(W, x + w)]
                    if line.size > 0 and _max_run(line) / w >= border_run_ratio:
                        border_row = row
                        break
                # Check 1: side lines above the bottom border
                has_sides = False
                for row in range(max(0, y), border_row):
                    line = binary[row, max(0, x) : min(W, x + w)]
                    if _max_run(line) >= side_run_min:
                        has_sides = True
                        break
                if has_sides:
                    result.append(c)
                # else: empty-bus artifact → reject
                continue

            # No border found in either zone → likely too small or unusual → keep
            result.append(c)

        return result

    def filter_chamfer_shape(
        self,
        candidates: List[dict],
        drawing: np.ndarray,
        template: np.ndarray,
        max_chamfer: float = 3.0,
        canny_lo: int = 30,
        canny_hi: int = 100,
    ) -> List[dict]:
        """Reject candidates whose bbox region does not match the template's edge structure.

        Chamfer distance measures how well the template's edge skeleton aligns with
        the drawing region's edge skeleton.  It is specifically suited to binary
        line-art drawings because it does not depend on intensity values — only on
        edge placement.

        For each candidate:
          1. Resize (and optionally rotate) the template to match the bbox dimensions.
          2. Extract Canny edges from both the template and the drawing region.
          3. Compute the distance transform of the drawing edges (each pixel stores
             its distance to the nearest edge).
          4. Sample the distance transform at every template-edge pixel location.
          5. The mean distance is the Chamfer score.  Low score = good structural match.

        Real components (IEC rectangle) have Chamfer ≈ 0.7–2.0 at the scales used.
        Wire junctions, L-corners, and other FPs yield Chamfer > 5.0 because their
        edge structure is fundamentally different from the full rectangle + lead template.

        Args:
            max_chamfer:  Mean pixel distance threshold.  Candidates above this are
                          rejected as structural FPs.
            canny_lo/hi:  Canny edge thresholds (applied to both template and region).
        """
        H, W = drawing.shape[:2]
        draw_gray = drawing if drawing.ndim == 2 else cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        draw_gray = draw_gray.astype(np.uint8)
        tmpl_gray = template if template.ndim == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        tmpl_gray = tmpl_gray.astype(np.uint8)

        result = []
        for c in candidates:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            angle = c.get("angle", 0)
            y1 = max(0, y); y2 = min(H, y + h)
            x1 = max(0, x); x2 = min(W, x + w)
            region = draw_gray[y1:y2, x1:x2]
            rh, rw = region.shape[:2]
            if rh < 4 or rw < 4:
                result.append(c)
                continue

            # Rotate template before resizing so the aspect ratio matches the bbox
            is_vert = 70 <= abs(angle) <= 110
            if is_vert:
                tmpl_s = cv2.resize(
                    cv2.rotate(tmpl_gray, cv2.ROTATE_90_CLOCKWISE), (rw, rh)
                )
            else:
                tmpl_s = cv2.resize(tmpl_gray, (rw, rh))

            tmpl_edges  = cv2.Canny(tmpl_s, canny_lo, canny_hi)
            region_edges = cv2.Canny(region, canny_lo, canny_hi)

            # Bidirectional Chamfer: average of template→region and region→template.
            # Using only template→region inflates the score when the bbox includes
            # wire-lead stubs that have no counterpart in the template (e.g. R5/R9).
            # The symmetric mean is more robust to minor structural asymmetries.
            dt_r = cv2.distanceTransform(
                (255 - region_edges).astype(np.uint8), cv2.DIST_L2, 5
            )
            pts = np.where(tmpl_edges > 0)
            if len(pts[0]) == 0:
                result.append(c)
                continue
            rows = np.clip(pts[0], 0, rh - 1)
            cols = np.clip(pts[1], 0, rw - 1)
            t2r = float(np.mean(dt_r[rows, cols]))

            dt_t = cv2.distanceTransform(
                (255 - tmpl_edges).astype(np.uint8), cv2.DIST_L2, 5
            )
            pts2 = np.where(region_edges > 0)
            if len(pts2[0]) == 0:
                chamfer_dist = t2r
            else:
                rows2 = np.clip(pts2[0], 0, rh - 1)
                cols2 = np.clip(pts2[1], 0, rw - 1)
                r2t = float(np.mean(dt_t[rows2, cols2]))
                chamfer_dist = (t2r + r2t) / 2

            c_out = dict(c)
            c_out["chamfer_dist"] = round(chamfer_dist, 3)
            if chamfer_dist <= max_chamfer:
                result.append(c_out)

        return result

    def filter_neighborhood_complexity(
        self,
        candidates: List[dict],
        drawing: np.ndarray,
        expand_ratio: float = 1.0,
        max_edge_density: float = 0.05,
    ) -> List[dict]:
        """Remove candidates whose surrounding ring has too many Canny edges.

        Standalone components (resistors) sit in clean white space; components
        embedded inside complex symbols (bridge rectifiers) have many adjacent
        edges in the ring around the bounding box.

        Args:
            expand_ratio: Width of the outer ring in units of the bbox dimensions.
            max_edge_density: Canny-edge fraction in the ring above which the
                              candidate is rejected.
        """
        H, W = drawing.shape[:2]
        gray = drawing if drawing.ndim == 2 else cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray.astype(np.uint8), 30, 100)
        result = []
        for c in candidates:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            exp_x = max(5, int(w * expand_ratio))
            exp_y = max(5, int(h * expand_ratio))
            x1 = max(0, x - exp_x)
            y1 = max(0, y - exp_y)
            x2 = min(W, x + w + exp_x)
            y2 = min(H, y + h + exp_y)

            outer = edges[y1:y2, x1:x2].copy()
            # Blank the inner detection box so we measure only the surrounding ring
            inner_y1 = max(0, y - y1)
            inner_x1 = max(0, x - x1)
            inner_y2 = inner_y1 + h
            inner_x2 = inner_x1 + w
            outer[inner_y1:inner_y2, inner_x1:inner_x2] = 0

            outer_pixels = outer.size - w * h
            if outer_pixels <= 0:
                result.append(c)
                continue

            edge_density = float(np.count_nonzero(outer)) / outer_pixels
            if edge_density <= max_edge_density:
                result.append(c)
        return result

    @staticmethod
    def _overlap_ratio(a: dict, b: dict) -> float:
        """Max of IoU and containment ratio (intersection / area of smaller box).
        Handles multi-scale duplicates: a small box fully inside a large one gets merged.
        """
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

        if inter_area == 0:
            return 0.0

        area_a = a["w"] * a["h"]
        area_b = b["w"] * b["h"]
        union_area = area_a + area_b - inter_area
        min_area = min(area_a, area_b)

        iou = inter_area / union_area if union_area > 0 else 0.0
        containment = inter_area / min_area if min_area > 0 else 0.0
        return max(iou, containment)

    @staticmethod
    def _is_rgb(img: np.ndarray) -> bool:
        """Heuristic: check if a 3-channel image is likely RGB (not BGR)."""
        # Cannot determine with certainty; assume caller passes BGR from OpenCV
        return False

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        """IoU between two candidate bbox dicts."""
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
