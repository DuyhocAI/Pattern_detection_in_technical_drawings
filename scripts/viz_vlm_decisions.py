"""Draw the VLM's KEEP (green) vs DROP (red) decisions on the full drawing so a
human can judge how many genuine resistors are being killed by the 2B model.

Green box = VLM kept (labelled resistor). Red box = VLM dropped (labelled other).
Red label shows what the VLM thought it was.
"""
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import PatternDetectionPipeline  # noqa: E402
from src.vlm_verifier import VLMVerifier  # noqa: E402

DRAWING = os.environ.get("DRAWING", r"D:\Sotatek_Assessment\drawings\1.png")
PATTERN = r"D:\Sotatek_Assessment\drawings\test_2.png"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "debug_output", "vlm_decisions.png")


def main():
    pipe = PatternDetectionPipeline(config={"use_vlm": False, "vlm_recall_boost": True})
    pipe.vlm_recall_boost = True

    drw_color = cv2.imread(DRAWING)
    drw = pipe.preprocessor.preprocess(DRAWING)["processed"]
    pat = pipe.preprocessor.preprocess(PATTERN)["processed"]

    res = pipe.detect_auto(PATTERN, DRAWING, return_visualization=False)
    cands = [{"x": d["bbox"]["x"], "y": d["bbox"]["y"], "w": d["bbox"]["w"],
              "h": d["bbox"]["h"], "angle": d.get("angle", 0),
              "confidence": d.get("confidence", 1.0)} for d in res["detections"]]

    vlm = VLMVerifier()
    tcls = vlm.classify_template(pat)

    keep = drop = 0
    for c in cands:
        crop = vlm._crop_for_vlm(drw, c)
        label, _ = vlm._classify_one(crop)
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        if label == tcls:
            cv2.rectangle(drw_color, (x, y), (x + w, y + h), (0, 170, 0), 2)
            keep += 1
        else:
            cv2.rectangle(drw_color, (x, y), (x + w, y + h), (0, 0, 220), 2)
            cv2.putText(drw_color, label[:4], (x, max(0, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
            drop += 1

    cv2.imwrite(OUT, drw_color)
    print(f"[VIZ] template={tcls}  KEEP(green)={keep}  DROP(red)={drop}")
    print(f"[VIZ] saved {OUT}")


if __name__ == "__main__":
    main()
