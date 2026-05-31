"""Evaluate the pipeline on the 6 assessment drawings.

Template pairings (established in session 5):
  Drawing 1 (resistor circuit)    + test_pattern_d1_2.png  -> baseline 10 dets
  Drawing 4 (bridge rectifier)    + test_pattern_d1.png    -> baseline  4 dets
  Drawings 2,3,5,6                + test_pattern_d1_2.png  -> expected  0 dets

DINODense Pass C was shown to be dormant (ON==OFF everywhere) in session 6.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import PatternDetectionPipeline  # noqa: E402

DRAWINGS_DIR = r"D:\Sotatek_Assessment\drawings"

# Primary evaluation: resistor template vs all 6 drawings
RESISTOR_PAT = os.path.join(DRAWINGS_DIR, "test_pattern_d1_2.png")
# Secondary: bridge rectifier template vs drawing 4
BRIDGE_PAT = os.path.join(DRAWINGS_DIR, "test_pattern_d1.png")

BASELINES = {
    ("resistor", "1.png"): 10,
    ("resistor", "4.png"): None,   # not expected
    ("bridge",   "4.png"): 4,
}


def main():
    pipe = PatternDetectionPipeline()
    summary = {}

    print("\n===== Resistor template (test_pattern_d1_2.png) =====")
    for i in range(1, 7):
        dpath = os.path.join(DRAWINGS_DIR, f"{i}.png")
        if not os.path.exists(dpath):
            print(f"  SKIP {i}.png (not found)")
            continue
        r = pipe.detect_auto(RESISTOR_PAT, dpath, return_visualization=False)
        n = r["total_detections"]
        summary[("resistor", f"{i}.png")] = n
        base = BASELINES.get(("resistor", f"{i}.png"))
        tag = f"  (baseline {base})" if base is not None else ""
        print(f"  {i}.png: {n}{tag}")

    print("\n===== Bridge-rectifier template (test_pattern_d1.png) vs drawing 4 =====")
    dpath = os.path.join(DRAWINGS_DIR, "4.png")
    r = pipe.detect_auto(BRIDGE_PAT, dpath, return_visualization=False)
    n = r["total_detections"]
    print(f"  4.png: {n}  (baseline 4)")

    print("\n===== SUMMARY =====")
    for i in range(1, 7):
        k = ("resistor", f"{i}.png")
        if k in summary:
            print(f"  resistor + {i}.png: {summary[k]}")


if __name__ == "__main__":
    main()
