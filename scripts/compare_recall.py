"""Compare VLM filter modes on the user's schematic (both recall-boost ON):
  - whitelist  : keep only VLM-labelled 'resistor'  (high precision, low recall)
  - reject-only: drop only confident non-target classes (high recall)
Saves annotated PNGs for visual comparison.
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import PatternDetectionPipeline  # noqa: E402

DRAWING = os.environ.get("DRAWING", r"D:\Sotatek_Assessment\drawings\1.png")
PATTERN = r"D:\Sotatek_Assessment\drawings\test_2.png"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "debug_output")


def run(reject_only: bool, tag: str):
    print(f"\n########## reject_only={reject_only} (boost+VLM) ##########")
    pipe = PatternDetectionPipeline(config={
        "use_vlm": True,
        "vlm_recall_boost": True,
        "vlm_reject_only": reject_only,
        "vlm_symbol_name": "a resistor (zigzag or plain rectangle)",
    })
    t0 = time.time()
    result = pipe.detect_auto(PATTERN, DRAWING, return_visualization=True)
    dt = time.time() - t0
    n = result["total_detections"]
    outpath = os.path.join(OUT, f"mode_{tag}.png")
    cv2.imwrite(outpath, result["visualization"])
    print(f"[CMP] {tag}: {n} detections in {dt:.1f}s -> {outpath}")
    return n


def main():
    if not os.path.exists(DRAWING) or not os.path.exists(PATTERN):
        print("MISSING input files"); return
    n_white = run(False, "whitelist")
    n_reject = run(True, "rejectonly")
    print(f"\n========== RESULT ==========")
    print(f"  whitelist   : {n_white}")
    print(f"  reject-only : {n_reject}")
    print(f"  recall delta: +{n_reject - n_white}")


if __name__ == "__main__":
    main()
