"""End-to-end comparison on the user's actual schematic: VLM off vs on.

Saves two annotated PNGs so the FP reduction is directly visible.
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import PatternDetectionPipeline  # noqa: E402

DRAWING = r"C:\Users\PC\.claude\image-cache\34c9f4cb-5a8b-4329-b243-cbaa9f386643\3.png"
PATTERN = r"C:\Users\PC\.claude\image-cache\34c9f4cb-5a8b-4329-b243-cbaa9f386643\2.png"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "debug_output")


def run(use_vlm: bool, tag: str):
    print(f"\n########## use_vlm={use_vlm} ##########")
    pipe = PatternDetectionPipeline(config={
        "use_vlm": use_vlm,
        "vlm_symbol_name": "a resistor (zigzag or plain rectangle)",
    })
    t0 = time.time()
    result = pipe.detect_auto(PATTERN, DRAWING, return_visualization=True)
    dt = time.time() - t0
    n = result["total_detections"]
    outpath = os.path.join(OUT, f"compare_{tag}.png")
    cv2.imwrite(outpath, result["visualization"])
    print(f"[CMP] {tag}: {n} detections in {dt:.1f}s -> {outpath}")
    # Print class breakdown if VLM ran
    classes = {}
    for d in result["detections"]:
        cl = d.get("vlm_class")
        if cl:
            classes[cl] = classes.get(cl, 0) + 1
    if classes:
        print(f"[CMP] {tag} vlm_class kept: {classes}")
    return n


def main():
    if not os.path.exists(DRAWING):
        print(f"MISSING drawing: {DRAWING}")
        return
    n_off = run(False, "vlm_off")
    n_on = run(True, "vlm_on")
    print(f"\n========== RESULT ==========")
    print(f"  VLM off: {n_off} detections")
    print(f"  VLM on : {n_on} detections")
    print(f"  removed by VLM: {n_off - n_on}")


if __name__ == "__main__":
    main()
