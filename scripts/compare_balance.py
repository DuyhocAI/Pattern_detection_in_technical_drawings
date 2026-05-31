"""Find the precision/recall sweet spot: reject-only WITHOUT recall-boost,
compared against the two extremes already measured."""
import os, sys, time
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import PatternDetectionPipeline  # noqa

DRAWING = os.environ.get("DRAWING", r"D:\Sotatek_Assessment\drawings\1.png")
PATTERN = r"D:\Sotatek_Assessment\drawings\test_2.png"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug_output")


def run(boost, reject, tag):
    print(f"\n##### boost={boost} reject_only={reject} #####")
    pipe = PatternDetectionPipeline(config={
        "use_vlm": True, "vlm_recall_boost": boost, "vlm_reject_only": reject,
        "vlm_symbol_name": "a resistor (zigzag or plain rectangle)"})
    t0 = time.time()
    r = pipe.detect_auto(PATTERN, DRAWING, return_visualization=True)
    cv2.imwrite(os.path.join(OUT, f"bal_{tag}.png"), r["visualization"])
    print(f"[CMP] {tag}: {r['total_detections']} dets in {time.time()-t0:.1f}s")
    return r["total_detections"]


def main():
    a = run(False, True, "noboost_reject")   # the missing data point
    print(f"\n========== balance: noboost+reject-only = {a} ==========")


if __name__ == "__main__":
    main()
