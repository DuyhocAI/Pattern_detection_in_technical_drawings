"""Quick evaluation harness: run the pipeline on the example pairs and print
detection counts. Compares the DINODenseMatcher path on vs off.

NOTE: PatternDetectionPipeline.detect_auto signature is (pattern_input, drawing_input).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import PatternDetectionPipeline  # noqa: E402


def run(pipe, name, ppath, dpath):
    result = pipe.detect_auto(ppath, dpath, return_visualization=False)
    dets = result.get("detections", [])
    print(f"\n=== {name}: {result.get('total_detections')} detections ===")
    for d in dets:
        bbox = d.get("bbox", (d.get("x"), d.get("y"), d.get("w"), d.get("h")))
        print(f"    bbox={bbox} conf={d.get('confidence')} dino={d.get('dino_score')} "
              f"angle={d.get('angle')}")
    return result.get("total_detections")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ex = os.path.join(root, "examples")
    pairs = [
        ("example1", os.path.join(ex, "example1_pattern.png"), os.path.join(ex, "example1_drawing.png")),
        ("example2", os.path.join(ex, "example2_pattern.png"), os.path.join(ex, "example2_drawing.png")),
    ]

    summary = {}
    for mode_name, flag in [("dense_ON", True), ("dense_OFF", False)]:
        print(f"\n########## MODE: {mode_name} (use_dino_dense={flag}) ##########")
        pipe = PatternDetectionPipeline(config={"use_dino_dense": flag})
        for name, ppath, dpath in pairs:
            if not (os.path.exists(dpath) and os.path.exists(ppath)):
                print(f"SKIP {name}: missing files")
                continue
            n = run(pipe, name, ppath, dpath)
            summary[(mode_name, name)] = n

    print("\n\n========== SUMMARY ==========")
    for (mode_name, name), n in summary.items():
        print(f"  {mode_name:10s} {name:10s} -> {n}")


if __name__ == "__main__":
    main()
