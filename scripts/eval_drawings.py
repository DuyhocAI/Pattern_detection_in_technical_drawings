"""Compare DINODense ON vs OFF on the 6 real drawings using the resistor
(simple) template — this is the path where DINODense Pass C actually activates.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import PatternDetectionPipeline  # noqa: E402

DRAWINGS_DIR = r"D:\Sotatek_Assessment\drawings"


def main():
    pattern = os.path.join(DRAWINGS_DIR, "test_1.png")  # resistor simple template
    drawings = [os.path.join(DRAWINGS_DIR, f"{i}.png") for i in range(1, 7)]

    if not os.path.exists(pattern):
        print(f"MISSING pattern: {pattern}")
        return

    summary = {}
    for mode_name, flag in [("dense_ON", True), ("dense_OFF", False)]:
        print(f"\n########## MODE: {mode_name} ##########")
        pipe = PatternDetectionPipeline(config={"use_dino_dense": flag})
        for dpath in drawings:
            name = os.path.basename(dpath)
            if not os.path.exists(dpath):
                print(f"SKIP {name}")
                continue
            try:
                result = pipe.detect_auto(pattern, dpath, return_visualization=False)
                n = result.get("total_detections")
            except Exception as e:
                n = f"ERR:{e}"
            summary[(mode_name, name)] = n
            print(f"  {name}: {n}")

    print("\n========== SUMMARY (resistor template) ==========")
    for i in range(1, 7):
        name = f"{i}.png"
        on = summary.get(("dense_ON", name))
        off = summary.get(("dense_OFF", name))
        flag = "  <-- DIFF" if on != off else ""
        print(f"  {name}:  ON={on}  OFF={off}{flag}")


if __name__ == "__main__":
    main()
