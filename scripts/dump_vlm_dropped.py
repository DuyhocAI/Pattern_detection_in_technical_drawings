"""Dump every candidate the VLM DROPS (and keeps) on the user's schematic, so we
can eyeball how many genuine resistors are being killed vs. correctly rejected.

Runs the full recall-boost pipeline up to the VLM stage, then classifies each
candidate and writes a labelled crop into debug_output/vlm_dropped/.
Filename: <KEEP|DROP>_<idx>_x<..>_y<..>_conf<..>_<vlmlabel>.png
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
                   "debug_output", "vlm_dropped")
os.makedirs(OUT, exist_ok=True)


def main():
    # Build the candidate set exactly as the recall-boost pipeline does, but stop
    # before the VLM so we can inspect every candidate it will judge.
    pipe = PatternDetectionPipeline(config={
        "use_vlm": False,            # we call the VLM manually below
        "vlm_recall_boost": True,    # but with boosted recall candidate set
    })
    pipe.vlm_recall_boost = True

    drw = pipe.preprocessor.preprocess(DRAWING)["processed"]
    pat = pipe.preprocessor.preprocess(PATTERN)["processed"]

    # Run detect_auto with VLM off but recall-boost on -> these are the candidates
    # that reach Stage 3 (before VLM filtering and final NMS).
    res = pipe.detect_auto(PATTERN, DRAWING, return_visualization=False)
    cands = [{"x": d["bbox"]["x"], "y": d["bbox"]["y"], "w": d["bbox"]["w"],
              "h": d["bbox"]["h"], "angle": d.get("angle", 0),
              "confidence": d.get("confidence", 1.0)} for d in res["detections"]]
    print(f"[DUMP] recall-boost (VLM off) yields {len(cands)} post-NMS detections")
    print("[DUMP] NOTE: to see ALL pre-VLM candidates, inspect the verbose log;")
    print("[DUMP] here we classify the post-NMS set so crops are de-duplicated.")

    vlm = VLMVerifier()
    tcls = vlm.classify_template(pat)

    keep = drop = 0
    for i, c in enumerate(cands):
        crop = vlm._crop_for_vlm(drw, c)
        label, raw = vlm._classify_one(crop)
        is_keep = (label == tcls)
        tag = "KEEP" if is_keep else "DROP"
        keep += is_keep
        drop += (not is_keep)
        fn = f"{tag}_{i:02d}_x{c['x']}_y{c['y']}_conf{c['confidence']:.2f}_{label}.png"
        cv2.imwrite(os.path.join(OUT, fn),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if crop.ndim == 3 else crop)
        print(f"  {tag} ({c['x']},{c['y']}) conf={c['confidence']:.2f} -> {label}")

    print(f"\n[DUMP] template class={tcls}  KEEP={keep}  DROP={drop}")
    print(f"[DUMP] crops in {OUT}")


if __name__ == "__main__":
    main()
