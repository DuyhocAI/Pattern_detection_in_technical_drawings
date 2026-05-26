"""Debug and threshold-tuning script for pattern detection pipeline."""

import argparse
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import PatternDetectionPipeline


def find_example_pairs(examples_dir: str) -> list:
    """Find (pattern, drawing) file pairs in examples_dir."""
    pairs = {}
    for fname in sorted(os.listdir(examples_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            continue
        name_no_ext = os.path.splitext(fname)[0]
        parts = name_no_ext.split("_")
        if len(parts) < 2:
            continue
        key = parts[0]
        tag = "_".join(parts[1:])
        pairs.setdefault(key, {})[tag] = os.path.join(examples_dir, fname)

    result = []
    for key in sorted(pairs):
        imgs = pairs[key]
        if "pattern" in imgs and "drawing" in imgs:
            result.append((key, imgs["pattern"], imgs["drawing"]))
    return result


def save_visualization(output_dir: str, name: str, vis: np.ndarray):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Tune thresholds for pattern detection pipeline.")
    parser.add_argument("--examples_dir", default="examples")
    parser.add_argument("--output_dir", default="debug_output")
    parser.add_argument("--max_pairs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Only test NCC=0.55, cosine=0.75")
    args = parser.parse_args()

    if not os.path.isdir(args.examples_dir):
        print(f"[ERROR] Examples directory not found: {args.examples_dir}")
        sys.exit(1)

    pairs = find_example_pairs(args.examples_dir)
    if not pairs:
        print(f"[ERROR] No pattern/drawing pairs found in {args.examples_dir}")
        sys.exit(1)

    pairs = pairs[: args.max_pairs]

    if args.quick:
        ncc_thresholds = [0.55]
        cosine_thresholds = [0.75]
    else:
        ncc_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
        cosine_thresholds = [0.65, 0.70, 0.75, 0.80]

    # Header
    header = f"{'Example':<12} {'NCC':>6} {'Cosine':>8} {'#Det':>6} {'Time(s)':>9}"
    print("\n" + header)
    print("-" * len(header))

    pipeline = PatternDetectionPipeline()

    total_runs = len(pairs) * len(ncc_thresholds) * len(cosine_thresholds)
    run_idx = 0

    for key, pattern_path, drawing_path in pairs:
        for ncc_t in ncc_thresholds:
            for cos_t in cosine_thresholds:
                run_idx += 1
                print(f"\r[{run_idx}/{total_runs}] Running...", end="", flush=True)

                pipeline.update_thresholds(ncc_threshold=ncc_t, cosine_threshold=cos_t)
                t0 = time.time()
                try:
                    result = pipeline.detect(pattern_path, drawing_path, return_visualization=True)
                except Exception as e:
                    print(f"\r[ERROR] {key} ncc={ncc_t} cos={cos_t}: {e}")
                    continue
                elapsed = time.time() - t0

                n_det = result["total_detections"]
                row = f"{key:<12} {ncc_t:>6.2f} {cos_t:>8.2f} {n_det:>6} {elapsed:>9.2f}"
                print(f"\r{row}")

                if "visualization" in result:
                    tag = f"{key}_ncc{ncc_t:.2f}_cos{cos_t:.2f}"
                    save_visualization(args.output_dir, tag, result["visualization"])

    print(f"\n[Done] Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
