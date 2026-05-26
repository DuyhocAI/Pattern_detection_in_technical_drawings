"""Generate synthetic example images for demo purposes."""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_resistor_symbol(size: int = 60) -> np.ndarray:
    """Draw a simple resistor symbol (rectangle with leads)."""
    img = np.full((size, size * 2), 255, dtype=np.uint8)
    h, w = img.shape
    cx, cy = w // 2, h // 2

    # Body rectangle
    cv2.rectangle(img, (cx - 20, cy - 8), (cx + 20, cy + 8), 0, 2)
    # Left lead
    cv2.line(img, (0, cy), (cx - 20, cy), 0, 2)
    # Right lead
    cv2.line(img, (cx + 20, cy), (w - 1, cy), 0, 2)

    return img


def make_capacitor_symbol(size: int = 60) -> np.ndarray:
    """Draw a simple capacitor symbol (two parallel lines)."""
    img = np.full((size, size * 2), 255, dtype=np.uint8)
    h, w = img.shape
    cx, cy = w // 2, h // 2

    # Left plate
    cv2.line(img, (cx - 4, cy - 20), (cx - 4, cy + 20), 0, 3)
    # Right plate
    cv2.line(img, (cx + 4, cy - 20), (cx + 4, cy + 20), 0, 3)
    # Left lead
    cv2.line(img, (0, cy), (cx - 4, cy), 0, 2)
    # Right lead
    cv2.line(img, (cx + 4, cy), (w - 1, cy), 0, 2)

    return img


def place_symbol_on_drawing(
    drawing: np.ndarray, symbol: np.ndarray, positions: list, angles: list = None
) -> np.ndarray:
    """Place symbol copies at specified positions."""
    if angles is None:
        angles = [0] * len(positions)

    sh, sw = symbol.shape
    for (px, py), angle in zip(positions, angles):
        if angle != 0:
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
            sym = cv2.warpAffine(symbol, M, (sw, sh), borderValue=255)
        else:
            sym = symbol

        x1, y1 = px, py
        x2, y2 = x1 + sw, y1 + sh

        if x2 > drawing.shape[1] or y2 > drawing.shape[0]:
            continue

        roi = drawing[y1:y2, x1:x2]
        mask = sym < 128
        roi[mask] = sym[mask]

    return drawing


def generate_examples(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Example 1: Resistor pattern on circuit drawing
    resistor = make_resistor_symbol(50)
    drawing1 = np.full((600, 900), 255, dtype=np.uint8)

    # Add a grid-like circuit background
    for x in range(0, 900, 150):
        cv2.line(drawing1, (x, 100), (x, 500), 200, 1)
    for y in range(100, 600, 100):
        cv2.line(drawing1, (50, y), (850, y), 200, 1)

    positions1 = [(100, 250), (300, 150), (500, 350), (700, 200), (200, 420)]
    drawing1 = place_symbol_on_drawing(drawing1, resistor, positions1, angles=[0, 2, -3, 1, 0])

    cv2.imwrite(os.path.join(output_dir, "example1_pattern.png"), resistor)
    cv2.imwrite(os.path.join(output_dir, "example1_drawing.png"), drawing1)
    print(f"Generated example1: resistor × {len(positions1)}")

    # Example 2: Capacitor pattern on PCB layout
    capacitor = make_capacitor_symbol(50)
    drawing2 = np.full((600, 900), 255, dtype=np.uint8)

    # Add some PCB traces
    for x in range(80, 820, 80):
        cv2.line(drawing2, (x, 50), (x, 550), 220, 1)
    cv2.rectangle(drawing2, (50, 50), (850, 550), 180, 2)

    positions2 = [(120, 200), (320, 300), (520, 180), (720, 400)]
    drawing2 = place_symbol_on_drawing(drawing2, capacitor, positions2, angles=[0, -2, 3, 0])

    cv2.imwrite(os.path.join(output_dir, "example2_pattern.png"), capacitor)
    cv2.imwrite(os.path.join(output_dir, "example2_drawing.png"), drawing2)
    print(f"Generated example2: capacitor × {len(positions2)}")

    print(f"\nExamples saved to: {output_dir}/")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "examples")
    generate_examples(out)
