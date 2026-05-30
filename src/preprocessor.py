import cv2
import numpy as np
from typing import Union


class Preprocessor:
    """Stage 0: Image preprocessing for BOM engineering drawings."""

    def load_image(self, path: str) -> np.ndarray:
        """Load image from path and return as grayscale uint8 array.

        Args:
            path: File path to image (PNG, JPG, TIFF supported).

        Returns:
            Grayscale numpy array (uint8, 0-255).

        Raises:
            ValueError: If image cannot be read.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return img

    def binarize(self, img: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """Binarize grayscale image.

        Args:
            img: Grayscale numpy array.
            method: "adaptive" | "otsu" | "none"

        Returns:
            Binary image with pixel values 0 or 255.
        """
        if method == "adaptive":
            return cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=15, C=4
            )
        elif method == "otsu":
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == "none":
            if len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            raise ValueError(f"Unknown binarize method: {method}. Use 'adaptive', 'otsu', or 'none'.")

    def denoise(self, img: np.ndarray, kernel_size: int = 2) -> np.ndarray:
        """Apply morphological closing to fill broken thin lines.

        Args:
            img: Binary or grayscale image.
            kernel_size: Size of the morphological kernel (keep small to preserve thin lines).

        Returns:
            Denoised image.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def resize_if_needed(self, img: np.ndarray, max_dim: int = 4096) -> np.ndarray:
        """Resize image if its largest dimension exceeds max_dim, preserving aspect ratio.

        Args:
            img: Input image.
            max_dim: Maximum allowed dimension in pixels.

        Returns:
            Resized image (or original if already within bounds).
        """
        h, w = img.shape[:2]
        largest = max(h, w)
        if largest <= max_dim:
            return img
        scale = max_dim / largest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def suppress_text_noise(self, img: np.ndarray, max_area: int = 180) -> np.ndarray:
        """Remove small isolated strokes (text labels) from a binary image.

        Uses connected-component analysis to erase blobs that are small and
        elongated — the typical signature of text characters — while preserving
        larger, rounder component symbols.

        Args:
            img: Binary image (white background, black strokes).
            max_area: Components with pixel area below this threshold AND
                      aspect ratio > 1.3 are treated as text and erased.

        Returns:
            Binary image with text-like blobs replaced by white.
        """
        inv = cv2.bitwise_not(img)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        result = img.copy()
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if area < max_area and aspect > 1.3:
                result[labels == lbl] = 255
        return result

    def dilate_strokes(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Thicken black strokes in a binary image by eroding (strokes are black on white bg).

        Useful when template uses thin stylized lines but drawing uses bold strokes.

        Args:
            img: Binary image (white background, black strokes).
            kernel_size: Size of dilation kernel.

        Returns:
            Image with thickened strokes.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(img, kernel)

    def clahe_enhance(
        self, img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Improves local contrast in drawings with uneven line density — faint
        strokes that are washed out globally become visible after CLAHE.
        Applied to grayscale images BEFORE binarization.

        Args:
            img: Grayscale uint8 image.
            clip_limit: Threshold for contrast limiting (higher = stronger).
            tile_size: Grid size in pixels for local histogram computation.

        Returns:
            Contrast-enhanced grayscale image.
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
        )
        return clahe.apply(img.astype(np.uint8))

    def normalize_strokes(
        self, img: np.ndarray, target_width: int = 2
    ) -> np.ndarray:
        """Normalize stroke width via thinning + uniform dilation.

        Engineering drawings may scan at different DPIs, producing thick or
        thin lines. This normalizes all strokes to `target_width` pixels:
          1. Morphological thinning (iterative erosion to approximate skeleton)
          2. Dilation to the desired target width

        This makes NCC matching invariant to line-width variation between the
        template and the drawing, which is a common source of false negatives.

        Args:
            img: Binary image (white background, black strokes).
            target_width: Desired uniform stroke width in pixels.

        Returns:
            Binary image with normalized stroke width.
        """
        inv = cv2.bitwise_not(img)

        # Iterative thinning: erode until no more pixels can be removed.
        # Stop after 20 iterations to bound runtime.
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        prev = np.zeros_like(inv)
        for _ in range(20):
            eroded = cv2.erode(inv, kernel)
            temp = cv2.dilate(eroded, kernel)
            diff = cv2.subtract(inv, temp)
            skeleton = cv2.bitwise_or(prev, diff)
            inv = eroded.copy()
            prev = skeleton.copy()
            if cv2.countNonZero(inv) == 0:
                break

        # Re-dilate skeleton to target width
        if target_width > 1:
            kw = target_width * 2 - 1
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kw))
            skeleton = cv2.dilate(skeleton, dk)

        return cv2.bitwise_not(skeleton)

    def preprocess(
        self,
        img_or_path: Union[str, np.ndarray],
        binarize_method: str = "adaptive",
        denoise: bool = True,
        dilate_strokes: int = 0,
        clahe: bool = False,
        normalize_stroke_width: int = 0,
    ) -> dict:
        """Full preprocessing pipeline.

        Args:
            img_or_path: File path string or numpy array.
            binarize_method: Binarization method passed to self.binarize().
            denoise: Whether to apply morphological denoising.

        Returns:
            Dict with keys:
                "original": original grayscale numpy array
                "processed": preprocessed numpy array
                "scale_factor": downscale ratio applied (1.0 if no resize)
        """
        if isinstance(img_or_path, str):
            original = self.load_image(img_or_path)
        elif isinstance(img_or_path, np.ndarray):
            if len(img_or_path.shape) == 3:
                original = cv2.cvtColor(img_or_path, cv2.COLOR_RGB2GRAY)
            else:
                original = img_or_path.copy()
        else:
            raise ValueError("img_or_path must be a file path string or numpy array.")

        resized = self.resize_if_needed(original)
        h_orig, w_orig = original.shape[:2]
        h_res, w_res = resized.shape[:2]
        scale_factor = h_res / h_orig if h_orig > 0 else 1.0

        # CLAHE before binarization: improves faint-stroke visibility
        to_binarize = self.clahe_enhance(resized) if clahe else resized

        processed = self.binarize(to_binarize, method=binarize_method)
        if denoise:
            processed = self.denoise(processed)
        if dilate_strokes > 0:
            processed = self.dilate_strokes(processed, kernel_size=dilate_strokes)
        if normalize_stroke_width > 0:
            processed = self.normalize_strokes(processed, target_width=normalize_stroke_width)

        return {
            "original": original,
            "processed": processed,
            "scale_factor": scale_factor,
        }
