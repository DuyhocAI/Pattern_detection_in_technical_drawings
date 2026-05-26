import sys
import types
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Provide a minimal torch stub so pipeline imports don't fail when torch is absent
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda x: x
    torch_stub.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
    torch_stub.cuda = MagicMock()
    torch_stub.cuda.is_available = MagicMock(return_value=False)
    torch_stub.hub = MagicMock()
    torch_stub.Tensor = MagicMock
    sys.modules["torch"] = torch_stub
    tv_stub = types.ModuleType("torchvision")
    tv_transforms = types.ModuleType("torchvision.transforms")
    tv_transforms.Compose = MagicMock
    tv_transforms.Resize = MagicMock
    tv_transforms.ToTensor = MagicMock
    tv_transforms.Normalize = MagicMock
    tv_stub.transforms = tv_transforms
    sys.modules["torchvision"] = tv_stub
    sys.modules["torchvision.transforms"] = tv_transforms


def _make_white_image(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _make_circle_image(h: int, w: int, radius: int = 30) -> np.ndarray:
    img = _make_white_image(h, w)
    import cv2
    cv2.circle(img, (w // 2, h // 2), radius, 0, 2)
    return img


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        from src.preprocessor import Preprocessor
        self.prep = Preprocessor()

    def test_load_grayscale(self):
        import tempfile, cv2
        img = np.zeros((100, 100), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            loaded = self.prep.load_image(f.name)
        self.assertEqual(loaded.shape, (100, 100))
        self.assertEqual(loaded.dtype, np.uint8)

    def test_binarize_adaptive(self):
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        binary = self.prep.binarize(img, method="adaptive")
        unique_vals = set(np.unique(binary))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_resize_large_image(self):
        img = np.zeros((4000, 6000), dtype=np.uint8)
        resized = self.prep.resize_if_needed(img, max_dim=4096)
        self.assertLessEqual(max(resized.shape[:2]), 4096)


class TestNCCMatcher(unittest.TestCase):
    def setUp(self):
        from src.ncc_matcher import NCCMatcher
        self.matcher = NCCMatcher(
            scales=[1.0],
            angles=[0],
            ncc_threshold=0.7,
            nms_iou_threshold=0.3,
        )

    def test_match_exact_template(self):
        import cv2
        drawing = _make_circle_image(500, 500, radius=40)
        # Crop around the circle as template
        template = drawing[200:300, 200:300]
        candidates = self.matcher.match(drawing, template)
        self.assertGreater(len(candidates), 0)

    def test_nms_removes_duplicates(self):
        cands = [
            {"x": 10, "y": 10, "w": 50, "h": 50, "ncc_score": 0.9, "scale": 1.0, "angle": 0},
            {"x": 12, "y": 12, "w": 50, "h": 50, "ncc_score": 0.8, "scale": 1.0, "angle": 0},
            {"x": 200, "y": 200, "w": 50, "h": 50, "ncc_score": 0.85, "scale": 1.0, "angle": 0},
        ]
        result = self.matcher._apply_nms(cands)
        # First two strongly overlap, should be merged to 1 + the distant one = 2
        self.assertEqual(len(result), 2)

    def test_no_match_returns_empty(self):
        # A circle pattern on a white drawing will not match a fully white region
        # Use a very high threshold so a completely different pattern returns nothing
        from src.ncc_matcher import NCCMatcher
        import cv2
        strict = NCCMatcher(scales=[1.0], angles=[0], ncc_threshold=0.99)
        # Drawing has a circle on the left; template is a rectangle on the right
        drawing = _make_white_image(300, 300)
        cv2.circle(drawing, (80, 150), 30, 0, 2)

        # Template: a very different shape (filled black square — not circle)
        template = _make_white_image(60, 60)
        cv2.rectangle(template, (5, 5), (55, 55), 0, -1)

        result = strict.match(drawing, template)
        self.assertEqual(len(result), 0)


class TestPostprocessor(unittest.TestCase):
    def setUp(self):
        from src.postprocessor import Postprocessor
        self.post = Postprocessor()

    def test_format_output_structure(self):
        candidates = [
            {"x": 10, "y": 20, "w": 50, "h": 60, "confidence": 0.85,
             "ncc_score": 0.8, "dino_score": 0.9, "scale": 1.0, "angle": 0},
        ]
        result = self.post.format_output(candidates, (300, 400))
        self.assertIn("detections", result)
        self.assertIn("total_detections", result)
        self.assertIn("image_size", result)
        self.assertEqual(result["total_detections"], 1)
        det = result["detections"][0]
        self.assertIn("bbox", det)
        self.assertIn("confidence", det)

    def test_draw_boxes_returns_rgb(self):
        drawing = _make_white_image(200, 300)
        detections = [
            {"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "confidence": 0.9},
        ]
        output = self.post.draw_boxes(drawing, detections)
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[2], 3)
        self.assertEqual(output.dtype, np.uint8)


class TestPipeline(unittest.TestCase):
    """Integration tests using synthetic images and mocked DINOv2."""

    def _get_mock_device(self):
        try:
            import torch
            return torch.device("cpu")
        except ImportError:
            return "cpu"

    def test_full_pipeline_runs(self):
        # Import pipeline module first so patch can resolve "src.pipeline.DINOVerifier"
        import importlib
        import src.pipeline as pipeline_mod  # noqa: ensure module is loaded

        device = self._get_mock_device()

        with patch.object(pipeline_mod, "DINOVerifier") as MockDINO:
            instance = MockDINO.return_value
            instance.device = device
            instance.cosine_threshold = 0.75
            instance.verify_candidates.side_effect = lambda d, t, c: [
                dict(x, dino_score=0.85, confidence=(x["ncc_score"] + 0.85) / 2) for x in c
            ]

            from src.pipeline import PatternDetectionPipeline
            pipeline = PatternDetectionPipeline()

            drawing = _make_circle_image(400, 400, radius=40)
            template = drawing[160:240, 160:240]

            result = pipeline.detect(template, drawing, return_visualization=False)

        self.assertIn("detections", result)
        self.assertIn("total_detections", result)

    def test_empty_result_if_no_match(self):
        import src.pipeline as pipeline_mod  # noqa: ensure module is loaded

        device = self._get_mock_device()

        with patch.object(pipeline_mod, "DINOVerifier") as MockDINO:
            instance = MockDINO.return_value
            instance.device = device
            instance.cosine_threshold = 0.99
            instance.verify_candidates.return_value = []

            from src.pipeline import PatternDetectionPipeline
            pipeline = PatternDetectionPipeline(config={"ncc_threshold": 0.99})

            drawing = _make_white_image(300, 300)
            import cv2
            cv2.rectangle(drawing, (10, 10), (60, 60), 0, 2)
            template = _make_white_image(80, 80)
            cv2.circle(template, (40, 40), 30, 0, 2)

            result = pipeline.detect(template, drawing, return_visualization=False)

        self.assertEqual(result["total_detections"], 0)


if __name__ == "__main__":
    unittest.main()
