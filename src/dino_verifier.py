import warnings
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from typing import List, Optional


class DINOVerifier:
    """Stage 2: Zero-shot verification using DINOv2 patch features."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: Optional[str] = None,
        cosine_threshold: float = 0.84,
    ):
        """Initialize DINOv2 model.

        Args:
            model_name: "dinov2_vits14" (fast, 21M) or "dinov2_vitb14" (accurate, 86M).
            device: "cuda" | "cpu" | None (auto-detect).
            cosine_threshold: Candidates below this cosine similarity are rejected.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[DINOVerifier] Loading {model_name} on {self.device}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        self.model.eval()
        self.model.to(self.device)

        self.cosine_threshold = cosine_threshold
        self.transform = self._get_transform()
        self._template_feat: Optional[torch.Tensor] = None
        print(f"[DINOVerifier] Model ready.")

    def _get_transform(self) -> T.Compose:
        """Standard ImageNet normalization transform for DINOv2."""
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode_image(self, img: np.ndarray) -> torch.Tensor:
        """Encode a grayscale image to a DINOv2 feature vector.

        Args:
            img: Grayscale numpy array (H, W) or (H, W, 3).

        Returns:
            Feature tensor of shape (384,) for ViT-S/14 or (768,) for ViT-B/14.
        """
        from PIL import Image as PILImage

        if len(img.shape) == 2:
            rgb = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            rgb = np.concatenate([img, img, img], axis=-1)
        else:
            rgb = img

        pil_img = PILImage.fromarray(rgb.astype(np.uint8))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.forward_features(tensor)
            # mean-pool patch tokens (exclude CLS token)
            patch_tokens = features["x_norm_patchtokens"]  # (1, N_patches, D)
            feat = patch_tokens.mean(dim=1).squeeze(0)  # (D,)

        return feat

    def embed_crops_normalized(
        self, crops: List[np.ndarray], batch_size: int = 32
    ) -> np.ndarray:
        """Like embed_crops_batch but normalises each crop to landscape orientation.

        Horizontal and vertical instances of the same symbol produce similar
        DINOv2 embeddings after normalisation, making prototype comparison
        orientation-invariant.
        """
        normalised = []
        for c in crops:
            img = c if c.ndim == 3 else c
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            normalised.append(img)
        return self.embed_crops_batch(normalised, batch_size=batch_size)

    def embed_crops_batch(
        self, crops: List[np.ndarray], batch_size: int = 32
    ) -> np.ndarray:
        """Encode a list of image crops and return unit-normalised embeddings.

        Args:
            crops: List of grayscale or RGB numpy arrays (any size).
            batch_size: Number of crops per GPU forward pass.

        Returns:
            (N, D) float32 array of L2-normalised feature vectors.
        """
        from PIL import Image as PILImage

        if not crops:
            return np.empty((0,), dtype=np.float32)

        tensors = []
        for crop in crops:
            if crop.ndim == 2:
                rgb = np.stack([crop, crop, crop], axis=-1)
            else:
                rgb = crop
            pil = PILImage.fromarray(rgb.astype(np.uint8))
            tensors.append(self.transform(pil))

        all_feats = []
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(self.device)
            with torch.no_grad():
                feats = self.model.forward_features(batch)
                patch_tokens = feats["x_norm_patchtokens"]   # (B, N, D)
                pooled = patch_tokens.mean(dim=1)            # (B, D)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
            all_feats.append(pooled.cpu().numpy())

        return np.concatenate(all_feats, axis=0)   # (N, D)

    def encode_template(self, template: np.ndarray) -> torch.Tensor:
        """Encode template and cache the result.

        Args:
            template: Grayscale template image.

        Returns:
            Feature tensor.
        """
        self._template_feat = self.encode_image(template)
        return self._template_feat

    def verify_candidates(
        self,
        drawing: np.ndarray,
        template: np.ndarray,
        candidates: List[dict],
        derotate: bool = False,
    ) -> List[dict]:
        """Filter NCC candidates using DINOv2 cosine similarity.

        Args:
            drawing: Full drawing image (grayscale).
            template: Template pattern image (grayscale).
            candidates: List of candidate dicts from NCCMatcher.
            derotate: If True, rotate each crop by -angle before DINOv2 encoding.
                      Important for rotation-sensitive ViT models.

        Returns:
            Filtered and sorted candidates with "dino_score" and "confidence" added.
        """
        if not candidates:
            return []

        th, tw = template.shape[:2]
        if tw < 28 or th < 28:
            print(f"[DINOVerifier] Warning: template too small ({tw}x{th}px), skipping DINOv2.")
            for c in candidates:
                c["dino_score"] = 1.0
                c["confidence"] = c["ncc_score"]
            return candidates

        template_feat = self.encode_template(template)

        dh, dw = drawing.shape[:2]
        crop_tensors = []
        valid_indices = []

        for i, cand in enumerate(candidates):
            crop = self._crop_with_padding(drawing, cand, dh, dw)
            # De-rotate: align crop orientation with template before DINOv2
            if derotate:
                angle = float(cand.get("angle", 0))
                if abs(angle) > 1.0:
                    crop = self._rotate_crop(crop, -angle)
            from PIL import Image as PILImage
            if len(crop.shape) == 2:
                rgb = np.stack([crop, crop, crop], axis=-1)
            else:
                rgb = crop
            pil_img = PILImage.fromarray(rgb.astype(np.uint8))
            tensor = self.transform(pil_img)
            crop_tensors.append(tensor)
            valid_indices.append(i)

        # Batch encode
        BATCH_SIZE = 16 if len(crop_tensors) > 10 else len(crop_tensors)
        all_feats = []
        for batch_start in range(0, len(crop_tensors), BATCH_SIZE):
            batch = torch.stack(crop_tensors[batch_start:batch_start + BATCH_SIZE]).to(self.device)
            with torch.no_grad():
                features = self.model.forward_features(batch)
                patch_tokens = features["x_norm_patchtokens"]  # (B, N, D)
                feats = patch_tokens.mean(dim=1)  # (B, D)
            all_feats.append(feats)

        all_feats = torch.cat(all_feats, dim=0)  # (N_candidates, D)

        # Also encode center-crops (strips wire leads / surrounding context)
        center_tensors = []
        for i, cand in enumerate(candidates):
            crop = self._center_crop(drawing, cand, dh, dw, ratio=0.70)
            if derotate:
                angle = float(cand.get("angle", 0))
                if abs(angle) > 1.0:
                    crop = self._rotate_crop(crop, -angle)
            from PIL import Image as PILImage
            if len(crop.shape) == 2:
                rgb = np.stack([crop, crop, crop], axis=-1)
            else:
                rgb = crop
            pil_img = PILImage.fromarray(rgb.astype(np.uint8))
            center_tensors.append(self.transform(pil_img))

        center_feats = []
        for batch_start in range(0, len(center_tensors), BATCH_SIZE):
            batch = torch.stack(center_tensors[batch_start:batch_start + BATCH_SIZE]).to(self.device)
            with torch.no_grad():
                features = self.model.forward_features(batch)
                patch_tokens = features["x_norm_patchtokens"]
                feats = patch_tokens.mean(dim=1)
            center_feats.append(feats)
        center_feats = torch.cat(center_feats, dim=0)

        # Cosine similarity — take best of full crop vs center crop
        tmpl_norm = torch.nn.functional.normalize(template_feat.unsqueeze(0), dim=1)
        cands_norm = torch.nn.functional.normalize(all_feats, dim=1)
        center_norm = torch.nn.functional.normalize(center_feats, dim=1)
        sim_full   = (cands_norm @ tmpl_norm.T).squeeze(1)
        sim_center = (center_norm @ tmpl_norm.T).squeeze(1)
        # Element-wise max: use best-matching crop for each candidate
        similarities = torch.maximum(sim_full, sim_center)

        filtered = []
        for idx, sim_val in zip(valid_indices, similarities.tolist()):
            cand = candidates[idx]
            cand["dino_score"] = round(float(sim_val), 4)
            cand["confidence"] = round((cand["ncc_score"] + float(sim_val)) / 2, 4)
            if float(sim_val) >= self.cosine_threshold:
                filtered.append(cand)

        filtered.sort(key=lambda c: c["confidence"], reverse=True)
        return filtered

    @staticmethod
    def _rotate_crop(crop: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a crop image by `angle` degrees, filling border with white."""
        h, w = crop.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(
            crop, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    def _center_crop(
        self, drawing: np.ndarray, cand: dict, dh: int, dw: int, ratio: float = 0.70
    ) -> np.ndarray:
        """Crop the inner `ratio` of the candidate bbox (removes wire leads / border context).

        Useful when the template is a bare symbol (e.g. fuse circle) but the drawing
        instance has connecting wires that extend beyond the symbol boundary.
        """
        x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
        shrink_x = int(w * (1 - ratio) / 2)
        shrink_y = int(h * (1 - ratio) / 2)
        x1 = min(dw - 1, x + shrink_x)
        y1 = min(dh - 1, y + shrink_y)
        x2 = max(x1 + 1, min(dw, x + w - shrink_x))
        y2 = max(y1 + 1, min(dh, y + h - shrink_y))
        crop = drawing[y1:y2, x1:x2]
        if crop.size == 0:
            crop = drawing[max(0, y):min(dh, y + h), max(0, x):min(dw, x + w)]
        return crop

    def _crop_with_padding(
        self, drawing: np.ndarray, cand: dict, dh: int, dw: int
    ) -> np.ndarray:
        """Crop candidate region with 10% padding, filling out-of-bound with white."""
        x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(dw, x + w + pad_x)
        y2 = min(dh, y + h + pad_y)

        crop = drawing[y1:y2, x1:x2]

        # Pad if needed
        top = max(0, pad_y - y)
        left = max(0, pad_x - x)
        bottom = max(0, (y + h + pad_y) - dh)
        right = max(0, (x + w + pad_x) - dw)

        if any([top, left, bottom, right]):
            crop = np.pad(
                crop,
                ((top, bottom), (left, right)),
                mode="constant",
                constant_values=255,
            )

        return crop
