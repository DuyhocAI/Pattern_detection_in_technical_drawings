"""Stage 3 (optional): Vision-Language Model verifier using Qwen2-VL-2B.

Why a VLM after NCC + DINOv2:
  NCC matches by pixel correlation and DINOv2 by patch-feature cosine. Both
  encode *how a symbol looks* (line density, spatial frequency). Two schematic
  symbols with similar low-level structure -- a zigzag/rectangle resistor vs a
  coiled inductor vs a crystal -- land close in both spaces, so the worst false
  positives (inductor, crystal, op-amp fragments) survive every spatial filter.

  A VLM reasons about *what the symbol is*. Asked "is this a resistor?", Qwen2-VL
  can use the learned semantic concept to reject an inductor even when its crop
  correlates highly with the resistor template. This is the discriminability CLIP
  image-to-image lacked (see design_spec/model_survey.md).

Design:
  * Drop-in Stage-3 filter: same signature shape as DINOVerifier.verify_candidates.
  * Lazy load: the 2B model (~4.5 GB bf16) is only loaded on first use, so the
    rest of the pipeline runs without it when `use_vlm` is off.
  * Each candidate crop is taken with generous context padding and upscaled so the
    small (30-80 px) line-art symbol is legible to the VLM, then the model is asked
    a constrained yes/no question. Only "yes" candidates survive.
  * The template image is shown to the VLM as a visual reference so the check stays
    zero-shot (no hardcoded symbol class name needed); an optional `symbol_name`
    sharpens the prompt when the class is known.

Tested on: RTX 3060 12 GB, transformers >= 4.45 (Qwen2-VL support).
"""
from __future__ import annotations

import re
from typing import List, Optional

import cv2
import numpy as np


class VLMVerifier:
    """Verify candidate crops with a local Qwen2-VL model (yes/no semantic check).

    Args:
        model_name: HuggingFace model ID. Default Qwen2-VL-2B-Instruct.
        device: "cuda" | "cpu" | None (auto-detect).
        symbol_name: Optional human name of the target symbol ("resistor"). When
            provided it is woven into the prompt for a sharper decision; when None
            the model compares each crop against the shown template image only.
        min_keep_conf: Candidates with confidence >= this are kept WITHOUT asking
            the VLM (trusted high-confidence detections — saves inference time).
        max_ask_conf: Candidates with confidence above this are auto-kept; only
            those in (min_keep_conf upper region downward) borderline band are asked.
            (See verify_candidates for the exact banding logic.)
        context_pad_ratio: Fraction of bbox size added on each side as context.
        upscale_to: Target longer-side pixel size for each crop shown to the VLM.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        symbol_name: Optional[str] = None,
        context_pad_ratio: float = 0.6,
        upscale_to: int = 224,
        max_new_tokens: int = 8,
    ):
        self.model_name = model_name
        self.symbol_name = symbol_name
        self.context_pad_ratio = context_pad_ratio
        self.upscale_to = upscale_to
        self.max_new_tokens = max_new_tokens

        self._device_arg = device
        self._model = None          # lazy
        self._processor = None      # lazy
        self.device = device or "cuda"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device = self._device_arg
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Free any cached allocations from earlier stages (DINOv2, NCC tensors)
        # before pulling the ~4.5 GB VLM onto the GPU. On a 12 GB card shared with
        # the OS display this reclaimed headroom is the difference between fitting
        # and an OOM.
        if device == "cuda":
            torch.cuda.empty_cache()

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"[VLMVerifier] Loading {self.model_name} on {device} ({dtype})...")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        # min_pixels/max_pixels keep token counts bounded for small crops
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=128 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self._torch = torch
        print("[VLMVerifier] Model ready.")

    # ------------------------------------------------------------------
    # Crop preparation
    # ------------------------------------------------------------------

    def _crop_for_vlm(self, drawing: np.ndarray, c: dict) -> np.ndarray:
        """Extract a candidate region with context padding, de-rotate, upscale.

        Small schematic symbols are illegible to a VLM at native resolution; we
        add surrounding context (helps the model see connecting wires) and upscale
        the longer side to `upscale_to` px with cubic interpolation.
        """
        H, W = drawing.shape[:2]
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        pad_x = int(w * self.context_pad_ratio)
        pad_y = int(h * self.context_pad_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        crop = drawing[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.full((self.upscale_to, self.upscale_to), 255, np.uint8)

        # De-rotate vertical candidates to a canonical horizontal orientation
        angle = c.get("angle", 0)
        if 70 <= abs(angle) <= 110 and crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Upscale longer side to upscale_to
        ch, cw = crop.shape[:2]
        longer = max(ch, cw)
        if longer > 0 and longer < self.upscale_to:
            f = self.upscale_to / longer
            crop = cv2.resize(crop, (max(1, int(cw * f)), max(1, int(ch * f))),
                              interpolation=cv2.INTER_CUBIC)
        # To 3-channel for the VLM
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        return crop

    def _prep_template(self, template: np.ndarray) -> np.ndarray:
        t = template
        if t.ndim == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)
        ch, cw = t.shape[:2]
        longer = max(ch, cw)
        if longer > 0 and longer < self.upscale_to:
            f = self.upscale_to / longer
            t = cv2.resize(t, (max(1, int(cw * f)), max(1, int(ch * f))),
                           interpolation=cv2.INTER_CUBIC)
        return t

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    # Component vocabulary for classification mode
    _CLASSES = [
        "resistor", "inductor", "capacitor", "diode", "crystal",
        "transistor", "op-amp", "logic-gate", "wire-junction", "other",
    ]

    def _build_prompt(self) -> str:
        sym = self.symbol_name or "the reference component symbol shown in the first image"
        return (
            "You are an expert at reading electronic schematic diagrams.\n"
            "The FIRST image is a reference symbol. The SECOND image is a candidate "
            "region cropped from a larger schematic.\n"
            f"Question: Does the SECOND image contain the SAME type of component as "
            f"{sym}?\n"
            "Pay attention to the exact symbol shape. A resistor is a zigzag or a "
            "plain rectangle. Do NOT confuse it with: an inductor (series of loops/"
            "humps), a crystal (rectangle between two capacitor plates), a capacitor "
            "(two parallel lines), a diode (triangle+bar), or an op-amp (large "
            "triangle). Answer with exactly one word: 'yes' or 'no'."
        )

    def _build_classify_prompt(self) -> str:
        """Open-classification prompt — avoids the yes/no agreement bias of small VLMs.

        Instead of confirming "is this X?" (which a 2B model tends to answer 'yes'
        to regardless), we force it to NAME the component from a closed vocabulary.
        The caller keeps only candidates classified as the target class.
        """
        classes = ", ".join(self._CLASSES)
        return (
            "You are an expert at reading electronic schematic diagrams. The image "
            "is a small region cropped from a schematic, possibly with connecting "
            "wires around the central component.\n"
            "Identify the SINGLE central electronic component. Shape guide:\n"
            "- resistor: a zigzag (sawtooth) line, OR a plain rectangle in series "
            "with a wire.\n"
            "- inductor: a series of rounded loops / humps / coils.\n"
            "- capacitor: two short parallel lines (or one curved) with a gap.\n"
            "- diode: a triangle pointing into a bar.\n"
            "- crystal: a small rectangle drawn between two capacitor plates.\n"
            "- transistor: a circle/junction with three leads.\n"
            "- op-amp: a large triangle.\n"
            "- logic-gate: AND/OR/XOR gate outline.\n"
            "- wire-junction: only wires/dots, no component body.\n"
            f"Answer with EXACTLY ONE word from this list: {classes}."
        )

    @staticmethod
    def _np_to_pil(arr: np.ndarray):
        from PIL import Image
        return Image.fromarray(arr.astype(np.uint8))

    def _generate(self, content: list) -> str:
        """Run one chat turn with the given content list, return decoded text."""
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # Release the per-crop activation/KV-cache spike so memory does not creep
        # up across a long candidate list on a tight (12 GB) card.
        del inputs, gen, trimmed
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
        return out.strip().lower()

    def _ask_one(self, template_rgb: np.ndarray, crop_rgb: np.ndarray):
        """Run a single yes/no query. Returns (decided_bool, raw_answer)."""
        ans = self._generate([
            {"type": "image", "image": self._np_to_pil(template_rgb)},
            {"type": "image", "image": self._np_to_pil(crop_rgb)},
            {"type": "text", "text": self._build_prompt()},
        ])
        m = re.search(r"\b(yes|no)\b", ans)
        decided = (m.group(1) == "yes") if m else ans.startswith("y")
        return decided, ans

    def _classify_one(self, crop_rgb: np.ndarray):
        """Open-classification query. Returns (class_label, raw_answer)."""
        ans = self._generate([
            {"type": "image", "image": self._np_to_pil(crop_rgb)},
            {"type": "text", "text": self._build_classify_prompt()},
        ])
        # Match the first known class token that appears in the answer
        label = "other"
        for cls in self._CLASSES:
            if re.search(rf"\b{re.escape(cls)}\b", ans):
                label = cls
                break
        return label, ans

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify_template(self, template: np.ndarray) -> str:
        """Classify the template itself to learn the target component class.

        Keeps the VLM stage zero-shot: we never hardcode "resistor"; instead the
        template's own classification defines what candidates must match.
        """
        self._ensure_loaded()
        t = self._prep_template(template)
        label, raw = self._classify_one(t)
        print(f"[VLMVerifier] Template classified as: {label!r} (raw {raw!r})")
        return label

    def filter_by_template_class(
        self,
        drawing: np.ndarray,
        template: np.ndarray,
        candidates: List[dict],
        target_class: Optional[str] = None,
        keep_min_conf: float = 1.01,
        verbose: bool = False,
    ) -> List[dict]:
        """Open-classification Stage-3 filter (robust to small-VLM yes-bias).

        Each candidate crop is classified into a closed component vocabulary; only
        those whose class matches the template's class survive. This avoids the
        yes/no agreement bias that made the confirm-style prompt keep 100% of
        candidates (see scripts/poc_vlm.py vs poc_vlm_classify.py).

        Args:
            target_class: Class candidates must match. If None, derived from the
                template via classify_template().
            keep_min_conf: Candidates with confidence >= this are kept WITHOUT
                asking the VLM (trusted high-conf detections; saves inference).
                Default 1.01 = always ask.
        """
        if not candidates:
            return []
        self._ensure_loaded()
        if target_class is None:
            target_class = self.classify_template(template)

        kept = []
        for c in candidates:
            if c.get("confidence", 0.0) >= keep_min_conf:
                c["vlm_class"] = "auto-keep"
                kept.append(c)
                continue
            crop = self._crop_for_vlm(drawing, c)
            label, raw = self._classify_one(crop)
            c["vlm_class"] = label
            if verbose:
                print(f"  [VLM] ({c['x']},{c['y']}) conf={c.get('confidence',0):.2f} "
                      f"-> {label!r} {'KEEP' if label == target_class else 'DROP'}")
            if label == target_class:
                kept.append(c)
        print(f"[VLMVerifier] class-filter ({target_class}): "
              f"{len(candidates)} -> {len(kept)} kept")
        return kept

    def verify_candidates(
        self,
        drawing: np.ndarray,
        template: np.ndarray,
        candidates: List[dict],
        ask_band: tuple = (0.0, 1.01),
        verbose: bool = False,
    ) -> List[dict]:
        """Filter candidates with the VLM.

        Args:
            drawing: Preprocessed (binarised) drawing, grayscale.
            template: Preprocessed template image, grayscale.
            candidates: Detection dicts (need x,y,w,h, optional angle/confidence).
            ask_band: (low, high) confidence band. Candidates whose confidence is
                INSIDE this band are sent to the VLM; candidates above `high` are
                auto-kept (trusted), candidates below `low` are auto-rejected.
                Default (0.0, 1.01) asks the VLM about every candidate.
            verbose: Print per-candidate VLM answers.

        Returns:
            Filtered list (VLM 'no' candidates removed). Each surviving candidate
            gains a 'vlm_pass' bool and 'vlm_answer' string when it was asked.
        """
        if not candidates:
            return []
        self._ensure_loaded()
        template_rgb = self._prep_template(template)

        low, high = ask_band
        kept = []
        for c in candidates:
            conf = c.get("confidence", 1.0)
            if conf >= high:
                c["vlm_pass"] = True
                c["vlm_answer"] = "auto-keep"
                kept.append(c)
                continue
            if conf < low:
                c["vlm_pass"] = False
                c["vlm_answer"] = "auto-reject"
                continue
            crop_rgb = self._crop_for_vlm(drawing, c)
            decided, raw = self._ask_one(template_rgb, crop_rgb)
            c["vlm_pass"] = decided
            c["vlm_answer"] = raw
            if verbose:
                print(f"  [VLM] ({c['x']},{c['y']}) conf={conf:.2f} -> {raw!r} "
                      f"=> {'KEEP' if decided else 'DROP'}")
            if decided:
                kept.append(c)
        print(f"[VLMVerifier] {len(candidates)} candidates -> {len(kept)} kept")
        return kept
