"""Local Qwen2-VL reranker with explicit degraded modes and failure isolation."""

from __future__ import annotations

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


class Qwen2VLReranker:
    """Reranks visually retrieved SKUs when the DINO result is ambiguous.

    A heuristic fallback remains available for observability and candidate
    ordering, but is never represented as VLM verification.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.model: Any = None
        self.processor: Any = None
        self.is_ready = True
        self.failure_threshold = 3
        self.cooldown_seconds = 60.0
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        # One worker prevents overlapping GPU generations. A timed-out
        # generation may finish in the background; the open circuit ensures no
        # further work queues behind it during the cooldown.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen2-vl")

    def initialize(self, config: Dict[str, Any]) -> None:
        self.model_id = config.get("model_id", self.model_id)
        self.device = config.get("device", self.device)
        self.failure_threshold = max(1, int(config.get("failure_threshold", 3)))
        self.cooldown_seconds = max(1.0, float(config.get("cooldown_seconds", 60.0)))
        local_files_only = bool(config.get("local_files_only", True))

        local_weights_dir = Path("configs/weights/qwen2_vl_2b_instruct")
        local_awq_dir = Path("configs/weights/qwen2_vl_awq")
        if (local_weights_dir / "config.json").exists():
            self.model_id = str(local_weights_dir.resolve())
        elif (local_awq_dir / "config.json").exists():
            self.model_id = str(local_awq_dir.resolve())
        elif local_files_only:
            # Do not hand a remote model ID to Transformers in strict offline
            # mode: some processor discovery paths still issue HEAD requests
            # before honoring local_files_only.
            self.model = None
            self.processor = None
            print(
                "[Qwen2VL] No local weights found under configs/weights; "
                "network access was not attempted."
            )
            return

        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                self.model_id, local_files_only=local_files_only
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                local_files_only=local_files_only,
            )
            if not torch.cuda.is_available():
                self.model.to(self.device)
            self.model.eval()
            print(f"[Qwen2VL] Offline model '{self.model_id}' ready on {self.device}.")
        except Exception as exc:
            self.model = None
            self.processor = None
            print(
                f"[Qwen2VL] Heavy model unavailable ({exc}); "
                "heuristic fallback will be labeled unverified."
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.model = None
        self.processor = None

    @property
    def circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self, reason: str) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._circuit_open_until = time.monotonic() + self.cooldown_seconds
            print(
                f"[Qwen2VL] Circuit opened for {self.cooldown_seconds:.0f}s "
                f"after {self._consecutive_failures} failures ({reason})."
            )

    @staticmethod
    def enhance_crop_for_vlm(crop_image: Image.Image) -> Image.Image:
        try:
            import cv2
            import numpy as np

            rgb = np.array(crop_image.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            denoised = cv2.bilateralFilter(bgr, d=5, sigmaColor=50, sigmaSpace=50)
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            lightness = cv2.createCLAHE(
                clipLimit=1.5, tileGridSize=(4, 4)
            ).apply(lightness)
            enhanced = cv2.cvtColor(
                cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2RGB
            )
            return Image.fromarray(enhanced)
        except Exception:
            return crop_image

    def _generate_option(self, crop_image: Image.Image, candidates: List[Dict[str, Any]]) -> int:
        options = [f"{index + 1}. {candidate['display_name']}" for index, candidate in enumerate(candidates)]
        options.append(f"{len(candidates) + 1}. Unknown")
        prompt = (
            "Look at this retail package carefully. Select the exact brand, variant, "
            "size and pack count. If none is exact, select Unknown.\n"
            + "\n".join(options)
            + f"\nRespond ONLY with one option number from 1 to {len(candidates) + 1}."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": crop_image},
                {"type": "text", "text": prompt},
            ],
        }]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text_prompt, images=crop_image, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=5)
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        match = re.search(r"\b(\d+)\b", output)
        if not match:
            raise ValueError(f"invalid VLM output: {output!r}")
        option = int(match.group(1))
        if not 1 <= option <= len(candidates) + 1:
            raise ValueError(f"VLM option out of range: {option}")
        return option - 1

    @staticmethod
    def _unknown(similarity: float, mode: str, verified: bool) -> Dict[str, Any]:
        return {
            "class_id": -1,
            "display_name": "Class Unknown",
            "similarity": float(similarity),
            "s_fused": float(similarity),
            "vlm_selected": bool(verified),
            "qwen2_vl_verified": bool(verified),
            "vlm_verified": bool(verified),
            "vlm_selected_rank": 1,
            "inference_mode": mode,
        }

    @staticmethod
    def _fuse(
        candidates: List[Dict[str, Any]],
        selected_index: int,
        mode: str,
        verified: bool,
    ) -> List[Dict[str, Any]]:
        reranked: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            item = dict(candidate)
            selected = index == selected_index
            similarity = float(item.get("similarity", 0.0))
            item["s_fused"] = 0.80 * similarity + (0.20 if selected else 0.0)
            item["vlm_selected"] = bool(selected and verified)
            item["qwen2_vl_verified"] = bool(selected and verified)
            item["vlm_verified"] = bool(selected and verified)
            item["vlm_selected_rank"] = selected_index + 1 if selected else None
            item["inference_mode"] = mode
            reranked.append(item)
        reranked.sort(key=lambda item: item.get("s_fused", 0.0), reverse=True)
        return reranked

    def _heuristic_fallback(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Preserve the existing title heuristic, but truthfully mark it as a
        # degraded non-visual fallback.
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(candidates):
            title = str(candidate.get("display_name", "")).lower()
            score = float(candidate.get("similarity", 0.0))
            for token, boost in (
                ("lemon", 0.05),
                ("mint", 0.05),
                ("earl grey", 0.05),
                ("green", 0.03),
                ("yellow", 0.02),
                ("black", 0.02),
            ):
                if token in title:
                    score += boost
            if score > best_score:
                best_score = score
                best_index = index
        return self._fuse(candidates, best_index, "heuristic_fallback", verified=False)

    def rerank_top5_candidates(
        self,
        crop_image: Image.Image,
        top5_candidates: List[Dict[str, Any]],
        timeout_ms: int = 400,
    ) -> List[Dict[str, Any]]:
        if not top5_candidates:
            return []

        top_similarity = float(top5_candidates[0].get("similarity", 0.0))
        if top_similarity < 0.62:
            return [self._unknown(top_similarity, "similarity_threshold", verified=False)]

        enhanced = self.enhance_crop_for_vlm(crop_image)
        can_generate = (
            self.model is not None
            and self.processor is not None
            and not self.circuit_open
        )
        if can_generate:
            future = self._executor.submit(self._generate_option, enhanced, top5_candidates)
            try:
                selected = future.result(timeout=max(1, int(timeout_ms)) / 1000.0)
                self._record_success()
                if selected == len(top5_candidates):
                    # Unknown is a final categorical decision. It never enters
                    # score sorting with known candidates.
                    return [self._unknown(top_similarity, "qwen2_vl", verified=True)]
                return self._fuse(top5_candidates, selected, "qwen2_vl", verified=True)
            except FutureTimeoutError:
                future.cancel()
                self._record_failure("timeout")
                print(f"[Qwen2VL] Generation exceeded the {timeout_ms}ms budget.")
            except Exception as exc:
                self._record_failure(type(exc).__name__)
                print(f"[Qwen2VL] Invalid/failed generation: {exc}")

        result = self._heuristic_fallback(top5_candidates)
        selected_name = next(
            (item["display_name"] for item in result if item.get("s_fused") == max(x["s_fused"] for x in result)),
            result[0]["display_name"],
        )
        print(
            f"[Qwen2VL] Heavy model unavailable; heuristic fallback selected "
            f"'{selected_name}' (not VLM verified)."
        )
        return result
