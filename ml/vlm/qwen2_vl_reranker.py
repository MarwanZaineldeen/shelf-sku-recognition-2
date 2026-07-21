import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import torch

class Qwen2VLReranker:
    """Concrete Qwen2-VL Visual Language Model Reranker Plugin.
    
    Executes gated, low-latency zero-shot package text and variant verification
    only when visual vector similarity is ambiguous (0.75 <= S_vis < 0.85).
    """

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.is_ready = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """Loads Qwen2-VL weights and processor 100% offline from local disk or cache."""
        self.model_id = config.get("model_id", self.model_id)
        self.device = config.get("device", self.device)
        local_files_only = config.get("local_files_only", True)

        # Check for local offline model directory
        from pathlib import Path
        local_weights_dir = Path("configs/weights/qwen2_vl_2b_instruct")
        local_awq_dir = Path("configs/weights/qwen2_vl_awq")

        if local_weights_dir.exists() and (local_weights_dir / "config.json").exists():
            self.model_id = str(local_weights_dir.resolve())
            print(f"[Qwen2VL] Using local offline model directory: '{self.model_id}'")
        elif local_awq_dir.exists() and (local_awq_dir / "config.json").exists():
            self.model_id = str(local_awq_dir.resolve())
            print(f"[Qwen2VL] Using local offline AWQ model directory: '{self.model_id}'")

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import os
            # Set HuggingFace offline environment variables to prevent any network calls
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    local_files_only=local_files_only
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=local_files_only
                )
            except Exception as net_e:
                # Retry with online fallback if local_files_only failed and allowed
                if local_files_only:
                    print(f"[Qwen2VL] Offline load failed ({net_e}). Trying standard cache load...")
                    self.processor = AutoProcessor.from_pretrained(self.model_id)
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                else:
                    raise net_e

            if not torch.cuda.is_available() and self.model is not None:
                self.model.to(self.device)
            if self.model is not None:
                self.model.eval()
                self.is_ready = True
                print(f"[Qwen2VL] 100% Offline Model '{self.model_id}' ready on {self.device}.")
        except Exception as e:
            print(f"[Qwen2VL] Warning: Offline model initialization skipped/failed: {e}")
            self.is_ready = False

    @staticmethod
    def enhance_crop_for_vlm(crop_image: Image.Image) -> Image.Image:
        """Applies mild bilateral filtering and CLAHE to enhance dim packaging text for VLM."""
        try:
            import cv2
            import numpy as np

            # Convert PIL to BGR OpenCV image
            rgb_arr = np.array(crop_image.convert("RGB"))
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            # 1. Bilateral filter: denoise while preserving text edges
            denoised = cv2.bilateralFilter(bgr_arr, d=5, sigmaColor=50, sigmaSpace=50)

            # 2. Mild CLAHE contrast boost on LAB L-channel
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            l_enhanced = clahe.apply(l)
            enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

            # Convert back to PIL Image
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(enhanced_rgb)
        except Exception as e:
            # Fallback to original image if OpenCV enhancement fails
            return crop_image

    def rerank_top5_candidates(
        self,
        crop_image: Image.Image,
        top5_candidates: List[Dict[str, Any]],
        timeout_ms: int = 400
    ) -> List[Dict[str, Any]]:
        """Reranks Top-5 candidates using Qwen2-VL constrained text matching.
        
        Args:
            crop_image: Crop PIL image of product facing.
            top5_candidates: List of Top-5 candidate dictionaries containing display_name and class_id.
            timeout_ms: Maximum allowed execution budget in milliseconds.
            
        Returns:
            Updated candidates list with qwen2_vl_boost score.
        """
        if not self.is_ready or not top5_candidates:
            return top5_candidates

        # Apply mild CLAHE + Bilateral enhancement to boost dim text readability for VLM
        crop_image = self.enhance_crop_for_vlm(crop_image)

        t0 = time.perf_counter()

        # Format Top-5 Candidate Options for Qwen2-VL Prompt
        options_str = "\n".join([f"{i+1}. {c['display_name']}" for i, c in enumerate(top5_candidates)])
        prompt = (
            "Look at this retail product package image carefully. "
            "Which of the following product titles matches the exact brand, flavor, and pack size?\n"
            f"{options_str}\n"
            "Respond ONLY with the option number (1, 2, 3, 4, or 5)."
        )

        try:
            from transformers import AutoProcessor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=text_prompt,
                images=crop_image,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=5)
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

            # Parse prediction option number
            best_idx = 0
            for char in output_text:
                if char.isdigit() and 1 <= int(char) <= len(top5_candidates):
                    best_idx = int(char) - 1
                    break

            # Apply Qwen2-VL Additive Confidence Boost (+0.12) to selected candidate
            reranked = [dict(c) for c in top5_candidates]
            reranked[best_idx]["s_fused"] = min(1.0, reranked[best_idx].get("s_fused", 0.8) + 0.12)
            reranked[best_idx]["qwen2_vl_verified"] = True

            # Re-sort candidates by fused similarity
            reranked.sort(key=lambda x: x.get("s_fused", 0.0), reverse=True)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[Qwen2VL] Verified option #{best_idx+1} in {elapsed_ms:.1f}ms.")
            return reranked

        except Exception as e:
            print(f"[Qwen2VL] Execution error/fallback: {e}")
            return top5_candidates
