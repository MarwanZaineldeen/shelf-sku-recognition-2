"""DINOv3 ViT-B/16 (768-D) Feature Extractor — Native transformers >= 4.56 support.

Uses the teammate's offline model weights with native DINOv3ViTModel from
the transformers library. Produces L2-normalised CLS-token embeddings.

CRITICAL: Uses the teammate's aspect-ratio-preserving resize + gray-canvas
padding preprocessing to match the DB reference embeddings exactly.
"""

import io
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import List, Dict, Any, Tuple
from pathlib import Path

from ml.base import BaseEmbedder, CropDTO, EmbeddingDTO

# Teammate's offline model weights location
_PKG_MODEL_DIR = Path(
    "d:/Marwan/ITI AI&ML/Transmid GP"
    "/scratch/teammate_pkg"
    "/dinov3_v2_exemplar_all_flat_offline_v1"
    "/dinov3_v2_exemplar_all_flat_offline_v1"
    "/model"
)

# Teammate's preprocessing constants (must match how DB embeddings were built)
_IMAGE_SIZE = 224
_PAD_COLOR = (124, 116, 104)  # ImageNet-mean gray


class DINOv3Extractor(BaseEmbedder):
    """DINOv3 ViT-B/16 768-D feature extractor using native HF support."""

    def __init__(
        self,
        model_dir: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self._model_dir = Path(model_dir) if model_dir else _PKG_MODEL_DIR
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self._dimension = 768
        self.initialize({})

    # ── BaseEmbedder interface ───────────────────────────────────────

    @property
    def dimension(self) -> int:
        return self._dimension

    def initialize(self, config: Dict[str, Any]) -> None:
        """Load DINOv3ViTModel from local safetensors via AutoModel."""
        from transformers import AutoModel, AutoImageProcessor

        model_path = str(self._model_dir)
        self.processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True
        )
        self.model.to(self.device).eval()
        self._dimension = self.model.config.hidden_size
        print(
            f"[DINOv3] Native DINOv3ViTModel loaded — "
            f"{type(self.model).__name__}, dim={self._dimension}, "
            f"device={self.device}"
        )

    def health_check(self) -> Tuple[bool, str]:
        if self.model is None:
            return False, "DINOv3 model not initialised."
        return True, "Healthy"

    def shutdown(self) -> None:
        self.model = None
        self.processor = None

    # ── Teammate's exact preprocessing ───────────────────────────────

    @staticmethod
    def _prepare_pil(image: Image.Image) -> Image.Image:
        """Aspect-ratio-preserving resize + gray-canvas padding.

        Reproduces the teammate's ``_prepare_pil`` exactly so that query
        embeddings are in the same distribution as the reference DB vectors.
        """
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = _IMAGE_SIZE / max(image.width, image.height)
        width = max(1, min(_IMAGE_SIZE, round(image.width * scale)))
        height = max(1, min(_IMAGE_SIZE, round(image.height * scale)))
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (_IMAGE_SIZE, _IMAGE_SIZE), color=_PAD_COLOR)
        canvas.paste(image, ((_IMAGE_SIZE - width) // 2, (_IMAGE_SIZE - height) // 2))
        return canvas

    # ── Core embedding logic ─────────────────────────────────────────

    def extract(self, images: List[Image.Image]) -> np.ndarray:
        """Return (N, 768) L2-normalised CLS-token embeddings."""
        if not images:
            return np.empty((0, self._dimension), dtype=np.float32)

        all_vecs: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch_pil = images[start: start + self.batch_size]

            # Apply teammate's exact preprocessing
            prepared = []
            for img in batch_pil:
                if isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = Image.fromarray(img[:, :, ::-1])  # BGR→RGB
                    else:
                        img = Image.fromarray(img)
                prepared.append(self._prepare_pil(img))

            # Use processor with resize/crop disabled since we already preprocessed
            inputs = self.processor(
                images=prepared,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )
            inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                out = self.model(**inputs)
                cls_tok = out.last_hidden_state[:, 0, :]
                cls_tok = torch.nn.functional.normalize(
                    cls_tok.float(), p=2, dim=1
                )
            all_vecs.append(cls_tok.cpu().numpy().astype(np.float32))

        return np.concatenate(all_vecs, axis=0)

    # ── DTO wrappers ─────────────────────────────────────────────────

    def extract_dto(self, crop: CropDTO) -> EmbeddingDTO:
        pil_img = Image.open(io.BytesIO(crop.image_bytes)).convert("RGB")
        vec = self.extract([pil_img])[0]
        return EmbeddingDTO(vector=vec.tolist(), dimension=self._dimension)

    def extract_batch_dto(self, crops: List[CropDTO]) -> List[EmbeddingDTO]:
        pil_images = [
            Image.open(io.BytesIO(c.image_bytes)).convert("RGB")
            for c in crops
        ]
        vectors = self.extract(pil_images)
        return [
            EmbeddingDTO(vector=v.tolist(), dimension=self._dimension)
            for v in vectors
        ]
