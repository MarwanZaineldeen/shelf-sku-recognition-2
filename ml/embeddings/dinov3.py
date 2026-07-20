import os
import io
import sys
import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from pathlib import Path

workspace_root = Path("d:/Marwan/ITI AI&ML/Transmid GP")
sys.path.append(str(workspace_root))

from ml.base import BaseEmbedder, CropDTO, EmbeddingDTO
from transformers import AutoImageProcessor, AutoModel

class DINOv3Extractor(BaseEmbedder):
    """Concrete DINOv3 (ViT-B/16) visual feature extractor plugin.
    
    Extracts 768-D L2-normalized visual feature embeddings for fine-grained FMCG product crops.
    """

    def __init__(
        self,
        model_name_or_path: str = str(workspace_root / "configs/weights/dinov3_vitb16"),
        device: str = "cpu",
        batch_size: int = 16
    ):
        super().__init__(dimension=768)
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self.initialize({"model_name_or_path": model_name_or_path, "device": device, "batch_size": batch_size})

    def initialize(self, config: Dict[str, Any]) -> None:
        """Loads DINOv3 model and processor weights in PyTorch evaluation mode."""
        self.model_name_or_path = config.get("model_name_or_path", self.model_name_or_path)
        self.device = config.get("device", self.device)
        self.batch_size = config.get("batch_size", self.batch_size)
        self.dimension = 768

        try:
            local_path = Path(self.model_name_or_path)
            if local_path.exists():
                self.processor = AutoImageProcessor.from_pretrained(str(local_path), trust_remote_code=True)
                self.model = AutoModel.from_pretrained(str(local_path), trust_remote_code=True)
            else:
                # Fallback to HuggingFace HUB if local folder is missing
                self.processor = AutoImageProcessor.from_pretrained("facebook/dinov3-base", trust_remote_code=True)
                self.model = AutoModel.from_pretrained("facebook/dinov3-base", trust_remote_code=True)

            self.model.to(self.device).eval()
            print(f"[DINOv3] Loaded model successfully on {self.device} (Dimension: {self.dimension}).")
        except Exception as e:
            # Flexible fallback for DINOv2 processor / vision backbone
            print(f"[DINOv3] Warning: Custom DINOv3 AutoModel fallback triggered ({e}). Using DINOv2-base 768-D fallback.")
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base")
            self.model.to(self.device).eval()

    def health_check(self) -> Tuple[bool, str]:
        if self.model is None:
            return False, "DINOv3 model not initialized."
        return True, "Healthy"

    def shutdown(self) -> None:
        self.model = None
        self.processor = None

    def extract(self, images: List[Image.Image]) -> np.ndarray:
        """Extracts 768-D L2-normalized feature vectors for a batch of PIL images."""
        if self.model is None:
            raise RuntimeError("DINOv3 model is not initialized. Call initialize() first.")
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]

            # Convert numpy images to PIL if needed
            processed_batch = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = Image.fromarray(img[:, :, ::-1])
                    else:
                        img = Image.fromarray(img)
                processed_batch.append(img)

            inputs = self.processor(images=processed_batch, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch_feats = outputs.pooler_output
                else:
                    batch_feats = outputs.last_hidden_state[:, 0, :]

                # Enforce L2 Normalization
                batch_feats = torch.nn.functional.normalize(batch_feats, p=2, dim=-1)
                all_embeddings.append(batch_feats.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def extract_dto(self, crop: CropDTO) -> EmbeddingDTO:
        pil_img = Image.open(io.BytesIO(crop.image_bytes)).convert("RGB")
        vec = self.extract([pil_img])[0]
        return EmbeddingDTO(vector=vec.tolist(), dimension=self.dimension)

    def extract_batch_dto(self, crops: List[CropDTO]) -> List[EmbeddingDTO]:
        pil_images = []
        for crop in crops:
            pil_img = Image.open(io.BytesIO(crop.image_bytes)).convert("RGB")
            pil_images.append(pil_img)
        vectors = self.extract(pil_images)
        return [EmbeddingDTO(vector=v.tolist(), dimension=self.dimension) for v in vectors]
