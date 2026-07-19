import io
from typing import List, Union, Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ml.embeddings.base import EmbeddingExtractor
from ml.base import BaseEmbedder, CropDTO, EmbeddingDTO


class CLIPExtractor(EmbeddingExtractor, BaseEmbedder):
    """Concrete implementation of the CLIP vision feature embedding extractor plugin."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu", batch_size: int = 32) -> None:
        """Initializes the CLIP extractor."""
        super().__init__(model_name=model_name, device=device, batch_size=batch_size)
        self.processor = None
        self.model = None
        
        # If initialized directly via constructor, load model immediately
        if model_name:
            self.initialize({"model_name": model_name, "device": device, "batch_size": batch_size})

    def initialize(self, config: Dict[str, Any]) -> None:
        """Loads weights, processor config, and sets eval state."""
        self.model_name = config.get("model_name", self.model_name)
        self.device = config.get("device", self.device)
        self.batch_size = config.get("batch_size", self.batch_size)
        
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model '{self.model_name}': {str(e)}")

    def health_check(self) -> Tuple[bool, str]:
        """Checks if model is loaded and ready."""
        if self.model is None or self.processor is None:
            return False, "Model or processor not initialized."
        return True, "Healthy"

    def shutdown(self) -> None:
        """Releases PyTorch tensors and CUDA memory."""
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def dimension(self) -> int:
        """Returns the embedding feature dimension (512 for clip-vit-base-patch32)."""
        # Retrieve projection dimension config or fall back to 512
        return getattr(self.model.config.projection_dim, "projection_dim", 512)

    def extract(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """Extracts and L2-normalizes embedding vectors using CLIP Vision.

        Args:
            images: List of images as numpy arrays or PIL Images.

        Returns:
            np.ndarray: Normalized float32 matrix of shape (N, D).

        Raises:
            ValueError: If the images list is empty.
        """
        if not images:
            raise ValueError("The images list for feature extraction cannot be empty.")

        all_embeddings = []

        # Process in batches to prevent GPU/CPU memory overflow
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            if (i // self.batch_size) % 10 == 0 or i + self.batch_size >= len(images):
                print(f"CLIP progress: {i}/{len(images)} crops processed...", flush=True)

            # Convert numpy images to PIL images to ensure uniform processing
            processed_batch = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    # Convert BGR (OpenCV standard) to RGB
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = Image.fromarray(img[:, :, ::-1])
                    else:
                        img = Image.fromarray(img)
                processed_batch.append(img)

            # Preprocess images using CLIP processor
            inputs = self.processor(images=processed_batch, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Extract image features directly from CLIP visual head projection space
                batch_feats = self.model.get_image_features(pixel_values=inputs.pixel_values)

                # L2 normalization of vectors
                batch_feats = torch.nn.functional.normalize(batch_feats, p=2, dim=1)
                all_embeddings.append(batch_feats.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def extract_dto(self, crop: CropDTO) -> EmbeddingDTO:
        """Extracts normalized visual feature vector for a single crop."""
        pil_img = Image.open(io.BytesIO(crop.image_bytes)).convert("RGB")
        vec = self.extract([pil_img])[0]
        return EmbeddingDTO(vector=vec.tolist(), dimension=self.dimension)

    def extract_batch_dto(self, crops: List[CropDTO]) -> List[EmbeddingDTO]:
        """Extracts features for a batch of crops safely."""
        pil_images = []
        for crop in crops:
            pil_images.append(Image.open(io.BytesIO(crop.image_bytes)).convert("RGB"))
        vectors = self.extract(pil_images)
        return [EmbeddingDTO(vector=v.tolist(), dimension=self.dimension) for v in vectors]
