from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from PIL import Image

class EmbeddingExtractor(ABC):
    """Abstract base class representing a pre-trained feature embedding extractor."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        """Initializes the base embedding extractor.

        Args:
            model_name: The identifier name of the pre-trained backbone model.
            device: Computing device to run inference on (e.g. 'cpu', 'cuda').
            batch_size: Processing batch size for data loaders.

        Raises:
            ValueError: If batch_size is non-positive or model_name is empty.
        """
        if not model_name.strip():
            raise ValueError("Model name cannot be empty.")
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the output feature vector dimension.

        Returns:
            int: Dimension of the extracted embeddings.
        """
        pass

    @abstractmethod
    def extract(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """Extracts and L2-normalizes embedding vectors from input images.

        Args:
            images: List of images represented as numpy arrays or PIL Images.

        Returns:
            np.ndarray: Normalized float32 matrix of shape (N, D), where N is the number 
            of images and D is the output dimension.

        Raises:
            ValueError: If images list is empty or contains invalid items.
        """
        pass
