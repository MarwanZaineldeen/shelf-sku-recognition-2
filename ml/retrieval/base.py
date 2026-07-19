from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import numpy as np

def verify_split_disjointness(train_metadata: List[Dict[str, Any]], eval_metadata: List[Dict[str, Any]]) -> None:
    """Verifies that train and evaluation splits share absolutely no visual duplicate family IDs.

    Args:
        train_metadata: List of dictionaries containing training metadata (must include 'family_id').
        eval_metadata: List of dictionaries containing evaluation metadata (must include 'family_id').

    Raises:
        AssertionError: If split contamination is detected.
        ValueError: If metadata records are missing the required 'family_id' field.
    """
    train_families = set()
    for idx, r in enumerate(train_metadata):
        if "family_id" not in r:
            raise ValueError(f"Train record at index {idx} is missing 'family_id'.")
        fid = r["family_id"]
        if fid:  # Only track non-empty family IDs
            train_families.add(fid)

    eval_families = set()
    for idx, r in enumerate(eval_metadata):
        if "family_id" not in r:
            raise ValueError(f"Evaluation record at index {idx} is missing 'family_id'.")
        fid = r["family_id"]
        if fid:  # Only track non-empty family IDs
            eval_families.add(fid)

    leaked = train_families.intersection(eval_families)
    if leaked:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED: Split contamination found! The following duplicate family IDs "
            f"exist in both train and evaluation splits: {leaked}. Re-run split reconciliation first."
        )


class VectorIndex(ABC):
    """Abstract base class representing the gallery vector database and search index."""

    def __init__(self, dimension: int) -> None:
        """Initializes the vector index with its expected feature dimension.

        Args:
            dimension: Feature dimensionality of registered vectors.

        Raises:
            ValueError: If feature dimension is non-positive.
        """
        if dimension <= 0:
            raise ValueError("Feature dimension must be a positive integer.")
        self.dimension = dimension

    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Adds reference vectors and associated metadata to the search index.

        Args:
            vectors: A float32 numpy array of shape (N, D) containing gallery embeddings.
            metadata: A list of N dictionaries containing crop metadata.

        Raises:
            ValueError: If inputs are invalid or counts do not match.
        """
        pass

    @abstractmethod
    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Searches the index for query nearest neighbors.

        Args:
            query_vectors: A float32 numpy array of shape (M, D) containing normalized query vectors.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of:
                - neighbor_indices: (M, K) integer array of retrieved gallery indices.
                - similarity_scores: (M, K) float32 array of cosine similarity scores.

        Raises:
            ValueError: If query_vectors shape is invalid or top_k is non-positive.
        """
        pass
