from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from ml.retrieval.base import VectorIndex
from ml.base import BaseRetriever, EmbeddingDTO, SearchResultDTO


class NumpyCosineIndex(VectorIndex, BaseRetriever):
    """Concrete NumPy implementation of exact cosine similarity search index plugin."""

    def __init__(self, dimension: int = 384) -> None:
        """Initializes the NumPy Cosine index."""
        super().__init__(dimension=dimension)
        self.gallery_vectors: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []

    def initialize(self, config: Dict[str, Any]) -> None:
        """Configures the search index. Optionally loads reference vectors from an SQLite database."""
        self.dimension = config.get("dimension", self.dimension)
        db_path = config.get("db_path")
        if db_path:
            from ml.retrieval.sqlite_registry import SQLiteGalleryStore
            store = SQLiteGalleryStore()
            store.initialize({"db_path": db_path})
            embeddings, metadata = store.fetch_all_references()
            store.shutdown()
            
            if embeddings:
                # Convert list of EmbeddingDTO to pre-allocated numpy matrix
                vectors = np.empty((len(embeddings), self.dimension), dtype=np.float32)
                for i, e in enumerate(embeddings):
                    vectors[i] = e.vector
                self.add(vectors, metadata)

    def health_check(self) -> Tuple[bool, str]:
        """Diagnostics checks."""
        if self.gallery_vectors is None or len(self.metadata) == 0:
            return False, "Index is empty."
        return True, "Healthy"

    def shutdown(self) -> None:
        """Releases vector memory allocation."""
        self.gallery_vectors = None
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Adds gallery reference vectors and metadata to the NumPy index.

        Args:
            vectors: Float32 numpy array of shape (N, D) containing embeddings.
            metadata: List of N dictionaries containing crop metadata.

        Raises:
            ValueError: If input dimensions do not match, or counts of vectors and metadata mismatch.
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("Gallery vectors must be a numpy array.")
        if len(vectors.shape) != 2:
            raise ValueError(f"Gallery vectors must be 2D array, got shape {vectors.shape}.")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}."
            )
        if len(metadata) != vectors.shape[0]:
            raise ValueError(
                f"Count mismatch. Got {vectors.shape[0]} vectors, but {len(metadata)} metadata records."
            )

        # Enforce float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # In-place L2 normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        vectors /= norms

        if self.gallery_vectors is None:
            self.gallery_vectors = vectors
            self.metadata = list(metadata)
        else:
            self.gallery_vectors = np.vstack([self.gallery_vectors, vectors])
            self.metadata.extend(metadata)

    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Searches the index for query nearest neighbors using Cosine Similarity.

        Args:
            query_vectors: Float32 numpy array of shape (M, D) containing query vectors.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two numpy arrays:
                - neighbor_indices: (M, K) integer array of retrieved gallery indices.
                - similarity_scores: (M, K) float32 array of cosine similarity scores.

        Raises:
            RuntimeError: If search is executed on an empty index.
            ValueError: If query_vectors shape is invalid or top_k is non-positive.
        """
        if self.gallery_vectors is None or len(self.metadata) == 0:
            raise RuntimeError("Index is empty. Call index.add() before search.")
        if not isinstance(query_vectors, np.ndarray):
            raise ValueError("Query vectors must be a numpy array.")
        if len(query_vectors.shape) != 2:
            raise ValueError(f"Query vectors must be 2D array, got shape {query_vectors.shape}.")
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch. Expected {self.dimension}, got {query_vectors.shape[1]}."
            )
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # Clamp top_k to size of gallery
        num_gallery = self.gallery_vectors.shape[0]
        if top_k > num_gallery:
            top_k = num_gallery

        # Normalize queries
        queries_clean = query_vectors.astype(np.float32)
        norms = np.linalg.norm(queries_clean, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        queries_normalized = queries_clean / norms

        # Process queries in batches to avoid high memory allocations (e.g. MemoryError)
        batch_size = 500
        all_sorted_indices = []
        all_sorted_scores = []

        for batch_start in range(0, queries_normalized.shape[0], batch_size):
            batch_queries = queries_normalized[batch_start : batch_start + batch_size]
            scores = np.dot(batch_queries, self.gallery_vectors.T)
            
            # Partition index array for top_k largest values (unsorted)
            partition_indices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
            
            # Precisely sort the top_k elements descending per row
            for row_idx, top_indices in enumerate(partition_indices):
                row_scores = scores[row_idx, top_indices]
                sort_order = np.argsort(-row_scores)
                all_sorted_indices.append(top_indices[sort_order])
                all_sorted_scores.append(row_scores[sort_order])

        return np.vstack(all_sorted_indices), np.vstack(all_sorted_scores)

    def search_dto(
        self,
        embedding: EmbeddingDTO,
        top_k: int,
        family_id: Optional[str] = None
    ) -> List[SearchResultDTO]:
        """Finds closest candidates in the gallery registry.

        Args:
            embedding: The DTO containing the query vector representation.
            top_k: Number of nearest neighbors to retrieve.
            family_id: Optional Brand / Family ID filter.

        Returns:
            List[SearchResultDTO]: Decoded search matching results.
        """
        query_vector = np.array([embedding.vector], dtype=np.float32)
        indices_matrix, scores_matrix = self.search(query_vector, top_k=top_k)
        
        results = []
        for idx, score in zip(indices_matrix[0], scores_matrix[0]):
            meta = self.metadata[idx]
            results.append(SearchResultDTO(
                remapped_class_id=int(meta["remapped_class_id"]),
                old_class_id=int(meta.get("old_class_id", -1)),
                similarity=float(score),
                metadata=meta
            ))
        return results
