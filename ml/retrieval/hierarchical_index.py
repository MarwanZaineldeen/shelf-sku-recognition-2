from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from ml.retrieval.base import VectorIndex
from ml.base import BaseRetriever, EmbeddingDTO, SearchResultDTO


class HierarchicalCosineIndex(VectorIndex, BaseRetriever):
    """2-Layer Hierarchical Search & Brand Clustering Retriever Plugin.

    Layer 1: Brand / Family Centroid Identification
    Layer 2: Partitioned Nearest-Neighbor SKU Search within Predicted Brand
    """

    def __init__(self, dimension: int = 384) -> None:
        super().__init__(dimension=dimension)
        self.gallery_vectors: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []

        # Partitioned index storage
        self.brand_vectors: Dict[str, np.ndarray] = {}
        self.brand_metadata: Dict[str, List[Dict[str, Any]]] = {}
        self.brand_centroids: Dict[str, np.ndarray] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """Configures search index and loads database reference vectors."""
        self.dimension = config.get("dimension", self.dimension)
        db_path = config.get("db_path")
        if db_path:
            from ml.retrieval.sqlite_registry import SQLiteGalleryStore
            store = SQLiteGalleryStore()
            store.initialize({"db_path": db_path})
            embeddings, metadata = store.fetch_all_references()
            store.shutdown()

            if len(embeddings) > 0:
                if isinstance(embeddings, np.ndarray):
                    vectors = embeddings
                else:
                    vectors = np.array([e.vector for e in embeddings], dtype=np.float32)
                self.add(vectors, metadata)

    def health_check(self) -> Tuple[bool, str]:
        if not self.brand_vectors or len(self.metadata) == 0:
            return False, "Hierarchical index is empty."
        return True, "Healthy"

    def shutdown(self) -> None:
        self.gallery_vectors = None
        self.metadata = []
        self.brand_vectors.clear()
        self.brand_metadata.clear()
        self.brand_centroids.clear()

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Adds reference vectors and partitions them into brand clusters."""
        if vectors.shape[0] != len(metadata):
            raise ValueError("Number of vectors must equal metadata items count.")

        # L2 normalize gallery vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        vectors_normalized = (vectors / norms).astype(np.float32)

        self.gallery_vectors = vectors_normalized
        self.metadata = metadata

        # Partition into brand clusters based on family_id
        temp_vectors: Dict[str, List[np.ndarray]] = {}
        temp_meta: Dict[str, List[Dict[str, Any]]] = {}

        for i, meta in enumerate(metadata):
            fam_id = str(meta.get("family_id", "default_brand"))
            if fam_id not in temp_vectors:
                temp_vectors[fam_id] = []
                temp_meta[fam_id] = []

            temp_vectors[fam_id].append(vectors_normalized[i])
            temp_meta[fam_id].append(meta)

        # Build brand centroids and partitioned matrices
        for fam_id, vec_list in temp_vectors.items():
            arr = np.vstack(vec_list).astype(np.float32)
            self.brand_vectors[fam_id] = arr
            self.brand_metadata[fam_id] = temp_meta[fam_id]

            # Compute normalized centroid vector for Layer 1
            mean_vec = np.mean(arr, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            self.brand_centroids[fam_id] = mean_vec.astype(np.float32)

    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Matrix-based query search implementation for VectorIndex compatibility."""
        if self.gallery_vectors is None or self.gallery_vectors.shape[0] == 0:
            raise ValueError("Search index is empty. Call add() first.")

        num_gallery = self.gallery_vectors.shape[0]
        actual_k = min(top_k, num_gallery)

        queries_clean = query_vectors.astype(np.float32)
        norms = np.linalg.norm(queries_clean, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        queries_normalized = queries_clean / norms

        scores = np.dot(queries_normalized, self.gallery_vectors.T)
        partition_indices = np.argpartition(-scores, actual_k - 1, axis=1)[:, :actual_k]

        all_sorted_indices = []
        all_sorted_scores = []
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
        """Executes 2-layer hierarchical retrieval.

        Args:
            embedding: Query embedding DTO.
            top_k: Number of SKU candidates to retrieve.
            family_id: Optional explicit brand ID to skip Layer 1 brand identification.

        Returns:
            List[SearchResultDTO]: Matching candidate results.
        """
        if not self.brand_vectors:
            return []

        query_vec = np.array(embedding.vector, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Layer 1: Identify Target Brand Partition
        target_brand = family_id

        if not target_brand or target_brand not in self.brand_vectors:
            # Match query against all Brand Centroids
            best_brand = None
            best_brand_sim = -1.0

            for fam_id, centroid in self.brand_centroids.items():
                sim = float(np.dot(query_vec, centroid))
                if sim > best_brand_sim:
                    best_brand_sim = sim
                    best_brand = fam_id

            target_brand = best_brand if best_brand else list(self.brand_vectors.keys())[0]

        # Layer 2: Partitioned SKU Nearest-Neighbor Search
        part_vectors = self.brand_vectors[target_brand]
        part_meta = self.brand_metadata[target_brand]

        scores = np.dot(part_vectors, query_vec)
        num_items = len(scores)
        actual_k = min(top_k, num_items)

        # Sort descending
        top_indices = np.argsort(-scores)[:actual_k]

        results = []
        for idx in top_indices:
            meta = part_meta[idx]
            sim_score = float(scores[idx])
            results.append(
                SearchResultDTO(
                    remapped_class_id=int(meta["remapped_class_id"]),
                    old_class_id=int(meta.get("old_class_id", -1)),
                    similarity=sim_score,
                    metadata=meta
                )
            )

        return results
