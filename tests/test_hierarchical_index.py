import unittest
import numpy as np
from ml.base import EmbeddingDTO
from ml.retrieval.hierarchical_index import HierarchicalCosineIndex


class TestHierarchicalCosineIndex(unittest.TestCase):
    """Unit tests for 2-Layer Hierarchical Search & Brand Clustering Retriever."""

    def setUp(self) -> None:
        self.index = HierarchicalCosineIndex(dimension=4)

        # Create mock vectors for 2 distinct brands
        # Brand A (Lipton): Centroid roughly around [1, 0, 0, 0]
        # Brand B (Nestle): Centroid roughly around [0, 1, 0, 0]
        vectors = np.array([
            [1.0, 0.1, 0.0, 0.0],  # Lipton SKU 1 (Class 0)
            [0.9, 0.2, 0.0, 0.0],  # Lipton SKU 2 (Class 1)
            [0.0, 0.9, 0.1, 0.0],  # Nestle SKU 1 (Class 10)
            [0.1, 1.0, 0.0, 0.0]   # Nestle SKU 2 (Class 11)
        ], dtype=np.float32)

        metadata = [
            {"remapped_class_id": 0, "old_class_id": 0, "family_id": "Lipton"},
            {"remapped_class_id": 1, "old_class_id": 1, "family_id": "Lipton"},
            {"remapped_class_id": 10, "old_class_id": 10, "family_id": "Nestle"},
            {"remapped_class_id": 11, "old_class_id": 11, "family_id": "Nestle"}
        ]

        self.index.add(vectors, metadata)

    def test_layer1_brand_centroid_and_sku_search(self) -> None:
        """Verifies Layer 1 correctly routes Lipton query to Lipton partition."""
        query_vec = [0.95, 0.05, 0.0, 0.0]  # Visual vector close to Lipton
        embedding = EmbeddingDTO(vector=query_vec, dimension=4)

        # Execute 2-layer search
        results = self.index.search_dto(embedding, top_k=2)
        
        self.assertEqual(len(results), 2)
        # Results should belong to Lipton brand family
        self.assertIn(results[0].remapped_class_id, [0, 1])
        self.assertEqual(results[0].metadata["family_id"], "Lipton")

    def test_explicit_family_id_filtering(self) -> None:
        """Verifies explicit family_id bypasses Layer 1 and forces search inside requested brand."""
        query_vec = [1.0, 0.0, 0.0, 0.0]
        embedding = EmbeddingDTO(vector=query_vec, dimension=4)

        # Force search inside Nestle brand
        results = self.index.search_dto(embedding, top_k=2, family_id="Nestle")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].metadata["family_id"], "Nestle")
        self.assertIn(results[0].remapped_class_id, [10, 11])

    def test_health_check_and_shutdown(self) -> None:
        """Verifies plugin diagnostics and memory cleanup."""
        ok, msg = self.index.health_check()
        self.assertTrue(ok)
        self.assertEqual(msg, "Healthy")

        self.index.shutdown()
        ok_after, _ = self.index.health_check()
        self.assertFalse(ok_after)


if __name__ == "__main__":
    unittest.main()
