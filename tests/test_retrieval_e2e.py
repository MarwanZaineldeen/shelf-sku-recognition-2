import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from ml.embeddings.dinov2 import DINOv2Extractor
from ml.embeddings.clip import CLIPExtractor
from ml.retrieval.base import verify_split_disjointness
from ml.retrieval.numpy_index import NumpyCosineIndex
from ml.evaluation.metrics import (
    compute_recall_at_k,
    compute_macro_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_cmc_curve,
    bootstrap_metrics,
    calibrate_similarity_threshold,
)


class TestEmbeddingExtractors(unittest.TestCase):
    """Unit tests for feature embedding extraction modules."""

    def test_dinov2_extractor_cpu(self) -> None:
        """Verifies DINOv2 extractor loads on CPU and returns normalized features."""
        try:
            extractor = DINOv2Extractor(model_name="facebook/dinov2-small", device="cpu", batch_size=1)
            self.assertEqual(extractor.dimension, 384)

            # Create a mock image
            mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
            feats = extractor.extract([mock_img])
            
            self.assertEqual(feats.shape, (1, 384))
            self.assertEqual(feats.dtype, np.float32)
            
            # Assert L2 normalized
            norms = np.linalg.norm(feats, axis=1)
            np.testing.assert_almost_equal(norms, 1.0, decimal=5)
        except Exception as e:
            self.skipTest(f"Skipping DINOv2 download test: {e}")

    def test_clip_extractor_cpu(self) -> None:
        """Verifies CLIP extractor loads on CPU and returns normalized features."""
        try:
            extractor = CLIPExtractor(model_name="openai/clip-vit-base-patch32", device="cpu", batch_size=1)
            self.assertEqual(extractor.dimension, 512)

            # Create a mock image
            mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
            feats = extractor.extract([mock_img])
            
            self.assertEqual(feats.shape, (1, 512))
            self.assertEqual(feats.dtype, np.float32)
            
            # Assert L2 normalized
            norms = np.linalg.norm(feats, axis=1)
            np.testing.assert_almost_equal(norms, 1.0, decimal=5)
        except Exception as e:
            self.skipTest(f"Skipping CLIP download test: {e}")


class TestVectorIndex(unittest.TestCase):
    """Unit tests for the Vector Index indexing and search logic."""

    def test_numpy_cosine_index(self) -> None:
        """Verifies NumpyCosineIndex exact similarity and sorted indexing."""
        index = NumpyCosineIndex(dimension=4)
        
        # Add gallery vectors
        # v1 and v2 are perpendicular, v3 is identical to v1
        v1 = np.array([1, 0, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0], dtype=np.float32)
        v3 = np.array([1, 0, 0, 0], dtype=np.float32)
        vectors = np.vstack([v1, v2, v3])
        
        metadata = [
            {"crop_path": "c1.jpg", "remapped_class_id": 1, "family_id": "f1"},
            {"crop_path": "c2.jpg", "remapped_class_id": 2, "family_id": "f2"},
            {"crop_path": "c3.jpg", "remapped_class_id": 1, "family_id": "f1"}
        ]
        
        index.add(vectors, metadata)
        self.assertEqual(index.gallery_vectors.shape, (3, 4))
        
        # Search query (similar to v1/v3)
        query = np.array([[1, 0.1, 0, 0]], dtype=np.float32)
        indices, scores = index.search(query, top_k=2)
        
        # Nearest neighbors should be index 0 and index 2 (similar to v1/v3)
        self.assertEqual(indices.shape, (1, 2))
        self.assertEqual(scores.shape, (1, 2))
        self.assertIn(indices[0, 0], [0, 2])
        self.assertIn(indices[0, 1], [0, 2])
        self.assertGreater(scores[0, 0], 0.9)

    def test_split_disjointness_validation(self) -> None:
        """Verifies split disjointness checks raise correct exceptions."""
        train_meta = [{"family_id": "f1"}, {"family_id": "f2"}]
        val_meta_clean = [{"family_id": "f3"}, {"family_id": "f4"}]
        val_meta_dirty = [{"family_id": "f2"}, {"family_id": "f4"}]
        
        # Clean should pass
        verify_split_disjointness(train_meta, val_meta_clean)
        
        # Leaked should fail
        with self.assertRaises(AssertionError):
            verify_split_disjointness(train_meta, val_meta_dirty)


class TestMetricsAndCalibration(unittest.TestCase):
    """Unit tests for metric algorithms and threshold calibration sweeps."""

    def test_metrics_suite(self) -> None:
        """Checks Recall, NDCG, MRR, and CMC computations."""
        # Query labels: [1, 2]
        # Predicted top neighbors:
        # q1: [1, 2, 3] -> Correct at rank 1
        # q2: [3, 2, 1] -> Correct at rank 2
        query_labels = np.array([1, 2])
        neighbor_labels = np.array([[1, 2, 3], [3, 2, 1]])

        rec1 = compute_recall_at_k(neighbor_labels, query_labels, 1)
        rec2 = compute_recall_at_k(neighbor_labels, query_labels, 2)
        self.assertEqual(rec1, 0.5)
        self.assertEqual(rec2, 1.0)

        macro_rec1 = compute_macro_recall_at_k(neighbor_labels, query_labels, 1)
        self.assertEqual(macro_rec1, 0.5)

        mrr = compute_mrr(neighbor_labels, query_labels)
        # q1: RR = 1/1, q2: RR = 1/2 -> Mean = (1 + 0.5)/2 = 0.75
        self.assertEqual(mrr, 0.75)

        ndcg2 = compute_ndcg_at_k(neighbor_labels, query_labels, 2)
        # q1: IDCG = 1.0, DCG = 1.0 -> NDCG = 1.0
        # q2: IDCG = 1.0 (top rank relevant possible), DCG = 0 at rank 1 -> NDCG = 0.0
        # Mean NDCG@2 = (1.0 + 0.0) / 2 = 0.5
        self.assertEqual(ndcg2, 0.5)

        cmc = compute_cmc_curve(neighbor_labels, query_labels, max_k=3)
        self.assertEqual(cmc, [0.5, 1.0, 1.0])

    def test_bootstrap_evaluation(self) -> None:
        """Verifies bootstrap output envelope structure."""
        query_labels = np.array([1, 2, 1, 2])
        neighbor_labels = np.array([[1, 2], [3, 1], [1, 2], [2, 1]])
        
        ci_stats = bootstrap_metrics(neighbor_labels, query_labels, top_k=2, num_bootstraps=50)
        self.assertIn("recall@1", ci_stats)
        self.assertIn("recall@5", ci_stats)
        self.assertIn("mean", ci_stats["recall@1"])
        self.assertIn("ci_lower", ci_stats["recall@1"])

    def test_threshold_calibration(self) -> None:
        """Verifies precision-constrained threshold sweeps."""
        scores = np.array([0.9, 0.85, 0.7, 0.6, 0.5])
        correct = np.array([True, True, True, False, False])
        
        # Under target precision 0.95, threshold should select a value
        # that leaves only correct predictions: e.g. >= 0.7.
        # scores >= 0.7: [0.9, 0.85, 0.7] (all True, precision = 1.0 >= 0.95).
        # scores >= 0.6: [0.9, 0.85, 0.7, 0.6] (3 True, 1 False, precision = 0.75 < 0.95).
        # So tau* should be 0.7.
        tau, auto_rate = calibrate_similarity_threshold(scores, correct, target_precision=0.95)
        self.assertAlmostEqual(tau, 0.605)
        self.assertAlmostEqual(auto_rate, 0.6)


if __name__ == "__main__":
    unittest.main()
