import os
import sys
import unittest
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from ml.base import SearchResultDTO, OCRResultDTO, CommercialSKUDTO
from ml.fusion.tfidf_ocr_matcher import TfidfOCRMatcher


class TestTfidfOCRFusion(unittest.TestCase):
    """Unit test suite for TF-IDF Character/Word N-Gram OCR Matcher."""

    def setUp(self):
        self.matcher = TfidfOCRMatcher(boost_alpha=0.15)
        self.matcher.initialize({
            "boost_alpha": 0.15,
            "gt_ocr_path": str(workspace_root / "configs/class_ocr_groundtruth.json")
        })

    def test_tfidf_vectorizer_initialization(self):
        """Verifies vectorizer fits on 67 class ground-truth profiles."""
        self.assertEqual(len(self.matcher.gt_profiles), 67)
        self.assertIsNotNone(self.matcher.vectorizer)
        self.assertEqual(self.matcher.gt_matrix.shape[0], 67)

    def test_compute_tfidf_similarity(self):
        """Validates TF-IDF cosine similarity computation against known text queries."""
        query_text = "Lipton Green Tea Lemon 50 tea bags"
        scores = self.matcher.compute_tfidf_similarity(query_text)
        
        self.assertIn(0, scores)
        # Class 0 is "Lipton Green Tea Lemon - 50 Tea Bags", so its similarity should be highest
        self.assertGreater(scores[0], 0.5)

    def test_fuse_reranking_and_boosting(self):
        """Validates candidates reranking and score boosting via TF-IDF fusion."""
        candidates = [
            SearchResultDTO(remapped_class_id=1, old_class_id=1, similarity=0.90, metadata={}),
            SearchResultDTO(remapped_class_id=0, old_class_id=0, similarity=0.89, metadata={})
        ]

        # Query text explicitly contains "Lemon 50 tea bags", matching Class 0
        ocr_result = OCRResultDTO(text="Lipton Green Tea Lemon 50 tea bags box", confidence=0.95)

        fused = self.matcher.fuse(candidates, ocr_result)
        self.assertEqual(len(fused), 2)
        # Class 0 should be boosted to rank 1 due to strong TF-IDF text match
        self.assertEqual(fused[0].remapped_class_id, 0)
        self.assertGreater(fused[0].similarity, fused[1].similarity)

    def test_commercial_sku_dto_serialization(self):
        """Validates CommercialSKUDTO serialization."""
        comm = CommercialSKUDTO(
            project_sku_id="TM_RAW_000",
            display_name="Lipton Green Tea Lemon - 50 Tea Bags",
            brand="Lipton",
            product_name="Green Tea",
            variant="Lemon",
            pack_count="50 tea bags",
            pack_type="box"
        )
        self.assertEqual(comm.project_sku_id, "TM_RAW_000")
        self.assertEqual(comm.brand, "Lipton")


if __name__ == "__main__":
    unittest.main()
