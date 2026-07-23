import unittest
from PIL import Image
import io
from ml.vlm.qwen2_vl_reranker import Qwen2VLReranker


class TestQwen2VLRerankerRejection(unittest.TestCase):
    """Unit tests to verify Qwen2-VL visual-language open-set rejection for unknown SKUs."""

    def setUp(self) -> None:
        self.reranker = Qwen2VLReranker()
        # Initialize with offline mode parameters
        self.reranker.initialize({"local_files_only": True})
        
        # Create a dummy image
        self.dummy_img = Image.new("RGB", (100, 100), color="red")

    def test_open_set_rejection_low_similarity(self) -> None:
        """Verifies that visual similarity < 0.62 triggers direct 'Class Unknown' (-1) assignment."""
        candidates = [
            {"class_id": 5, "display_name": "Lipton Mint", "similarity": 0.58},
            {"class_id": 12, "display_name": "Lipton Lemon", "similarity": 0.51},
        ]
        
        reranked = self.reranker.rerank_top5_candidates(self.dummy_img, candidates)
        
        # Must return Class Unknown (-1) as the top choice
        self.assertTrue(len(reranked) > 0)
        self.assertEqual(reranked[0]["class_id"], -1)
        self.assertEqual(reranked[0]["display_name"], "Class Unknown")
        self.assertTrue(reranked[0]["vlm_selected"])

    def test_no_rejection_high_similarity(self) -> None:
        """Verifies that visual similarity >= 0.62 does not trigger immediate open-set rejection."""
        candidates = [
            {"class_id": 5, "display_name": "Lipton Mint", "similarity": 0.75},
            {"class_id": 12, "display_name": "Lipton Lemon", "similarity": 0.70},
        ]
        
        reranked = self.reranker.rerank_top5_candidates(self.dummy_img, candidates)
        
        # First choice should be one of the catalog candidates, not -1
        self.assertNotEqual(reranked[0]["class_id"], -1)
        # Verify fusion score boosts were calculated
        self.assertIn("s_fused", reranked[0])


if __name__ == "__main__":
    unittest.main()
