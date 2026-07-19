import unittest
from ml.base import SearchResultDTO, OCRResultDTO
from ml.fusion.lexicon_fusion import LexiconLateFusion


class TestLexiconLateFusion(unittest.TestCase):
    """Unit tests for the visual + OCR lexicon late-fusion strategy plugin."""

    def setUp(self) -> None:
        self.fusion = LexiconLateFusion()
        self.config = {
            "boost_alpha": 0.05,
            "lexicons": {
                "0": ["green", "mint"],
                "1": ["lemon", "50", "bags"],
                "2": ["peach"]
            }
        }
        self.fusion.initialize(self.config)

    def test_fusion_without_ocr(self) -> None:
        """Verifies candidate matches are returned unmodified if OCR output is empty."""
        matches = [
            SearchResultDTO(remapped_class_id=0, old_class_id=10, similarity=0.90, metadata={}),
            SearchResultDTO(remapped_class_id=1, old_class_id=20, similarity=0.85, metadata={})
        ]
        
        fused = self.fusion.fuse(matches, ocr=None)
        self.assertEqual(fused, matches)
        
        ocr_empty = OCRResultDTO(text="", confidence=0.0)
        fused_empty = self.fusion.fuse(matches, ocr=ocr_empty)
        self.assertEqual(fused_empty, matches)

    def test_fusion_with_lexicon_matches(self) -> None:
        """Verifies score boost and reranking triggers when OCR text matches lexicon."""
        matches = [
            SearchResultDTO(remapped_class_id=0, old_class_id=10, similarity=0.92, metadata={}),
            SearchResultDTO(remapped_class_id=1, old_class_id=20, similarity=0.90, metadata={})
        ]
        
        # OCR detects "lemon 50 bags pack" (associated with class_id 1)
        ocr = OCRResultDTO(text="lemon 50 bags pack", confidence=1.0)
        fused = self.fusion.fuse(matches, ocr=ocr)
        
        # Class 1 matching keywords: "lemon", "50", "bags" -> 3 matches
        # Score boost = 0.05 * 1.0 * 3 = 0.15
        # Class 1 similarity = 0.90 + 0.15 = 1.05 (clamped to 1.0)
        # Class 0 matching keywords: None -> score unchanged (0.92)
        # Final reranking should place Class 1 (similarity 1.0) above Class 0 (similarity 0.92)
        
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].remapped_class_id, 1)
        self.assertAlmostEqual(fused[0].similarity, 1.0)
        self.assertEqual(fused[1].remapped_class_id, 0)
        self.assertAlmostEqual(fused[1].similarity, 0.92)

    def test_partial_match_and_scaling(self) -> None:
        """Verifies that boost scales with OCR model confidence."""
        matches = [
            SearchResultDTO(remapped_class_id=0, old_class_id=10, similarity=0.80, metadata={})
        ]
        
        # Only "green" matches (1 match), confidence is 0.80
        # Boost = 0.05 * 0.80 * 1 = 0.04
        # Expected score = 0.80 + 0.04 = 0.84
        ocr = OCRResultDTO(text="pure green tea", confidence=0.80)
        fused = self.fusion.fuse(matches, ocr=ocr)
        
        self.assertAlmostEqual(fused[0].similarity, 0.84)


if __name__ == "__main__":
    unittest.main()
