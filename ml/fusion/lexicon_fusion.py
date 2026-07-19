from typing import Dict, List, Any, Tuple, Optional
from ml.base import BaseFusionStrategy, SearchResultDTO, OCRResultDTO


class LexiconLateFusion(BaseFusionStrategy):
    """Late fusion strategy boosting similarity based on crop text matches."""

    def __init__(self) -> None:
        self.lexicons: Dict[int, List[str]] = {}
        self.boost_alpha = 0.05

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes class lexicons and boost parameters.

        Config schema:
            boost_alpha: 0.05
            lexicons: { "class_id_1": ["lemon", "50"], ... }
        """
        self.boost_alpha = config.get("boost_alpha", self.boost_alpha)
        
        raw_lexicons = config.get("lexicons", {})
        for cid_str, keywords in raw_lexicons.items():
            try:
                cid = int(cid_str)
                self.lexicons[cid] = [str(k).lower() for k in keywords]
            except ValueError:
                continue

    def health_check(self) -> Tuple[bool, str]:
        return True, "Healthy"

    def shutdown(self) -> None:
        pass

    def fuse(self, matches: List[SearchResultDTO], ocr: Optional[OCRResultDTO]) -> List[SearchResultDTO]:
        """Boosts candidate similarities if OCR text matches target class keywords.

        Args:
            matches: List of candidate SearchResultDTOs.
            ocr: Extracted OCRResultDTO.

        Returns:
            List[SearchResultDTO]: Fused and reranked match list.
        """
        if not ocr or not ocr.text.strip():
            return matches

        ocr_text_lower = ocr.text.lower()
        fused_matches = []

        for match in matches:
            cid = match.remapped_class_id
            similarity = match.similarity
            
            lexicon_keywords = self.lexicons.get(cid, [])
            if lexicon_keywords:
                # Count matching keywords
                match_count = sum(1 for kw in lexicon_keywords if kw in ocr_text_lower)
                if match_count > 0:
                    # Boost similarity scaled by OCR confidence, clamped to 1.0
                    similarity += self.boost_alpha * ocr.confidence * match_count
                    similarity = min(similarity, 1.0)
                    
            fused_matches.append(
                SearchResultDTO(
                    remapped_class_id=match.remapped_class_id,
                    old_class_id=match.old_class_id,
                    similarity=similarity,
                    metadata=match.metadata
                )
            )

        # Rerank candidates based on fused similarities
        fused_matches.sort(key=lambda x: x.similarity, reverse=True)
        return fused_matches
