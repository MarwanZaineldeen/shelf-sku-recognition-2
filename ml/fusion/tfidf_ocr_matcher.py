import json
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml.base import BaseFusionStrategy, SearchResultDTO, OCRResultDTO


class TfidfOCRMatcher(BaseFusionStrategy):
    """
    TF-IDF Character/Word N-Gram OCR Matcher & Fusion Strategy.
    
    Compares query crop extracted OCR text against 67 precalculated Ground-Truth
    Class OCR text profiles using TF-IDF character N-gram cosine similarity.
    Fuses text match scores with visual embedding similarities to resolve fine-grained
    packaging variants (flavor, weight, pack size).
    """

    def __init__(self, boost_alpha: float = 0.15):
        self.boost_alpha = boost_alpha
        self.gt_profiles: Dict[str, Dict[str, Any]] = {}
        self.vectorizer: TfidfVectorizer = None
        self.class_indices: List[int] = []
        self.gt_matrix = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes TF-IDF vectorizer on 67 class OCR ground-truth profiles."""
        self.boost_alpha = config.get("boost_alpha", 0.15)
        gt_json_path = config.get("gt_ocr_path", "configs/class_ocr_groundtruth.json")

        with open(gt_json_path, "r", encoding="utf-8") as f:
            self.gt_profiles = json.load(f)

        # Prepare corpus for TF-IDF Vectorizer
        corpus = []
        self.class_indices = []

        for cid_str, profile in self.gt_profiles.items():
            cid = int(cid_str)
            text = profile.get("precalculated_ocr_text", "")
            corpus.append(text)
            self.class_indices.append(cid)

        # Fit TfidfVectorizer with character-level 3-gram analyzer (handles noisy OCR OCR typos)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True
        )
        self.gt_matrix = self.vectorizer.fit_transform(corpus)

    def compute_tfidf_similarity(self, query_text: str) -> Dict[int, float]:
        """Computes TF-IDF cosine similarity scores for query text against all 67 classes."""
        if not query_text.strip() or self.vectorizer is None:
            return {}

        query_vec = self.vectorizer.transform([query_text.strip()])
        sim_scores = cosine_similarity(query_vec, self.gt_matrix)[0]

        scores_dict = {}
        for idx, cid in enumerate(self.class_indices):
            scores_dict[cid] = float(sim_scores[idx])

        return scores_dict

    def fuse(
        self,
        search_results: List[SearchResultDTO],
        ocr_result: OCRResultDTO
    ) -> List[SearchResultDTO]:
        """
        Fuses visual similarity scores with TF-IDF OCR text matching scores.
        
        Formula: S_fused = (1 - alpha) * S_visual + alpha * S_tfidf
        """
        if not search_results:
            return search_results

        if not ocr_result or not ocr_result.text.strip():
            return search_results

        tfidf_scores = self.compute_tfidf_similarity(ocr_result.text)

        fused_results = []
        for res in search_results:
            cid = res.remapped_class_id
            s_visual = res.similarity
            s_tfidf = tfidf_scores.get(cid, 0.0)

            # Compute additive positive-only fusion: OCR text match adds boost to visual similarity without penalizing if OCR is empty or garbled
            s_fused = min(1.0, max(0.0, float(s_visual + self.boost_alpha * s_tfidf)))

            meta = dict(res.metadata or {})
            meta["s_visual"] = s_visual
            meta["s_tfidf"] = s_tfidf

            fused_dto = SearchResultDTO(
                remapped_class_id=res.remapped_class_id,
                old_class_id=res.old_class_id,
                similarity=s_fused,
                metadata=meta
            )
            fused_results.append(fused_dto)

        # Rerank candidates by fused similarity descending
        fused_results.sort(key=lambda x: x.similarity, reverse=True)
        return fused_results

    def health_check(self) -> Tuple[bool, str]:
        """Runs diagnostics checks."""
        if self.vectorizer is None or self.gt_matrix is None:
            return False, "TF-IDF Vectorizer not initialized."
        return True, "TF-IDF OCR Matcher operating normally."

    def shutdown(self) -> None:
        """Safely releases memory references."""
        self.gt_profiles.clear()
        self.vectorizer = None
        self.gt_matrix = None
