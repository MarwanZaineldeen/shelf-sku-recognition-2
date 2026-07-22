"""Bridge from the live audit pipeline into the Pipeline 3 review store.

The dashboard posts a reviewer's verdict with only the crop's identity and
the class they chose. Everything else worth recording — the query embedding,
the Top-K slate, the similarity that triggered the review — was computed
during the audit and is gone by the time the human clicks Save.

``ReviewContextCache`` holds that context between the audit response and the
review callback, so the embedding is captured for free rather than being
recomputed with a second backbone pass (or lost, which would leave the
review unusable for curation and fine-tuning).

The cache is a best-effort accelerator, never a dependency: if it misses —
a server restart between audit and review, or an eviction — the review is
still recorded from the fields the client supplies. Losing a human label is
worse than losing an embedding.
"""

from collections import OrderedDict
from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field

from ml.active_learning.store import (
    ReviewStore,
    DECISION_APPROVED,
    DECISION_CORRECTED,
    DECISION_NOT_IN_CATALOG,
    DEFAULT_MODEL_VERSION,
)


# Class IDs below zero mean "no catalog SKU describes this crop" — the
# dashboard's "Unknown / Non-Catalog Competitor SKU" option.
NOT_IN_CATALOG_SENTINEL = -1

# Roughly forty shelf audits' worth of crops. Bounded so a long-lived server
# cannot accumulate embeddings indefinitely.
DEFAULT_CACHE_ENTRIES = 2000


class ReviewContext(BaseModel):
    """Audit-time context for one crop, awaiting a human verdict."""
    source_image: str = Field(..., description="Parent shelf image name.")
    crop_id: str = Field(..., description="Crop identifier within that image.")
    predicted_class_id: int = Field(..., description="Class the pipeline ranked first.")
    top1_similarity: float = Field(..., description="Rank-1 cosine similarity.")
    calibrated_probability: Optional[float] = Field(None, description="Platt-calibrated probability.")
    embedding: Optional[List[float]] = Field(None, description="Query embedding from the audit.")
    candidates: Optional[List[Dict[str, Any]]] = Field(None, description="Top-K slate.")
    crop_path: Optional[str] = Field(None, description="Crop path, if persisted to disk.")


class ReviewContextCache:
    """Bounded LRU cache of audit context, keyed by (source image, crop id)."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive.")
        self.max_entries = max_entries
        self._entries: "OrderedDict[Tuple[str, str], ReviewContext]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, context: ReviewContext) -> None:
        """Stores context, evicting the least recently used entry when full."""
        key = (context.source_image, context.crop_id)
        self._entries[key] = context
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get(self, source_image: str, crop_id: str) -> Optional[ReviewContext]:
        """Returns cached context, or None on a miss."""
        key = (source_image, crop_id)
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def put_predictions(self, source_image: str, predictions: List[Any]) -> int:
        """Caches every prediction from one audit.

        Args:
            source_image: Shelf image the predictions came from.
            predictions: PredictionDTOs, typically the HITL queue. Entries
                without a crop_id are skipped.

        Returns:
            int: Entries cached.
        """
        cached = 0
        for pred in predictions:
            crop_id = getattr(pred, "crop_id", None)
            if not crop_id:
                continue
            self.put(ReviewContext(
                source_image=source_image,
                crop_id=crop_id,
                predicted_class_id=int(getattr(pred, "predicted_class_id", -1)),
                top1_similarity=float(getattr(pred.bbox, "confidence", 0.0)),
                calibrated_probability=float(getattr(pred, "confidence_probability", 0.0)),
                embedding=getattr(pred, "embedding", None),
                candidates=getattr(pred, "top5_candidates", None),
            ))
            cached += 1
        return cached

    def clear(self) -> None:
        self._entries.clear()


def decision_for(
    assigned_class_id: int,
    predicted_class_id: int,
) -> Tuple[str, Optional[int]]:
    """Maps a reviewer's choice onto a decision and a ground-truth class.

    Args:
        assigned_class_id: Class the reviewer selected; negative means the
            crop matches no catalog SKU.
        predicted_class_id: Class the pipeline ranked first.

    Returns:
        Tuple[str, Optional[int]]: (decision, true_class_id). The class is
        None for an open-set rejection, which has no ground-truth SKU.
    """
    if assigned_class_id < 0:
        return DECISION_NOT_IN_CATALOG, None
    if assigned_class_id == predicted_class_id:
        return DECISION_APPROVED, assigned_class_id
    return DECISION_CORRECTED, assigned_class_id


def record_review(
    store: ReviewStore,
    source_image: str,
    crop_id: str,
    assigned_class_id: int,
    reviewer_id: str,
    context: Optional[ReviewContext] = None,
    predicted_class_id: Optional[int] = None,
    top1_similarity: Optional[float] = None,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> str:
    """Records one human review, enriched with audit context where available.

    Cached context supplies the embedding, the candidate slate, and the
    calibrated probability. Client-supplied values fill in on a cache miss;
    the review is still stored, just without the vector, and
    ``memory.promote_reviews`` will later skip it with NO_EMBEDDING rather
    than promoting something it cannot represent.

    Args:
        store: Destination review store.
        source_image: Parent shelf image name.
        crop_id: Crop identifier within that image.
        assigned_class_id: Reviewer's class choice; negative means not in catalog.
        reviewer_id: Who reviewed it.
        context: Cached audit context, if any.
        predicted_class_id: Fallback rank-1 class when context is missing.
        top1_similarity: Fallback rank-1 similarity when context is missing.
        model_version: Embedding model identifier.

    Returns:
        str: The stored review ID.

    Raises:
        ValueError: If neither context nor a fallback predicted class is given —
            without it there is no way to tell an approval from a correction.
    """
    resolved_prediction = (
        context.predicted_class_id if context is not None else predicted_class_id
    )
    if resolved_prediction is None:
        raise ValueError(
            "Cannot record a review without a predicted class: supply cached "
            "context or an explicit predicted_class_id, otherwise an approval "
            "cannot be distinguished from a correction."
        )

    resolved_similarity = (
        context.top1_similarity if context is not None else (top1_similarity or 0.0)
    )
    decision, true_class_id = decision_for(assigned_class_id, int(resolved_prediction))

    return store.log_review(
        source_image=source_image,
        decision=decision,
        top1_predicted_class_id=int(resolved_prediction),
        top1_similarity=float(resolved_similarity),
        reviewer_id=reviewer_id,
        true_class_id=true_class_id,
        candidates=context.candidates if context is not None else None,
        embedding=context.embedding if context is not None else None,
        crop_path=context.crop_path if context is not None else None,
        calibrated_probability=(
            context.calibrated_probability if context is not None else None
        ),
        model_version=model_version,
    )
