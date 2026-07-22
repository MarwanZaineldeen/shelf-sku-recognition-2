"""Hard negative confusion mining for Pipeline 3.

Reads the Top-K candidate slates recorded alongside human reviews and
recovers which SKU pairs the retriever actually confuses — the fine-grained
packaging variants (same brand, different flavour or pack size) that visual
similarity alone cannot separate.

Two consumers:
  * ``finetune.py`` builds SupCon batches around these pairs, so the loss
    spends its gradient where the embedding space is genuinely ambiguous
    rather than on classes that were never in danger of being confused.
  * Operators read the report directly. It is useful on its own, with the
    projection head switched off, because it names the variants worth
    photographing more reference crops of.
"""

from typing import List, Dict, Tuple, Any, Optional, Iterable
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, Field

from ml.active_learning.store import ReviewStore, ReviewRecord, CandidateRecord


# A competitor within this cosine margin of the true class is treated as a
# near-miss even when it did not actually outrank the truth. Confusions that
# almost happened are the ones about to happen on the next shelf.
DEFAULT_NEAR_MISS_MARGIN = 0.05


class ConfusionPair(BaseModel):
    """An ordered (true class, competitor) pair the retriever confuses."""
    true_class_id: int = Field(..., description="Verified SKU class.")
    confused_class_id: int = Field(..., description="Competing SKU class.")
    frequency: int = Field(..., description="Reviews exhibiting this confusion.")
    n_outranked: int = Field(..., description="Times the competitor beat the true class.")
    mean_similarity: float = Field(..., description="Mean competitor similarity.")
    mean_margin: float = Field(
        ...,
        description="Mean (true - competitor) similarity; negative means the competitor won.",
    )


class ConfusionReport(BaseModel):
    """Aggregate confusion structure mined from a review corpus."""
    pairs: List[ConfusionPair] = Field(default_factory=list, description="Sorted by frequency.")
    n_reviews: int = Field(0, description="Verified reviews examined.")
    n_reviews_with_candidates: int = Field(0, description="Reviews carrying a candidate slate.")
    n_true_class_absent: int = Field(
        0, description="Reviews whose true class was missing from the slate entirely."
    )

    def top_confusers(self, class_id: int, k: int = 5) -> List[int]:
        """Returns the k classes most often confused with `class_id`."""
        matches = [p for p in self.pairs if p.true_class_id == class_id]
        matches.sort(key=lambda p: p.frequency, reverse=True)
        return [p.confused_class_id for p in matches[:k]]

    def confusion_groups(self, min_frequency: int = 1) -> List[Tuple[int, ...]]:
        """Groups each class with its confusers, for batch construction.

        Returns one tuple per class that has at least one confuser, the true
        class first. A SupCon batch built from such a group contains the
        positives and their hardest negatives together.
        """
        grouped: Dict[int, List[int]] = defaultdict(list)
        for pair in self.pairs:
            if pair.frequency >= min_frequency:
                grouped[pair.true_class_id].append(pair.confused_class_id)

        return [
            (class_id, *confusers)
            for class_id, confusers in sorted(grouped.items())
            if confusers
        ]

    def summary(self, top_n: int = 20) -> str:
        """Renders the most frequent confusions as a text table."""
        lines = [
            f"Confusion report: {len(self.pairs)} pairs from {self.n_reviews} verified reviews "
            f"({self.n_true_class_absent} with the true class absent from the slate)",
            f"{'true':>6} {'confused':>9} {'freq':>6} {'outranked':>10} {'mean_sim':>9} {'mean_margin':>12}",
        ]
        for pair in self.pairs[:top_n]:
            lines.append(
                f"{pair.true_class_id:>6} {pair.confused_class_id:>9} {pair.frequency:>6} "
                f"{pair.n_outranked:>10} {pair.mean_similarity:>9.4f} {pair.mean_margin:>+12.4f}"
            )
        return "\n".join(lines)


def mine_confusion_pairs(
    reviews: Iterable[Tuple[ReviewRecord, List[CandidateRecord]]],
    near_miss_margin: float = DEFAULT_NEAR_MISS_MARGIN,
    min_frequency: int = 1,
) -> ConfusionReport:
    """Mines confusion pairs from reviews and their candidate slates.

    A competitor counts as a confusion when it either outranked the verified
    class, or sat within `near_miss_margin` cosine of it.

    Args:
        reviews: (review, candidates) pairs, as returned by
            ReviewStore.fetch_reviews_with_candidates.
        near_miss_margin: Cosine margin defining a near miss.
        min_frequency: Drop pairs seen fewer times than this.

    Returns:
        ConfusionReport: Pairs sorted by frequency, then by mean margin
        (tightest confusions first), then by class ID for determinism.
    """
    similarity_sums: Dict[Tuple[int, int], float] = defaultdict(float)
    margin_sums: Dict[Tuple[int, int], float] = defaultdict(float)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    outranked: Dict[Tuple[int, int], int] = defaultdict(int)

    n_reviews = 0
    n_with_candidates = 0
    n_true_absent = 0

    for review, candidates in reviews:
        if review.true_class_id is None:
            continue
        n_reviews += 1
        if not candidates:
            continue
        n_with_candidates += 1

        true_class = int(review.true_class_id)
        true_entry = next((c for c in candidates if c.class_id == true_class), None)
        if true_entry is None:
            n_true_absent += 1

        for candidate in candidates:
            if candidate.class_id == true_class:
                continue

            if true_entry is None:
                # The true class never made the slate, so every candidate
                # beat it. Margin is unknown; charge the full similarity.
                beat_truth = True
                margin = -float(candidate.similarity)
            else:
                beat_truth = candidate.rank < true_entry.rank
                margin = float(true_entry.similarity) - float(candidate.similarity)

            if not beat_truth and margin > near_miss_margin:
                continue

            key = (true_class, int(candidate.class_id))
            counts[key] += 1
            similarity_sums[key] += float(candidate.similarity)
            margin_sums[key] += margin
            if beat_truth:
                outranked[key] += 1

    pairs = [
        ConfusionPair(
            true_class_id=true_class,
            confused_class_id=confused_class,
            frequency=count,
            n_outranked=outranked[(true_class, confused_class)],
            mean_similarity=similarity_sums[(true_class, confused_class)] / count,
            mean_margin=margin_sums[(true_class, confused_class)] / count,
        )
        for (true_class, confused_class), count in counts.items()
        if count >= min_frequency
    ]

    # Frequency first, then tightest margin, then IDs so ties are stable.
    pairs.sort(key=lambda p: (-p.frequency, p.mean_margin, p.true_class_id, p.confused_class_id))

    return ConfusionReport(
        pairs=pairs,
        n_reviews=n_reviews,
        n_reviews_with_candidates=n_with_candidates,
        n_true_class_absent=n_true_absent,
    )


def mine_from_store(
    store: ReviewStore,
    near_miss_margin: float = DEFAULT_NEAR_MISS_MARGIN,
    min_frequency: int = 1,
) -> ConfusionReport:
    """Convenience wrapper mining confusions straight from a review store."""
    return mine_confusion_pairs(
        store.fetch_reviews_with_candidates(only_verified=True),
        near_miss_margin=near_miss_margin,
        min_frequency=min_frequency,
    )
