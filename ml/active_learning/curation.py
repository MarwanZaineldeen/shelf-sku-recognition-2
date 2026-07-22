"""Non-parametric gallery memory curation for Pipeline 3.

Keeps the reference gallery diverse and bounded: rejects near-duplicate
crops, then caps each SKU class at N vectors chosen by k-center greedy
max-min selection so the survivors span the class's appearance variation
instead of clustering around whichever facings happened to be photographed
most often.

All selection is deterministic. The pipeline reports bootstrap confidence
intervals downstream, and a non-reproducible gallery would make those
numbers meaningless.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


# Cosine similarity above which two crops are treated as the same view.
DEFAULT_NEAR_DUP_THRESHOLD = 0.98

# Per-class vector cap. Provisional: ratify against class_size_histogram()
# on the real registry before committing (see final_pipeline3_claude_plan.md).
DEFAULT_CLASS_CAP = 500

# Unit-norm tolerance. DINOv3 emits L2-normalized vectors, but gallery BLOBs
# are only re-normalized inside NumpyCosineIndex.add, so curation cannot
# assume its input arrived through that path.
_NORM_TOLERANCE = 1e-3


class CurationDecision(BaseModel):
    """Outcome of curating one SKU class."""
    keep_indices: List[int] = Field(..., description="Input row indices retained.")
    prune_indices: List[int] = Field(..., description="Input row indices to soft-delete.")
    n_input: int = Field(..., description="Vectors considered.")
    n_near_duplicate: int = Field(..., description="Dropped as near-duplicates.")
    n_over_cap: int = Field(..., description="Dropped by the diversity cap.")


def _validate_unit_norm(vectors: np.ndarray) -> None:
    """Raises if vectors are not L2-normalized.

    Cosine distance is computed as ``1 - dot``, which is only a distance on
    the unit sphere. Un-normalized input would silently produce a nonsense
    ranking rather than an error.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2-D (N, D) array, got shape {vectors.shape}.")
    if vectors.shape[0] == 0:
        return

    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
        worst = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(
            f"Vectors must be L2-normalized for cosine curation "
            f"(max deviation {worst:.6f}). Normalize before curating."
        )


def medoid_index(vectors: np.ndarray) -> int:
    """Returns the index of the vector closest to the class centroid.

    Used as the deterministic k-center seed. The medoid is the most typical
    crop of the class, which makes it a defensible starting point — unlike
    a random seed, which would make curation irreproducible.
    """
    _validate_unit_norm(vectors)
    if vectors.shape[0] == 0:
        raise ValueError("Cannot compute a medoid of an empty array.")

    centroid = vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        # Antipodal vectors cancelling out; index 0 keeps this deterministic.
        return 0
    return int(np.argmax(vectors @ (centroid / norm)))


def kcenter_greedy_select(
    vectors: np.ndarray,
    k: int,
    initial_indices: Optional[List[int]] = None,
    seed_strategy: str = "medoid",
) -> List[int]:
    """Greedy k-center max-min diversity selection under cosine distance.

    Repeatedly picks the vector farthest from everything already selected,
    yielding a subset that covers the class's appearance space.

    Args:
        vectors: (N, D) L2-normalized array.
        k: Number of vectors to select.
        initial_indices: Pre-selected seeds (e.g. an existing curated
            gallery being extended with a new review batch). All seeds
            constrain the selection, not just the last.
        seed_strategy: 'medoid' (default, deterministic) or 'first'.

    Returns:
        List[int]: Selected row indices. Seeds come first, then picks in
        selection order. Never contains duplicates.

    Raises:
        ValueError: If vectors are not unit-norm, or seed_strategy is unknown.
    """
    _validate_unit_norm(vectors)
    n_samples = vectors.shape[0]

    if k <= 0:
        return []
    if k >= n_samples and not initial_indices:
        return list(range(n_samples))

    # Seed selection, de-duplicated while preserving order.
    if initial_indices:
        selected: List[int] = list(dict.fromkeys(int(i) for i in initial_indices))
        for idx in selected:
            if not 0 <= idx < n_samples:
                raise ValueError(f"initial_indices contains out-of-range index {idx}.")
    elif seed_strategy == "medoid":
        selected = [medoid_index(vectors)]
    elif seed_strategy == "first":
        selected = [0]
    else:
        raise ValueError(f"Unknown seed_strategy '{seed_strategy}'. Use 'medoid' or 'first'.")

    if len(selected) >= k:
        return selected

    # Prime the distance map against EVERY seed. Folding in only the last
    # seed would let the selection drift back toward the earlier ones —
    # the defect in the original specification's implementation.
    min_distances = np.full(n_samples, np.inf, dtype=np.float64)
    for idx in selected:
        min_distances = np.minimum(min_distances, 1.0 - vectors @ vectors[idx])

    # Mask selected rows so they cannot be picked again. -inf survives the
    # np.minimum updates below, so masking once per row is enough.
    min_distances[selected] = -np.inf

    while len(selected) < k:
        nxt = int(np.argmax(min_distances))
        if not np.isfinite(min_distances[nxt]):
            # Every remaining row is already selected: the class has fewer
            # distinct vectors than k.
            break
        selected.append(nxt)
        min_distances = np.minimum(min_distances, 1.0 - vectors @ vectors[nxt])
        min_distances[nxt] = -np.inf

    return selected


def reject_near_duplicates(
    vectors: np.ndarray,
    threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> np.ndarray:
    """Greedily drops crops that duplicate an already-kept crop.

    Args:
        vectors: (N, D) L2-normalized array.
        threshold: Cosine similarity at or above which a vector is a duplicate.

    Returns:
        np.ndarray: Boolean keep-mask of length N. The first vector of each
        duplicate group is kept, so the result is order-deterministic.
    """
    _validate_unit_norm(vectors)
    n_samples = vectors.shape[0]
    keep_mask = np.zeros(n_samples, dtype=bool)
    if n_samples == 0:
        return keep_mask

    kept: List[int] = []
    for i in range(n_samples):
        if kept and float(np.max(vectors[kept] @ vectors[i])) >= threshold:
            continue
        kept.append(i)
        keep_mask[i] = True

    return keep_mask


def curate_class(
    vectors: np.ndarray,
    cap: int = DEFAULT_CLASS_CAP,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    initial_indices: Optional[List[int]] = None,
    seed_strategy: str = "medoid",
) -> CurationDecision:
    """Curates one SKU class: near-duplicate rejection, then a diversity cap.

    Args:
        vectors: (N, D) L2-normalized array for a single class.
        cap: Maximum vectors to retain.
        near_dup_threshold: Cosine similarity treated as duplication.
        initial_indices: Rows that must be retained (indices into `vectors`).
        seed_strategy: k-center seed strategy.

    Returns:
        CurationDecision: Keep/prune indices into `vectors`, plus counts
        attributing each drop to duplication or to the cap.
    """
    _validate_unit_norm(vectors)
    n_input = int(vectors.shape[0])
    if n_input == 0:
        return CurationDecision(
            keep_indices=[], prune_indices=[], n_input=0,
            n_near_duplicate=0, n_over_cap=0,
        )

    # Stage 1 — near-duplicate rejection. Forced-keep rows are exempt.
    keep_mask = reject_near_duplicates(vectors, threshold=near_dup_threshold)
    if initial_indices:
        keep_mask[list(initial_indices)] = True

    survivor_idx = np.flatnonzero(keep_mask)
    n_near_duplicate = n_input - len(survivor_idx)

    # Stage 2 — diversity cap over the survivors.
    if len(survivor_idx) <= cap:
        final_idx = survivor_idx.tolist()
        n_over_cap = 0
    else:
        # k-center runs in survivor space, so remap the forced-keep rows in
        # and the selection back out.
        remap = {int(orig): pos for pos, orig in enumerate(survivor_idx)}
        seeds = [remap[int(i)] for i in initial_indices] if initial_indices else None

        picked = kcenter_greedy_select(
            vectors[survivor_idx], k=cap,
            initial_indices=seeds, seed_strategy=seed_strategy,
        )
        final_idx = sorted(int(survivor_idx[p]) for p in picked)
        n_over_cap = len(survivor_idx) - len(final_idx)

    keep_set = set(final_idx)
    prune_idx = [i for i in range(n_input) if i not in keep_set]

    return CurationDecision(
        keep_indices=sorted(keep_set),
        prune_indices=prune_idx,
        n_input=n_input,
        n_near_duplicate=n_near_duplicate,
        n_over_cap=n_over_cap,
    )


def class_size_histogram(db_path: str) -> Dict[int, int]:
    """Counts active reference crops per class in a gallery registry.

    Convenience wrapper for CLI inspection; opens and closes its own store.
    Run this before ratifying a cap — with ~472 vectors per class on
    average, a cap of 500 prunes nothing unless the distribution is skewed.

    Args:
        db_path: Path to the gallery SQLite registry.

    Returns:
        Dict[int, int]: class_id -> active crop count.
    """
    from ml.retrieval.sqlite_registry import SQLiteGalleryStore

    store = SQLiteGalleryStore()
    store.initialize({"db_path": db_path})
    try:
        return store.class_size_histogram()
    finally:
        store.shutdown()
