"""Non-parametric gallery memory updater for Pipeline 3.

Applies curation decisions and promotes reviewed crops into the reference
gallery. Every write is versioned and reversible: pruning soft-deletes,
insertion stamps a new gallery version, and both are undone by
``SQLiteGalleryStore.rollback_version``.

Dry-run is the default. Nothing writes unless the caller explicitly asks.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from pydantic import BaseModel, Field

from ml.base import IPlugin
from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from ml.active_learning.curation import (
    curate_class,
    CurationDecision,
    DEFAULT_CLASS_CAP,
    DEFAULT_NEAR_DUP_THRESHOLD,
)
from ml.active_learning.store import ReviewRecord


class ClassCurationReport(BaseModel):
    """Per-class outcome of a curation pass."""
    class_id: int = Field(..., description="Remapped SKU class ID.")
    n_input: int = Field(..., description="Active vectors before curation.")
    n_kept: int = Field(..., description="Vectors retained.")
    n_near_duplicate: int = Field(..., description="Dropped as near-duplicates.")
    n_over_cap: int = Field(..., description="Dropped by the diversity cap.")


class CurationPassReport(BaseModel):
    """Outcome of curating the whole gallery."""
    applied: bool = Field(..., description="False for a dry run.")
    version: Optional[int] = Field(None, description="Gallery version stamped on pruned rows.")
    cap: int = Field(..., description="Per-class cap used.")
    near_dup_threshold: float = Field(..., description="Near-duplicate threshold used.")
    total_before: int = Field(..., description="Active vectors before.")
    total_kept: int = Field(..., description="Active vectors after.")
    total_pruned: int = Field(..., description="Vectors soft-deleted.")
    per_class: List[ClassCurationReport] = Field(default_factory=list)


class PromotionReport(BaseModel):
    """Outcome of promoting reviewed crops into the gallery."""
    applied: bool = Field(..., description="False for a dry run.")
    version: Optional[int] = Field(None, description="Gallery version of inserted rows.")
    n_promoted: int = Field(..., description="Review crops inserted.")
    n_skipped: int = Field(..., description="Reviews skipped, with reasons in skipped_reasons.")
    skipped_reasons: Dict[str, int] = Field(default_factory=dict)


class GalleryMemoryUpdater(IPlugin):
    """Applies curation and review promotion to the reference gallery."""

    def __init__(self, gallery_store: Optional[SQLiteGalleryStore] = None) -> None:
        self.gallery_store = gallery_store
        self._owns_store = False

    # ── IPlugin lifecycle ────────────────────────────────────────────

    def initialize(self, config: Dict[str, Any]) -> None:
        """Connects to the gallery registry.

        Config schema:
            db_path: str — required unless a store was injected.
        """
        if self.gallery_store is None:
            db_path = config.get("db_path")
            if not db_path:
                raise ValueError("Configuration must specify 'db_path'.")
            self.gallery_store = SQLiteGalleryStore()
            self.gallery_store.initialize({"db_path": db_path})
            self._owns_store = True

    def health_check(self) -> Tuple[bool, str]:
        if self.gallery_store is None:
            return False, "Gallery store not connected."
        return self.gallery_store.health_check()

    def shutdown(self) -> None:
        if self.gallery_store is not None and self._owns_store:
            self.gallery_store.shutdown()
            self.gallery_store = None

    # ── Curation ─────────────────────────────────────────────────────

    def curate_gallery(
        self,
        cap: int = DEFAULT_CLASS_CAP,
        near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
        apply: bool = False,
        class_ids: Optional[List[int]] = None,
    ) -> CurationPassReport:
        """Curates every class, optionally applying the result.

        Args:
            cap: Per-class vector cap.
            near_dup_threshold: Cosine similarity treated as duplication.
            apply: Write the pruning. Default False — inspect the report first.
            class_ids: Restrict to these classes; None curates all.

        Returns:
            CurationPassReport: Per-class and total counts. On an applied
            pass, `version` is the rollback point.
        """
        if self.gallery_store is None:
            raise RuntimeError("Gallery store not connected.")

        histogram = self.gallery_store.class_size_histogram()
        targets = sorted(histogram.keys()) if class_ids is None else sorted(class_ids)

        per_class: List[ClassCurationReport] = []
        prune_ids: List[str] = []
        total_before = 0
        total_kept = 0

        for class_id in targets:
            vectors, metadata = self.gallery_store.fetch_active_by_class(class_id)
            if vectors.shape[0] == 0:
                continue

            # Gallery BLOBs are stored as written; NumpyCosineIndex normalizes
            # at load time, so curation must normalize for itself.
            vectors = _l2_normalize(vectors)

            decision: CurationDecision = curate_class(
                vectors, cap=cap, near_dup_threshold=near_dup_threshold,
            )

            total_before += decision.n_input
            total_kept += len(decision.keep_indices)
            prune_ids.extend(metadata[i]["id"] for i in decision.prune_indices)

            per_class.append(ClassCurationReport(
                class_id=class_id,
                n_input=decision.n_input,
                n_kept=len(decision.keep_indices),
                n_near_duplicate=decision.n_near_duplicate,
                n_over_cap=decision.n_over_cap,
            ))

        version: Optional[int] = None
        if apply and prune_ids:
            version = self.gallery_store.prune_references(prune_ids)

        return CurationPassReport(
            applied=bool(apply and prune_ids),
            version=version,
            cap=cap,
            near_dup_threshold=near_dup_threshold,
            total_before=total_before,
            total_kept=total_kept,
            total_pruned=len(prune_ids),
            per_class=per_class,
        )

    # ── Review promotion ─────────────────────────────────────────────

    def promote_reviews(
        self,
        records: List[ReviewRecord],
        apply: bool = False,
        expected_dim: Optional[int] = None,
    ) -> PromotionReport:
        """Inserts verified review crops into the gallery as continual rows.

        Reviews without a stored vector, without a verified class, or with a
        mismatched dimension are skipped rather than silently coerced. Rows
        land with origin='continual' and carry their source_image, which the
        promotion gate needs to enforce gallery/test-set disjointness.

        Args:
            records: Reviews to promote.
            apply: Write the rows. Default False.
            expected_dim: Required embedding dimension. Inferred from the
                current gallery when omitted.

        Returns:
            PromotionReport: Counts, with skip reasons attributed.
        """
        if self.gallery_store is None:
            raise RuntimeError("Gallery store not connected.")

        if expected_dim is None:
            expected_dim = self._infer_gallery_dimension()

        references: List[Tuple[Any, ...]] = []
        skipped: Dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        for record in records:
            if record.embedding is None:
                skip("NO_EMBEDDING")
                continue
            # NOT_IN_CATALOG reviews have no class and must never become
            # gallery references — they are open-set rejections.
            if record.true_class_id is None:
                skip("NO_VERIFIED_CLASS")
                continue
            if expected_dim is not None and len(record.embedding) != expected_dim:
                skip("DIMENSION_MISMATCH")
                continue

            references.append((
                int(record.true_class_id),
                int(record.true_class_id),
                record.crop_path or f"review/{record.review_id}.jpg",
                str(record.true_class_id),
                record.source_image,
                # Reviews carry no bbox; retrieval never reads these, and a
                # zero box is honest about the crop already being extracted.
                [0.0, 0.0, 0.0, 0.0],
                record.embedding,
            ))

        version: Optional[int] = None
        if apply and references:
            version = self.gallery_store.save_references_bulk(references, origin="continual")

        return PromotionReport(
            applied=bool(apply and references),
            version=version,
            n_promoted=len(references),
            n_skipped=sum(skipped.values()),
            skipped_reasons=skipped,
        )

    # ── Index refresh ────────────────────────────────────────────────

    def rebuild_index(self, index: Any) -> int:
        """Reloads a vector index from the current active gallery.

        Args:
            index: A retriever exposing shutdown() and add(), e.g.
                NumpyCosineIndex.

        Returns:
            int: Vectors loaded into the index.
        """
        if self.gallery_store is None:
            raise RuntimeError("Gallery store not connected.")

        vectors, metadata = self.gallery_store.fetch_all_references()
        index.shutdown()
        if vectors.shape[0] > 0:
            index.add(vectors, metadata)
        return int(vectors.shape[0])

    def _infer_gallery_dimension(self) -> Optional[int]:
        """Reads the embedding width of the current gallery, or None if empty."""
        if self.gallery_store is None:
            raise RuntimeError("Gallery store not connected.")

        vectors, _ = self.gallery_store.fetch_all_references()
        if vectors.shape[0] == 0:
            return None
        return int(vectors.shape[1])


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Returns a unit-norm copy, guarding against zero vectors."""
    arr = vectors.astype(np.float32, copy=True)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return arr / norms
