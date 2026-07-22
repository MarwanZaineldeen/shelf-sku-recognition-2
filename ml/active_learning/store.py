"""SQLite review store for the Pipeline 3 active continual learning loop.

Logs human HITL decisions plus the Top-K candidate slate that produced them,
so downstream curation, hard-negative mining, and SupCon fine-tuning can run
without re-reading crop files or re-running the DINOv3 backbone.

Lives in its own database (``reviews.db``), deliberately separate from the
read-mostly gallery registry, so review churn never touches the vectors the
search index loads from.
"""

import os
import uuid
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field

from ml.base import IPlugin


# Repository root, resolved relative to this file so the module is portable
# across machines (ml/active_learning/store.py -> ml/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _REPO_ROOT / "data" / "processed" / "active_learning" / "reviews.db"

# Reviewer decision vocabulary. NOT_IN_CATALOG is the open-set rejection case:
# the crop is a real product, but no catalog SKU describes it.
DECISION_APPROVED = "APPROVED"
DECISION_CORRECTED = "CORRECTED"
DECISION_NOT_IN_CATALOG = "NOT_IN_CATALOG"
VALID_DECISIONS = (DECISION_APPROVED, DECISION_CORRECTED, DECISION_NOT_IN_CATALOG)

# Decisions that imply the top-1 prediction was wrong.
_CORRECTION_DECISIONS = (DECISION_CORRECTED, DECISION_NOT_IN_CATALOG)

DEFAULT_MODEL_VERSION = "dinov3_vitb16_raw768"


class CandidateRecord(BaseModel):
    """One entry of the Top-K candidate slate shown to the reviewer."""
    rank: int = Field(..., description="1-indexed position in the retrieved slate.")
    class_id: int = Field(..., description="Candidate SKU class ID.")
    similarity: float = Field(..., description="Cosine similarity of the candidate.")


class ReviewRecord(BaseModel):
    """A single human review decision and the context that produced it."""
    review_id: str = Field(..., description="Unique review identifier.")
    crop_path: Optional[str] = Field(None, description="Path to the crop on disk, if persisted.")
    source_image: str = Field(..., description="Parent shelf image name — required for leakage control.")
    decision: str = Field(..., description="APPROVED | CORRECTED | NOT_IN_CATALOG.")
    true_class_id: Optional[int] = Field(None, description="Verified SKU class ID; None when NOT_IN_CATALOG.")
    top1_predicted_class_id: int = Field(..., description="Class ID the pipeline predicted at rank 1.")
    top1_similarity: float = Field(..., description="Cosine similarity of the rank-1 prediction.")
    calibrated_probability: Optional[float] = Field(None, description="Platt-calibrated probability, if available.")
    is_correction: int = Field(..., description="1 if the top-1 prediction was wrong, else 0.")
    embedding: Optional[List[float]] = Field(None, description="Query embedding captured at audit time.")
    embedding_dim: Optional[int] = Field(None, description="Length of the stored embedding.")
    model_version: str = Field(..., description="Embedding model that produced the vector.")
    reviewer_id: str = Field(..., description="Identifier of the human reviewer.")
    consumed_in_batch: Optional[str] = Field(None, description="Batch ID that consumed this review; None if unconsumed.")
    created_at: Optional[str] = Field(None, description="Insertion timestamp.")


class ReviewStore(IPlugin):
    """SQLite-backed repository for HITL review decisions and candidate slates."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path: str = str(db_path) if db_path else str(_DEFAULT_DB_PATH)
        self.conn: Optional[sqlite3.Connection] = None

    # ── IPlugin lifecycle ────────────────────────────────────────────

    def initialize(self, config: Dict[str, Any]) -> None:
        """Opens the review database and creates the schema if missing.

        Config schema:
            db_path: str (default: data/processed/active_learning/reviews.db)
        """
        self.db_path = str(config.get("db_path", self.db_path))

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # SQLite leaves foreign keys off by default; candidate rows must not
        # be able to outlive their parent review.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def health_check(self) -> Tuple[bool, str]:
        """Verifies the connection is live and the expected tables exist."""
        if not self.conn:
            return False, "Review store not connected."
        try:
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('reviews', 'review_candidates')"
            )
            tables = {row["name"] for row in cursor.fetchall()}
            missing = {"reviews", "review_candidates"} - tables
            if missing:
                return False, f"Missing tables: {sorted(missing)}"
            return True, "Healthy"
        except Exception as e:
            return False, f"Review store check failed: {str(e)}"

    def shutdown(self) -> None:
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _create_tables(self) -> None:
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")

        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id               TEXT PRIMARY KEY,
                    crop_path               TEXT,
                    source_image            TEXT NOT NULL,
                    decision                TEXT NOT NULL,
                    true_class_id           INTEGER,
                    top1_predicted_class_id INTEGER NOT NULL,
                    top1_similarity         REAL NOT NULL,
                    calibrated_probability  REAL,
                    is_correction           INTEGER NOT NULL,
                    embedding               BLOB,
                    embedding_dim           INTEGER,
                    model_version           TEXT NOT NULL,
                    reviewer_id             TEXT NOT NULL,
                    consumed_in_batch       TEXT,
                    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS review_candidates (
                    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id    TEXT NOT NULL,
                    rank         INTEGER NOT NULL,
                    class_id     INTEGER NOT NULL,
                    similarity   REAL NOT NULL,
                    FOREIGN KEY(review_id) REFERENCES reviews(review_id)
                )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reviews_unconsumed ON reviews(consumed_in_batch)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_review ON review_candidates(review_id)"
            )

    # ── Write path ───────────────────────────────────────────────────

    def log_review(
        self,
        source_image: str,
        decision: str,
        top1_predicted_class_id: int,
        top1_similarity: float,
        reviewer_id: str,
        true_class_id: Optional[int] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        embedding: Optional[List[float]] = None,
        crop_path: Optional[str] = None,
        calibrated_probability: Optional[float] = None,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> str:
        """Persists one review decision and its candidate slate atomically.

        Args:
            source_image: Parent shelf image name. Required — the promotion
                gate uses it to guarantee gallery/test-set disjointness.
            decision: One of VALID_DECISIONS.
            top1_predicted_class_id: Class the pipeline ranked first.
            top1_similarity: Cosine similarity of that rank-1 prediction.
            reviewer_id: Identifier of the human who made the call.
            true_class_id: Verified class. Must be None for NOT_IN_CATALOG
                and provided otherwise.
            candidates: Top-K slate as dicts carrying 'class_id' and
                'similarity'. Accepts the orchestrator's top5_candidates
                shape directly; rank is taken from list order.
            embedding: Query vector captured during the audit, so curation
                and fine-tuning never need to re-run the backbone.
            crop_path: Optional path to the crop on disk.
            calibrated_probability: Platt-calibrated probability, if computed.
            model_version: Embedding model identifier for the stored vector.

        Returns:
            str: The generated review ID.

        Raises:
            ValueError: If the decision is unknown or true_class_id is
                inconsistent with it.
        """
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")

        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"Unknown decision '{decision}'. Expected one of {list(VALID_DECISIONS)}."
            )

        # An open-set rejection has no true class by definition; every other
        # decision must name one. Enforcing this here keeps downstream
        # curation from silently training on a NULL label.
        if decision == DECISION_NOT_IN_CATALOG:
            if true_class_id is not None:
                raise ValueError(
                    "true_class_id must be None when decision is NOT_IN_CATALOG."
                )
        elif true_class_id is None:
            raise ValueError(f"true_class_id is required when decision is '{decision}'.")

        # Derived, never accepted from the caller, so decision and
        # is_correction cannot drift apart.
        is_correction = 1 if decision in _CORRECTION_DECISIONS else 0

        review_id = f"rev_{uuid.uuid4().hex}"

        embedding_blob: Optional[bytes] = None
        embedding_dim: Optional[int] = None
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.ndim != 1:
                raise ValueError(f"Embedding must be 1-D, got shape {vec.shape}.")
            embedding_blob = vec.tobytes()
            embedding_dim = int(vec.shape[0])

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO reviews (
                    review_id, crop_path, source_image, decision, true_class_id,
                    top1_predicted_class_id, top1_similarity, calibrated_probability,
                    is_correction, embedding, embedding_dim, model_version,
                    reviewer_id, consumed_in_batch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    review_id, crop_path, source_image, decision, true_class_id,
                    int(top1_predicted_class_id), float(top1_similarity),
                    calibrated_probability, is_correction, embedding_blob,
                    embedding_dim, model_version, reviewer_id,
                ),
            )

            if candidates:
                rows = [
                    (
                        review_id,
                        int(cand.get("rank", idx + 1)),
                        int(cand["class_id"]),
                        float(cand["similarity"]),
                    )
                    for idx, cand in enumerate(candidates)
                ]
                self.conn.executemany(
                    "INSERT INTO review_candidates (review_id, rank, class_id, similarity) VALUES (?, ?, ?, ?)",
                    rows,
                )

        return review_id

    def mark_consumed(self, review_ids: List[str], batch_id: str) -> int:
        """Stamps reviews as consumed by a curation or fine-tuning batch.

        Args:
            review_ids: Review IDs to stamp.
            batch_id: Identifier of the consuming batch.

        Returns:
            int: Number of rows updated.
        """
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")
        if not review_ids:
            return 0

        with self.conn:
            cursor = self.conn.executemany(
                "UPDATE reviews SET consumed_in_batch = ? WHERE review_id = ? AND consumed_in_batch IS NULL",
                [(batch_id, rid) for rid in review_ids],
            )
            return cursor.rowcount if cursor.rowcount is not None else 0

    # ── Read path ────────────────────────────────────────────────────

    def _row_to_record(self, row: sqlite3.Row) -> ReviewRecord:
        embedding: Optional[List[float]] = None
        if row["embedding"] is not None:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()

        return ReviewRecord(
            review_id=row["review_id"],
            crop_path=row["crop_path"],
            source_image=row["source_image"],
            decision=row["decision"],
            true_class_id=row["true_class_id"],
            top1_predicted_class_id=row["top1_predicted_class_id"],
            top1_similarity=row["top1_similarity"],
            calibrated_probability=row["calibrated_probability"],
            is_correction=row["is_correction"],
            embedding=embedding,
            embedding_dim=row["embedding_dim"],
            model_version=row["model_version"],
            reviewer_id=row["reviewer_id"],
            consumed_in_batch=row["consumed_in_batch"],
            created_at=str(row["created_at"]) if row["created_at"] is not None else None,
        )

    def fetch_review(self, review_id: str) -> Optional[ReviewRecord]:
        """Fetches a single review by ID, or None if absent."""
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")

        cursor = self.conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,))
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def fetch_candidates(self, review_id: str) -> List[CandidateRecord]:
        """Fetches the Top-K candidate slate for a review, ordered by rank."""
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")

        cursor = self.conn.execute(
            "SELECT rank, class_id, similarity FROM review_candidates WHERE review_id = ? ORDER BY rank ASC",
            (review_id,),
        )
        return [
            CandidateRecord(rank=row["rank"], class_id=row["class_id"], similarity=row["similarity"])
            for row in cursor.fetchall()
        ]

    def fetch_reviews_with_candidates(
        self,
        only_verified: bool = True,
    ) -> List[Tuple[ReviewRecord, List[CandidateRecord]]]:
        """Fetches reviews paired with their candidate slates.

        Args:
            only_verified: Restrict to reviews carrying a true_class_id.
                Open-set rejections have no ground-truth class, so confusion
                mining cannot use them.

        Returns:
            List of (review, candidates) with candidates ordered by rank.
        """
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")

        query = "SELECT * FROM reviews"
        if only_verified:
            query += " WHERE true_class_id IS NOT NULL"
        query += " ORDER BY rowid ASC"

        reviews = [self._row_to_record(row) for row in self.conn.execute(query).fetchall()]
        if not reviews:
            return []

        # One pass over candidates, bucketed by review, rather than a query
        # per review.
        buckets: Dict[str, List[CandidateRecord]] = {r.review_id: [] for r in reviews}
        cursor = self.conn.execute(
            "SELECT review_id, rank, class_id, similarity FROM review_candidates ORDER BY review_id, rank ASC"
        )
        for row in cursor.fetchall():
            bucket = buckets.get(row["review_id"])
            if bucket is not None:
                bucket.append(CandidateRecord(
                    rank=row["rank"], class_id=row["class_id"], similarity=row["similarity"]
                ))

        return [(review, buckets[review.review_id]) for review in reviews]

    def fetch_unconsumed(self, limit: Optional[int] = None) -> List[ReviewRecord]:
        """Fetches reviews not yet consumed by any batch, oldest first.

        Args:
            limit: Optional cap on rows returned (e.g. the 50-review session size).

        Returns:
            List[ReviewRecord]: Unconsumed reviews in deterministic order.
        """
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")

        # Ordered by rowid, not created_at: CURRENT_TIMESTAMP has second
        # granularity, so a batch written inside one second would come back
        # in arbitrary order. rowid is assigned monotonically at insert, so
        # it is both deterministic and true insertion order.
        query = "SELECT * FROM reviews WHERE consumed_in_batch IS NULL ORDER BY rowid ASC"
        params: Tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)

        cursor = self.conn.execute(query, params)
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def count_unconsumed(self) -> int:
        """Counts unconsumed reviews. Drives the N>=500 fine-tune trigger."""
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")

        cursor = self.conn.execute("SELECT COUNT(*) FROM reviews WHERE consumed_in_batch IS NULL")
        return int(cursor.fetchone()[0])

    def fetch_embeddings_by_class(self, class_id: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Fetches verified review embeddings for one SKU class.

        Only rows carrying a stored vector and a verified true_class_id are
        returned, so NOT_IN_CATALOG rejections never enter curation.

        Args:
            class_id: Verified SKU class ID to fetch.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: An (N, D) float32 array
            and N metadata dicts in matching order. The array is empty with
            shape (0, 0) when the class has no stored vectors.
        """
        if not self.conn:
            raise RuntimeError("Review store connection not initialized.")

        cursor = self.conn.execute(
            """
            SELECT review_id, crop_path, source_image, embedding, embedding_dim,
                   true_class_id, model_version, decision
            FROM reviews
            WHERE true_class_id = ? AND embedding IS NOT NULL
            ORDER BY rowid ASC
            """,
            (int(class_id),),
        )
        rows = cursor.fetchall()
        if not rows:
            return np.empty((0, 0), dtype=np.float32), []

        dim = int(rows[0]["embedding_dim"])
        vectors = np.empty((len(rows), dim), dtype=np.float32)
        metadata: List[Dict[str, Any]] = []

        for idx, row in enumerate(rows):
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if vec.shape[0] != dim:
                raise ValueError(
                    f"Inconsistent embedding dimension for class {class_id}: "
                    f"expected {dim}, got {vec.shape[0]} in review '{row['review_id']}'."
                )
            vectors[idx] = vec
            metadata.append({
                "review_id": row["review_id"],
                "crop_path": row["crop_path"],
                "source_image": row["source_image"],
                "remapped_class_id": row["true_class_id"],
                "model_version": row["model_version"],
                "decision": row["decision"],
            })

        return vectors, metadata
