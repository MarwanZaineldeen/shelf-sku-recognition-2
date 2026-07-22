"""Unit tests for the Pipeline 3 review store.

Model-free by design: no gallery DB, no model weights, no image I/O.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.active_learning.store import (
    ReviewStore,
    DECISION_APPROVED,
    DECISION_CORRECTED,
    DECISION_NOT_IN_CATALOG,
)


class ReviewStoreTestCase(unittest.TestCase):
    """Base fixture providing a throwaway review database."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "reviews.db")
        self.store = ReviewStore()
        self.store.initialize({"db_path": self.db_path})

    def tearDown(self) -> None:
        self.store.shutdown()
        self._tmp.cleanup()

    def _log(self, **overrides):
        """Logs a review with sensible defaults, overridable per test."""
        payload = {
            "source_image": "shelf_001.jpg",
            "decision": DECISION_APPROVED,
            "top1_predicted_class_id": 12,
            "top1_similarity": 0.91,
            "reviewer_id": "merchandiser_user",
            "true_class_id": 12,
        }
        payload.update(overrides)
        return self.store.log_review(**payload)


class TestLifecycle(ReviewStoreTestCase):
    """Plugin lifecycle and schema creation."""

    def test_health_check_after_initialize(self) -> None:
        ok, msg = self.store.health_check()
        self.assertTrue(ok, msg)

    def test_health_check_before_initialize(self) -> None:
        fresh = ReviewStore()
        ok, _ = fresh.health_check()
        self.assertFalse(ok)

    def test_initialize_creates_parent_directory(self) -> None:
        nested = Path(self._tmp.name) / "a" / "b" / "reviews.db"
        store = ReviewStore()
        store.initialize({"db_path": str(nested)})
        self.assertTrue(nested.exists())
        store.shutdown()

    def test_default_db_path_is_repo_relative(self) -> None:
        """Guards against the hardcoded absolute paths found elsewhere in the repo."""
        default_path = ReviewStore().db_path
        self.assertNotIn("Marwan", default_path)
        self.assertTrue(default_path.endswith("reviews.db"))
        self.assertIn("active_learning", default_path)


class TestRoundTrip(ReviewStoreTestCase):
    """Write/read fidelity for reviews and their candidate slates."""

    def test_review_round_trip(self) -> None:
        review_id = self._log(
            crop_path="crops/c_001.jpg",
            calibrated_probability=0.87,
        )
        record = self.store.fetch_review(review_id)

        self.assertIsNotNone(record)
        self.assertEqual(record.review_id, review_id)
        self.assertEqual(record.source_image, "shelf_001.jpg")
        self.assertEqual(record.decision, DECISION_APPROVED)
        self.assertEqual(record.true_class_id, 12)
        self.assertEqual(record.top1_predicted_class_id, 12)
        self.assertAlmostEqual(record.top1_similarity, 0.91, places=6)
        self.assertAlmostEqual(record.calibrated_probability, 0.87, places=6)
        self.assertIsNone(record.consumed_in_batch)

    def test_fetch_missing_review_returns_none(self) -> None:
        self.assertIsNone(self.store.fetch_review("rev_does_not_exist"))

    def test_candidates_stored_in_orchestrator_shape(self) -> None:
        """The orchestrator's top5_candidates dicts must insert without translation."""
        candidates = [
            {"class_id": 12, "display_name": "Lipton Yellow Label 50s", "similarity": 0.91},
            {"class_id": 34, "display_name": "Lipton Green Tea 25s", "similarity": 0.88},
            {"class_id": 7, "display_name": "Lipton Earl Grey 25s", "similarity": 0.81},
        ]
        review_id = self._log(candidates=candidates)
        fetched = self.store.fetch_candidates(review_id)

        self.assertEqual(len(fetched), 3)
        self.assertEqual([c.rank for c in fetched], [1, 2, 3])
        self.assertEqual([c.class_id for c in fetched], [12, 34, 7])
        self.assertAlmostEqual(fetched[0].similarity, 0.91, places=6)

    def test_explicit_rank_overrides_list_order(self) -> None:
        review_id = self._log(candidates=[
            {"class_id": 5, "similarity": 0.7, "rank": 2},
            {"class_id": 9, "similarity": 0.9, "rank": 1},
        ])
        fetched = self.store.fetch_candidates(review_id)
        self.assertEqual([c.class_id for c in fetched], [9, 5])

    def test_review_without_candidates(self) -> None:
        review_id = self._log()
        self.assertEqual(self.store.fetch_candidates(review_id), [])


class TestEmbeddingFidelity(ReviewStoreTestCase):
    """Float32 BLOB round-trip — curation and SupCon depend on these vectors."""

    def test_embedding_survives_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        vec = rng.normal(size=768).astype(np.float32)
        vec /= np.linalg.norm(vec)

        review_id = self._log(embedding=vec.tolist())
        record = self.store.fetch_review(review_id)

        self.assertEqual(record.embedding_dim, 768)
        np.testing.assert_allclose(
            np.asarray(record.embedding, dtype=np.float32), vec, rtol=0, atol=0
        )

    def test_embedding_preserves_unit_norm(self) -> None:
        """Curation asserts unit norms; the store must not perturb them."""
        rng = np.random.default_rng(7)
        vec = rng.normal(size=768).astype(np.float32)
        vec /= np.linalg.norm(vec)

        record = self.store.fetch_review(self._log(embedding=vec.tolist()))
        norm = float(np.linalg.norm(np.asarray(record.embedding, dtype=np.float32)))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_review_without_embedding(self) -> None:
        record = self.store.fetch_review(self._log())
        self.assertIsNone(record.embedding)
        self.assertIsNone(record.embedding_dim)

    def test_multidimensional_embedding_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._log(embedding=np.zeros((2, 768), dtype=np.float32).tolist())


class TestDecisionValidation(ReviewStoreTestCase):
    """Decision vocabulary, open-set rejection, and derived is_correction."""

    def test_not_in_catalog_stores_null_true_class(self) -> None:
        review_id = self._log(
            decision=DECISION_NOT_IN_CATALOG,
            true_class_id=None,
            top1_predicted_class_id=12,
        )
        record = self.store.fetch_review(review_id)
        self.assertEqual(record.decision, DECISION_NOT_IN_CATALOG)
        self.assertIsNone(record.true_class_id)

    def test_not_in_catalog_rejects_true_class(self) -> None:
        with self.assertRaises(ValueError):
            self._log(decision=DECISION_NOT_IN_CATALOG, true_class_id=12)

    def test_approved_requires_true_class(self) -> None:
        with self.assertRaises(ValueError):
            self._log(decision=DECISION_APPROVED, true_class_id=None)

    def test_corrected_requires_true_class(self) -> None:
        with self.assertRaises(ValueError):
            self._log(decision=DECISION_CORRECTED, true_class_id=None)

    def test_unknown_decision_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._log(decision="MAYBE_PROBABLY")

    def test_is_correction_derived_from_decision(self) -> None:
        """is_correction is derived, never passed in, so the two cannot drift apart."""
        approved = self.store.fetch_review(self._log(decision=DECISION_APPROVED, true_class_id=12))
        corrected = self.store.fetch_review(self._log(decision=DECISION_CORRECTED, true_class_id=34))
        rejected = self.store.fetch_review(
            self._log(decision=DECISION_NOT_IN_CATALOG, true_class_id=None)
        )

        self.assertEqual(approved.is_correction, 0)
        self.assertEqual(corrected.is_correction, 1)
        self.assertEqual(rejected.is_correction, 1)

    def test_rejected_review_invisible_to_curation(self) -> None:
        """NOT_IN_CATALOG has no true class, so it must never enter a class gallery."""
        rng = np.random.default_rng(3)
        vec = rng.normal(size=8).astype(np.float32)
        vec /= np.linalg.norm(vec)
        self._log(decision=DECISION_NOT_IN_CATALOG, true_class_id=None, embedding=vec.tolist())

        for class_id in range(67):
            vectors, meta = self.store.fetch_embeddings_by_class(class_id)
            self.assertEqual(len(meta), 0)
            self.assertEqual(vectors.shape[0], 0)


class TestUnconsumedAccounting(ReviewStoreTestCase):
    """Batch consumption bookkeeping — drives the D2/D3 triggers."""

    def test_count_and_fetch_unconsumed(self) -> None:
        self.assertEqual(self.store.count_unconsumed(), 0)
        ids = [self._log() for _ in range(5)]
        self.assertEqual(self.store.count_unconsumed(), 5)
        self.assertEqual(len(self.store.fetch_unconsumed()), 5)
        self.assertEqual({r.review_id for r in self.store.fetch_unconsumed()}, set(ids))

    def test_mark_consumed_removes_from_unconsumed(self) -> None:
        ids = [self._log() for _ in range(5)]
        updated = self.store.mark_consumed(ids[:3], "batch_001")

        self.assertEqual(updated, 3)
        self.assertEqual(self.store.count_unconsumed(), 2)
        remaining = {r.review_id for r in self.store.fetch_unconsumed()}
        self.assertEqual(remaining, set(ids[3:]))

    def test_consumed_batch_id_recorded(self) -> None:
        review_id = self._log()
        self.store.mark_consumed([review_id], "batch_042")
        self.assertEqual(self.store.fetch_review(review_id).consumed_in_batch, "batch_042")

    def test_reconsumption_is_a_no_op(self) -> None:
        """A second batch must not steal reviews already credited to the first."""
        review_id = self._log()
        self.store.mark_consumed([review_id], "batch_001")
        updated = self.store.mark_consumed([review_id], "batch_002")

        self.assertEqual(updated, 0)
        self.assertEqual(self.store.fetch_review(review_id).consumed_in_batch, "batch_001")

    def test_mark_consumed_empty_list(self) -> None:
        self.assertEqual(self.store.mark_consumed([], "batch_001"), 0)

    def test_fetch_unconsumed_respects_limit(self) -> None:
        for _ in range(10):
            self._log()
        self.assertEqual(len(self.store.fetch_unconsumed(limit=4)), 4)

    def test_fetch_unconsumed_is_deterministic(self) -> None:
        """Rows written inside one second must still come back in a stable order."""
        for _ in range(20):
            self._log()
        first = [r.review_id for r in self.store.fetch_unconsumed()]
        second = [r.review_id for r in self.store.fetch_unconsumed()]
        self.assertEqual(first, second)


class TestFetchEmbeddingsByClass(ReviewStoreTestCase):
    """Vector retrieval feeding curation."""

    def _unit(self, seed: int, dim: int = 16) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_returns_only_requested_class(self) -> None:
        for seed in range(3):
            self._log(true_class_id=5, embedding=self._unit(seed).tolist())
        for seed in range(2):
            self._log(true_class_id=9, embedding=self._unit(100 + seed).tolist())

        vectors, meta = self.store.fetch_embeddings_by_class(5)
        self.assertEqual(vectors.shape, (3, 16))
        self.assertEqual(len(meta), 3)
        self.assertTrue(all(m["remapped_class_id"] == 5 for m in meta))

    def test_skips_reviews_without_embeddings(self) -> None:
        self._log(true_class_id=5, embedding=self._unit(1).tolist())
        self._log(true_class_id=5)  # no vector

        vectors, meta = self.store.fetch_embeddings_by_class(5)
        self.assertEqual(vectors.shape[0], 1)
        self.assertEqual(len(meta), 1)

    def test_empty_class_returns_empty_array(self) -> None:
        vectors, meta = self.store.fetch_embeddings_by_class(999)
        self.assertEqual(vectors.shape[0], 0)
        self.assertEqual(meta, [])

    def test_metadata_carries_source_image_for_leakage_control(self) -> None:
        self._log(true_class_id=5, source_image="shelf_042.jpg", embedding=self._unit(1).tolist())
        _, meta = self.store.fetch_embeddings_by_class(5)
        self.assertEqual(meta[0]["source_image"], "shelf_042.jpg")

    def test_returned_in_insertion_order(self) -> None:
        """Ordered by rowid, so a batch written inside one second stays stable."""
        ids = [self._log(true_class_id=5, embedding=self._unit(s).tolist()) for s in range(4)]
        _, meta = self.store.fetch_embeddings_by_class(5)
        self.assertEqual([m["review_id"] for m in meta], ids)

    def test_vectors_and_metadata_are_row_aligned(self) -> None:
        ids = [self._log(true_class_id=5, embedding=self._unit(s).tolist()) for s in range(4)]
        vectors, meta = self.store.fetch_embeddings_by_class(5)

        self.assertEqual([m["review_id"] for m in meta], ids)
        for idx, review_id in enumerate(ids):
            expected = np.asarray(self.store.fetch_review(review_id).embedding, dtype=np.float32)
            np.testing.assert_allclose(vectors[idx], expected, rtol=0, atol=0)

    def test_mixed_dimensions_raise(self) -> None:
        """A backbone change mid-stream must fail loudly, not silently corrupt the batch."""
        self._log(true_class_id=5, embedding=self._unit(1, dim=16).tolist())
        self._log(true_class_id=5, embedding=self._unit(2, dim=32).tolist())

        with self.assertRaises(ValueError):
            self.store.fetch_embeddings_by_class(5)


class TestReferentialIntegrity(ReviewStoreTestCase):
    """Foreign keys are off by default in SQLite; confirm the PRAGMA took effect."""

    def test_orphan_candidate_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store.conn:
                self.store.conn.execute(
                    "INSERT INTO review_candidates (review_id, rank, class_id, similarity) VALUES (?, ?, ?, ?)",
                    ("rev_orphan", 1, 3, 0.5),
                )

    def test_candidates_isolated_per_review(self) -> None:
        first = self._log(candidates=[{"class_id": 1, "similarity": 0.9}])
        second = self._log(candidates=[{"class_id": 2, "similarity": 0.8}])

        self.assertEqual([c.class_id for c in self.store.fetch_candidates(first)], [1])
        self.assertEqual([c.class_id for c in self.store.fetch_candidates(second)], [2])


class TestPersistence(ReviewStoreTestCase):
    """Data must survive a shutdown/reopen cycle."""

    def test_reopen_preserves_reviews(self) -> None:
        rng = np.random.default_rng(11)
        vec = rng.normal(size=768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        review_id = self._log(embedding=vec.tolist(), candidates=[{"class_id": 12, "similarity": 0.91}])
        self.store.shutdown()

        reopened = ReviewStore()
        reopened.initialize({"db_path": self.db_path})
        try:
            record = reopened.fetch_review(review_id)
            self.assertIsNotNone(record)
            np.testing.assert_allclose(
                np.asarray(record.embedding, dtype=np.float32), vec, rtol=0, atol=0
            )
            self.assertEqual(len(reopened.fetch_candidates(review_id)), 1)
            self.assertEqual(reopened.count_unconsumed(), 1)
        finally:
            reopened.shutdown()


if __name__ == "__main__":
    unittest.main()
