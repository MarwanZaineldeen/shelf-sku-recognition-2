"""Unit tests for the audit-to-review-store bridge.

Includes the equivalence coverage that gates deleting hitl_store.py: every
capability that module offered must be demonstrably served here.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.base import BBoxDTO, PredictionDTO
from ml.active_learning.store import (
    ReviewStore,
    DECISION_APPROVED,
    DECISION_CORRECTED,
    DECISION_NOT_IN_CATALOG,
)
from ml.active_learning.ingest import (
    ReviewContext,
    ReviewContextCache,
    decision_for,
    record_review,
    NOT_IN_CATALOG_SENTINEL,
)

DIM = 16


def unit_vector(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def make_prediction(crop_id: str, class_id: int = 12, with_embedding: bool = True) -> PredictionDTO:
    return PredictionDTO(
        crop_id=crop_id,
        bbox=BBoxDTO(x1=0, y1=0, x2=10, y2=10, confidence=0.91),
        predicted_class_id=class_id,
        confidence_probability=0.78,
        automated=False,
        reject_reason="LOW_VISUAL_CONFIDENCE",
        top5_candidates=[
            {"class_id": class_id, "display_name": "A", "similarity": 0.91},
            {"class_id": class_id + 1, "display_name": "B", "similarity": 0.88},
        ],
        embedding=unit_vector(1).tolist() if with_embedding else None,
    )


class TestDecisionMapping(unittest.TestCase):

    def test_matching_class_is_an_approval(self) -> None:
        self.assertEqual(decision_for(12, 12), (DECISION_APPROVED, 12))

    def test_different_class_is_a_correction(self) -> None:
        self.assertEqual(decision_for(34, 12), (DECISION_CORRECTED, 34))

    def test_negative_class_is_an_open_set_rejection(self) -> None:
        """The dashboard's 'Unknown / Non-Catalog Competitor SKU' option."""
        self.assertEqual(
            decision_for(NOT_IN_CATALOG_SENTINEL, 12), (DECISION_NOT_IN_CATALOG, None)
        )

    def test_rejection_wins_even_when_prediction_also_negative(self) -> None:
        self.assertEqual(decision_for(-1, -1), (DECISION_NOT_IN_CATALOG, None))

    def test_assignment_over_a_failed_prediction_is_a_correction(self) -> None:
        """predicted_class_id = -1 means NO_MATCHING_CANDIDATES."""
        self.assertEqual(decision_for(12, -1), (DECISION_CORRECTED, 12))


class TestReviewContextCache(unittest.TestCase):

    def test_put_and_get(self) -> None:
        cache = ReviewContextCache()
        cache.put(ReviewContext(
            source_image="shelf.jpg", crop_id="crop_1",
            predicted_class_id=12, top1_similarity=0.9,
        ))
        self.assertEqual(cache.get("shelf.jpg", "crop_1").predicted_class_id, 12)

    def test_miss_returns_none(self) -> None:
        self.assertIsNone(ReviewContextCache().get("shelf.jpg", "crop_1"))

    def test_keyed_by_image_and_crop(self) -> None:
        """The same crop_id recurs across shelf images and must not collide."""
        cache = ReviewContextCache()
        for image, class_id in (("a.jpg", 1), ("b.jpg", 2)):
            cache.put(ReviewContext(
                source_image=image, crop_id="crop_1",
                predicted_class_id=class_id, top1_similarity=0.9,
            ))

        self.assertEqual(cache.get("a.jpg", "crop_1").predicted_class_id, 1)
        self.assertEqual(cache.get("b.jpg", "crop_1").predicted_class_id, 2)

    def test_evicts_least_recently_used(self) -> None:
        cache = ReviewContextCache(max_entries=2)
        for i in range(3):
            cache.put(ReviewContext(
                source_image="s.jpg", crop_id=f"crop_{i}",
                predicted_class_id=i, top1_similarity=0.9,
            ))

        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get("s.jpg", "crop_0"))
        self.assertIsNotNone(cache.get("s.jpg", "crop_2"))

    def test_reading_an_entry_protects_it_from_eviction(self) -> None:
        cache = ReviewContextCache(max_entries=2)
        for i in range(2):
            cache.put(ReviewContext(
                source_image="s.jpg", crop_id=f"crop_{i}",
                predicted_class_id=i, top1_similarity=0.9,
            ))

        cache.get("s.jpg", "crop_0")  # refresh crop_0
        cache.put(ReviewContext(
            source_image="s.jpg", crop_id="crop_2",
            predicted_class_id=2, top1_similarity=0.9,
        ))

        self.assertIsNotNone(cache.get("s.jpg", "crop_0"))
        self.assertIsNone(cache.get("s.jpg", "crop_1"))

    def test_put_predictions_captures_embedding_and_slate(self) -> None:
        cache = ReviewContextCache()
        cached = cache.put_predictions("shelf.jpg", [make_prediction("crop_1")])

        self.assertEqual(cached, 1)
        context = cache.get("shelf.jpg", "crop_1")
        self.assertIsNotNone(context.embedding)
        self.assertEqual(len(context.candidates), 2)
        self.assertEqual(context.predicted_class_id, 12)

    def test_put_predictions_skips_entries_without_crop_id(self) -> None:
        prediction = make_prediction("crop_1")
        prediction.crop_id = None
        self.assertEqual(ReviewContextCache().put_predictions("s.jpg", [prediction]), 0)

    def test_invalid_capacity_raises(self) -> None:
        with self.assertRaises(ValueError):
            ReviewContextCache(max_entries=0)

    def test_clear(self) -> None:
        cache = ReviewContextCache()
        cache.put_predictions("s.jpg", [make_prediction("crop_1")])
        cache.clear()
        self.assertEqual(len(cache), 0)


class RecordReviewTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ReviewStore()
        self.store.initialize({"db_path": str(Path(self._tmp.name) / "reviews.db")})
        self.cache = ReviewContextCache()

    def tearDown(self) -> None:
        self.store.shutdown()
        self._tmp.cleanup()


class TestRecordReview(RecordReviewTestCase):

    def test_context_supplies_embedding_and_candidates(self) -> None:
        """The whole point of the cache: the vector is captured for free."""
        self.cache.put_predictions("shelf.jpg", [make_prediction("crop_1")])

        review_id = record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=34, reviewer_id="qa",
            context=self.cache.get("shelf.jpg", "crop_1"),
        )
        record = self.store.fetch_review(review_id)

        self.assertEqual(record.decision, DECISION_CORRECTED)
        self.assertEqual(record.true_class_id, 34)
        self.assertEqual(record.top1_predicted_class_id, 12)
        self.assertIsNotNone(record.embedding)
        self.assertEqual(record.embedding_dim, DIM)
        self.assertEqual(len(self.store.fetch_candidates(review_id)), 2)

    def test_approval_recorded_when_reviewer_agrees(self) -> None:
        self.cache.put_predictions("shelf.jpg", [make_prediction("crop_1", class_id=12)])

        review_id = record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=12, reviewer_id="qa",
            context=self.cache.get("shelf.jpg", "crop_1"),
        )
        record = self.store.fetch_review(review_id)

        self.assertEqual(record.decision, DECISION_APPROVED)
        self.assertEqual(record.is_correction, 0)

    def test_open_set_rejection_recorded_with_null_class(self) -> None:
        self.cache.put_predictions("shelf.jpg", [make_prediction("crop_1")])

        review_id = record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=NOT_IN_CATALOG_SENTINEL, reviewer_id="qa",
            context=self.cache.get("shelf.jpg", "crop_1"),
        )
        record = self.store.fetch_review(review_id)

        self.assertEqual(record.decision, DECISION_NOT_IN_CATALOG)
        self.assertIsNone(record.true_class_id)
        self.assertEqual(record.is_correction, 1)

    def test_cache_miss_still_records_the_human_label(self) -> None:
        """Losing an embedding is acceptable; losing a human label is not."""
        review_id = record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=34, reviewer_id="qa",
            context=None, predicted_class_id=12, top1_similarity=0.91,
        )
        record = self.store.fetch_review(review_id)

        self.assertEqual(record.decision, DECISION_CORRECTED)
        self.assertEqual(record.top1_predicted_class_id, 12)
        self.assertIsNone(record.embedding)

    def test_cache_miss_review_is_skipped_by_promotion(self) -> None:
        """An embedding-less review must not silently corrupt the gallery."""
        from ml.active_learning.memory import GalleryMemoryUpdater
        from ml.retrieval.sqlite_registry import SQLiteGalleryStore

        record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=34, reviewer_id="qa",
            context=None, predicted_class_id=12, top1_similarity=0.9,
        )

        gallery = SQLiteGalleryStore()
        gallery.initialize({"db_path": str(Path(self._tmp.name) / "gallery.db")})
        updater = GalleryMemoryUpdater(gallery_store=gallery)
        updater.initialize({})
        try:
            report = updater.promote_reviews(
                self.store.fetch_unconsumed(), apply=True, expected_dim=DIM
            )
            self.assertEqual(report.n_promoted, 0)
            self.assertEqual(report.skipped_reasons.get("NO_EMBEDDING"), 1)
        finally:
            gallery.shutdown()

    def test_missing_prediction_entirely_raises(self) -> None:
        """Without a predicted class, an approval is indistinguishable from a correction."""
        with self.assertRaises(ValueError):
            record_review(
                store=self.store, source_image="shelf.jpg", crop_id="crop_1",
                assigned_class_id=34, reviewer_id="qa",
                context=None, predicted_class_id=None,
            )

    def test_context_overrides_client_supplied_prediction(self) -> None:
        """Server-side context is authoritative; a client cannot rewrite history."""
        self.cache.put_predictions("shelf.jpg", [make_prediction("crop_1", class_id=12)])

        review_id = record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=12, reviewer_id="qa",
            context=self.cache.get("shelf.jpg", "crop_1"),
            predicted_class_id=99, top1_similarity=0.1,
        )
        record = self.store.fetch_review(review_id)

        self.assertEqual(record.top1_predicted_class_id, 12)
        self.assertEqual(record.decision, DECISION_APPROVED)

    def test_source_image_recorded_for_leakage_control(self) -> None:
        self.cache.put_predictions("shelf_042.jpg", [make_prediction("crop_1")])

        review_id = record_review(
            store=self.store, source_image="shelf_042.jpg", crop_id="crop_1",
            assigned_class_id=34, reviewer_id="qa",
            context=self.cache.get("shelf_042.jpg", "crop_1"),
        )
        self.assertEqual(self.store.fetch_review(review_id).source_image, "shelf_042.jpg")

    def test_reviewer_id_recorded(self) -> None:
        review_id = record_review(
            store=self.store, source_image="s.jpg", crop_id="crop_1",
            assigned_class_id=12, reviewer_id="merchandiser_user",
            predicted_class_id=12,
        )
        self.assertEqual(self.store.fetch_review(review_id).reviewer_id, "merchandiser_user")


class TestHitlStoreEquivalence(RecordReviewTestCase):
    """Gates the removal of hitl_store.py.

    Each test maps one HITLActiveLearningStore capability onto its
    replacement. hitl_store's own continual-learning path never worked —
    approve_task and correct_task both called SQLiteGalleryStore.insert_crop,
    which is defined nowhere — so these are the first working versions.
    """

    def test_replaces_log_hitl_task(self) -> None:
        """hitl_store.log_hitl_task -> record_review, and it keeps the embedding."""
        self.cache.put_predictions("shelf.jpg", [make_prediction("crop_1")])
        review_id = record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=12, reviewer_id="qa",
            context=self.cache.get("shelf.jpg", "crop_1"),
        )
        record = self.store.fetch_review(review_id)

        self.assertEqual(record.source_image, "shelf.jpg")
        self.assertEqual(record.top1_predicted_class_id, 12)
        self.assertIsNotNone(record.calibrated_probability)
        self.assertIsNotNone(record.embedding)

    def test_replaces_approve_task(self) -> None:
        """hitl_store.approve_task -> an APPROVED review that actually promotes."""
        from ml.active_learning.memory import GalleryMemoryUpdater
        from ml.retrieval.sqlite_registry import SQLiteGalleryStore

        self.cache.put_predictions("shelf.jpg", [make_prediction("crop_1", class_id=12)])
        record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=12, reviewer_id="qa",
            context=self.cache.get("shelf.jpg", "crop_1"),
        )

        gallery = SQLiteGalleryStore()
        gallery.initialize({"db_path": str(Path(self._tmp.name) / "gallery.db")})
        updater = GalleryMemoryUpdater(gallery_store=gallery)
        updater.initialize({})
        try:
            report = updater.promote_reviews(
                self.store.fetch_unconsumed(), apply=True, expected_dim=DIM
            )
            self.assertEqual(report.n_promoted, 1)

            _, metadata = gallery.fetch_all_references()
            self.assertEqual(metadata[0]["remapped_class_id"], 12)
            self.assertEqual(metadata[0]["origin"], "continual")
        finally:
            gallery.shutdown()

    def test_replaces_correct_task(self) -> None:
        """hitl_store.correct_task -> a CORRECTED review promoted under the true class."""
        from ml.active_learning.memory import GalleryMemoryUpdater
        from ml.retrieval.sqlite_registry import SQLiteGalleryStore

        self.cache.put_predictions("shelf.jpg", [make_prediction("crop_1", class_id=12)])
        record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=34, reviewer_id="qa",
            context=self.cache.get("shelf.jpg", "crop_1"),
        )

        gallery = SQLiteGalleryStore()
        gallery.initialize({"db_path": str(Path(self._tmp.name) / "gallery.db")})
        updater = GalleryMemoryUpdater(gallery_store=gallery)
        updater.initialize({})
        try:
            updater.promote_reviews(
                self.store.fetch_unconsumed(), apply=True, expected_dim=DIM
            )
            _, metadata = gallery.fetch_all_references()
            self.assertEqual(
                metadata[0]["remapped_class_id"], 34,
                "the corrected class must be stored, not the prediction",
            )
        finally:
            gallery.shutdown()

    def test_provides_capabilities_hitl_store_lacked(self) -> None:
        """Open-set rejection, candidate slates, and batch accounting are new."""
        self.cache.put_predictions("shelf.jpg", [make_prediction("crop_1")])
        review_id = record_review(
            store=self.store, source_image="shelf.jpg", crop_id="crop_1",
            assigned_class_id=NOT_IN_CATALOG_SENTINEL, reviewer_id="qa",
            context=self.cache.get("shelf.jpg", "crop_1"),
        )

        self.assertEqual(
            self.store.fetch_review(review_id).decision, DECISION_NOT_IN_CATALOG
        )
        self.assertEqual(len(self.store.fetch_candidates(review_id)), 2)
        self.assertEqual(self.store.count_unconsumed(), 1)

    def test_health_check_and_shutdown_lifecycle(self) -> None:
        """hitl_store implemented IPlugin; ReviewStore does too."""
        ok, _ = self.store.health_check()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
