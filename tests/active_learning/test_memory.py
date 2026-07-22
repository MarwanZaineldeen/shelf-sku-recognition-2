"""Unit tests for the Pipeline 3 gallery memory updater and schema migration.

Model-free: temporary SQLite registries, synthetic vectors, no weights.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from ml.retrieval.numpy_index import NumpyCosineIndex
from ml.active_learning.memory import GalleryMemoryUpdater
from ml.active_learning.store import (
    ReviewStore,
    DECISION_APPROVED,
    DECISION_NOT_IN_CATALOG,
)

DIM = 16


def unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def basis(i: int, dim: int = DIM) -> np.ndarray:
    """Returns an orthonormal basis vector — pairwise cosine 0, never a near-dup."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[i] = 1.0
    return vec


class GalleryTestCase(unittest.TestCase):
    """Fixture providing a throwaway gallery registry."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "gallery.db")
        self.store = SQLiteGalleryStore()
        self.store.initialize({"db_path": self.db_path})
        self.updater = GalleryMemoryUpdater(gallery_store=self.store)
        self.updater.initialize({})

    def tearDown(self) -> None:
        self.store.shutdown()
        self._tmp.cleanup()

    def _bulk(self, entries):
        """entries: list of (class_id, vector). Returns the gallery version."""
        return self.store.save_references_bulk([
            (cid, cid, f"crop_{i}.jpg", f"fam_{cid}", f"shelf_{i}.jpg",
             [0.0, 0.0, 10.0, 10.0], vec.tolist())
            for i, (cid, vec) in enumerate(entries)
        ])

    def _active_count(self) -> int:
        vectors, _ = self.store.fetch_all_references()
        return int(vectors.shape[0])


class TestSchemaMigration(GalleryTestCase):
    """Additive columns must be idempotent and preserve existing rows."""

    def test_curation_columns_created(self) -> None:
        cursor = self.store.conn.execute("PRAGMA table_info(sku_crops)")
        columns = {row["name"] for row in cursor.fetchall()}
        self.assertIn("active", columns)
        self.assertIn("pruned_in_version", columns)
        self.assertIn("origin", columns)

    def test_migration_is_idempotent(self) -> None:
        self._bulk([(1, basis(0))])
        # Re-opening runs the migration again; it must not raise or lose rows.
        self.store.shutdown()
        reopened = SQLiteGalleryStore()
        reopened.initialize({"db_path": self.db_path})
        try:
            vectors, _ = reopened.fetch_all_references()
            self.assertEqual(vectors.shape[0], 1)
        finally:
            reopened.shutdown()

    def test_existing_rows_default_to_active(self) -> None:
        self._bulk([(1, basis(0)), (2, basis(1))])
        self.assertEqual(self._active_count(), 2)

    def test_rows_default_to_seed_origin(self) -> None:
        self._bulk([(1, basis(0))])
        _, metadata = self.store.fetch_all_references()
        self.assertEqual(metadata[0]["origin"], "seed")


class TestFetchAllReferencesContract(GalleryTestCase):
    """fetch_all_references returns an ndarray, matching NumpyCosineIndex."""

    def test_returns_ndarray_not_dtos(self) -> None:
        self._bulk([(1, basis(0)), (2, basis(1))])
        vectors, metadata = self.store.fetch_all_references()

        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(vectors.shape, (2, DIM))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(len(metadata), 2)

    def test_feeds_numpy_index_directly(self) -> None:
        """The production path in server/app.py — must work without conversion."""
        self._bulk([(1, basis(0)), (2, basis(1))])
        vectors, metadata = self.store.fetch_all_references()

        index = NumpyCosineIndex(dimension=DIM)
        index.add(vectors, metadata)
        ok, _ = index.health_check()
        self.assertTrue(ok)

    def test_metadata_carries_id_for_pruning(self) -> None:
        self._bulk([(1, basis(0))])
        _, metadata = self.store.fetch_all_references()
        self.assertIn("id", metadata[0])


class TestSoftDeleteAndRollback(GalleryTestCase):
    """Pruning must be reversible — the reason for the soft-delete migration."""

    def test_prune_hides_rows(self) -> None:
        self._bulk([(1, basis(0)), (1, basis(1)), (2, basis(2))])
        _, metadata = self.store.fetch_all_references()

        self.store.prune_references([metadata[0]["id"]])
        self.assertEqual(self._active_count(), 2)

    def test_rollback_restores_pruned_rows(self) -> None:
        baseline = self._bulk([(1, basis(i)) for i in range(5)])
        self.assertEqual(self._active_count(), 5)

        _, metadata = self.store.fetch_all_references()
        self.store.prune_references([m["id"] for m in metadata[:3]])
        self.assertEqual(self._active_count(), 2)

        self.store.rollback_version(baseline)
        self.assertEqual(self._active_count(), 5)

    def test_rollback_hides_rows_inserted_later(self) -> None:
        first = self._bulk([(1, basis(0))])
        self._bulk([(2, basis(1))])
        self.assertEqual(self._active_count(), 2)

        self.store.rollback_version(first)
        vectors, metadata = self.store.fetch_all_references()
        self.assertEqual(vectors.shape[0], 1)
        self.assertEqual(metadata[0]["remapped_class_id"], 1)

    def test_delete_sku_is_reversible(self) -> None:
        baseline = self._bulk([(1, basis(0)), (2, basis(1))])
        self.store.delete_sku(class_id=1)
        self.assertEqual(self._active_count(), 1)

        self.store.rollback_version(baseline)
        self.assertEqual(self._active_count(), 2)

    def test_prune_empty_list_is_a_no_op(self) -> None:
        self._bulk([(1, basis(0))])
        self.store.prune_references([])
        self.assertEqual(self._active_count(), 1)


class TestClassHistogram(GalleryTestCase):

    def test_counts_per_class(self) -> None:
        self._bulk([(1, basis(0)), (1, basis(1)), (2, basis(2))])
        self.assertEqual(self.store.class_size_histogram(), {1: 2, 2: 1})

    def test_excludes_pruned_rows(self) -> None:
        self._bulk([(1, basis(0)), (1, basis(1))])
        _, metadata = self.store.fetch_all_references()
        self.store.prune_references([metadata[0]["id"]])
        self.assertEqual(self.store.class_size_histogram(), {1: 1})


class TestFetchActiveByClass(GalleryTestCase):

    def test_returns_only_requested_class(self) -> None:
        self._bulk([(1, basis(0)), (1, basis(1)), (2, basis(2))])
        vectors, metadata = self.store.fetch_active_by_class(1)

        self.assertEqual(vectors.shape, (2, DIM))
        self.assertTrue(all(m["remapped_class_id"] == 1 for m in metadata))

    def test_empty_class_returns_empty(self) -> None:
        vectors, metadata = self.store.fetch_active_by_class(42)
        self.assertEqual(vectors.shape[0], 0)
        self.assertEqual(metadata, [])


class TestCurateGallery(GalleryTestCase):
    """End-to-end curation over a registry."""

    def test_dry_run_writes_nothing(self) -> None:
        self._bulk([(1, unit(basis(0) + 0.001 * basis(1))) for _ in range(5)])
        report = self.updater.curate_gallery(cap=2)

        self.assertFalse(report.applied)
        self.assertIsNone(report.version)
        self.assertGreater(report.total_pruned, 0)
        self.assertEqual(self._active_count(), 5, "dry run must not prune")

    def test_apply_prunes_duplicates(self) -> None:
        self._bulk([(1, basis(0)) for _ in range(5)] + [(1, basis(1))])
        report = self.updater.curate_gallery(cap=100, apply=True)

        self.assertTrue(report.applied)
        self.assertEqual(report.total_before, 6)
        self.assertEqual(report.total_kept, 2)   # 5 identical collapse to 1, plus 1 distinct
        self.assertEqual(self._active_count(), 2)

    def test_apply_enforces_cap(self) -> None:
        self._bulk([(1, basis(i)) for i in range(10)])
        report = self.updater.curate_gallery(cap=4, apply=True)

        self.assertEqual(report.total_kept, 4)
        self.assertEqual(self._active_count(), 4)

    def test_curation_is_reversible(self) -> None:
        baseline = self._bulk([(1, basis(i)) for i in range(10)])
        report = self.updater.curate_gallery(cap=3, apply=True)

        self.assertEqual(self._active_count(), 3)
        self.store.rollback_version(baseline)
        self.assertEqual(self._active_count(), 10)

    def test_classes_curated_independently(self) -> None:
        self._bulk(
            [(1, basis(i)) for i in range(6)] + [(2, basis(i)) for i in range(3)]
        )
        self.updater.curate_gallery(cap=2, apply=True)
        self.assertEqual(self.store.class_size_histogram(), {1: 2, 2: 2})

    def test_class_ids_filter(self) -> None:
        self._bulk([(1, basis(i)) for i in range(5)] + [(2, basis(i)) for i in range(5)])
        self.updater.curate_gallery(cap=2, apply=True, class_ids=[1])

        histogram = self.store.class_size_histogram()
        self.assertEqual(histogram[1], 2)
        self.assertEqual(histogram[2], 5, "unlisted classes must be untouched")

    def test_report_totals_are_consistent(self) -> None:
        self._bulk([(1, basis(i)) for i in range(8)])
        report = self.updater.curate_gallery(cap=3)
        self.assertEqual(report.total_before, report.total_kept + report.total_pruned)

    def test_empty_gallery(self) -> None:
        report = self.updater.curate_gallery(cap=10, apply=True)
        self.assertFalse(report.applied)
        self.assertEqual(report.total_before, 0)


class TestPromoteReviews(GalleryTestCase):
    """Promotion of HITL-reviewed crops into the gallery."""

    def setUp(self) -> None:
        super().setUp()
        self._review_tmp = tempfile.TemporaryDirectory()
        self.reviews = ReviewStore()
        self.reviews.initialize({
            "db_path": str(Path(self._review_tmp.name) / "reviews.db")
        })

    def tearDown(self) -> None:
        self.reviews.shutdown()
        self._review_tmp.cleanup()
        super().tearDown()

    def _log(self, **overrides) -> None:
        payload = {
            "source_image": "shelf_new.jpg",
            "decision": DECISION_APPROVED,
            "top1_predicted_class_id": 1,
            "top1_similarity": 0.9,
            "reviewer_id": "reviewer",
            "true_class_id": 1,
            "embedding": basis(5).tolist(),
        }
        payload.update(overrides)
        self.reviews.log_review(**payload)

    def test_dry_run_writes_nothing(self) -> None:
        self._bulk([(1, basis(0))])
        self._log()
        report = self.updater.promote_reviews(self.reviews.fetch_unconsumed())

        self.assertFalse(report.applied)
        self.assertEqual(report.n_promoted, 1)
        self.assertEqual(self._active_count(), 1)

    def test_apply_inserts_with_continual_origin(self) -> None:
        self._bulk([(1, basis(0))])
        self._log()
        report = self.updater.promote_reviews(self.reviews.fetch_unconsumed(), apply=True)

        self.assertTrue(report.applied)
        _, metadata = self.store.fetch_all_references()
        origins = {m["origin"] for m in metadata}
        self.assertEqual(origins, {"seed", "continual"})

    def test_source_image_preserved_for_leakage_control(self) -> None:
        """The promotion gate needs this to enforce gallery/test disjointness."""
        self._bulk([(1, basis(0))])
        self._log(source_image="shelf_042.jpg")
        self.updater.promote_reviews(self.reviews.fetch_unconsumed(), apply=True)

        _, metadata = self.store.fetch_all_references()
        promoted = [m for m in metadata if m["origin"] == "continual"]
        self.assertEqual(promoted[0]["source_image_name"], "shelf_042.jpg")

    def test_open_set_rejections_never_promoted(self) -> None:
        self._bulk([(1, basis(0))])
        self._log(decision=DECISION_NOT_IN_CATALOG, true_class_id=None)
        report = self.updater.promote_reviews(self.reviews.fetch_unconsumed(), apply=True)

        self.assertEqual(report.n_promoted, 0)
        self.assertEqual(report.skipped_reasons.get("NO_VERIFIED_CLASS"), 1)
        self.assertEqual(self._active_count(), 1)

    def test_reviews_without_embeddings_skipped(self) -> None:
        self._bulk([(1, basis(0))])
        self._log(embedding=None)
        report = self.updater.promote_reviews(self.reviews.fetch_unconsumed(), apply=True)

        self.assertEqual(report.n_promoted, 0)
        self.assertEqual(report.skipped_reasons.get("NO_EMBEDDING"), 1)

    def test_dimension_mismatch_skipped_not_coerced(self) -> None:
        """A backbone change must not silently corrupt the gallery."""
        self._bulk([(1, basis(0))])
        self._log(embedding=np.zeros(DIM * 2, dtype=np.float32).tolist())
        report = self.updater.promote_reviews(self.reviews.fetch_unconsumed(), apply=True)

        self.assertEqual(report.n_promoted, 0)
        self.assertEqual(report.skipped_reasons.get("DIMENSION_MISMATCH"), 1)
        self.assertEqual(self._active_count(), 1)

    def test_promotion_is_reversible(self) -> None:
        baseline = self._bulk([(1, basis(0))])
        self._log()
        self.updater.promote_reviews(self.reviews.fetch_unconsumed(), apply=True)
        self.assertEqual(self._active_count(), 2)

        self.store.rollback_version(baseline)
        self.assertEqual(self._active_count(), 1)

    def test_promoted_vectors_are_searchable(self) -> None:
        """The point of promotion: the new crop must be retrievable."""
        self._bulk([(1, basis(0))])
        query = basis(5)
        self._log(true_class_id=7, embedding=query.tolist())
        self.updater.promote_reviews(self.reviews.fetch_unconsumed(), apply=True)

        index = NumpyCosineIndex(dimension=DIM)
        index.initialize({"dimension": DIM, "db_path": self.db_path})
        indices, scores = index.search(query.reshape(1, -1), top_k=1)

        _, metadata = self.store.fetch_all_references()
        self.assertEqual(metadata[int(indices[0, 0])]["remapped_class_id"], 7)
        self.assertAlmostEqual(float(scores[0, 0]), 1.0, places=5)


class TestRebuildIndex(GalleryTestCase):

    def test_rebuild_reflects_pruning(self) -> None:
        self._bulk([(1, basis(i)) for i in range(6)])
        index = NumpyCosineIndex(dimension=DIM)

        self.assertEqual(self.updater.rebuild_index(index), 6)
        self.updater.curate_gallery(cap=2, apply=True)
        self.assertEqual(self.updater.rebuild_index(index), 2)

    def test_rebuild_on_empty_gallery(self) -> None:
        index = NumpyCosineIndex(dimension=DIM)
        self.assertEqual(self.updater.rebuild_index(index), 0)
        ok, _ = index.health_check()
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
