"""Unit tests for the Pipeline 3 orchestration loop."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.active_learning.loop import (
    should_finetune,
    run_session,
    run_finetune,
    run_report,
    run_rollback,
    build_parser,
    main,
    DEFAULT_FINETUNE_THRESHOLD,
)
from ml.active_learning.store import ReviewStore, DECISION_APPROVED, DECISION_NOT_IN_CATALOG
from ml.retrieval.sqlite_registry import SQLiteGalleryStore

DIM = 16


def basis(i: int, dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[i % dim] = 1.0
    return vec


class TestFinetuneTrigger(unittest.TestCase):
    """D3: volume-based trigger with a force override."""

    def test_below_threshold_does_not_trigger(self) -> None:
        self.assertFalse(should_finetune(499, threshold=500))

    def test_at_threshold_triggers(self) -> None:
        self.assertTrue(should_finetune(500, threshold=500))

    def test_force_overrides(self) -> None:
        self.assertTrue(should_finetune(0, threshold=500, force=True))

    def test_default_threshold_is_500(self) -> None:
        self.assertEqual(DEFAULT_FINETUNE_THRESHOLD, 500)


class LoopTestCase(unittest.TestCase):
    """Fixture with a populated gallery and review store."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.gallery_db = str(root / "gallery.db")
        self.review_db = str(root / "reviews.db")

        gallery = SQLiteGalleryStore()
        gallery.initialize({"db_path": self.gallery_db})
        gallery.save_references_bulk([
            (1, 1, f"crop_{i}.jpg", "fam_1", f"seed_shelf_{i}.jpg",
             [0.0, 0.0, 10.0, 10.0], basis(i).tolist())
            for i in range(8)
        ])
        gallery.shutdown()

        self.reviews = ReviewStore()
        self.reviews.initialize({"db_path": self.review_db})

    def tearDown(self) -> None:
        self.reviews.shutdown()
        self._tmp.cleanup()

    def _log(self, class_id: int = 1, vector_index: int = 9, **kw) -> str:
        payload = {
            "source_image": "review_shelf.jpg",
            "decision": DECISION_APPROVED,
            "top1_predicted_class_id": class_id,
            "top1_similarity": 0.9,
            "reviewer_id": "reviewer",
            "true_class_id": class_id,
            "embedding": basis(vector_index).tolist(),
            "candidates": [
                {"class_id": class_id, "similarity": 0.9},
                {"class_id": class_id + 1, "similarity": 0.88},
            ],
        }
        payload.update(kw)
        return self.reviews.log_review(**payload)

    def _active_count(self) -> int:
        gallery = SQLiteGalleryStore()
        gallery.initialize({"db_path": self.gallery_db})
        try:
            vectors, _ = gallery.fetch_all_references()
            return int(vectors.shape[0])
        finally:
            gallery.shutdown()


class TestRunSession(LoopTestCase):

    def test_dry_run_changes_nothing(self) -> None:
        self._log()
        before = self._active_count()

        result = run_session(self.review_db, self.gallery_db, cap=100)

        self.assertFalse(result["applied"])
        self.assertEqual(result["n_promoted"], 1)
        self.assertEqual(result["n_reviews_consumed"], 0)
        self.assertEqual(self._active_count(), before)
        self.assertEqual(self.reviews.count_unconsumed(), 1, "backlog must be untouched")

    def test_dry_run_predicts_the_applied_outcome_exactly(self) -> None:
        """A preview that disagrees with the real run is worse than no preview.

        Skipping the writes would let curation run against a gallery without
        the pending promotions, so it would report different counts than the
        applied session actually produces.
        """
        for i in range(5):
            self._log(vector_index=i)

        preview = run_session(self.review_db, self.gallery_db, cap=6)
        applied = run_session(self.review_db, self.gallery_db, cap=6, apply=True)

        for key in ("n_promoted", "gallery_before", "gallery_after", "n_pruned"):
            self.assertEqual(preview[key], applied[key], f"{key} differs between preview and apply")

    def test_dry_run_leaves_gallery_version_recoverable(self) -> None:
        """The preview's transient writes must be fully undone."""
        self._log()
        before = self._active_count()

        run_session(self.review_db, self.gallery_db, cap=3)
        self.assertEqual(self._active_count(), before)

        # And a second preview must behave identically to the first.
        second = run_session(self.review_db, self.gallery_db, cap=3)
        self.assertEqual(second["gallery_before"], before + 1)
        self.assertEqual(self._active_count(), before)

    def test_apply_promotes_and_consumes(self) -> None:
        self._log()
        before = self._active_count()

        result = run_session(self.review_db, self.gallery_db, cap=100, apply=True)

        self.assertTrue(result["applied"])
        self.assertEqual(result["n_promoted"], 1)
        self.assertEqual(result["n_reviews_consumed"], 1)
        self.assertEqual(self._active_count(), before + 1)
        self.assertEqual(self.reviews.count_unconsumed(), 0)

    def test_session_size_bounds_the_batch(self) -> None:
        for i in range(10):
            self._log(vector_index=i)

        result = run_session(self.review_db, self.gallery_db, session_size=4, cap=100, apply=True)

        self.assertEqual(result["n_reviews_in_batch"], 4)
        self.assertEqual(result["remaining_backlog"], 6)

    def test_curation_applies_within_the_session(self) -> None:
        self._log()
        result = run_session(self.review_db, self.gallery_db, cap=3, apply=True)

        self.assertEqual(result["gallery_after"], 3)
        self.assertEqual(self._active_count(), 3)

    def test_promoted_crops_compete_for_gallery_slots(self) -> None:
        """Promotion runs before curation, so new crops are capped too."""
        for i in range(5):
            self._log(vector_index=i)

        result = run_session(self.review_db, self.gallery_db, cap=4, apply=True)
        self.assertEqual(result["gallery_after"], 4)

    def test_open_set_rejections_are_consumed_but_not_promoted(self) -> None:
        self._log(decision=DECISION_NOT_IN_CATALOG, true_class_id=None)

        result = run_session(self.review_db, self.gallery_db, cap=100, apply=True)

        self.assertEqual(result["n_promoted"], 0)
        self.assertEqual(result["n_skipped"], 1)
        self.assertEqual(result["n_reviews_consumed"], 1, "must not be reprocessed forever")

    def test_empty_backlog(self) -> None:
        result = run_session(self.review_db, self.gallery_db, cap=100, apply=True)
        self.assertEqual(result["n_reviews_in_batch"], 0)
        self.assertEqual(result["n_promoted"], 0)

    def test_session_is_reversible(self) -> None:
        self._log()
        before = self._active_count()
        result = run_session(self.review_db, self.gallery_db, cap=3, apply=True)

        gallery = SQLiteGalleryStore()
        gallery.initialize({"db_path": self.gallery_db})
        try:
            gallery.rollback_version(result["baseline_version"])
        finally:
            gallery.shutdown()

        self.assertEqual(self._active_count(), before)


class TestRunFinetune(LoopTestCase):

    def test_below_threshold_skips_training(self) -> None:
        self._log()
        result = run_finetune(self.review_db, str(Path(self._tmp.name) / "ckpt"))

        self.assertFalse(result["trained"])
        self.assertIn("threshold", result["reason"])

    def test_force_trains_and_writes_a_challenger(self) -> None:
        for class_id in (1, 2, 3):
            for i in range(4):
                self._log(class_id=class_id, vector_index=class_id * 4 + i)

        checkpoint_dir = str(Path(self._tmp.name) / "ckpt")
        result = run_finetune(self.review_db, checkpoint_dir, force=True, epochs=3)

        self.assertTrue(result["trained"])
        self.assertTrue(Path(result["checkpoint_path"]).exists())
        self.assertEqual(result["n_classes"], 3)

    def test_checkpoint_is_marked_unpromoted(self) -> None:
        """Training must never imply activation."""
        from ml.active_learning.finetune import load_checkpoint

        for class_id in (1, 2):
            for i in range(4):
                self._log(class_id=class_id, vector_index=class_id * 4 + i)

        result = run_finetune(
            self.review_db, str(Path(self._tmp.name) / "ckpt"), force=True, epochs=2
        )
        _, info = load_checkpoint(result["checkpoint_path"])

        self.assertEqual(info["metadata"]["status"], "CHALLENGER_UNPROMOTED")

    def test_no_embeddings_reports_cleanly(self) -> None:
        self.reviews.log_review(
            source_image="s.jpg", decision=DECISION_APPROVED,
            top1_predicted_class_id=1, top1_similarity=0.9,
            reviewer_id="r", true_class_id=1,
        )
        result = run_finetune(
            self.review_db, str(Path(self._tmp.name) / "ckpt"), force=True
        )
        self.assertFalse(result["trained"])


class TestRunReportAndRollback(LoopTestCase):

    def test_report_renders_gallery_and_backlog(self) -> None:
        self._log()
        report = run_report(self.review_db, self.gallery_db)

        self.assertIn("Gallery:", report)
        self.assertIn("Reviews:", report)
        self.assertIn("1 unconsumed", report)

    def test_report_handles_missing_databases(self) -> None:
        report = run_report("nonexistent_reviews.db", "nonexistent_gallery.db")
        self.assertIn("not found", report)

    def test_report_includes_confusion_table(self) -> None:
        for _ in range(3):
            self._log(class_id=1, candidates=[
                {"class_id": 2, "similarity": 0.95},
                {"class_id": 1, "similarity": 0.90},
            ])
        self.assertIn("Confusion report", run_report(self.review_db, self.gallery_db))

    def test_rollback_dry_run_changes_nothing(self) -> None:
        self._log()
        run_session(self.review_db, self.gallery_db, cap=100, apply=True)
        before = self._active_count()

        result = run_rollback(self.gallery_db, version=1)

        self.assertFalse(result["applied"])
        self.assertEqual(self._active_count(), before)

    def test_rollback_apply_restores(self) -> None:
        gallery = SQLiteGalleryStore()
        gallery.initialize({"db_path": self.gallery_db})
        baseline = gallery.get_current_version()
        gallery.shutdown()

        self._log()
        run_session(self.review_db, self.gallery_db, cap=3, apply=True)
        self.assertEqual(self._active_count(), 3)

        result = run_rollback(self.gallery_db, version=baseline, apply=True)

        self.assertTrue(result["applied"])
        self.assertEqual(result["vectors_after"], 8)


class TestCLI(LoopTestCase):

    def test_parser_requires_a_subcommand(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_session_defaults_to_dry_run(self) -> None:
        self.assertFalse(build_parser().parse_args(["session"]).apply)

    def test_apply_flag_parsed(self) -> None:
        self.assertTrue(build_parser().parse_args(["session", "--apply"]).apply)

    def test_force_flag_parsed(self) -> None:
        self.assertTrue(build_parser().parse_args(["finetune", "--force"]).force)

    def test_default_paths_are_repo_relative(self) -> None:
        """Guards against the hardcoded absolute paths found elsewhere in the repo."""
        args = build_parser().parse_args(["report"])
        self.assertNotIn("Marwan", args.gallery_db)
        self.assertNotIn("Marwan", args.review_db)

    def test_main_report_returns_zero(self) -> None:
        self._log()
        exit_code = main([
            "--review-db", self.review_db, "--gallery-db", self.gallery_db, "report",
        ])
        self.assertEqual(exit_code, 0)

    def test_main_session_dry_run_returns_zero(self) -> None:
        self._log()
        exit_code = main([
            "--review-db", self.review_db, "--gallery-db", self.gallery_db, "session",
        ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(self.reviews.count_unconsumed(), 1, "dry run must not consume")


if __name__ == "__main__":
    unittest.main()
