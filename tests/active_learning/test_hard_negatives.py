"""Unit tests for Pipeline 3 hard negative confusion mining."""

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
from ml.active_learning.hard_negatives import (
    mine_confusion_pairs,
    mine_from_store,
    ConfusionReport,
)


class HardNegativeTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ReviewStore()
        self.store.initialize({"db_path": str(Path(self._tmp.name) / "reviews.db")})

    def tearDown(self) -> None:
        self.store.shutdown()
        self._tmp.cleanup()

    def _log(self, true_class: int, candidates, decision: str = DECISION_CORRECTED, **kw):
        """Logs a review whose slate is given as [(class_id, similarity), ...]."""
        payload = {
            "source_image": "shelf.jpg",
            "decision": decision,
            "top1_predicted_class_id": candidates[0][0],
            "top1_similarity": candidates[0][1],
            "reviewer_id": "reviewer",
            "true_class_id": true_class,
            "candidates": [
                {"class_id": cid, "similarity": sim} for cid, sim in candidates
            ],
        }
        payload.update(kw)
        return self.store.log_review(**payload)


class TestMining(HardNegativeTestCase):

    def test_planted_confusion_is_recovered(self) -> None:
        """Class 12 is repeatedly beaten by class 34."""
        for _ in range(5):
            self._log(12, [(34, 0.91), (12, 0.88), (7, 0.60)])

        report = mine_from_store(self.store)
        pair = next(p for p in report.pairs if p.confused_class_id == 34)

        self.assertEqual(pair.true_class_id, 12)
        self.assertEqual(pair.frequency, 5)
        self.assertEqual(pair.n_outranked, 5)
        self.assertAlmostEqual(pair.mean_similarity, 0.91, places=6)
        self.assertAlmostEqual(pair.mean_margin, -0.03, places=6)

    def test_pairs_ordered_by_frequency(self) -> None:
        for _ in range(6):
            self._log(12, [(34, 0.91), (12, 0.88)])
        for _ in range(2):
            self._log(12, [(50, 0.90), (12, 0.88)])

        report = mine_from_store(self.store)
        self.assertEqual(report.pairs[0].confused_class_id, 34)
        self.assertEqual(report.pairs[0].frequency, 6)

    def test_distant_candidates_ignored(self) -> None:
        """A candidate far below the true class is not a confusion."""
        self._log(12, [(12, 0.95), (34, 0.40)])

        report = mine_confusion_pairs(
            self.store.fetch_reviews_with_candidates(), near_miss_margin=0.05
        )
        self.assertEqual(report.pairs, [])

    def test_near_miss_counted_without_outranking(self) -> None:
        """A close second is a confusion waiting to happen."""
        self._log(12, [(12, 0.90), (34, 0.88)])

        report = mine_confusion_pairs(
            self.store.fetch_reviews_with_candidates(), near_miss_margin=0.05
        )
        pair = report.pairs[0]

        self.assertEqual(pair.confused_class_id, 34)
        self.assertEqual(pair.n_outranked, 0, "it did not outrank, only came close")
        self.assertGreater(pair.mean_margin, 0.0)

    def test_margin_controls_sensitivity(self) -> None:
        self._log(12, [(12, 0.90), (34, 0.83)])

        tight = mine_confusion_pairs(
            self.store.fetch_reviews_with_candidates(), near_miss_margin=0.05
        )
        loose = mine_confusion_pairs(
            self.store.fetch_reviews_with_candidates(), near_miss_margin=0.10
        )
        self.assertEqual(len(tight.pairs), 0)
        self.assertEqual(len(loose.pairs), 1)

    def test_true_class_absent_from_slate(self) -> None:
        """Every candidate beat a truth that never made the Top-K."""
        self._log(99, [(12, 0.80), (34, 0.75)])

        report = mine_from_store(self.store)
        self.assertEqual(report.n_true_class_absent, 1)
        self.assertEqual(len(report.pairs), 2)
        self.assertTrue(all(p.n_outranked == 1 for p in report.pairs))
        self.assertTrue(all(p.mean_margin < 0 for p in report.pairs))

    def test_correct_predictions_produce_no_confusion(self) -> None:
        for _ in range(3):
            self._log(12, [(12, 0.97), (34, 0.30)], decision=DECISION_APPROVED)

        self.assertEqual(mine_from_store(self.store).pairs, [])

    def test_open_set_rejections_excluded(self) -> None:
        """NOT_IN_CATALOG has no ground truth, so it cannot define a confusion."""
        self._log(
            None, [(12, 0.80), (34, 0.75)],
            decision=DECISION_NOT_IN_CATALOG, true_class_id=None,
        )
        report = mine_from_store(self.store)

        self.assertEqual(report.n_reviews, 0)
        self.assertEqual(report.pairs, [])

    def test_reviews_without_candidates_counted_but_unmined(self) -> None:
        self.store.log_review(
            source_image="shelf.jpg", decision=DECISION_APPROVED,
            top1_predicted_class_id=12, top1_similarity=0.9,
            reviewer_id="reviewer", true_class_id=12,
        )
        report = mine_from_store(self.store)

        self.assertEqual(report.n_reviews, 1)
        self.assertEqual(report.n_reviews_with_candidates, 0)

    def test_min_frequency_filter(self) -> None:
        for _ in range(4):
            self._log(12, [(34, 0.91), (12, 0.88)])
        self._log(12, [(50, 0.91), (12, 0.88)])

        report = mine_confusion_pairs(
            self.store.fetch_reviews_with_candidates(), min_frequency=2
        )
        self.assertEqual([p.confused_class_id for p in report.pairs], [34])

    def test_empty_corpus(self) -> None:
        report = mine_from_store(self.store)
        self.assertEqual(report.pairs, [])
        self.assertEqual(report.n_reviews, 0)

    def test_mining_is_deterministic(self) -> None:
        for _ in range(3):
            self._log(12, [(34, 0.91), (12, 0.88)])
            self._log(5, [(9, 0.85), (5, 0.84)])

        first = mine_from_store(self.store)
        second = mine_from_store(self.store)
        self.assertEqual(
            [(p.true_class_id, p.confused_class_id) for p in first.pairs],
            [(p.true_class_id, p.confused_class_id) for p in second.pairs],
        )


class TestReportHelpers(HardNegativeTestCase):

    def test_top_confusers(self) -> None:
        for _ in range(5):
            self._log(12, [(34, 0.91), (12, 0.88)])
        for _ in range(2):
            self._log(12, [(50, 0.90), (12, 0.88)])

        report = mine_from_store(self.store)
        self.assertEqual(report.top_confusers(12, k=2), [34, 50])
        self.assertEqual(report.top_confusers(999), [])

    def test_confusion_groups_lead_with_true_class(self) -> None:
        for _ in range(3):
            self._log(12, [(34, 0.91), (12, 0.88)])

        groups = mine_from_store(self.store).confusion_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], 12)
        self.assertIn(34, groups[0])

    def test_summary_renders(self) -> None:
        for _ in range(3):
            self._log(12, [(34, 0.91), (12, 0.88)])

        summary = mine_from_store(self.store).summary()
        self.assertIn("Confusion report", summary)
        self.assertIn("34", summary)

    def test_summary_on_empty_report(self) -> None:
        self.assertIn("0 pairs", ConfusionReport().summary())


if __name__ == "__main__":
    unittest.main()
