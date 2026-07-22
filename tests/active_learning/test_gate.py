"""Unit tests for the Pipeline 3 statistical promotion gate.

Model-free: synthetic retrieval outputs, no DB and no weights.
"""

import unittest

import numpy as np

from ml.active_learning.gate import (
    SystemEvaluation,
    GateDecision,
    LeakageError,
    evaluate_promotion,
    paired_bootstrap_delta,
    assert_gallery_test_disjoint,
    recall_indicators,
    top1_correct,
    fit_platt,
    platt_probabilities,
)

N_CLASSES = 67


def make_neighbors(
    labels: np.ndarray,
    top1_hit: np.ndarray,
    top5_hit: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Builds a (M, k) neighbor-label matrix with prescribed hit patterns.

    Misses are filled with a label that is never the true one, so Top-1 and
    Top-5 correctness are exactly what the flags say.
    """
    neighbors = ((labels[:, None] + 1) % N_CLASSES) + np.zeros((1, k), dtype=int)
    neighbors[top1_hit, 0] = labels[top1_hit]
    # A Top-5-only hit sits at rank 2, out of the Top-1 slot.
    top5_only = top5_hit & ~top1_hit
    neighbors[top5_only, 2] = labels[top5_only]
    return neighbors


def make_split(
    n: int,
    accuracy: float,
    correct_loc: float,
    correct_scale: float,
    wrong_loc: float,
    wrong_scale: float,
    seed: int,
    top5_bonus: float = 0.9,
):
    """Generates one split: labels, neighbor matrix, and rank-1 scores.

    Args:
        accuracy: Fraction of queries answered correctly at rank 1.
        correct_loc/correct_scale: Score distribution for correct answers.
        wrong_loc/wrong_scale: Score distribution for wrong answers.
        top5_bonus: Fraction of Top-1 misses recovered within the Top-5.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, N_CLASSES, size=n)

    top1_hit = rng.random(n) < accuracy
    top5_hit = top1_hit | (rng.random(n) < top5_bonus)

    scores = np.where(
        top1_hit,
        rng.normal(correct_loc, correct_scale, size=n),
        rng.normal(wrong_loc, wrong_scale, size=n),
    )
    return labels, make_neighbors(labels, top1_hit, top5_hit), np.clip(scores, 0.0, 1.0)


def make_system(
    name: str,
    accuracy: float,
    seed: int,
    correct_loc: float = 0.95,
    correct_scale: float = 0.01,
    wrong_loc: float = 0.55,
    wrong_scale: float = 0.03,
    n_val: int = 2000,
    n_test: int = 6000,
    test_labels: np.ndarray = None,
    top5_bonus: float = 0.9,
) -> SystemEvaluation:
    """Builds a SystemEvaluation with well-separated scores by default.

    Passing `test_labels` forces the test split to reuse a given label
    vector, which the gate requires for pairing.

    Split sizes mirror the project's real evaluation data (147 validation /
    150 test images, ~6,100 boxes each). Size matters here: the Top-5
    non-inferiority margin is finer than the sampling noise of a small
    evaluation set, so an undersized fixture would fail criteria for lack of
    data rather than lack of quality.
    """
    val_labels, val_neighbors, val_scores = make_split(
        n_val, accuracy, correct_loc, correct_scale, wrong_loc, wrong_scale, seed,
        top5_bonus=top5_bonus,
    )
    tst_labels, tst_neighbors, tst_scores = make_split(
        n_test, accuracy, correct_loc, correct_scale, wrong_loc, wrong_scale, seed + 1,
        top5_bonus=top5_bonus,
    )

    if test_labels is not None:
        # Re-key the test split onto shared labels, preserving hit pattern.
        hit1 = tst_neighbors[:, 0] == tst_labels
        hit5 = (tst_neighbors == tst_labels[:, None]).any(axis=1)
        tst_labels = test_labels
        tst_neighbors = make_neighbors(tst_labels, hit1, hit5)

    return SystemEvaluation(
        name=name,
        val_neighbor_labels=val_neighbors,
        val_query_labels=val_labels,
        val_top_scores=val_scores,
        test_neighbor_labels=tst_neighbors,
        test_query_labels=tst_labels,
        test_top_scores=tst_scores,
    )


class TestIndicators(unittest.TestCase):

    def test_recall_indicators(self) -> None:
        neighbors = np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]])
        labels = np.array([1, 1, 1])

        np.testing.assert_array_equal(recall_indicators(neighbors, labels, 1), [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(recall_indicators(neighbors, labels, 3), [1.0, 1.0, 0.0])

    def test_recall_indicators_rejects_bad_k(self) -> None:
        with self.assertRaises(ValueError):
            recall_indicators(np.array([[1, 2]]), np.array([1]), 0)

    def test_top1_correct(self) -> None:
        neighbors = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(top1_correct(neighbors, np.array([1, 4])), [True, False])


class TestPlattCalibration(unittest.TestCase):

    def test_fit_recovers_a_positive_slope(self) -> None:
        """Higher similarity must map to higher probability."""
        rng = np.random.default_rng(0)
        correct = rng.random(500) < 0.6
        sims = np.where(correct, rng.normal(0.9, 0.03, 500), rng.normal(0.6, 0.03, 500))

        a, b = fit_platt(sims, correct)
        self.assertGreater(a, 0.0)
        self.assertGreater(
            platt_probabilities(np.array([0.9]), a, b)[0],
            platt_probabilities(np.array([0.6]), a, b)[0],
        )

    def test_all_correct_split_saturates_high(self) -> None:
        a, b = fit_platt(np.linspace(0.7, 0.99, 50), np.ones(50, dtype=bool))
        self.assertGreater(platt_probabilities(np.array([0.8]), a, b)[0], 0.99)

    def test_all_wrong_split_saturates_low(self) -> None:
        a, b = fit_platt(np.linspace(0.1, 0.5, 50), np.zeros(50, dtype=bool))
        self.assertLess(platt_probabilities(np.array([0.3]), a, b)[0], 0.01)

    def test_empty_split_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_platt(np.array([]), np.array([]))

    def test_matches_production_calibrator(self) -> None:
        """Gate estimates must reflect what PlattCalibrator does at serve time."""
        from ml.calibrators.platt import PlattCalibrator

        calibrator = PlattCalibrator()
        calibrator.initialize({"global_coefs": {"a": 15.0, "b": -11.0}})

        sims = np.array([0.5, 0.75, 0.85, 0.92, 0.99])
        expected = [calibrator.calibrate(float(s), 0) for s in sims]
        np.testing.assert_allclose(platt_probabilities(sims, 15.0, -11.0), expected, rtol=1e-9)


class TestPairedBootstrap(unittest.TestCase):

    def test_identical_systems_give_zero_delta(self) -> None:
        values = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        result = paired_bootstrap_delta(values, values, n_boot=200)

        self.assertEqual(result.delta_mean, 0.0)
        self.assertEqual(result.ci_lower, 0.0)
        self.assertEqual(result.ci_upper, 0.0)

    def test_uniformly_better_challenger_has_positive_ci(self) -> None:
        champion = np.zeros(200)
        challenger = np.ones(200)
        result = paired_bootstrap_delta(champion, challenger, n_boot=500)

        self.assertAlmostEqual(result.delta_mean, 1.0)
        self.assertGreater(result.ci_lower, 0.0)

    def test_is_reproducible(self) -> None:
        rng = np.random.default_rng(1)
        champion = (rng.random(300) < 0.8).astype(float)
        challenger = (rng.random(300) < 0.85).astype(float)

        first = paired_bootstrap_delta(champion, challenger, seed=7)
        second = paired_bootstrap_delta(champion, challenger, seed=7)
        self.assertEqual(first.ci_lower, second.ci_lower)

    def test_pairing_tightens_the_interval(self) -> None:
        """The reason the gate pairs: shared query difficulty cancels out.

        Two systems that agree on most queries have a much better-determined
        difference than either system's absolute accuracy suggests.
        """
        rng = np.random.default_rng(2)
        champion = (rng.random(600) < 0.7).astype(float)
        # Challenger tracks the champion, flipping a few misses to hits.
        challenger = champion.copy()
        flip = np.flatnonzero(champion == 0.0)[:30]
        challenger[flip] = 1.0

        paired = paired_bootstrap_delta(champion, challenger, n_boot=1000, seed=3)

        # Unpaired: resample each system independently, destroying the pairing.
        gen = np.random.default_rng(3)
        n = champion.shape[0]
        unpaired_deltas = (
            challenger[gen.integers(0, n, size=(1000, n))].mean(axis=1)
            - champion[gen.integers(0, n, size=(1000, n))].mean(axis=1)
        )
        unpaired_width = float(
            np.percentile(unpaired_deltas, 97.5) - np.percentile(unpaired_deltas, 2.5)
        )

        self.assertLess(paired.ci_upper - paired.ci_lower, unpaired_width)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_delta(np.zeros(5), np.zeros(6))

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_delta(np.array([]), np.array([]))


class TestLeakageAssertion(unittest.TestCase):

    def test_disjoint_sets_pass(self) -> None:
        assert_gallery_test_disjoint(["a.jpg", "b.jpg"], ["c.jpg", "d.jpg"])

    def test_overlap_raises(self) -> None:
        with self.assertRaises(LeakageError):
            assert_gallery_test_disjoint(["a.jpg", "b.jpg"], ["b.jpg", "c.jpg"])

    def test_message_names_the_offenders(self) -> None:
        with self.assertRaises(LeakageError) as ctx:
            assert_gallery_test_disjoint(["shelf_042.jpg"], ["shelf_042.jpg"])
        self.assertIn("shelf_042.jpg", str(ctx.exception))

    def test_large_overlap_is_truncated_but_counted(self) -> None:
        images = [f"shelf_{i}.jpg" for i in range(25)]
        with self.assertRaises(LeakageError) as ctx:
            assert_gallery_test_disjoint(images, images)
        self.assertIn("25 shelf image(s)", str(ctx.exception))
        self.assertIn("more", str(ctx.exception))

    def test_empty_inputs_pass(self) -> None:
        assert_gallery_test_disjoint([], [])


class TestPromotionGate(unittest.TestCase):
    """End-to-end verdicts."""

    def test_identical_systems_are_not_promoted(self) -> None:
        """No evidence of improvement must mean no promotion."""
        system = make_system("baseline", accuracy=0.88, seed=10)
        challenger = SystemEvaluation(
            name="challenger",
            val_neighbor_labels=system.val_neighbor_labels,
            val_query_labels=system.val_query_labels,
            val_top_scores=system.val_top_scores,
            test_neighbor_labels=system.test_neighbor_labels,
            test_query_labels=system.test_query_labels,
            test_top_scores=system.test_top_scores,
        )

        decision = evaluate_promotion(system, challenger, n_boot=300)

        self.assertFalse(decision.promoted)
        self.assertEqual(decision.top1.delta_mean, 0.0)
        self.assertEqual(decision.top1.ci_lower, 0.0)
        failed = [c.name for c in decision.criteria if not c.passed]
        self.assertEqual(failed, ["top1_gain"])

    def test_genuinely_better_challenger_is_promoted(self) -> None:
        champion = make_system("champion", accuracy=0.84, seed=20)
        challenger = make_system(
            "challenger", accuracy=0.93, seed=30,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=300)

        self.assertTrue(decision.promoted, decision.summary())
        self.assertGreater(decision.top1.delta_mean, 0.0)
        self.assertTrue(all(c.passed for c in decision.criteria))

    def test_worse_challenger_is_rejected(self) -> None:
        champion = make_system("champion", accuracy=0.92, seed=40)
        challenger = make_system(
            "challenger", accuracy=0.80, seed=50,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=300)
        self.assertFalse(decision.promoted)

    def test_marginal_gain_below_effect_size_is_rejected(self) -> None:
        """Guards the repeated-testing problem: > 0 is not enough."""
        champion = make_system("champion", accuracy=0.880, seed=60)
        challenger = make_system(
            "challenger", accuracy=0.884, seed=70,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=300)

        self.assertFalse(decision.promoted)
        top1 = next(c for c in decision.criteria if c.name == "top1_gain")
        self.assertFalse(top1.passed)

    def test_top1_gain_with_automation_collapse_is_rejected(self) -> None:
        """The failure Top-1 alone cannot see.

        The challenger is more accurate but its similarity scores no longer
        separate right from wrong, so reaching the precision target forces a
        threshold that automates almost nothing. Promoting it would gut the
        auto-approval rate while every accuracy metric looked better.
        """
        champion = make_system(
            "champion", accuracy=0.84, seed=80,
            correct_loc=0.95, correct_scale=0.01,
            wrong_loc=0.55, wrong_scale=0.03,
        )
        challenger = make_system(
            "challenger", accuracy=0.93, seed=90,
            correct_loc=0.80, correct_scale=0.08,
            wrong_loc=0.78, wrong_scale=0.08,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=300)

        self.assertGreater(decision.top1.ci_lower, 0.005, "challenger should win on Top-1")
        automation = next(
            c for c in decision.criteria if c.name == "automation_rate_no_regression"
        )
        self.assertFalse(automation.passed, decision.summary())
        self.assertFalse(decision.promoted)
        self.assertLess(
            decision.automation.challenger_mean,
            decision.automation.champion_mean,
        )

    def test_negligible_top5_regression_passes_but_would_fail_a_zero_margin(self) -> None:
        """Top-5 is a non-inferiority test, not a superiority test.

        The challenger gains 10pp of Top-1 while giving up 0.1% of Top-5 —
        6 queries in 6000. That is comfortably inside what we are willing to
        lose, but its confidence interval still dips below zero, so a
        `ci_lower >= 0` rule would reject it. This test pins both halves:
        the margin admits it, a zero margin does not.
        """
        champion = make_system("champion", accuracy=0.84, seed=280)

        neighbors = champion.test_neighbor_labels.copy()
        labels = champion.test_query_labels
        hit1 = neighbors[:, 0] == labels
        hit5 = (neighbors == labels[:, None]).any(axis=1)
        top5_only = np.flatnonzero(hit5 & ~hit1)

        # Promote 600 rank-2 hits to rank 1: Top-1 improves, Top-5 unchanged.
        for i in top5_only[:600]:
            neighbors[i, 0] = labels[i]
        # Drop 6 queries out of the Top-5 entirely: a 0.1% regression.
        for i in top5_only[-6:]:
            neighbors[i, :] = (labels[i] + 1) % N_CLASSES

        challenger = SystemEvaluation(
            name="challenger",
            val_neighbor_labels=champion.val_neighbor_labels,
            val_query_labels=champion.val_query_labels,
            val_top_scores=champion.val_top_scores,
            test_neighbor_labels=neighbors,
            test_query_labels=labels,
            test_top_scores=champion.test_top_scores,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=500)
        top5 = next(c for c in decision.criteria if c.name == "top5_no_regression")

        self.assertLess(top5.observed, 0.0, "the CI lower bound should dip below zero")
        self.assertTrue(top5.passed, "a 0.1% Top-5 loss is inside the margin")
        self.assertFalse(top5.underpowered)
        self.assertTrue(decision.promoted, decision.summary())

        # The same challenger under a superiority rule: rejected on noise.
        strict = evaluate_promotion(
            champion, challenger, max_top5_regression=0.0, n_boot=500
        )
        strict_top5 = next(c for c in strict.criteria if c.name == "top5_no_regression")
        self.assertFalse(strict_top5.passed)
        self.assertFalse(strict.promoted)

    def test_undersized_evaluation_set_is_flagged_underpowered(self) -> None:
        """A margin finer than the noise must diagnose itself, not look like a bad challenger.

        Two equally-good systems on only 300 queries: the Top-5 interval is
        far wider than the 0.5pp margin and straddles the decision boundary,
        so the criterion fails for want of evidence. The operator needs to
        read 'your test set is too small', not 'your model regressed'.
        """
        champion = make_system("champion", accuracy=0.84, seed=260, n_val=200, n_test=300)
        challenger = make_system(
            "challenger", accuracy=0.84, seed=270, n_val=200, n_test=300,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=300)
        top5 = next(c for c in decision.criteria if c.name == "top5_no_regression")

        self.assertFalse(top5.passed)
        self.assertTrue(top5.underpowered)
        self.assertIn("UNDERPOWERED", top5.detail)
        self.assertIn("300 queries", top5.detail)

    def test_conclusive_regression_is_not_flagged_underpowered(self) -> None:
        """A proven collapse is evidence of harm, not a shortage of evidence.

        Its interval is wide in absolute terms but lies entirely below the
        margin, so calling it underpowered would send the operator hunting
        for more data instead of rejecting a genuinely broken challenger.
        """
        champion = make_system(
            "champion", accuracy=0.84, seed=290,
            correct_loc=0.95, correct_scale=0.01, wrong_loc=0.55, wrong_scale=0.03,
        )
        challenger = make_system(
            "challenger", accuracy=0.93, seed=300,
            correct_loc=0.80, correct_scale=0.08, wrong_loc=0.78, wrong_scale=0.08,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=300)
        automation = next(
            c for c in decision.criteria if c.name == "automation_rate_no_regression"
        )

        self.assertFalse(automation.passed)
        self.assertFalse(automation.underpowered)
        self.assertNotIn("UNDERPOWERED", automation.detail)

    def test_real_top5_regression_is_rejected(self) -> None:
        """The margin must still catch a regression that genuinely matters."""
        champion = make_system("champion", accuracy=0.84, seed=240, top5_bonus=0.95)
        challenger = make_system(
            "challenger", accuracy=0.93, seed=250, top5_bonus=0.10,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=300)
        top5 = next(c for c in decision.criteria if c.name == "top5_no_regression")

        self.assertFalse(top5.passed, decision.summary())
        self.assertFalse(decision.promoted)

    def test_leakage_blocks_evaluation_entirely(self) -> None:
        champion = make_system("champion", accuracy=0.84, seed=100)
        challenger = make_system(
            "challenger", accuracy=0.99, seed=110,
            test_labels=champion.test_query_labels,
        )

        with self.assertRaises(LeakageError):
            evaluate_promotion(
                champion, challenger,
                gallery_source_images=["shelf_01.jpg", "shelf_02.jpg"],
                test_source_images=["shelf_02.jpg", "shelf_03.jpg"],
            )

    def test_clean_source_images_permit_promotion(self) -> None:
        champion = make_system("champion", accuracy=0.84, seed=120)
        challenger = make_system(
            "challenger", accuracy=0.93, seed=130,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(
            champion, challenger,
            gallery_source_images=["gallery_01.jpg"],
            test_source_images=["test_01.jpg"],
            n_boot=300,
        )
        self.assertTrue(decision.promoted, decision.summary())

    def test_mismatched_test_sets_raise(self) -> None:
        champion = make_system("champion", accuracy=0.9, seed=140, n_test=6000)
        challenger = make_system("challenger", accuracy=0.9, seed=150, n_test=5000)

        with self.assertRaises(ValueError):
            evaluate_promotion(champion, challenger)

    def test_different_queries_same_length_raise(self) -> None:
        """Equal counts are not enough — the pairing must be real."""
        champion = make_system("champion", accuracy=0.9, seed=160)
        challenger = make_system("challenger", accuracy=0.9, seed=170)

        with self.assertRaises(ValueError):
            evaluate_promotion(champion, challenger)

    def test_decision_reports_calibration_artifacts(self) -> None:
        """A promotion must carry the coefficients it was validated with."""
        champion = make_system("champion", accuracy=0.84, seed=180)
        challenger = make_system(
            "challenger", accuracy=0.93, seed=190,
            test_labels=champion.test_query_labels,
        )

        decision = evaluate_promotion(champion, challenger, n_boot=300)

        self.assertIsNotNone(decision.challenger_platt)
        self.assertIsNotNone(decision.challenger_threshold)
        self.assertEqual(len(decision.challenger_platt), 2)
        self.assertEqual(decision.n_test_queries, 6000)

    def test_summary_is_renderable(self) -> None:
        champion = make_system("champion", accuracy=0.84, seed=200)
        challenger = make_system(
            "challenger", accuracy=0.93, seed=210,
            test_labels=champion.test_query_labels,
        )

        summary = evaluate_promotion(champion, challenger, n_boot=200).summary()
        self.assertIn("top1_gain", summary)
        self.assertIn("automation_rate_no_regression", summary)

    def test_is_reproducible(self) -> None:
        champion = make_system("champion", accuracy=0.84, seed=220)
        challenger = make_system(
            "challenger", accuracy=0.90, seed=230,
            test_labels=champion.test_query_labels,
        )

        first = evaluate_promotion(champion, challenger, n_boot=300, seed=11)
        second = evaluate_promotion(champion, challenger, n_boot=300, seed=11)
        self.assertEqual(first.top1.ci_lower, second.top1.ci_lower)
        self.assertEqual(first.promoted, second.promoted)


class TestSystemEvaluationValidation(unittest.TestCase):

    def test_row_count_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            SystemEvaluation(
                name="bad",
                val_neighbor_labels=np.zeros((10, 5), dtype=int),
                val_query_labels=np.zeros(9, dtype=int),
                val_top_scores=np.zeros(10),
                test_neighbor_labels=np.zeros((10, 5), dtype=int),
                test_query_labels=np.zeros(10, dtype=int),
                test_top_scores=np.zeros(10),
            )

    def test_one_dimensional_neighbors_raise(self) -> None:
        with self.assertRaises(ValueError):
            SystemEvaluation(
                name="bad",
                val_neighbor_labels=np.zeros(10, dtype=int),
                val_query_labels=np.zeros(10, dtype=int),
                val_top_scores=np.zeros(10),
                test_neighbor_labels=np.zeros((10, 5), dtype=int),
                test_query_labels=np.zeros(10, dtype=int),
                test_top_scores=np.zeros(10),
            )


if __name__ == "__main__":
    unittest.main()
