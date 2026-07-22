"""Unit tests for Pipeline 3 gallery curation.

Model-free: pure numpy, no DB and no weights.
"""

import unittest

import numpy as np

from ml.active_learning.curation import (
    kcenter_greedy_select,
    reject_near_duplicates,
    curate_class,
    medoid_index,
)


def unit(vectors) -> np.ndarray:
    """L2-normalizes rows of a float32 array."""
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def random_unit(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return unit(rng.normal(size=(n, dim)))


class TestInputValidation(unittest.TestCase):
    """Cosine distance is only a distance on the unit sphere."""

    def test_unnormalized_vectors_rejected(self) -> None:
        with self.assertRaises(ValueError):
            kcenter_greedy_select(np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32), k=1)

    def test_one_dimensional_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reject_near_duplicates(np.array([1.0, 0.0], dtype=np.float32))

    def test_empty_input_accepted(self) -> None:
        empty = np.empty((0, 4), dtype=np.float32)
        self.assertEqual(kcenter_greedy_select(empty, k=3), [])
        self.assertEqual(reject_near_duplicates(empty).tolist(), [])
        self.assertEqual(curate_class(empty).n_input, 0)


class TestMedoidSeed(unittest.TestCase):
    """The medoid replaces the spec's np.random.randint seed."""

    def test_medoid_is_the_central_vector(self) -> None:
        # Two vectors flank a central one; the middle is the medoid.
        vectors = unit([[1.0, -0.3], [1.0, 0.0], [1.0, 0.3]])
        self.assertEqual(medoid_index(vectors), 1)

    def test_medoid_of_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            medoid_index(np.empty((0, 4), dtype=np.float32))

    def test_antipodal_cancellation_is_deterministic(self) -> None:
        """A zero centroid must not produce a random or NaN index."""
        vectors = unit([[1.0, 0.0], [-1.0, 0.0]])
        self.assertEqual(medoid_index(vectors), 0)


class TestKCenterGreedy(unittest.TestCase):
    """Selection correctness, including the multi-seed defect from the spec."""

    def test_selects_extremes_of_a_spread(self) -> None:
        # Three tight clusters; k=3 must take one from each rather than
        # three from the densest.
        vectors = unit([
            [1.0, 0.0], [0.99, 0.01], [0.98, 0.02],   # cluster A (dense)
            [0.0, 1.0],                               # cluster B
            [-1.0, 0.0],                              # cluster C
        ])
        picked = set(kcenter_greedy_select(vectors, k=3))

        self.assertEqual(len(picked), 3)
        self.assertTrue(picked & {0, 1, 2}, "expected a representative of cluster A")
        self.assertIn(3, picked)
        self.assertIn(4, picked)

    def test_all_seeds_constrain_selection(self) -> None:
        """Regression: the spec folded in only selected[-1], ignoring earlier seeds.

        With seeds [0, 1] covering both ends of the spread, the next pick must
        be the midpoint (2). The buggy version measured distance only from
        seed 1, and so picked the far end of the axis instead.
        """
        vectors = unit([
            [1.0, 0.0],    # 0: seed, one extreme
            [-1.0, 0.0],   # 1: seed, opposite extreme
            [0.0, 1.0],    # 2: equidistant from both seeds — the correct pick
            [0.98, 0.199], # 3: near seed 0; only correct if seed 0 is ignored
        ])
        picked = kcenter_greedy_select(vectors, k=3, initial_indices=[0, 1])

        self.assertEqual(picked[:2], [0, 1], "seeds must be preserved in order")
        self.assertEqual(picked[2], 2, "seed 0 was ignored — the multi-seed bug")

    def test_no_duplicate_indices_when_k_exceeds_distinct(self) -> None:
        """Repeated vectors must not be selected twice."""
        vectors = unit([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
        picked = kcenter_greedy_select(vectors, k=8)
        self.assertEqual(len(picked), len(set(picked)))

    def test_selection_is_deterministic(self) -> None:
        vectors = random_unit(60, 16, seed=1)
        self.assertEqual(
            kcenter_greedy_select(vectors, k=10),
            kcenter_greedy_select(vectors, k=10),
        )

    def test_k_at_or_above_n_returns_everything(self) -> None:
        vectors = random_unit(5, 4, seed=2)
        self.assertEqual(kcenter_greedy_select(vectors, k=5), list(range(5)))
        self.assertEqual(kcenter_greedy_select(vectors, k=99), list(range(5)))

    def test_k_zero_or_negative_returns_empty(self) -> None:
        vectors = random_unit(5, 4, seed=3)
        self.assertEqual(kcenter_greedy_select(vectors, k=0), [])
        self.assertEqual(kcenter_greedy_select(vectors, k=-1), [])

    def test_seeds_beyond_k_are_preserved(self) -> None:
        """Seeds cannot be un-selected, so the result may exceed k."""
        vectors = random_unit(10, 4, seed=4)
        picked = kcenter_greedy_select(vectors, k=2, initial_indices=[0, 1, 2, 3])
        self.assertEqual(picked, [0, 1, 2, 3])

    def test_duplicate_seeds_deduplicated(self) -> None:
        vectors = random_unit(10, 4, seed=5)
        picked = kcenter_greedy_select(vectors, k=4, initial_indices=[2, 2, 2])
        self.assertEqual(len(picked), len(set(picked)))
        self.assertEqual(picked[0], 2)

    def test_out_of_range_seed_raises(self) -> None:
        with self.assertRaises(ValueError):
            kcenter_greedy_select(random_unit(5, 4, seed=6), k=3, initial_indices=[99])

    def test_unknown_seed_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            kcenter_greedy_select(random_unit(5, 4, seed=7), k=3, seed_strategy="lottery")

    def test_first_strategy_seeds_at_zero(self) -> None:
        vectors = random_unit(20, 8, seed=8)
        self.assertEqual(kcenter_greedy_select(vectors, k=5, seed_strategy="first")[0], 0)

    def test_beats_head_truncation_on_coverage(self) -> None:
        """The point of k-center: better coverage than taking the first k rows."""
        rng = np.random.default_rng(9)
        dense = unit(np.tile([1.0, 0.0, 0.0], (40, 1)) + rng.normal(scale=0.01, size=(40, 3)))
        spread = unit(rng.normal(size=(20, 3)))
        vectors = np.vstack([dense, spread]).astype(np.float32)

        picked = kcenter_greedy_select(vectors, k=10)

        def worst_case_coverage(subset: np.ndarray) -> float:
            # Largest distance from any vector to its nearest selected vector.
            return float(np.max(1.0 - np.max(vectors @ subset.T, axis=1)))

        self.assertLess(worst_case_coverage(vectors[picked]), worst_case_coverage(vectors[:10]))


class TestNearDuplicateRejection(unittest.TestCase):

    def test_exact_duplicates_collapse_to_one(self) -> None:
        vectors = unit([[1.0, 0.0]] * 4)
        self.assertEqual(int(reject_near_duplicates(vectors).sum()), 1)

    def test_keeps_first_of_each_group(self) -> None:
        vectors = unit([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(reject_near_duplicates(vectors).tolist(), [True, False, True])

    def test_distinct_vectors_all_kept(self) -> None:
        vectors = unit([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.assertTrue(reject_near_duplicates(vectors).all())

    def test_threshold_boundary(self) -> None:
        # Construct a pair at cosine ~0.99: dropped at 0.98, kept at 0.995.
        theta = np.arccos(0.99)
        vectors = unit([[1.0, 0.0], [np.cos(theta), np.sin(theta)]])

        self.assertEqual(int(reject_near_duplicates(vectors, threshold=0.98).sum()), 1)
        self.assertEqual(int(reject_near_duplicates(vectors, threshold=0.995).sum()), 2)


class TestCurateClass(unittest.TestCase):
    """Stage composition: duplicates first, then the diversity cap."""

    def test_under_cap_and_distinct_keeps_all(self) -> None:
        decision = curate_class(random_unit(20, 8, seed=10), cap=100)
        self.assertEqual(len(decision.keep_indices), 20)
        self.assertEqual(decision.prune_indices, [])
        self.assertEqual(decision.n_near_duplicate, 0)
        self.assertEqual(decision.n_over_cap, 0)

    def test_cap_enforced(self) -> None:
        decision = curate_class(random_unit(50, 8, seed=11), cap=10)
        self.assertEqual(len(decision.keep_indices), 10)
        self.assertEqual(len(decision.prune_indices), 40)
        self.assertEqual(decision.n_over_cap, 40)

    def test_keep_and_prune_partition_the_input(self) -> None:
        decision = curate_class(random_unit(30, 8, seed=12), cap=7)
        self.assertEqual(
            sorted(decision.keep_indices + decision.prune_indices), list(range(30))
        )
        self.assertFalse(set(decision.keep_indices) & set(decision.prune_indices))

    def test_drops_attributed_to_the_right_stage(self) -> None:
        # 10 copies of one basis vector + 20 mutually orthogonal ones. Using
        # an orthonormal basis rather than random vectors makes the expected
        # counts exact: random low-dimensional vectors land inside the 0.98
        # threshold often enough to make the assertion flaky.
        basis = np.eye(24, dtype=np.float32)
        duplicates = np.tile(basis[0], (10, 1))
        distinct = basis[4:24]
        vectors = np.vstack([duplicates, distinct]).astype(np.float32)

        decision = curate_class(vectors, cap=5)

        self.assertEqual(decision.n_input, 30)
        self.assertEqual(decision.n_near_duplicate, 9)  # 10 copies -> 1 survivor
        self.assertEqual(decision.n_over_cap, 16)       # 21 survivors -> 5
        self.assertEqual(len(decision.keep_indices), 5)

    def test_forced_keeps_survive_both_stages(self) -> None:
        """Rows in initial_indices are exempt from deduplication and the cap."""
        vectors = unit(np.tile([1.0, 0.0, 0.0], (10, 1)))  # all duplicates
        decision = curate_class(vectors, cap=2, initial_indices=[3, 7])

        self.assertIn(3, decision.keep_indices)
        self.assertIn(7, decision.keep_indices)

    def test_curation_is_deterministic(self) -> None:
        vectors = random_unit(80, 16, seed=14)
        first = curate_class(vectors, cap=20)
        second = curate_class(vectors, cap=20)
        self.assertEqual(first.keep_indices, second.keep_indices)

    def test_keep_indices_sorted(self) -> None:
        decision = curate_class(random_unit(40, 8, seed=15), cap=9)
        self.assertEqual(decision.keep_indices, sorted(decision.keep_indices))


if __name__ == "__main__":
    unittest.main()
