"""Unit tests for the Pipeline 3 SupCon projection head.

CPU-only, synthetic embeddings, no backbone and no DB.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.active_learning.finetune import (
    SupConProjectionHead,
    supcon_loss,
    build_confusion_batches,
    train_projection_head,
    save_checkpoint,
    load_checkpoint,
)
from ml.active_learning.hard_negatives import ConfusionReport, ConfusionPair


def separable_embeddings(n_classes: int = 6, per_class: int = 10, dim: int = 32, seed: int = 0):
    """Builds well-separated class clusters on the unit sphere."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    vectors, labels = [], []
    for class_id in range(n_classes):
        noisy = centers[class_id] + rng.normal(scale=0.05, size=(per_class, dim))
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        vectors.append(noisy)
        labels.extend([class_id] * per_class)

    return np.vstack(vectors).astype(np.float32), np.asarray(labels)


def overlapping_embeddings(
    n_classes: int = 5, per_class: int = 24, dim: int = 32, seed: int = 0
):
    """Builds heavily overlapping class clusters, split into train and eval halves.

    Every class sits near a shared base direction and is distinguished only
    by a small offset — the fine-grained variant case (same brand and
    packaging, different flavour) where raw cosine similarity struggles and
    a projection head has something to learn.

    Returns:
        (train_x, train_y, eval_x, eval_y)
    """
    rng = np.random.default_rng(seed)

    base = rng.normal(size=dim)
    base /= np.linalg.norm(base)
    offsets = rng.normal(size=(n_classes, dim))
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)

    centers = base + 0.35 * offsets
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    train_x, train_y, eval_x, eval_y = [], [], [], []
    half = per_class // 2
    for class_id in range(n_classes):
        samples = centers[class_id] + rng.normal(scale=0.10, size=(per_class, dim))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        train_x.append(samples[:half])
        eval_x.append(samples[half:])
        train_y.extend([class_id] * half)
        eval_y.extend([class_id] * (per_class - half))

    return (
        np.vstack(train_x).astype(np.float32), np.asarray(train_y),
        np.vstack(eval_x).astype(np.float32), np.asarray(eval_y),
    )


class TestProjectionHead(unittest.TestCase):

    def test_output_shape_and_normalization(self) -> None:
        head = SupConProjectionHead(in_dim=768, out_dim=128)
        out = head(torch.randn(16, 768))

        self.assertEqual(out.shape, (16, 128))
        np.testing.assert_allclose(
            torch.linalg.norm(out, dim=1).detach().numpy(),
            np.ones(16), rtol=1e-5, atol=1e-5,
        )

    def test_layernorm_gives_identical_train_and_eval_output(self) -> None:
        """The reason for LayerNorm over BatchNorm: no train/serve drift.

        BatchNorm would use batch statistics in train mode and running
        statistics in eval, so the same crop could project differently
        depending on what it was batched with.
        """
        head = SupConProjectionHead(in_dim=32, hidden_dim=16, out_dim=8)
        sample = torch.randn(4, 32)

        head.train()
        train_out = head(sample).detach().numpy()
        head.eval()
        eval_out = head(sample).detach().numpy()

        np.testing.assert_allclose(train_out, eval_out, rtol=1e-6, atol=1e-6)

    def test_projection_independent_of_batch_composition(self) -> None:
        """A vector must project the same alone as in company."""
        head = SupConProjectionHead(in_dim=32, hidden_dim=16, out_dim=8).eval()
        sample = torch.randn(1, 32)
        padded = torch.cat([sample, torch.randn(7, 32)], dim=0)

        np.testing.assert_allclose(
            head(sample).detach().numpy()[0],
            head(padded).detach().numpy()[0],
            rtol=1e-6, atol=1e-6,
        )

    def test_project_array_round_trip(self) -> None:
        head = SupConProjectionHead(in_dim=32, hidden_dim=16, out_dim=8)
        projected = head.project_array(np.random.randn(100, 32).astype(np.float32))

        self.assertEqual(projected.shape, (100, 8))
        self.assertEqual(projected.dtype, np.float32)
        np.testing.assert_allclose(
            np.linalg.norm(projected, axis=1), np.ones(100), rtol=1e-5, atol=1e-5
        )

    def test_project_empty_array(self) -> None:
        head = SupConProjectionHead(in_dim=32, hidden_dim=16, out_dim=8)
        self.assertEqual(
            head.project_array(np.empty((0, 32), dtype=np.float32)).shape, (0, 8)
        )


class TestSupConLoss(unittest.TestCase):

    def test_tight_clusters_score_lower_than_scrambled(self) -> None:
        """The loss must actually prefer same-class vectors to be close."""
        labels = torch.tensor([0, 0, 1, 1])
        clustered = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]), dim=1
        )
        scrambled = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]]), dim=1
        )

        self.assertLess(supcon_loss(clustered, labels), supcon_loss(scrambled, labels))

    def test_loss_is_positive_and_finite(self) -> None:
        projections = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
        loss = supcon_loss(projections, torch.tensor([0, 1] * 8))

        self.assertGreater(float(loss), 0.0)
        self.assertTrue(np.isfinite(float(loss)))

    def test_batch_without_positives_returns_zero(self) -> None:
        """All-distinct labels give no positive pairs; must not divide by zero."""
        projections = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
        loss = supcon_loss(projections, torch.tensor([0, 1, 2, 3]))

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(np.isfinite(float(loss)))

    def test_lower_temperature_sharpens_contrast(self) -> None:
        projections = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]), dim=1
        )
        labels = torch.tensor([0, 0, 1, 1])

        self.assertNotAlmostEqual(
            float(supcon_loss(projections, labels, temperature=0.07)),
            float(supcon_loss(projections, labels, temperature=0.5)),
        )

    def test_invalid_temperature_raises(self) -> None:
        projections = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
        with self.assertRaises(ValueError):
            supcon_loss(projections, torch.tensor([0, 0, 1, 1]), temperature=0.0)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            supcon_loss(torch.randn(4, 8), torch.tensor([0, 1]))

    def test_gradients_flow(self) -> None:
        head = SupConProjectionHead(in_dim=16, hidden_dim=8, out_dim=4)
        loss = supcon_loss(head(torch.randn(8, 16)), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
        loss.backward()

        self.assertIsNotNone(head.fc1.weight.grad)
        self.assertGreater(float(head.fc1.weight.grad.abs().sum()), 0.0)


class TestBatchConstruction(unittest.TestCase):

    def test_every_batch_contains_positive_pairs(self) -> None:
        _, labels = separable_embeddings(n_classes=6, per_class=10)
        batches = build_confusion_batches(labels, batch_classes=3, samples_per_class=4)

        self.assertGreater(len(batches), 0)
        for batch in batches:
            counts = np.bincount(labels[batch])
            self.assertTrue((counts[counts > 0] >= 2).all(), "every class needs a positive")

    def test_batches_group_confusable_classes(self) -> None:
        """Hard negatives must share a batch with their positives."""
        _, labels = separable_embeddings(n_classes=6, per_class=10)
        confusion = ConfusionReport(pairs=[
            ConfusionPair(
                true_class_id=0, confused_class_id=5, frequency=10,
                n_outranked=8, mean_similarity=0.9, mean_margin=-0.02,
            )
        ])

        batches = build_confusion_batches(
            labels, batch_classes=2, samples_per_class=4, confusion=confusion
        )
        self.assertTrue(
            any({0, 5} <= set(labels[b].tolist()) for b in batches),
            "class 0 and its confuser 5 should co-occur in a batch",
        )

    def test_all_classes_covered_without_confusion_data(self) -> None:
        _, labels = separable_embeddings(n_classes=6, per_class=10)
        batches = build_confusion_batches(labels, batch_classes=2, samples_per_class=4)

        covered = set()
        for batch in batches:
            covered.update(labels[batch].tolist())
        self.assertEqual(covered, set(range(6)))

    def test_singleton_classes_cannot_seed_a_batch(self) -> None:
        """One sample cannot form a positive pair."""
        labels = np.array([0, 0, 0, 1, 2, 2, 2])
        batches = build_confusion_batches(labels, batch_classes=2, samples_per_class=2)

        for batch in batches:
            counts = np.bincount(labels[batch], minlength=3)
            self.assertNotEqual(counts[1], 1, "class 1 has one sample and cannot be a positive")

    def test_short_class_sampled_with_replacement(self) -> None:
        labels = np.array([0, 0, 1, 1])
        batches = build_confusion_batches(labels, batch_classes=2, samples_per_class=4)

        self.assertGreater(len(batches), 0)
        self.assertEqual(len(batches[0]), 8)

    def test_is_deterministic(self) -> None:
        _, labels = separable_embeddings(n_classes=6, per_class=10)
        first = build_confusion_batches(labels, seed=7)
        second = build_confusion_batches(labels, seed=7)

        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_no_eligible_classes_yields_no_batches(self) -> None:
        self.assertEqual(build_confusion_batches(np.array([0, 1, 2])), [])

    def test_invalid_parameters_raise(self) -> None:
        labels = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            build_confusion_batches(labels, batch_classes=1)
        with self.assertRaises(ValueError):
            build_confusion_batches(labels, samples_per_class=1)


class TestTraining(unittest.TestCase):

    def test_loss_decreases_on_separable_data(self) -> None:
        embeddings, labels = separable_embeddings(n_classes=6, per_class=12)
        _, report = train_projection_head(
            embeddings, labels, epochs=15, batch_classes=3, samples_per_class=4
        )

        self.assertLess(report.final_loss, report.initial_loss)
        self.assertEqual(report.n_classes, 6)
        self.assertEqual(report.n_samples, 72)

    def test_training_separates_overlapping_classes_on_held_out_data(self) -> None:
        """The point of the head, measured where it matters.

        Uses classes that genuinely overlap — the confusable-variant case the
        head exists for — and scores separation on samples withheld from
        training, so this tests learned structure rather than memorization.
        Well-separated clusters would prove nothing: there is no headroom
        above a gap that already sits near 1.0.
        """
        train_x, train_y, eval_x, eval_y = overlapping_embeddings(
            n_classes=5, per_class=24, seed=3
        )

        def mean_gap(vectors: np.ndarray, labels: np.ndarray) -> float:
            sims = vectors @ vectors.T
            same = labels[:, None] == labels[None, :]
            np.fill_diagonal(same, False)
            return float(sims[same].mean() - sims[~same].mean())

        baseline = mean_gap(eval_x, eval_y)
        self.assertLess(baseline, 0.5, "fixture must start overlapping, or the test is vacuous")

        head, _ = train_projection_head(
            train_x, train_y, epochs=40, batch_classes=3, samples_per_class=4, seed=3
        )
        self.assertGreater(mean_gap(head.project_array(eval_x), eval_y), baseline)

    def test_is_reproducible(self) -> None:
        embeddings, labels = separable_embeddings()
        _, first = train_projection_head(embeddings, labels, epochs=5, seed=11)
        _, second = train_projection_head(embeddings, labels, epochs=5, seed=11)

        self.assertEqual(first.loss_history, second.loss_history)

    def test_head_matches_embedding_width(self) -> None:
        embeddings, labels = separable_embeddings(dim=64)
        head, _ = train_projection_head(embeddings, labels, epochs=2)
        self.assertEqual(head.in_dim, 64)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            train_projection_head(np.random.randn(10, 8).astype(np.float32), np.zeros(9))

    def test_insufficient_positives_raises_clearly(self) -> None:
        """One review per class cannot train a contrastive objective."""
        embeddings = np.random.randn(3, 8).astype(np.float32)
        with self.assertRaises(ValueError) as ctx:
            train_projection_head(embeddings, np.array([0, 1, 2]), epochs=1)
        self.assertIn("positive pair", str(ctx.exception))


class TestCheckpoints(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_trip_preserves_projections(self) -> None:
        embeddings, labels = separable_embeddings()
        head, _ = train_projection_head(embeddings, labels, epochs=3)

        path = save_checkpoint(head, str(Path(self._tmp.name) / "head.pt"))
        restored, info = load_checkpoint(path)

        np.testing.assert_allclose(
            head.project_array(embeddings), restored.project_array(embeddings),
            rtol=1e-6, atol=1e-6,
        )
        self.assertEqual(info["in_dim"], head.in_dim)
        self.assertEqual(info["out_dim"], head.out_dim)

    def test_metadata_is_preserved(self) -> None:
        head = SupConProjectionHead(in_dim=32, hidden_dim=16, out_dim=8)
        path = save_checkpoint(
            head, str(Path(self._tmp.name) / "head.pt"),
            metadata={"status": "CHALLENGER_UNPROMOTED", "final_loss": 0.5},
        )
        _, info = load_checkpoint(path)

        self.assertEqual(info["metadata"]["status"], "CHALLENGER_UNPROMOTED")
        self.assertIn("created_at", info)

    def test_creates_missing_directories(self) -> None:
        head = SupConProjectionHead(in_dim=32, hidden_dim=16, out_dim=8)
        nested = Path(self._tmp.name) / "a" / "b" / "head.pt"
        save_checkpoint(head, str(nested))
        self.assertTrue(nested.exists())

    def test_restored_head_is_in_eval_mode(self) -> None:
        head = SupConProjectionHead(in_dim=32, hidden_dim=16, out_dim=8)
        path = save_checkpoint(head, str(Path(self._tmp.name) / "head.pt"))
        restored, _ = load_checkpoint(path)
        self.assertFalse(restored.training)


if __name__ == "__main__":
    unittest.main()
