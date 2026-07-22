"""Master orchestration CLI for the Pipeline 3 continual learning loop.

Ties the phases together:

    session   consume a batch of reviews -> curate -> promote -> mark consumed
    finetune  mine confusions -> train a challenger projection head
    report    gallery histogram, review backlog, confusion table
    rollback  restore the gallery to a previous version

Every write path is dry-run by default; ``--apply`` is required to touch the
database. Promotion of a trained head is deliberately *not* a subcommand
here — it runs through ``gate.evaluate_promotion``, which must recalibrate
and re-measure first.

Usage:
    python -m ml.active_learning.loop report
    python -m ml.active_learning.loop session --apply
    python -m ml.active_learning.loop finetune --force
    python -m ml.active_learning.loop rollback --version 42 --apply
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from ml.active_learning.store import ReviewStore
from ml.active_learning.curation import DEFAULT_CLASS_CAP, DEFAULT_NEAR_DUP_THRESHOLD
from ml.active_learning.memory import GalleryMemoryUpdater
from ml.active_learning.hard_negatives import mine_from_store, ConfusionReport
from ml.retrieval.sqlite_registry import SQLiteGalleryStore


_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GALLERY_DB = _REPO_ROOT / "data/processed/crops/gt_clean/retail_sku_registry_dinov3.db"
DEFAULT_REVIEW_DB = _REPO_ROOT / "data/processed/active_learning/reviews.db"
DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "data/processed/active_learning/checkpoints"

# D2: reviews are consumed in bounded sessions, giving clean transaction
# boundaries and letting k-center diversity act across a whole batch.
DEFAULT_SESSION_SIZE = 50

# D3: the projection head retrains on review volume, not on correction-rate
# spikes, which are noisy.
DEFAULT_FINETUNE_THRESHOLD = 500


def should_finetune(
    n_unconsumed: int,
    threshold: int = DEFAULT_FINETUNE_THRESHOLD,
    force: bool = False,
) -> bool:
    """Applies the D3 volume trigger.

    Args:
        n_unconsumed: Reviews not yet consumed by a batch.
        threshold: Minimum new reviews before training is worthwhile.
        force: Override the threshold.

    Returns:
        bool: Whether a fine-tune run should proceed.
    """
    return force or n_unconsumed >= threshold


def run_session(
    review_db: str,
    gallery_db: str,
    session_size: int = DEFAULT_SESSION_SIZE,
    cap: int = DEFAULT_CLASS_CAP,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    apply: bool = False,
) -> Dict[str, Any]:
    """Runs one review session: promote reviewed crops, then re-curate.

    Promotion runs before curation on purpose — new crops should compete for
    gallery slots on the same footing as incumbents, rather than being
    appended past the cap.

    A dry run executes both stages for real and then rolls the gallery back
    to its baseline version. Simply skipping the writes would misreport the
    outcome: curation would see a gallery without the pending promotions and
    predict different counts than the applied run produces. Preview fidelity
    matters more than avoiding a transient write, and the rollback path is
    the same one an operator would use to undo a real session.

    Returns:
        Dict[str, Any]: Session summary suitable for logging.

    Raises:
        RuntimeError: If a dry run's rollback fails to restore the baseline.
    """
    batch_id = f"batch_{int(time.time())}"

    reviews = ReviewStore()
    reviews.initialize({"db_path": review_db})
    gallery = SQLiteGalleryStore()
    gallery.initialize({"db_path": gallery_db})
    updater = GalleryMemoryUpdater(gallery_store=gallery)
    updater.initialize({})

    try:
        records = reviews.fetch_unconsumed(limit=session_size)
        baseline_version = gallery.get_current_version()
        baseline_vectors, _ = gallery.fetch_all_references()

        # Both stages always execute, so the reported numbers are the real
        # ones; a dry run undoes them below.
        promotion = updater.promote_reviews(records, apply=True)
        curation = updater.curate_gallery(
            cap=cap, near_dup_threshold=near_dup_threshold, apply=True
        )

        consumed = 0
        if apply:
            if records:
                consumed = reviews.mark_consumed([r.review_id for r in records], batch_id)
        else:
            gallery.rollback_version(baseline_version)
            restored, _ = gallery.fetch_all_references()
            if restored.shape[0] != baseline_vectors.shape[0]:
                raise RuntimeError(
                    f"Dry-run rollback failed to restore the gallery: "
                    f"{baseline_vectors.shape[0]} vectors before, "
                    f"{restored.shape[0]} after rollback to version "
                    f"{baseline_version}. The gallery may need manual rollback."
                )

        return {
            "batch_id": batch_id,
            "applied": apply,
            "baseline_version": baseline_version,
            "n_reviews_in_batch": len(records),
            "n_reviews_consumed": consumed,
            "n_promoted": promotion.n_promoted,
            "n_skipped": promotion.n_skipped,
            "skipped_reasons": promotion.skipped_reasons,
            "gallery_before": curation.total_before,
            "gallery_after": curation.total_kept,
            "n_pruned": curation.total_pruned,
            "remaining_backlog": reviews.count_unconsumed(),
        }
    finally:
        reviews.shutdown()
        gallery.shutdown()


def run_finetune(
    review_db: str,
    checkpoint_dir: str,
    threshold: int = DEFAULT_FINETUNE_THRESHOLD,
    force: bool = False,
    epochs: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Mines confusions and trains a challenger projection head.

    The checkpoint is written but never activated: promotion requires
    gate.evaluate_promotion, which recalibrates and re-measures first.
    """
    reviews = ReviewStore()
    reviews.initialize({"db_path": review_db})

    try:
        n_unconsumed = reviews.count_unconsumed()
        if not should_finetune(n_unconsumed, threshold=threshold, force=force):
            return {
                "trained": False,
                "reason": (
                    f"Only {n_unconsumed} new reviews; threshold is {threshold}. "
                    f"Pass --force to train anyway."
                ),
                "n_unconsumed": n_unconsumed,
            }

        confusion = mine_from_store(reviews)

        # Gather every verified embedding, class by class.
        histogram = _review_class_histogram(reviews)
        vectors: List[np.ndarray] = []
        labels: List[int] = []
        for class_id in sorted(histogram):
            class_vectors, _ = reviews.fetch_embeddings_by_class(class_id)
            if class_vectors.shape[0]:
                vectors.append(class_vectors)
                labels.extend([class_id] * class_vectors.shape[0])

        if not vectors:
            return {"trained": False, "reason": "No verified review embeddings available."}

        # Imported here so the rest of the CLI runs without torch loaded.
        from ml.active_learning.finetune import train_projection_head, save_checkpoint

        head, report = train_projection_head(
            np.vstack(vectors), np.asarray(labels),
            confusion=confusion, epochs=epochs, seed=seed,
        )
        path = save_checkpoint(
            head,
            str(Path(checkpoint_dir) / f"supcon_head_{int(time.time())}.pt"),
            metadata={
                "n_samples": report.n_samples,
                "n_classes": report.n_classes,
                "final_loss": report.final_loss,
                "n_confusion_pairs": len(confusion.pairs),
                "status": "CHALLENGER_UNPROMOTED",
            },
        )
        return {
            "trained": True,
            "checkpoint_path": path,
            "n_samples": report.n_samples,
            "n_classes": report.n_classes,
            "initial_loss": report.initial_loss,
            "final_loss": report.final_loss,
            "n_confusion_pairs": len(confusion.pairs),
        }
    finally:
        reviews.shutdown()


def run_report(review_db: str, gallery_db: str) -> str:
    """Renders gallery, backlog, and confusion status as text."""
    lines: List[str] = ["=" * 68, "Pipeline 3 — Active Continual Learning Status", "=" * 68]

    if Path(gallery_db).exists():
        gallery = SQLiteGalleryStore()
        gallery.initialize({"db_path": gallery_db})
        try:
            histogram = gallery.class_size_histogram()
            if histogram:
                counts = np.array(sorted(histogram.values()))
                lines += [
                    "",
                    f"Gallery: {sum(histogram.values())} active vectors across "
                    f"{len(histogram)} classes",
                    f"  per class  min {counts.min()} / median {int(np.median(counts))} "
                    f"/ mean {counts.mean():.1f} / max {counts.max()}",
                    f"  over cap {DEFAULT_CLASS_CAP}: "
                    f"{int((counts > DEFAULT_CLASS_CAP).sum())} classes",
                    f"  version {gallery.get_current_version()}",
                ]
            else:
                lines += ["", "Gallery: empty"]
        finally:
            gallery.shutdown()
    else:
        lines += ["", f"Gallery: not found at {gallery_db}"]

    if Path(review_db).exists():
        reviews = ReviewStore()
        reviews.initialize({"db_path": review_db})
        try:
            backlog = reviews.count_unconsumed()
            lines += [
                "",
                f"Reviews: {backlog} unconsumed",
                f"  fine-tune trigger at {DEFAULT_FINETUNE_THRESHOLD}: "
                f"{'READY' if backlog >= DEFAULT_FINETUNE_THRESHOLD else 'waiting'}",
            ]
            confusion = mine_from_store(reviews)
            if confusion.pairs:
                lines += ["", confusion.summary(top_n=15)]
        finally:
            reviews.shutdown()
    else:
        lines += ["", f"Reviews: not found at {review_db}"]

    return "\n".join(lines)


def run_rollback(gallery_db: str, version: int, apply: bool = False) -> Dict[str, Any]:
    """Restores the gallery to a previous version."""
    gallery = SQLiteGalleryStore()
    gallery.initialize({"db_path": gallery_db})
    try:
        before, _ = gallery.fetch_all_references()
        current = gallery.get_current_version()
        if apply:
            gallery.rollback_version(version)
        after, _ = gallery.fetch_all_references()

        return {
            "applied": apply,
            "current_version": current,
            "target_version": version,
            "vectors_before": int(before.shape[0]),
            "vectors_after": int(after.shape[0]),
        }
    finally:
        gallery.shutdown()


def _review_class_histogram(store: ReviewStore) -> Dict[int, int]:
    """Counts verified reviews per class."""
    histogram: Dict[int, int] = {}
    for review, _ in store.fetch_reviews_with_candidates(only_verified=True):
        if review.true_class_id is not None and review.embedding is not None:
            histogram[review.true_class_id] = histogram.get(review.true_class_id, 0) + 1
    return histogram


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.active_learning.loop",
        description="Pipeline 3 active continual learning loop.",
    )
    parser.add_argument("--review-db", default=str(DEFAULT_REVIEW_DB))
    parser.add_argument("--gallery-db", default=str(DEFAULT_GALLERY_DB))

    sub = parser.add_subparsers(dest="command", required=True)

    session = sub.add_parser("session", help="Consume a review batch, promote and re-curate.")
    session.add_argument("--session-size", type=int, default=DEFAULT_SESSION_SIZE)
    session.add_argument("--cap", type=int, default=DEFAULT_CLASS_CAP)
    session.add_argument("--near-dup", type=float, default=DEFAULT_NEAR_DUP_THRESHOLD)
    session.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")

    finetune = sub.add_parser("finetune", help="Train a challenger projection head (opt-in).")
    finetune.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    finetune.add_argument("--threshold", type=int, default=DEFAULT_FINETUNE_THRESHOLD)
    finetune.add_argument("--force", action="store_true", help="Ignore the volume trigger.")
    finetune.add_argument("--epochs", type=int, default=20)
    finetune.add_argument("--seed", type=int, default=42)

    sub.add_parser("report", help="Show gallery, backlog, and confusion status.")

    rollback = sub.add_parser("rollback", help="Restore the gallery to a previous version.")
    rollback.add_argument("--version", type=int, required=True)
    rollback.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "report":
        print(run_report(args.review_db, args.gallery_db))
        return 0

    if args.command == "session":
        result = run_session(
            args.review_db, args.gallery_db,
            session_size=args.session_size, cap=args.cap,
            near_dup_threshold=args.near_dup, apply=args.apply,
        )
    elif args.command == "finetune":
        result = run_finetune(
            args.review_db, args.checkpoint_dir,
            threshold=args.threshold, force=args.force,
            epochs=args.epochs, seed=args.seed,
        )
    elif args.command == "rollback":
        result = run_rollback(args.gallery_db, args.version, apply=args.apply)
    else:
        return 1

    for key, value in result.items():
        print(f"  {key}: {value}")
    if not result.get("applied", True):
        baseline = result.get("baseline_version")
        rolled_back = f" (changes rolled back to version {baseline})" if baseline else ""
        print(f"\n  DRY RUN — no lasting changes{rolled_back}. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
