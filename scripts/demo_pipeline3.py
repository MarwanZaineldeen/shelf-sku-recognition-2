"""Self-contained demonstration of the Pipeline 3 continual learning loop.

Builds a synthetic gallery and review corpus, then exercises every stage:
review ingest, confusion mining, curation, promotion, rollback, SupCon
training, and the statistical promotion gate.

Requires no model weights, no production database, and no network. Writes
only inside data/processed/pipeline3_demo/, which is gitignored.

    python scripts/demo_pipeline3.py
    python scripts/demo_pipeline3.py --keep    # leave the demo databases behind
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.base import BBoxDTO, PredictionDTO
from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from ml.active_learning.store import ReviewStore
from ml.active_learning.ingest import ReviewContextCache, record_review
from ml.active_learning.hard_negatives import mine_from_store
from ml.active_learning.gate import SystemEvaluation, evaluate_promotion, LeakageError
from ml.active_learning import loop

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "data" / "processed" / "pipeline3_demo"
GALLERY_DB = DEMO_DIR / "demo_gallery.db"
REVIEW_DB = DEMO_DIR / "demo_reviews.db"
CHECKPOINT_DIR = DEMO_DIR / "checkpoints"

DIM = 32
N_CLASSES = 6
CROPS_PER_CLASS = 14


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def seed_gallery(rng: np.random.Generator) -> None:
    """Creates a gallery with deliberate near-duplicates for curation to prune."""
    store = SQLiteGalleryStore()
    store.initialize({"db_path": str(GALLERY_DB)})
    try:
        references = []
        for class_id in range(N_CLASSES):
            center = unit(rng.normal(size=DIM))
            for j in range(CROPS_PER_CLASS):
                # Half of each class is near-identical to its centre, so the
                # 0.98 near-duplicate rule has something to find.
                jitter = 0.005 if j < CROPS_PER_CLASS // 2 else 0.25
                references.append((
                    class_id, class_id, f"crop_{class_id}_{j}.jpg", f"family_{class_id}",
                    f"seed_shelf_{class_id}_{j}.jpg", [0.0, 0.0, 10.0, 10.0],
                    unit(center + jitter * rng.normal(size=DIM)).tolist(),
                ))
        store.save_references_bulk(references)
        print(f"  Seeded {len(references)} vectors across {N_CLASSES} classes.")
    finally:
        store.shutdown()


def simulate_reviews(rng: np.random.Generator) -> None:
    """Simulates audits and human verdicts through the real server ingest path."""
    store = ReviewStore()
    store.initialize({"db_path": str(REVIEW_DB)})
    cache = ReviewContextCache()

    try:
        # Class 3 is persistently mistaken for class 2 — a confusable variant.
        predictions = [
            PredictionDTO(
                crop_id=f"crop_{i}",
                bbox=BBoxDTO(x1=0, y1=0, x2=10, y2=10, confidence=0.93),
                predicted_class_id=2,
                confidence_probability=0.72,
                automated=False,
                reject_reason="LOW_VISUAL_CONFIDENCE",
                top5_candidates=[
                    {"class_id": 2, "display_name": "SKU 2", "similarity": 0.93},
                    {"class_id": 3, "display_name": "SKU 3", "similarity": 0.91},
                    {"class_id": 5, "display_name": "SKU 5", "similarity": 0.55},
                ],
                embedding=unit(rng.normal(size=DIM)).tolist(),
            )
            for i in range(12)
        ]
        cache.put_predictions("audit_shelf.jpg", predictions)

        # 8 corrections, 3 approvals, 1 open-set rejection.
        verdicts = [3] * 8 + [2] * 3 + [-1]
        counts: dict = {}
        for i, assigned in enumerate(verdicts):
            review_id = record_review(
                store=store, source_image="audit_shelf.jpg", crop_id=f"crop_{i}",
                assigned_class_id=assigned, reviewer_id="merchandiser_user",
                context=cache.get("audit_shelf.jpg", f"crop_{i}"),
            )
            decision = store.fetch_review(review_id).decision
            counts[decision] = counts.get(decision, 0) + 1

        for decision, count in sorted(counts.items()):
            print(f"  {decision:<16} {count}")
        print(f"  Embeddings captured from the audit cache: {len(cache)} crops.")
    finally:
        store.shutdown()


def show_confusions() -> None:
    store = ReviewStore()
    store.initialize({"db_path": str(REVIEW_DB)})
    try:
        print(mine_from_store(store).summary(top_n=10))
        print("\n  Read: class 3 keeps losing to class 2 — the pair worth more crops.")
    finally:
        store.shutdown()


def demo_gate() -> None:
    """Runs the promotion gate on synthetic champion/challenger evaluations."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from active_learning.test_gate import make_system

    champion = make_system("champion_dinov3_768", accuracy=0.84, seed=80,
                           correct_loc=0.95, correct_scale=0.01,
                           wrong_loc=0.55, wrong_scale=0.03)

    print("\n--- A genuinely better challenger ---")
    better = make_system("challenger_good", accuracy=0.93, seed=30,
                         test_labels=champion.test_query_labels)
    print(evaluate_promotion(champion, better, n_boot=500).summary())

    print("\n--- Better Top-1, but the automation rate collapses ---")
    print("    (the failure Top-1 accuracy alone cannot see)")
    miscalibrated = make_system("challenger_miscalibrated", accuracy=0.93, seed=90,
                                correct_loc=0.80, correct_scale=0.08,
                                wrong_loc=0.78, wrong_scale=0.08,
                                test_labels=champion.test_query_labels)
    print(evaluate_promotion(champion, miscalibrated, n_boot=500).summary())

    print("\n--- Gallery crops leaking into the test set ---")
    try:
        evaluate_promotion(
            champion, better,
            gallery_source_images=["shelf_01.jpg", "shelf_02.jpg"],
            test_source_images=["shelf_02.jpg", "shelf_09.jpg"],
        )
        print("  ERROR: leakage was not detected!")
    except LeakageError as e:
        print(f"  LeakageError raised, as intended:\n    {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline 3 end-to-end demonstration.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the demo databases instead of deleting them.")
    args = parser.parse_args()

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    banner("STEP 1 — Seed a synthetic reference gallery")
    seed_gallery(rng)

    banner("STEP 2 — Simulate a shelf audit and human reviews")
    simulate_reviews(rng)

    banner("STEP 3 — Status report")
    print(loop.run_report(str(REVIEW_DB), str(GALLERY_DB)))

    banner("STEP 4 — Hard negative confusion mining")
    show_confusions()

    banner("STEP 5 — Session, dry run (preview only)")
    for key, value in loop.run_session(str(REVIEW_DB), str(GALLERY_DB), cap=8).items():
        print(f"  {key}: {value}")
    print("\n  Nothing was written. These counts match the applied run exactly.")

    banner("STEP 6 — Session, applied")
    applied = loop.run_session(str(REVIEW_DB), str(GALLERY_DB), cap=8, apply=True)
    for key, value in applied.items():
        print(f"  {key}: {value}")

    banner("STEP 7 — Roll the gallery back")
    print("  Curation and promotion are versioned and reversible.")
    for key, value in loop.run_rollback(
        str(GALLERY_DB), version=applied["baseline_version"], apply=True
    ).items():
        print(f"  {key}: {value}")

    banner("STEP 8 — Train a challenger SupCon head (opt-in, never auto-promoted)")
    for key, value in loop.run_finetune(
        str(REVIEW_DB), str(CHECKPOINT_DIR), force=True, epochs=10
    ).items():
        print(f"  {key}: {value}")

    banner("STEP 9 — Statistical promotion gate")
    demo_gate()

    banner("DONE")
    if args.keep:
        print(f"  Demo databases kept in: {DEMO_DIR}")
    else:
        shutil.rmtree(DEMO_DIR)
        print("  Demo databases removed. Re-run with --keep to inspect them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
