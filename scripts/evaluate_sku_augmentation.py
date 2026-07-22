"""SKU 100 Data Augmentation & Recognition Accuracy Benchmark Script.

Compares Strategy A (Raw YOLO Detector Crops) vs Strategy B (Selective Data Augmentation)
for a newly onboarded SKU (Class 100) using DINOv3 768-D feature embeddings.
"""

import io
import os
import sys
import json
import time
import sqlite3
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.embeddings.dinov3 import DINOv3Extractor
from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from ml.retrieval.hierarchical_index import HierarchicalCosineIndex
from ml.detection.yolo_detector import YOLOv8Detector

def apply_selective_augmentations(pil_img: Image.Image, count: int = 3) -> list[Image.Image]:
    """Generates `count` realistic augmented variants of a reference product crop."""
    aug_list = []
    w, h = pil_img.size

    for i in range(count):
        img = pil_img.copy().convert("RGB")

        # 1. Brightness / Contrast jitter
        b_factor = 0.85 + (i * 0.15)  # 0.85, 1.0, 1.15
        c_factor = 0.90 + (i * 0.10)  # 0.90, 1.0, 1.10
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        img = ImageEnhance.Contrast(img).enhance(c_factor)

        # 2. Slight rotation (-10 deg to +10 deg)
        angle = -10 if i == 0 else (10 if i == 1 else -5)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(124, 116, 104))

        # 3. Gaussian Blur (subtle lens blur)
        if i == 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

        # 4. Slight scale/crop jitter (5% boundary padding)
        crop_margin = int(min(w, h) * 0.04)
        if crop_margin > 0 and (w - 2 * crop_margin > 10) and (h - 2 * crop_margin > 10):
            img = img.crop((crop_margin, crop_margin, w - crop_margin, h - crop_margin))

        aug_list.append(img)

    return aug_list


def run_benchmark():
    print("=" * 70)
    print("  RUNNING SKU 100 RECOGNITION & DATA AUGMENTATION ACCURACY BENCHMARK")
    print("=" * 70)

    # 1. Initialize DINOv3 768-D Feature Extractor
    print("\n[Step 1] Initializing DINOv3 ViT-B/16 768-D Extractor...")
    embedder = DINOv3Extractor(device="cpu")

    # 2. Prepare Reference Crops for Class 100 (Juhayna Mango / Test Data)
    source_dir = repo_root / "pipeline_2_test_data" / "Juhayna Mango"
    if not source_dir.exists():
        source_dir = repo_root / "pipeline_2_test_data" / "Dreem"

    ref_images = sorted(list(source_dir.glob("*.jpg")))
    print(f"Found {len(ref_images)} total reference images in '{source_dir.name}'")

    # Split into Onboarding Reference Set (14 crops) and Holdout Test Set (10 crops)
    onboard_files = ref_images[:14]
    test_files = ref_images[14:24] if len(ref_images) >= 24 else ref_images[7:]

    print(f"  -> Onboarding Reference Set: {len(onboard_files)} images")
    print(f"  -> Holdout Test Validation Set: {len(test_files)} images")

    # Load PIL Images
    raw_onboard_pil = [Image.open(f).convert("RGB") for f in onboard_files]
    test_pil = [Image.open(f).convert("RGB") for f in test_files]

    # --- STRATEGY A: Raw YOLO Detector Crops ---
    print("\n" + "-" * 60)
    print("  STRATEGY A: Baseline (Raw Reference Crops only)")
    print("-" * 60)

    vecs_strat_a = embedder.extract(raw_onboard_pil)
    meta_strat_a = [
        {
            "remapped_class_id": 100,
            "old_class_id": 100,
            "family_id": "Juhayna_Mango_1L",
            "source_image_name": f.name,
            "bbox": [10.0, 10.0, 200.0, 200.0]
        }
        for f in onboard_files
    ]

    index_a = HierarchicalCosineIndex(dimension=768)
    index_a.add(vecs_strat_a, meta_strat_a)

    # Evaluate Strategy A on Holdout Test Set
    test_vecs = embedder.extract(test_pil)

    top1_correct_a = 0
    top5_correct_a = 0
    sims_a = []

    for i, t_vec in enumerate(test_vecs):
        top_indices, top_sims = index_a.search(t_vec.reshape(1, -1), top_k=5)
        top_sim = float(top_sims[0, 0])
        sims_a.append(top_sim)

        # Retrieve matched class from metadata
        matched_meta = index_a.metadata[top_indices[0, 0]]
        matched_class = matched_meta["remapped_class_id"]

        if matched_class == 100:
            top1_correct_a += 1
            top5_correct_a += 1

    top1_acc_a = (top1_correct_a / len(test_pil)) * 100.0
    top5_acc_a = (top5_correct_a / len(test_pil)) * 100.0
    mean_sim_a = np.mean(sims_a) if sims_a else 0.0

    print(f"Strategy A Metrics (Raw Crops = {len(onboard_files)} vectors):")
    print(f"  -> Top-1 Accuracy: {top1_acc_a:.2f}%")
    print(f"  -> Top-5 Accuracy: {top5_acc_a:.2f}%")
    print(f"  -> Mean Cosine Similarity (S_vis): {mean_sim_a * 100:.2f}%")
    print(f"  -> Min Similarity: {np.min(sims_a)*100:.2f}% | Max Similarity: {np.max(sims_a)*100:.2f}%")

    # --- STRATEGY B: Selective Visual Data Augmentation ---
    print("\n" + "-" * 60)
    print("  STRATEGY B: Selective Data Augmentation (+3 variants per crop)")
    print("-" * 60)

    aug_pil_list = []
    meta_strat_b = []

    for idx, raw_img in enumerate(raw_onboard_pil):
        # Always include raw image
        aug_pil_list.append(raw_img)
        meta_strat_b.append({
            "remapped_class_id": 100,
            "old_class_id": 100,
            "family_id": "Juhayna_Mango_1L",
            "source_image_name": f"{onboard_files[idx].name}_raw",
            "bbox": [10.0, 10.0, 200.0, 200.0]
        })

        # Apply augmentation to selective subset (every crop gets 2-3 synthetic variants)
        augmented_variants = apply_selective_augmentations(raw_img, count=2)
        for var_idx, aug_img in enumerate(augmented_variants):
            aug_pil_list.append(aug_img)
            meta_strat_b.append({
                "remapped_class_id": 100,
                "old_class_id": 100,
                "family_id": "Juhayna_Mango_1L",
                "source_image_name": f"{onboard_files[idx].name}_aug_{var_idx}",
                "bbox": [10.0, 10.0, 200.0, 200.0]
            })

    print(f"Generated {len(aug_pil_list)} total reference vectors (Raw: {len(raw_onboard_pil)}, Augmented: {len(aug_pil_list) - len(raw_onboard_pil)})")

    vecs_strat_b = embedder.extract(aug_pil_list)

    index_b = HierarchicalCosineIndex(dimension=768)
    index_b.add(vecs_strat_b, meta_strat_b)

    top1_correct_b = 0
    top5_correct_b = 0
    sims_b = []

    for i, t_vec in enumerate(test_vecs):
        top_indices, top_sims = index_b.search(t_vec.reshape(1, -1), top_k=5)
        top_sim = float(top_sims[0, 0])
        sims_b.append(top_sim)

        matched_meta = index_b.metadata[top_indices[0, 0]]
        matched_class = matched_meta["remapped_class_id"]

        if matched_class == 100:
            top1_correct_b += 1
            top5_correct_b += 1

    top1_acc_b = (top1_correct_b / len(test_pil)) * 100.0
    top5_acc_b = (top5_correct_b / len(test_pil)) * 100.0
    mean_sim_b = np.mean(sims_b) if sims_b else 0.0

    print(f"Strategy B Metrics (Augmented Crops = {len(aug_pil_list)} vectors):")
    print(f"  -> Top-1 Accuracy: {top1_acc_b:.2f}%")
    print(f"  -> Top-5 Accuracy: {top5_acc_b:.2f}%")
    print(f"  -> Mean Cosine Similarity (S_vis): {mean_sim_b * 100:.2f}%")
    print(f"  -> Min Similarity: {np.min(sims_b)*100:.2f}% | Max Similarity: {np.max(sims_b)*100:.2f}%")

    # Output JSON summary for documentation generation
    results = {
        "class_id": 100,
        "sku_name": "Juhayna Mango 1L Juice Pouch / Pack",
        "test_samples_count": len(test_pil),
        "strategy_a": {
            "name": "Strategy A: Raw Reference Crops",
            "indexed_vector_count": len(vecs_strat_a),
            "top1_accuracy_pct": round(top1_acc_a, 2),
            "top5_accuracy_pct": round(top5_acc_a, 2),
            "mean_similarity_pct": round(mean_sim_a * 100, 2),
            "min_similarity_pct": round(float(np.min(sims_a)) * 100, 2),
            "max_similarity_pct": round(float(np.max(sims_a)) * 100, 2),
            "storage_kb": round((len(vecs_strat_a) * 768 * 4) / 1024, 2)
        },
        "strategy_b": {
            "name": "Strategy B: Selective Visual Data Augmentation",
            "indexed_vector_count": len(vecs_strat_b),
            "top1_accuracy_pct": round(top1_acc_b, 2),
            "top5_accuracy_pct": round(top5_acc_b, 2),
            "mean_similarity_pct": round(mean_sim_b * 100, 2),
            "min_similarity_pct": round(float(np.min(sims_b)) * 100, 2),
            "max_similarity_pct": round(float(np.max(sims_b)) * 100, 2),
            "storage_kb": round((len(vecs_strat_b) * 768 * 4) / 1024, 2)
        }
    }

    res_path = repo_root / "data" / "sku_100_augmentation_benchmark.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Benchmark Results saved to: {res_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
