import argparse
import csv
import datetime
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import cv2

# Add workspace root to python path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from ml.retrieval.base import verify_split_disjointness
from ml.retrieval.numpy_index import NumpyCosineIndex
from ml.embeddings.dinov2 import DINOv2Extractor
from ml.embeddings.clip import CLIPExtractor
from ml.evaluation.metrics import (
    compute_recall_at_k,
    compute_macro_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_cmc_curve,
    bootstrap_metrics,
    calibrate_similarity_threshold,
)


def load_class_mapping(mapping_path: Path) -> Dict[str, int]:
    """Loads the class mapping configuration file.

    Args:
        mapping_path: Path to configs/class_id_mapping.json.

    Returns:
        Dict[str, int]: Mapping dictionary.
    """
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_query_crops(
    metadata_csv: Path,
    split: str,
    workspace_root: Path
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Loads crop images and metadata for evaluation queries.

    Args:
        metadata_csv: Path to crop_metadata.csv.
        split: Split filter ('val' or 'test').
        workspace_root: Project directory.

    Returns:
        Tuple of loaded images and metadata records.
    """
    images = []
    records = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue

            crop_path_rel = row.get("crop_path")
            if not crop_path_rel:
                continue
            crop_path_abs = workspace_root / crop_path_rel

            img = cv2.imread(str(crop_path_abs))
            if img is None:
                continue

            images.append(img)
            records.append({
                "crop_path": crop_path_rel,
                "source_image_name": row.get("source_image_name", ""),
                "family_id": row.get("family_id", ""),
                "remapped_class_id": int(row.get("remapped_class_id", -1)),
                "old_class_id": int(row.get("old_class_id", -1)),
                "bbox": [
                    float(row.get("x1", 0)),
                    float(row.get("y1", 0)),
                    float(row.get("x2", 0)),
                    float(row.get("y2", 0))
                ],
                "blur_score": float(row.get("blur_score", 0.0)),
                "quality_flag": row.get("quality_flag", "ok"),
                "height": img.shape[0],
                "width": img.shape[1]
            })
    return images, records


def main():
    parser = argparse.ArgumentParser(description="Milestone 4: Evaluation & Threshold Calibration CLI")
    parser.add_argument(
        "--gallery-path",
        type=str,
        required=True,
        help="Path to serialized gallery embeddings (.pkl)"
    )
    parser.add_argument(
        "--queries-dir",
        type=str,
        default="data/processed/crops/gt_clean",
        help="Path to crops directory containing crop_metadata.csv"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split to evaluate against gallery (default: val)"
    )
    parser.add_argument(
        "--query-embeddings-path",
        type=str,
        default="",
        help="Optional path to pre-extracted query embeddings cache (.pkl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/experiments/embedding_matching",
        help="Directory to save output reports"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K neighbors to retrieve"
    )
    parser.add_argument(
        "--calibration-precision",
        type=float,
        default=0.95,
        help="Target precision constraint for threshold calibration (default: 0.95)"
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for standard error computation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for extraction if queries are computed on-the-fly"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Processing batch size for on-the-fly extraction"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Gallery
    gallery_file = Path(args.gallery_path)
    if not gallery_file.exists():
        print(f"ERROR: Gallery file missing at {gallery_file}")
        sys.exit(1)

    print(f"Loading Gallery database from {gallery_file}...")
    with open(gallery_file, "rb") as f:
        gallery_db = pickle.load(f)
    
    gallery_meta = gallery_db["metadata"]
    gallery_vectors = gallery_db["embeddings"]
    gallery_records = gallery_db["crop_records"]
    
    print(f"Loaded gallery database containing {gallery_vectors.shape[0]} vectors of dimension {gallery_vectors.shape[1]}.")

    # 2. Load or Compute Queries
    query_vectors = None
    query_records = []
    
    if args.query_embeddings_path:
        q_file = Path(args.query_embeddings_path)
        if q_file.exists():
            print(f"Loading cached query embeddings from {q_file}...")
            with open(q_file, "rb") as f:
                q_db = pickle.load(f)
            query_vectors = q_db["embeddings"]
            query_records = q_db["crop_records"]
            print(f"Loaded {query_vectors.shape[0]} cached query vectors.")
        else:
            print(f"WARNING: Cache file {q_file} missing. Extracting queries on-the-fly...")

    if query_vectors is None:
        crops_dir = Path(args.queries_dir)
        metadata_csv = crops_dir / "crop_metadata.csv"
        if not metadata_csv.exists():
            print(f"ERROR: Metadata catalog missing at {metadata_csv}.")
            sys.exit(1)

        images, query_records = load_query_crops(metadata_csv, args.split, workspace_root)
        if not images:
            print(f"ERROR: No crops found for split: '{args.split}' in {metadata_csv}")
            sys.exit(1)

        model_name = gallery_meta["model_name"]
        backbone = gallery_meta["backbone"]
        print(f"Extracting query features using model: '{model_name}' ({backbone}) on device: '{args.device}'...")
        
        # Override environment variables for Hugging Face local caching path
        os.environ["HF_HOME"] = str(workspace_root / ".cache" / "huggingface")
        os.environ["HF_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
        os.environ["TORCH_HOME"] = str(workspace_root / ".cache" / "torch")
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        if model_name.lower() == "dinov2":
            extractor = DINOv2Extractor(model_name=backbone, device=args.device, batch_size=args.batch_size)
        else:
            extractor = CLIPExtractor(model_name=backbone, device=args.device, batch_size=args.batch_size)

        query_vectors = extractor.extract(images)

    # 3. Assert Split Disjointness (Family-ID Validation)
    print("Verifying split disjointness (assertion check)...")
    try:
        verify_split_disjointness(gallery_records, query_records)
        print("SUCCESS: Splits are completely disjoint. No cross-split leakage.")
    except AssertionError as e:
        print(f"ERROR: Split validation failed: {e}")
        sys.exit(1)

    # 4. Search Gallery
    print("Building Cosine Search Index...")
    index = NumpyCosineIndex(dimension=gallery_meta["embedding_dimension"])
    index.add(gallery_vectors, gallery_records)

    print(f"Querying nearest neighbors (top-k={args.top_k})...")
    neighbor_indices, similarity_scores = index.search(query_vectors, top_k=args.top_k)

    # Map neighbor indexes to class labels
    neighbor_labels = np.array([
        [gallery_records[idx]["remapped_class_id"] for idx in row]
        for row in neighbor_indices
    ])
    query_labels = np.array([r["remapped_class_id"] for r in query_records])

    # 5. Calculate Metrics
    print("Calculating metrics and bootstrap intervals...")
    rec1 = compute_recall_at_k(neighbor_labels, query_labels, 1)
    rec5 = compute_recall_at_k(neighbor_labels, query_labels, args.top_k)
    mac1 = compute_macro_recall_at_k(neighbor_labels, query_labels, 1)
    mac5 = compute_macro_recall_at_k(neighbor_labels, query_labels, args.top_k)
    mrr_val = compute_mrr(neighbor_labels, query_labels)
    ndcg_val = compute_ndcg_at_k(neighbor_labels, query_labels, args.top_k)
    cmc_vals = compute_cmc_curve(neighbor_labels, query_labels, max_k=10)

    # Run Bootstrap error bound calculations
    boot_stats = bootstrap_metrics(
        neighbor_labels, query_labels, top_k=args.top_k, num_bootstraps=args.bootstrap_iters
    )

    # 6. Calibrate Threshold
    top_scores = similarity_scores[:, 0]
    top_correct = (neighbor_labels[:, 0] == query_labels)
    calib_threshold, automation_rate = calibrate_similarity_threshold(
        top_scores, top_correct, target_precision=args.calibration_precision
    )

    print("================ EVALUATION SUMMARY ================")
    print(f"Recall@1: {rec1 * 100:.2f}% | Class-Balanced: {mac1 * 100:.2f}%")
    print(f"Recall@5: {rec5 * 100:.2f}% | Class-Balanced: {mac5 * 100:.2f}%")
    print(f"NDCG@5:   {ndcg_val * 100:.2f}% | MRR: {mrr_val * 100:.2f}%")
    print(f"Calibrated Threshold: {calib_threshold:.3f} | Automation Rate: {automation_rate * 100:.2f}%")
    print("====================================================")

    # 7. Generate Failure Diagnostics CSV
    experiment_id = gallery_meta["experiment_id"]
    failure_csv_path = output_dir / f"{experiment_id}_failure_analysis.csv"
    print(f"Compiling failure analysis log to: {failure_csv_path}")

    with open(failure_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_crop_path", "gt_class_id", "pred_class_id", 
            "top_similarity", "delta_similarity", "width", "height", "blur_score"
        ])

        for i in range(len(query_labels)):
            if not top_correct[i]:
                # Find index of first correct match in retrieved candidates (if any)
                correct_row_idx = np.where(neighbor_labels[i] == query_labels[i])[0]
                if len(correct_row_idx) > 0:
                    delta_sim = float(top_scores[i] - similarity_scores[i, correct_row_idx[0]])
                else:
                    delta_sim = 1.0  # correct not found in top_k

                rec = query_records[i]
                width = rec.get("width", int(rec["bbox"][2] - rec["bbox"][0]) if "bbox" in rec else -1)
                height = rec.get("height", int(rec["bbox"][3] - rec["bbox"][1]) if "bbox" in rec else -1)
                writer.writerow([
                    rec["crop_path"],
                    rec["remapped_class_id"],
                    neighbor_labels[i, 0],
                    f"{top_scores[i]:.4f}",
                    f"{delta_sim:.4f}",
                    width,
                    height,
                    f"{rec['blur_score']:.2f}"
                ])

    # 8. Compile Per-Class Performance Table
    class_mapping = load_class_mapping(workspace_root / "configs/class_id_mapping.json")
    new_to_old = class_mapping.get("new_to_old", {})
    
    unique_classes = np.unique(query_labels)
    per_class_summary = []
    
    for c in range(67):
        class_mask = (query_labels == c)
        support = int(np.sum(class_mask))
        if support == 0:
            continue
            
        c_neighbors = neighbor_labels[class_mask]
        c_queries = query_labels[class_mask]
        c_scores = top_scores[class_mask]

        c_rec1 = compute_recall_at_k(c_neighbors, c_queries, 1)
        c_rec5 = compute_recall_at_k(c_neighbors, c_queries, args.top_k)
        c_mean_sim = float(np.mean(c_scores))

        # Find most confused class
        incorrect_predictions = c_neighbors[c_neighbors[:, 0] != c, 0]
        if len(incorrect_predictions) > 0:
            most_confused = int(np.argmax(np.bincount(incorrect_predictions)))
        else:
            most_confused = -1

        old_id = new_to_old.get(str(c), c)

        per_class_summary.append({
            "remapped_class_id": c,
            "old_class_id": old_id,
            "support": support,
            "recall_1": c_rec1,
            "recall_5": c_rec5,
            "mean_similarity": c_mean_sim,
            "most_confused_class": most_confused
        })

    # 9. Save Summary JSON
    summary_json_path = output_dir / f"{experiment_id}_summary.json"
    summary_data = {
        "experiment_id": experiment_id,
        "model_name": gallery_meta["model_name"],
        "backbone": gallery_meta["backbone"],
        "split": args.split,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": {
            "recall_1": rec1,
            "recall_5": rec5,
            "macro_recall_1": mac1,
            "macro_recall_5": mac5,
            "mrr": mrr_val,
            "ndcg_5": ndcg_val,
            "cmc_curve": cmc_vals
        },
        "bootstrap": boot_stats,
        "calibration": {
            "target_precision": args.calibration_precision,
            "calibrated_threshold": calib_threshold,
            "automation_rate": automation_rate
        },
        "per_class": per_class_summary
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved machine-readable summary to: {summary_json_path}")

    # 10. Generate Markdown Narrative Report
    report_md_path = output_dir / f"{experiment_id}_evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Retrieval Evaluation Report - {gallery_meta['model_name'].upper()}\n\n")
        f.write(f"- **Experiment ID**: `{experiment_id}`\n")
        f.write(f"- **Backbone Model**: `{gallery_meta['backbone']}`\n")
        f.write(f"- **Evaluation Split**: `{args.split}`\n")
        f.write(f"- **Timestamp**: `{summary_data['timestamp']}`\n\n")

        f.write("## 1. Global Retrieval Performance\n\n")
        f.write("| Metric | Mean Accuracy | 95% Bootstrap Confidence Interval | Standard Error (Std) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        for name, key in [("Recall@1 (Top-1 Accuracy)", "recall@1"), 
                          ("Recall@5 (Top-5 Accuracy)", "recall@5"), 
                          ("NDCG@5 (Discounted Gain)", "ndcg"), 
                          ("MRR (Reciprocal Rank)", "mrr")]:
            stat = boot_stats[key]
            f.write(
                f"| {name} | {stat['mean']*100:.2f}% | "
                f"[{stat['ci_lower']*100:.2f}%, {stat['ci_upper']*100:.2f}%] | "
                f"{stat['std']*100:.2f}% |\n"
            )
        f.write("\n")

        f.write("### Class-Balanced Retrieval Outcomes\n")
        f.write(f"- **Class-Balanced (Macro) Recall@1**: {mac1 * 100:.2f}%\n")
        f.write(f"- **Class-Balanced (Macro) Recall@5**: {mac5 * 100:.2f}%\n\n")

        f.write("## 2. Threshold Calibration Summary\n\n")
        f.write("The similarity threshold is dynamically calibrated on the validation split to guarantee high-precision labels for the daily auto-annotation loop:\n\n")
        f.write(f"- **Target Labeling Precision Constraint**: $\ge {args.calibration_precision * 100:.1f}\%$\n")
        f.write(f"- **Calibrated Similarity Threshold ($\\tau^*$)**: **{calib_threshold:.3f}**\n")
        f.write(f"- **Automation Coverage Rate**: **{automation_rate * 100:.2f}%** (The percentage of items verified automatically without routing to HITL).\n\n")

        f.write("## 3. Per-Class Retrieval Performance\n\n")
        f.write("| Remapped Class ID | Old Class ID | Support Count | Recall@1 | Recall@5 | Mean Similarity | Most Confused Class |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for pc in per_class_summary:
            f.write(
                f"| {pc['remapped_class_id']} | {pc['old_class_id']} | {pc['support']} | "
                f"{pc['recall_1']*100:.2f}% | {pc['recall_5']*100:.2f}% | "
                f"{pc['mean_similarity']:.4f} | {pc['most_confused_class'] if pc['most_confused_class'] != -1 else 'None'} |\n"
            )
        f.write("\n")

        f.write("## 4. Failure Analysis Insights\n\n")
        f.write(f"A detailed list of failed query items has been compiled and saved to `{failure_csv_path.name}`. ")
        f.write("Use the coordinates and blur scores in this CSV sheet to identify low-resolution or blurry crop patterns affecting classification accuracy.")

    print(f"SUCCESS: Saved narrative report to: {report_md_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
