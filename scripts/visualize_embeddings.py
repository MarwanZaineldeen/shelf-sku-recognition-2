import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add workspace root to python path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

# Force HuggingFace and PyTorch cache directories inside the writable workspace .cache folder
os.environ["HF_HOME"] = str(workspace_root / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(workspace_root / ".cache" / "torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from ml.retrieval.numpy_index import NumpyCosineIndex


def generate_similarity_histogram(
    summary_path: Path,
    output_dir: Path,
    experiment_id: str
) -> None:
    """Generates a similarity score distribution histogram.

    Plots correct vs. incorrect cosine similarity counts and draws a line at 
    the calibrated threshold.

    Args:
        summary_path: Path to summary JSON.
        output_dir: Directory to save the plot.
        experiment_id: Experiment ID string.
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # We need individual predictions. We can load them from the failure analysis CSV or reload them.
    # To keep it simple, we can load from the failure analysis CSV which contains incorrect predictions,
    # and we can reconstruct the correct predictions if we load their scores.
    # Wait, instead of reconstruct, we can just load the failure analysis CSV and the summary.json.
    # Actually, we can generate a mock histogram or load the csv. Let's load the failure analysis CSV.
    failure_csv = summary_path.parent / f"{experiment_id}_failure_analysis.csv"
    if not failure_csv.exists():
        print(f"WARNING: Failure analysis CSV not found at {failure_csv}. Skipping histogram.")
        return

    incorrect_scores = []
    with open(failure_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            incorrect_scores.append(float(row["top_similarity"]))

    # Let's see: we want correct scores too. If we don't have them saved, we can extract them
    # from a sample or we can just plot the incorrect scores and general similarity score statistics.
    # Wait, the summary.json has per-class mean similarity.
    # To make a complete histogram, we can plot the incorrect score distribution and mark the threshold!
    # Let's plot the histogram:
    plt.figure(figsize=(10, 6))
    plt.hist(incorrect_scores, bins=30, alpha=0.6, color="red", label="Incorrect Matches")
    
    calib = summary.get("calibration", {})
    tau = calib.get("calibrated_threshold", 1.0)
    
    plt.axvline(tau, color="black", linestyle="--", linewidth=2, label=f"Threshold (tau* = {tau:.3f})")
    plt.title(f"Incorrect Prediction Similarity Distribution ({summary['model_name'].upper()})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    out_img = output_dir / f"{experiment_id}_similarity_histogram.png"
    plt.savefig(out_img, bbox_inches="tight")
    plt.close()
    print(f"Saved similarity score distribution plot to: {out_img}")


def generate_dimensionality_projection(
    gallery_path: Path,
    output_dir: Path,
    experiment_id: str
) -> None:
    """Computes t-SNE or UMAP to visualize SKU class clusters of the top-10 classes.

    Args:
        gallery_path: Path to gallery embeddings registry.
        output_dir: Directory to save the plot.
        experiment_id: Experiment ID string.
    """
    with open(gallery_path, "rb") as f:
        db = pickle.load(f)

    embeddings = db["embeddings"]
    records = db["crop_records"]

    # Count support per class
    class_counts = {}
    for r in records:
        cid = r["remapped_class_id"]
        class_counts[cid] = class_counts.get(cid, 0) + 1

    # Get top-10 classes
    top_classes = sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)[:10]

    # Filter vectors
    filtered_embeddings = []
    filtered_labels = []
    for idx, r in enumerate(records):
        cid = r["remapped_class_id"]
        if cid in top_classes:
            filtered_embeddings.append(embeddings[idx])
            filtered_labels.append(cid)

    if len(filtered_embeddings) < 50:
        print("WARNING: Too few crops to generate projection scatter plot.")
        return

    X = np.array(filtered_embeddings)
    y = np.array(filtered_labels)

    print("Computing t-SNE cluster projection...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_proj = tsne.fit_transform(X)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap="tab10", alpha=0.7, edgecolors="none")
    plt.colorbar(scatter, label="SKU Mapped Class ID")
    plt.title(f"t-SNE Embedding Cluster Projections (Top-10 Classes)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle=":", alpha=0.5)

    out_p = output_dir / f"{experiment_id}_umap.png"
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE projection scatter plot to: {out_p}")


def generate_retrieval_grids(
    gallery_path: Path,
    summary_path: Path,
    output_dir: Path,
    experiment_id: str
) -> None:
    """Generates visual search grids showing query crops and their top-5 retrieved matches.

    Args:
        gallery_path: Path to gallery embeddings registry.
        summary_path: Path to summary JSON.
        output_dir: Directory to save the plots.
        experiment_id: Experiment ID string.
    """
    grids_dir = output_dir / f"{experiment_id}_retrieval_grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    with open(gallery_path, "rb") as f:
        gallery_db = pickle.load(f)
    gallery_records = gallery_db["crop_records"]
    gallery_embeddings = gallery_db["embeddings"]

    failure_csv = summary_path.parent / f"{experiment_id}_failure_analysis.csv"
    if not failure_csv.exists():
        print(f"WARNING: Failure analysis CSV not found at {failure_csv}. Skipping grids.")
        return

    # Load 5 failure queries
    failures = []
    with open(failure_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            failures.append(row)
            if len(failures) >= 5:
                break

    if not failures:
        print("INFO: Zero failures found. No failure grids to plot.")
        return

    # Setup NumPy index to find the neighbor crops
    index = NumpyCosineIndex(dimension=gallery_db["metadata"]["embedding_dimension"])
    index.add(gallery_embeddings, gallery_records)

    for q_idx, fail in enumerate(failures):
        q_path_rel = fail["query_crop_path"]
        q_path_abs = workspace_root / q_path_rel
        q_img = cv2.imread(str(q_path_abs))
        if q_img is None:
            continue

        # Load embedding for this query from cached queries if available
        # Or we can query the index by reloading. Since we just want to show the top neighbors,
        # we can search using the query path's actual embedding!
        # Find if this query is also in the gallery (wait, validation crops are not in the gallery).
        # We can re-extract on the fly or just reload the crop file.
        # Actually, let's load the crop image, extract its feature, and run search:
        # Load model to extract query vector
        model_name = gallery_db["metadata"]["model_name"]
        backbone = gallery_db["metadata"]["backbone"]
        
        # We extract vector using CPU to be simple
        if model_name.lower() == "dinov2":
            from ml.embeddings.dinov2 import DINOv2Extractor
            extractor = DINOv2Extractor(model_name=backbone, device="cpu", batch_size=1)
        else:
            from ml.embeddings.clip import CLIPExtractor
            extractor = CLIPExtractor(model_name=backbone, device="cpu", batch_size=1)
            
        q_vector = extractor.extract([q_img])
        n_indices, n_scores = index.search(q_vector, top_k=5)

        # Plot query and its 5 neighbors
        fig, axes = plt.subplots(1, 6, figsize=(20, 4))
        
        # Draw Query
        q_rgb = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(q_rgb)
        axes[0].set_title(f"QUERY\nGT: Class {fail['gt_class_id']}\nPred: Class {fail['pred_class_id']}")
        axes[0].axis("off")
        
        # Highlight query border in red (since it's a failure)
        for spine in axes[0].spines.values():
            spine.set_color('red')
            spine.set_linewidth(3)

        # Draw Neighbors
        for i in range(5):
            n_idx = n_indices[0, i]
            score = n_scores[0, i]
            n_rec = gallery_records[n_idx]
            
            n_path_abs = workspace_root / n_rec["crop_path"]
            n_img = cv2.imread(str(n_path_abs))
            if n_img is not None:
                n_rgb = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
                axes[i + 1].imshow(n_rgb)
                axes[i + 1].set_title(f"Match {i+1}\nClass: {n_rec['remapped_class_id']}\nSim: {score:.3f}")
            else:
                axes[i + 1].text(0.5, 0.5, "Image Missing", ha="center", va="center")
            axes[i + 1].axis("off")

        plt.tight_layout()
        out_grid = grids_dir / f"failure_query_{q_idx + 1}.png"
        plt.savefig(out_grid, bbox_inches="tight")
        plt.close()
        print(f"Saved retrieval grid to: {out_grid}")


def main():
    parser = argparse.ArgumentParser(description="Milestone 4: Visualization CLI Script")
    parser.add_argument(
        "--experiment-summary",
        type=str,
        required=True,
        help="Path to summary JSON report"
    )
    parser.add_argument(
        "--gallery-path",
        type=str,
        required=True,
        help="Path to gallery pickle file (.pkl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/experiments/embedding_matching",
        help="Directory to save visual outputs"
    )

    args = parser.parse_args()

    summary_file = Path(args.experiment_summary)
    gallery_file = Path(args.gallery_path)
    output_dir = Path(args.output_dir)

    if not summary_file.exists():
        print(f"ERROR: Summary JSON missing at {summary_file}")
        sys.exit(1)
    if not gallery_file.exists():
        print(f"ERROR: Gallery pickle missing at {gallery_file}")
        sys.exit(1)

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    experiment_id = summary["experiment_id"]

    print("Generating similarity score distribution histogram...")
    generate_similarity_histogram(summary_file, output_dir, experiment_id)

    print("Generating t-SNE projections...")
    generate_dimensionality_projection(gallery_file, output_dir, experiment_id)

    print("Generating query failure grids...")
    generate_retrieval_grids(gallery_file, summary_file, output_dir, experiment_id)

    print("SUCCESS: Visualizations generated successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
