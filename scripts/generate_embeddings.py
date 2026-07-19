import argparse
import csv
import datetime
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import cv2

# Add workspace root to python path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

# Force HuggingFace and PyTorch cache directories inside the writable workspace .cache folder
os.environ["HF_HOME"] = str(workspace_root / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(workspace_root / ".cache" / "torch")

from ml.embeddings.dinov2 import DINOv2Extractor
from ml.embeddings.clip import CLIPExtractor


def get_git_commit() -> str:
    """Retrieves the current Git commit hash if available.

    Returns:
        str: Git hash or 'unknown'.
    """
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def get_system_info() -> Dict[str, str]:
    """Compiles library and environment system versions.

    Returns:
        Dict[str, str]: Library versions dictionary.
    """
    import transformers
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None"
    }


def load_crops_and_metadata(
    metadata_csv: Path,
    split: str,
    workspace_root: Path
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Loads crop images and associated metadata for a specific split.

    Args:
        metadata_csv: Path to crop_metadata.csv.
        split: Split filter ('train', 'val', or 'test').
        workspace_root: Absolute path to project directory.

    Returns:
        Tuple of:
            - images: List of numpy arrays representing loaded crop images.
            - records: List of dictionaries matching the loaded images.
    """
    images = []
    records = []

    print(f"Reading crop metadata from {metadata_csv} for split: '{split}'...")
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if row.get("split") != split:
                continue

            # Resolve crop path
            crop_path_rel = row.get("crop_path")
            if not crop_path_rel:
                continue
            crop_path_abs = workspace_root / crop_path_rel

            # Read image using OpenCV
            img = cv2.imread(str(crop_path_abs))
            if img is None:
                print(f"WARNING: Failed to load image at {crop_path_abs}. Skipping.")
                continue

            images.append(img)
            # Store metadata dictionary
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
                "quality_flag": row.get("quality_flag", "ok")
            })

    print(f"Successfully loaded {len(images)} crops for split: '{split}'.")
    return images, records


def main():
    parser = argparse.ArgumentParser(description="Milestone 4: Gallery & Query Embedding Generator")
    parser.add_argument(
        "--crops-dir",
        type=str,
        default="data/processed/crops/gt_clean",
        help="Path to clean crops directory containing crop_metadata.csv"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split to generate embeddings for (default: train)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2",
        choices=["dinov2", "clip"],
        help="Embedding model backbone (default: dinov2)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Target filepath for saved pickle database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference processing batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force regeneration of features and overwrite existing cache"
    )

    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    crops_dir = Path(args.crops_dir)
    metadata_csv = crops_dir / "crop_metadata.csv"

    if not metadata_csv.exists():
        print(f"ERROR: Metadata catalog missing at {metadata_csv}.")
        sys.exit(1)

    # Set default output path if not specified
    model_repr = args.model_name.lower()
    output_path = args.output_path
    if not output_path:
        output_path = f"data/processed/crops/gt_clean/embeddings_{model_repr}_{args.split}.pkl"
    output_file = Path(output_path)

    # Resolve backbone identifier
    backbone = "facebook/dinov2-small" if model_repr == "dinov2" else "openai/clip-vit-base-patch32"

    # Check cache validity
    if output_file.exists() and not args.overwrite:
        try:
            print(f"Cache file found at {output_file}. Checking compatibility headers...")
            with open(output_file, "rb") as f:
                cached_db = pickle.load(f)
            
            cached_meta = cached_db.get("metadata", {})
            if (
                cached_meta.get("backbone") == backbone and
                cached_meta.get("split") == args.split and
                cached_meta.get("dataset_version") == "v1.5_clean"
            ):
                print(f"SUCCESS: Compatible cache found. Reusing cached features ({len(cached_db['embeddings'])} vectors).")
                sys.exit(0)
            else:
                print("WARNING: Cache configuration mismatch. Regenerating embeddings...")
        except Exception as e:
            print(f"WARNING: Failed to read cache file: {e}. Regenerating...")

    # Load crops
    images, crop_records = load_crops_and_metadata(metadata_csv, args.split, workspace_root)
    if not images:
        print("ERROR: No crops found to process.")
        sys.exit(1)

    # Load extractor
    print(f"Loading {args.model_name} extractor on device: '{args.device}'...")
    if model_repr == "dinov2":
        extractor = DINOv2Extractor(model_name=backbone, device=args.device, batch_size=args.batch_size)
    else:
        extractor = CLIPExtractor(model_name=backbone, device=args.device, batch_size=args.batch_size)

    # Extract features
    print(f"Extracting visual features (dimension: {extractor.dimension})...")
    try:
        embeddings = extractor.extract(images)
    except Exception as e:
        print(f"ERROR: Extraction failed: {e}")
        sys.exit(1)

    # Compile reproducibility envelope
    timestamp = datetime.datetime.now().isoformat()
    experiment_id = f"EXP_{args.model_name.upper()}_V1.5_CLEAN_{args.split.upper()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    metadata_envelope = {
        "experiment_id": experiment_id,
        "model_name": args.model_name,
        "backbone": backbone,
        "embedding_dimension": extractor.dimension,
        "preprocessing": {
            "resize_size": [224, 224],
            "interpolation": "cv2.INTER_AREA",
            "normalization": "imagenet"
        },
        "dataset_version": "v1.5_clean",
        "split": args.split,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "system_info": get_system_info()
    }

    # Package database
    db_package = {
        "metadata": metadata_envelope,
        "embeddings": embeddings,
        "crop_records": crop_records
    }

    # Save to disk
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(db_package, f)

    print(f"SUCCESS: Saved embeddings and metadata to: {output_file}")
    print(f"Vector count: {embeddings.shape[0]}, Dimension: {embeddings.shape[1]}")
    sys.exit(0)


if __name__ == "__main__":
    main()
