import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.embeddings.dinov2 import DINOv2Extractor
from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from ml.retrieval.numpy_index import NumpyCosineIndex
from ml.retrieval.hierarchical_index import HierarchicalCosineIndex
from ml.onboarding.onboarder import SKUOnboarder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pipeline 2: New SKU Onboarding CLI Tool")
    parser.add_argument(
        "--mode",
        choices=["crops", "shelf"],
        default="crops",
        help="Input mode: 'crops' for ready cropped product images, 'shelf' for full shelf images + txt bounding boxes."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to folder containing cropped images or shelf images."
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Path to folder containing YOLO .txt label files (required if mode is 'shelf')."
    )
    parser.add_argument(
        "--class-id",
        type=int,
        required=True,
        help="Numeric class ID for the new SKU."
    )
    parser.add_argument(
        "--old-class-id",
        type=int,
        default=None,
        help="Original class ID (defaults to class-id if not specified)."
    )
    parser.add_argument(
        "--family-id",
        type=str,
        required=True,
        help="Brand family / cluster name (e.g. 'Nesquik', 'Heinz tomato ketchup')."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/processed/crops/gt_clean/retail_sku_registry_dinov3.db",
        help="Path to target SQLite vector database registry."
    )
    parser.add_argument(
        "--backbone",
        choices=["dinov2", "dinov3"],
        default="dinov3",
        help="Visual embedding backbone model to use."
    )

    args = parser.parse_args()

    old_class_id = args.old_class_id if args.old_class_id is not None else args.class_id

    print("==================================================")
    print("Executing Pipeline 2: New SKU Onboarding")
    print(f"Mode:        {args.mode}")
    print(f"Data Dir:    {args.data_dir}")
    print(f"Class ID:    {args.class_id}")
    print(f"Family ID:   {args.family_id}")
    print(f"DB Path:     {args.db_path}")
    print(f"Backbone:    {args.backbone}")
    print("==================================================")

    # 1. Initialize embedder
    if args.backbone == "dinov2":
        embedder = DINOv2Extractor(model_name="facebook/dinov2-small", device="cpu")
    else:
        from ml.embeddings.dinov3 import DINOv3Extractor
        embedder = DINOv3Extractor(device="cpu")

    # 2. Initialize vector store
    store = SQLiteGalleryStore()
    store.initialize({"db_path": args.db_path})

    # 3. Initialize search retriever index
    retriever = HierarchicalCosineIndex(dimension=embedder.dimension)
    retriever.initialize({"dimension": embedder.dimension, "db_path": args.db_path})

    # 4. Initialize YOLOv8 Detector if available
    detector = None
    yolo_weights = Path("runs/detect/yolo8l_sku110k/yolov8l-sku110k.pt")
    if not yolo_weights.exists():
        yolo_weights = Path("D:/ITI/Graduation Project/Project GitHub/yolov8l-sku110k.pt")
    if yolo_weights.exists():
        try:
            from ml.detection.yolo_detector import YOLOv8Detector
            detector = YOLOv8Detector()
            detector.initialize({"weights_path": str(yolo_weights), "confidence_threshold": 0.25, "imgsz": 640})
            print(f"Loaded YOLOv8l product detector from: {yolo_weights}")
        except Exception as e:
            print(f"Warning: Could not initialize YOLOv8 Detector: {e}")

    # 5. Instantiate onboarder with detector
    onboarder = SKUOnboarder(embedder=embedder, store=store, retriever=retriever, detector=detector)

    # 6. Run onboarding
    if args.mode == "crops":
        results = onboarder.onboard_from_crops(
            crops_dir=args.data_dir,
            class_id=args.class_id,
            old_class_id=old_class_id,
            family_id=args.family_id,
            detector=detector
        )
    else:
        results = onboarder.onboard_from_shelf_images(
            shelf_dir=args.data_dir,
            labels_dir=args.labels_dir,
            class_id=args.class_id,
            old_class_id=old_class_id,
            family_id=args.family_id,
            detector=detector
        )

    print("\nOnboarding Results:")
    print(f"  Status:       {results['status']}")
    print(f"  Crops Added:  {results['crops_added']}")
    print(f"  Crops Rejected:{results['rejected']}")
    print(f"  DB Version:   {results['db_version']}")
    print("Pipeline 2 Execution Completed Successfully!")


if __name__ == "__main__":
    main()
