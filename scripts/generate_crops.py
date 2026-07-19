import argparse
import csv
import json
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.preprocessing.box_providers import YOLOLabelBoxProvider
from ml.preprocessing.crop_generator import CropGenerator
from ml.preprocessing.quality_checks import QualityChecker

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Milestone 3: Ground-Truth Crop Generation Pipeline")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/yolo_remapped",
        help="Path to preprocessed YOLO dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/crops/gt",
        help="Target folder to save product crop folders"
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default="configs/class_id_mapping.json",
        help="Path to class remapping configs"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Bounding box padding ratio"
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=20,
        help="Minimum height/width threshold in pixels"
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=5.0,
        help="Maximum width-to-height ratio allowed"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=30.0,
        help="Laplacian variance threshold for blur checking"
    )
    parser.add_argument(
        "--save-grids",
        action="store_true",
        default=True,
        help="Save preview grids of crops"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for grid selection reproducibility"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output directory"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    mapping_path = Path(args.mapping_path)
    reports_dir = Path("reports/experiments/crop_generation")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("Starting Crop Generation Pipeline")
    print(f"Dataset Dir:   {dataset_dir.resolve()}")
    print(f"Output Dir:    {output_dir.resolve()}")
    print(f"Padding:       {args.padding}")
    print(f"Min Size:      {args.min_crop_size}px")
    print(f"Max Aspect:    {args.max_aspect_ratio}")
    print(f"Blur Thresh:   {args.blur_threshold}")
    print("==================================================")

    # Overwrite check
    if output_dir.exists():
        if args.force:
            logger.info(f"Overwriting existing output folder: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logger.error(f"Output folder already exists: {output_dir}. Use --force to overwrite.")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read class mapping config
    if not mapping_path.exists():
        logger.error(f"Class mapping config missing at: {mapping_path}")
        sys.exit(1)

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    new_to_old = mapping_data.get("new_to_old", {})

    # Instantiate modules
    crop_generator = CropGenerator(padding=args.padding)
    quality_checker = QualityChecker(
        min_crop_size=args.min_crop_size,
        max_aspect_ratio=args.max_aspect_ratio,
        blur_threshold=args.blur_threshold
    )

    # Initialize statistics counters
    stats = {
        "total_crops_generated": 0,
        "total_skipped_boxes": 0,
        "empty_label_images": 0,
        "malformed_label_rows": 0,
        "invalid_coordinate_rows": 0,
        "crops_per_split": defaultdict(int),
        "crops_per_class": defaultdict(int),
        "quality_flag_counts": defaultdict(int)
    }

    metadata_records = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Sorted splits processing
    splits = sorted(["train", "val", "test"])

    for split in splits:
        img_split_dir = dataset_dir / "images" / split
        lbl_split_dir = dataset_dir / "labels" / split

        if not img_split_dir.exists() or not lbl_split_dir.exists():
            logger.warning(f"Split folders missing for '{split}'. Skipping.")
            continue

        box_provider = YOLOLabelBoxProvider(lbl_split_dir)

        # Get sorted images
        images = sorted([p for p in img_split_dir.iterdir() if p.suffix.lower() in supported_extensions])
        # Find unsupported extensions to log warnings
        all_files = sorted(list(img_split_dir.iterdir()))
        unsupported_files = [p for p in all_files if p.suffix.lower() not in supported_extensions and p.is_file()]
        for p in unsupported_files:
            logger.warning(f"Unsupported image file extension skipped: {p.name}")

        for img_path in images:
            # Check corresponding label file exists
            lbl_path = lbl_split_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                logger.warning(f"Missing label file for image: {img_path.name}")
                stats["total_skipped_boxes"] += 1
                continue

            # Read image safely
            image = cv2.imread(str(img_path))
            if image is None:
                logger.error(f"Image read failure on file: {img_path.name}")
                continue
            img_h, img_w = image.shape[:2]
            img_area = img_h * img_w

            # Retrieve bounding boxes
            boxes = box_provider.get_boxes(img_path)

            if len(boxes) == 0:
                stats["empty_label_images"] += 1
                continue

            # Sort boxes deterministically by line index in file
            sorted_boxes = sorted(boxes, key=lambda b: b.get("line_idx", 0))

            bbox_idx = 1
            for box in sorted_boxes:
                status = box.get("status")
                
                # Handled skipped, malformed, or invalid lines
                if status == "malformed":
                    stats["malformed_label_rows"] += 1
                    stats["total_skipped_boxes"] += 1
                    continue
                elif status == "invalid_coords":
                    stats["invalid_coordinate_rows"] += 1
                    stats["total_skipped_boxes"] += 1
                    continue

                # Extract crop
                crop, (x1, y1, x2, y2) = crop_generator.extract_crop(image, box)
                crop_h, crop_w = crop.shape[:2]
                crop_area = crop_h * crop_w
                aspect_ratio = max(crop_h / crop_w, crop_w / crop_h) if crop_area > 0 else 0.0
                bbox_area_ratio = crop_area / img_area if img_area > 0 else 0.0

                # Run quality checks
                quality_flag, quality_notes = quality_checker.check_crop(crop)

                remapped_class_id = box["class_id"]
                old_class_id = int(new_to_old.get(str(remapped_class_id), remapped_class_id))

                # Handle crop with zero area safely
                if crop_area == 0 or quality_flag == "invalid_crop":
                    logger.warning(f"Extracted crop has zero area or is invalid: {img_path.name} bbox {bbox_idx}")
                    stats["total_skipped_boxes"] += 1
                    continue

                # Filename conventions
                crop_filename = f"{img_path.stem}_box{bbox_idx}_class{remapped_class_id}.jpg"
                crop_sub_dir = output_dir / split / f"class_{remapped_class_id}"
                crop_sub_dir.mkdir(parents=True, exist_ok=True)
                crop_filepath = crop_sub_dir / crop_filename

                # Write crop image file
                try:
                    cv2.imwrite(str(crop_filepath), crop)
                    stats["total_crops_generated"] += 1
                    stats["crops_per_split"][split] += 1
                    stats["crops_per_class"][remapped_class_id] += 1
                    stats["quality_flag_counts"][quality_flag] += 1
                except Exception as e:
                    logger.error(f"Failed to write crop image: {crop_filepath.name}. Error: {str(e)}")
                    stats["total_skipped_boxes"] += 1
                    continue

                # Populate relative path metadata records
                rel_crop_path = f"data/processed/crops/gt/{split}/class_{remapped_class_id}/{crop_filename}"
                rel_source_path = f"data/processed/yolo_remapped/images/{split}/{img_path.name}"

                metadata_records.append({
                    "crop_id": f"{img_path.stem}_{bbox_idx}",
                    "crop_path": rel_crop_path,
                    "source_image_path": rel_source_path,
                    "source_image_name": img_path.name,
                    "split": split,
                    "remapped_class_id": remapped_class_id,
                    "old_class_id": old_class_id,
                    "bbox_index": bbox_idx,
                    "x_center_norm": box["x_center_norm"],
                    "y_center_norm": box["y_center_norm"],
                    "width_norm": box["width_norm"],
                    "height_norm": box["height_norm"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop_width": crop_w,
                    "crop_height": crop_h,
                    "crop_area": crop_area,
                    "bbox_area_ratio": bbox_area_ratio,
                    "aspect_ratio": aspect_ratio,
                    "padding": args.padding,
                    "quality_flag": quality_flag,
                    "quality_notes": quality_notes
                })

                bbox_idx += 1

    # Write metadata CSV
    metadata_csv_path = output_dir / "crop_metadata.csv"
    if metadata_records:
        keys = metadata_records[0].keys()
        with open(metadata_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata_records)
        logger.info(f"Saved crops metadata catalog to: {metadata_csv_path}")

    # Identify rare class counts (classes with <= 10 crops)
    rare_classes_crops = {int(cls): int(count) for cls, count in stats["crops_per_class"].items() if count <= 10}

    # Generate sample grids of crops using matplotlib
    if args.save_grids and metadata_records:
        previews_dir = reports_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating crop visual preview grids in: {previews_dir}")

        # Grid 1: Sample of random OK crops
        ok_crops = [r for r in metadata_records if r["quality_flag"] == "ok"]
        if ok_crops:
            sample_size = min(25, len(ok_crops))
            sampled = random.sample(ok_crops, sample_size)
            cols = 5
            rows = (sample_size + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            axes = axes.flatten()
            for idx, record in enumerate(sampled):
                # Read relative path
                abs_crop_p = Path(__file__).resolve().parent.parent / record["crop_path"]
                crop_img = cv2.imread(str(abs_crop_p))
                if crop_img is not None:
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    axes[idx].imshow(crop_img)
                    axes[idx].set_title(f"Class {record['remapped_class_id']}\n{record['crop_width']}x{record['crop_height']}")
                axes[idx].axis("off")
            for idx in range(sample_size, len(axes)):
                axes[idx].axis("off")
            plt.tight_layout()
            plt.savefig(previews_dir / "ok_crops_grid.png", bbox_inches="tight")
            plt.close()

        # Grid 2: Sample of flagged crops (blurry, too_small, extreme_aspect_ratio)
        flagged_crops = [r for r in metadata_records if r["quality_flag"] != "ok"]
        if flagged_crops:
            sample_size = min(16, len(flagged_crops))
            sampled = random.sample(flagged_crops, sample_size)
            cols = 4
            rows = (sample_size + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
            axes = axes.flatten()
            for idx, record in enumerate(sampled):
                abs_crop_p = Path(__file__).resolve().parent.parent / record["crop_path"]
                crop_img = cv2.imread(str(abs_crop_p))
                if crop_img is not None:
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    axes[idx].imshow(crop_img)
                    axes[idx].set_title(f"{record['quality_flag']}\nClass {record['remapped_class_id']}")
                axes[idx].axis("off")
            for idx in range(sample_size, len(axes)):
                axes[idx].axis("off")
            plt.tight_layout()
            plt.savefig(previews_dir / "flagged_crops_grid.png", bbox_inches="tight")
            plt.close()

    # Save summary JSON
    summary_json_path = reports_dir / "crop_generation_summary.json"
    summary_data = {
        "execution_timestamp": datetime.now().isoformat(),
        "parameters": {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "padding": args.padding,
            "min_crop_size": args.min_crop_size,
            "max_aspect_ratio": args.max_aspect_ratio,
            "blur_threshold": args.blur_threshold
        },
        "metrics": {
            "total_crops_generated": stats["total_crops_generated"],
            "total_skipped_boxes": stats["total_skipped_boxes"],
            "empty_label_images": stats["empty_label_images"],
            "malformed_label_rows": stats["malformed_label_rows"],
            "invalid_coordinate_rows": stats["invalid_coordinate_rows"],
            "crops_per_split": dict(stats["crops_per_split"]),
            "crops_per_class": {str(k): int(v) for k, v in stats["crops_per_class"].items()},
            "quality_flag_counts": dict(stats["quality_flag_counts"]),
            "rare_class_counts": rare_classes_crops
        }
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved summary JSON report to: {summary_json_path}")

    # Write Markdown Report
    report_md_path = reports_dir / "crop_generation_report.md"
    
    # Calculate crop height/width stats
    widths = [r["crop_width"] for r in metadata_records]
    heights = [r["crop_height"] for r in metadata_records]
    areas = [r["crop_area"] for r in metadata_records]
    mean_w = np.mean(widths) if widths else 0
    mean_h = np.mean(heights) if heights else 0
    mean_a = np.mean(areas) if areas else 0
    min_a = np.min(areas) if areas else 0
    max_a = np.max(areas) if areas else 0

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Ground-Truth Crop Generation Report\n\n")
        f.write("This document summarizes the execution results of Milestone 3: Crop Generation Pipeline.\n\n")
        
        f.write("## 1. Executive Metrics\n\n")
        f.write(f"- **Total Crops Generated**: {stats['total_crops_generated']}\n")
        f.write(f"- **Total Skipped Boxes**: {stats['total_skipped_boxes']}\n")
        f.write(f"- **Empty-Label Images Handled**: {stats['empty_label_images']}\n")
        f.write(f"- **Malformed-Label Rows Skipped**: {stats['malformed_label_rows']}\n")
        f.write(f"- **Invalid Coordinate Rows Skipped**: {stats['invalid_coordinate_rows']}\n\n")
        
        f.write("## 2. Crops per Split\n\n")
        f.write("| Split | Crop Count | Percentage |\n")
        f.write("| :--- | :---: | :---: |\n")
        for s in sorted(stats["crops_per_split"].keys()):
            cnt = stats["crops_per_split"][s]
            pct = (cnt / stats["total_crops_generated"]) * 100 if stats["total_crops_generated"] else 0
            f.write(f"| **{s.capitalize()}** | {cnt} | {pct:.2f}% |\n")
        f.write("\n")

        f.write("## 3. Crop Geometry & Size Statistics\n\n")
        f.write(f"- **Mean Crop Width**: {mean_w:.2f} px\n")
        f.write(f"- **Mean Crop Height**: {mean_h:.2f} px\n")
        f.write(f"- **Mean Crop Area**: {mean_a:.2f} px²\n")
        f.write(f"- **Minimum Crop Area**: {min_a} px²\n")
        f.write(f"- **Maximum Crop Area**: {max_a} px²\n\n")

        f.write("## 4. Quality Checker Metrics\n\n")
        f.write("| Quality Flag | Count | Description |\n")
        f.write("| :--- | :---: | :--- |\n")
        for flag in sorted(stats["quality_flag_counts"].keys()):
            f.write(f"| `{flag}` | {stats['quality_flag_counts'][flag]} | ")
            if flag == "ok":
                f.write("Passed all size, aspect, and blur checks. |\n")
            elif flag == "too_small":
                f.write(f"Dimension below limit ({args.min_crop_size}px). |\n")
            elif flag == "extreme_aspect_ratio":
                f.write(f"Aspect ratio exceeds maximum threshold ({args.max_aspect_ratio}). |\n")
            elif flag == "blurry":
                f.write(f"Laplacian blur score below threshold ({args.blur_threshold}). |\n")
            else:
                f.write("Invalid crop structures. |\n")
        f.write("\n")

        f.write("## 5. Rare Class Audit (Classes with <= 10 crops)\n\n")
        if rare_classes_crops:
            f.write("| Remapped Class ID | Original Class ID | Crop Count |\n")
            f.write("| :---: | :---: | :---: |\n")
            for r_cls, cnt in sorted(rare_classes_crops.items()):
                o_cls = int(new_to_old.get(str(r_cls), r_cls))
                f.write(f"| {r_cls} | {o_cls} | {cnt} |\n")
        else:
            f.write("No rare classes with <= 10 crops found.\n")
        f.write("\n")

        f.write("## 6. How Crops Prepare Milestone 4 embedding-based SKU matching\n\n")
        f.write("Generating ground-truth product crops isolates the classification challenge from localization errors. ")
        f.write("These cropped image files serve as input for Milestone 4, where we will:\n")
        f.write("1. Compute feature embeddings using pretrained vision encoders (e.g. CLIP/DINOv2).\n")
        f.write("2. Create a lookup index in a Vector Database using these crop embeddings.\n")
        f.write("3. Evaluate KNN few-shot classifications on the val/test crop distributions to assess retrieval reliability.\n")

    logger.info(f"Saved Markdown report to: {report_md_path}")
    print("==================================================")
    print("Crop Generation Completed Successfully!")
    print(f"Total Crops Generated: {stats['total_crops_generated']}")
    print(f"Skipped boxes count:   {stats['total_skipped_boxes']}")
    print("==================================================")


if __name__ == "__main__":
    main()
