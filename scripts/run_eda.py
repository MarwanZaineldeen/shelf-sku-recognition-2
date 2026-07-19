import argparse
import sys
from pathlib import Path

# Add src to python path to import eda
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.eda import DatasetValidator


def generate_markdown_report(stats, errors_count, report_path: Path, output_dir: Path):
    """
    Saves a clean Markdown report summarizing validation and EDA stats.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare markdown table for general statistics
    general_stats_table = f"""
| Metric | Value |
| :--- | :--- |
| **Total Images Scanned** | {stats['num_images']} |
| **Total Label Files Scanned** | {stats['num_label_files']} |
| **Valid Bounding Boxes** | {stats['num_valid_annotated_boxes']} |
| **Unique Class IDs** | {stats['num_unique_classes']} |
| **Mean Boxes per Image** | {stats.get('mean_boxes_per_image', 0.0):.2f} |
| **Min Boxes per Image** | {stats.get('min_boxes_per_image', 0)} |
| **Max Boxes per Image** | {stats.get('max_boxes_per_image', 0)} |
"""

    # Prepare markdown table for data validation issues
    validation_table = f"""
| Issue Type | File Count / Row Count | Action Needed |
| :--- | :--- | :--- |
| **Corrupt Images** (unreadable by OpenCV) | {stats['num_corrupt_images']} | Re-upload or discard image |
| **Missing Label Files** (image exists, no .txt) | {stats['num_missing_label_files']} | Check export pipeline or label these images |
| **Labels without Images** (.txt exists, no image) | {stats['num_labels_without_images']} | Find matching image or delete orphan labels |
| **Empty Label Files** (.txt file is empty) | {stats['num_empty_label_files']} | Check if image indeed has 0 products or label file is corrupt |
| **Invalid Annotation Rows** (formatting/value errors) | {stats['num_invalid_annotation_rows']} | Inspect and fix format error details |
| **Images with 0 Valid Bounding Boxes** | {stats['num_images_without_labels']} | Verify target products are missing or label issues |
"""

    # Class distribution details
    class_dist_rows = []
    sorted_classes = sorted(stats.get("boxes_per_class", {}).items(), key=lambda x: x[1], reverse=True)
    for class_id, count in sorted_classes:
        class_dist_rows.append(f"| Class {class_id} | {count} | {count / stats['num_valid_annotated_boxes'] * 100:.2f}% |")
    class_dist_table = "\n".join(class_dist_rows)

    # Box dimension distributions table
    w_norm = stats.get("bbox_width_norm", {})
    h_norm = stats.get("bbox_height_norm", {})
    w_px = stats.get("bbox_width_px", {})
    h_px = stats.get("bbox_height_px", {})
    a_px = stats.get("bbox_area_px", {})
    ar = stats.get("bbox_aspect_ratio", {})

    dim_stats_table = f"""
| Metric | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Width (normalized)** | {w_norm.get('mean', 0.0):.4f} | {w_norm.get('std', 0.0):.4f} | {w_norm.get('min', 0.0):.4f} | {w_norm.get('25%', 0.0):.4f} | {w_norm.get('50%', 0.0):.4f} | {w_norm.get('75%', 0.0):.4f} | {w_norm.get('max', 0.0):.4f} |
| **Height (normalized)** | {h_norm.get('mean', 0.0):.4f} | {h_norm.get('std', 0.0):.4f} | {h_norm.get('min', 0.0):.4f} | {h_norm.get('25%', 0.0):.4f} | {h_norm.get('50%', 0.0):.4f} | {h_norm.get('75%', 0.0):.4f} | {h_norm.get('max', 0.0):.4f} |
| **Width (pixels)** | {w_px.get('mean', 0.0):.2f} | {w_px.get('std', 0.0):.2f} | {w_px.get('min', 0.0):.2f} | {w_px.get('25%', 0.0):.2f} | {w_px.get('50%', 0.0):.2f} | {w_px.get('75%', 0.0):.2f} | {w_px.get('max', 0.0):.2f} |
| **Height (pixels)** | {h_px.get('mean', 0.0):.2f} | {h_px.get('std', 0.0):.2f} | {h_px.get('min', 0.0):.2f} | {h_px.get('25%', 0.0):.2f} | {h_px.get('50%', 0.0):.2f} | {h_px.get('75%', 0.0):.2f} | {h_px.get('max', 0.0):.2f} |
| **Area (pixels²)** | {a_px.get('mean', 0.0):.2f} | {a_px.get('std', 0.0):.2f} | {a_px.get('min', 0.0):.2f} | {a_px.get('25%', 0.0):.2f} | {a_px.get('50%', 0.0):.2f} | {a_px.get('75%', 0.0):.2f} | {a_px.get('max', 0.0):.2f} |
| **Aspect Ratio (W/H)** | {ar.get('mean', 0.0):.2f} | {ar.get('std', 0.0):.2f} | {ar.get('min', 0.0):.2f} | {ar.get('25%', 0.0):.2f} | {ar.get('50%', 0.0):.2f} | {ar.get('75%', 0.0):.2f} | {ar.get('max', 0.0):.2f} |
"""

    # Construct the full markdown content
    content = f"""# Dataset Exploratory Data Analysis & Validation Report

This report summarizes the findings of the automated scan and validation pipeline run on the Lipton brand retail shelf SKU recognition pilot dataset.

---

## 1. Executive Summary

- **Total issues identified**: {errors_count}
- **Status**: {"⚠️ ACTION REQUIRED" if errors_count > 0 else "✅ PASS"}
- The pilot dataset consists of **{stats['num_images']} images** containing **{stats['num_valid_annotated_boxes']} valid bounding boxes** across **{stats['num_unique_classes']} unique classes**.
- On average, each image contains **{stats.get('mean_boxes_per_image', 0.0):.2f} annotations** (ranging from {stats.get('min_boxes_per_image', 0)} to {stats.get('max_boxes_per_image', 0)}).

---

## 2. Dataset Overview Statistics

{general_stats_table}

---

## 3. Data Integrity & Validation Auditing

Each image-annotation pair was validated against file consistency rules and YOLO format standards:
1. Missing, mismatched, or empty files.
2. OpenCV image readability.
3. Box values format (exactly 5 tokens, class_id as integer, coordinates as floats in range `[0.0, 1.0]`, and strictly positive dimensions).

{validation_table}

> [!NOTE]
> A detailed line-by-line list of all errors can be found in `eda_outputs/validation_errors.csv`.

---

## 4. Class Distribution Analysis

Below is the frequency mapping of boxes per class:

| Class ID | Box Count | Percentage (%) |
| :--- | :--- | :--- |
{class_dist_table}

---

## 5. Bounding Box Distributions

Summary statistics of bounding box shapes (width, height, area, and aspect ratio):

{dim_stats_table}

*Aspect ratio is computed as width / height. An aspect ratio around 1.0 indicates square boxes, > 1.0 indicates wide boxes, and < 1.0 indicates tall/narrow boxes.*

---

## 6. Generated Visual Artifacts

The following visualization plots have been generated and saved to the `eda_outputs/` folder:

- **Class Distribution Bar Chart** ([class_distribution.png](file:///{output_dir.resolve().as_posix()}/class_distribution.png))
- **Boxes per Image Histogram** ([boxes_per_image.png](file:///{output_dir.resolve().as_posix()}/boxes_per_image.png))
- **BBox Area Distribution Histogram** ([bbox_area_distribution.png](file:///{output_dir.resolve().as_posix()}/bbox_area_distribution.png))
- **Width vs. Height Scatter Plot** ([bbox_scatter.png](file:///{output_dir.resolve().as_posix()}/bbox_scatter.png))

In addition, random annotated preview images with bounding boxes drawn can be found in the `eda_outputs/previews/` directory.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Markdown report generated successfully at: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run EDA and Validation on YOLO Dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="d:/Marwan/ITI AI&ML/Transmid GP/Transmed Lipton - Dataset",
        help="Path to the folder containing images and txt files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="d:/Marwan/ITI AI&ML/Transmid GP/eda_outputs",
        help="Path to the directory where EDA output files will be saved"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="d:/Marwan/ITI AI&ML/Transmid GP/reports/eda_report.md",
        help="Path to save the generated markdown report"
    )
    parser.add_argument(
        "--num-previews",
        type=int,
        default=5,
        help="Number of random annotated preview images to save"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    report_path = Path(args.report_path)

    print("==================================================")
    print("Starting EDA & Data Validation Pipeline")
    print(f"Dataset Dir: {dataset_path.resolve()}")
    print(f"Output Dir:  {output_path.resolve()}")
    print(f"Report Path: {report_path.resolve()}")
    print("==================================================")

    # 1. Initialize pipeline
    validator = DatasetValidator(dataset_path, output_path)

    # 2. Scan dataset
    print("Scanning dataset directory...")
    validator.scan_dataset()
    print(f"Found {len(validator.images_found)} images and {len(validator.labels_found)} label files.")

    # 3. Validate
    print("Running validation rules...")
    validator.run_validation()

    # 4. Generate statistics
    print("Calculating statistics...")
    stats = validator.compute_statistics()

    # 5. Write error logs
    print("Writing error log CSV...")
    errors_count = validator.write_error_log()

    # 6. Generate plots
    print("Creating visualization plots...")
    validator.generate_visualizations()

    # 7. Generate previews
    print(f"Generating {args.num_previews} random previews with drawn bounding boxes...")
    validator.generate_previews(args.num_previews)

    # 8. Generate Markdown Report
    print("Generating report markdown...")
    generate_markdown_report(stats, errors_count, report_path, output_path)

    print("\nEDA Pipeline Completed Successfully!")
    print(f"Total invalid rows or integrity issues detected: {errors_count}")
    print(f"Review findings in the markdown report: {report_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
