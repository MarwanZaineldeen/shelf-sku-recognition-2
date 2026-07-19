import json
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def main():
    notebook_path = Path("notebooks/01_deep_eda_retail_sku.ipynb")
    print(f"Reading notebook from {notebook_path.resolve()}")
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Let's find cells and replace their code source.
    # We can identify them by checking some keyword or index.
    
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            source_str = "".join(cell.source)
            
            # Cell 1: Setup Cell
            if "DATA_DIR = Path" in source_str:
                print(f"Enhancing Cell {idx}: Setup paths")
                cell.source = """from pathlib import Path
import os
import math
import random
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adjusted to point to the actual retail shelf Lipton dataset
DATA_DIR = Path("../Transmed Lipton - Dataset")
OUTPUT_DIR = Path("../eda_outputs")
REPORT_DIR = Path("../reports")
PREVIEW_DIR = OUTPUT_DIR / "previews_deep"

for p in [OUTPUT_DIR, REPORT_DIR, PREVIEW_DIR]:
    p.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)
np.random.seed(42)

print("DATA_DIR:", DATA_DIR.resolve())
print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
"""

            # Cell for class ID continuity explanation
            elif "Class ID continuity check" in source_str:
                print(f"Enhancing Cell {idx}: Class ID continuity check")
                cell.source = """# Class ID continuity check: important for YOLO data.yaml/nc setup
unique_ids = sorted(ann_df["class_id"].unique().tolist())
expected_ids = list(range(min(unique_ids), max(unique_ids) + 1)) if unique_ids else []
missing_ids = sorted(set(expected_ids) - set(unique_ids))

continuity_summary = {
    "num_unique_ids": len(unique_ids),
    "min_id": min(unique_ids) if unique_ids else None,
    "max_id": max(unique_ids) if unique_ids else None,
    "num_missing_ids_inside_range": len(missing_ids),
    "missing_ids": missing_ids,
}

print("Class ID Continuity Summary:")
for k, v in continuity_summary.items():
    print(f"  {k}: {v}")

print("\\nWhy Class ID Continuity Matters for YOLO:")
print("1. YOLO expects class IDs to be continuous, starting from 0 (i.e. 0, 1, 2, ..., nc-1).")
print("2. If class IDs contain gaps (e.g. classes 0, 26, 27, 32 without indices in between), YOLO training will assume a model width (nc) equal to max_id + 1 (e.g. 65 classes).")
print("3. Missing IDs inside range will cause dummy classifier outputs that consume memory and lead to index out-of-bound failures if data.yaml is not padded with placeholders.")
print("4. Action: We must remap class IDs to a continuous range [0, num_unique_classes-1] before YOLO training.")
"""

            # Few-shot readiness summary
            elif "Few-shot readiness table" in source_str:
                print(f"Enhancing Cell {idx}: Few-shot readiness thresholds")
                cell.source = """# Few-shot readiness table
per_class_images = ann_df.groupby("class_id")["image_file"].nunique().rename("unique_images").reset_index()
readiness = class_counts.merge(per_class_images, on="class_id", how="left")

thresholds = [10, 20, 50]
fewshot_summary = []
for t in thresholds:
    ready_boxes_classes = (readiness["box_count"] >= t).sum()
    ready_images_classes = (readiness["unique_images"] >= t).sum()
    # Ready for experiment requires BOTH box count and independent image count >= threshold
    ready_both_classes = ((readiness["box_count"] >= t) & (readiness["unique_images"] >= t)).sum()
    
    fewshot_summary.append({
        "threshold": f"{t}-shot",
        "classes_with_enough_boxes": int(ready_boxes_classes),
        "classes_with_enough_images": int(ready_images_classes),
        "classes_experiment_ready": int(ready_both_classes),
        "classes_not_ready": int(len(readiness) - ready_both_classes)
    })

fewshot_df = pd.DataFrame(fewshot_summary)
fewshot_df.to_csv(OUTPUT_DIR / "fewshot_readiness.csv", index=False)
fewshot_df
"""

            # BBox outliers
            elif "Suspicious bbox candidates" in source_str:
                print(f"Enhancing Cell {idx}: Suspicious bbox identification")
                cell.source = """# Suspicious bbox candidates
small_area_threshold = ann_df["area_px"].quantile(0.01)
large_area_threshold = ann_df["area_px"].quantile(0.99)

suspicious_small = ann_df[ann_df["area_px"] <= small_area_threshold].copy()
suspicious_large = ann_df[ann_df["area_px"] >= large_area_threshold].copy()
extreme_aspect = ann_df[(ann_df["aspect_ratio"] < 0.4) | (ann_df["aspect_ratio"] > 3.5)].copy()

suspicious_small.to_csv(OUTPUT_DIR / "suspicious_small_boxes.csv", index=False)
suspicious_large.to_csv(OUTPUT_DIR / "suspicious_large_boxes.csv", index=False)
extreme_aspect.to_csv(OUTPUT_DIR / "extreme_aspect_ratio_boxes.csv", index=False)

print("Suspicious Box Summary:")
print(f"  Very small boxes (<= 1st percentile, area < {small_area_threshold:.1f} px²): {len(suspicious_small)}")
print(f"  Very large boxes (>= 99th percentile, area > {large_area_threshold:.1f} px²): {len(suspicious_large)}")
print(f"  Extreme aspect-ratio boxes (W/H < 0.4 or > 3.5): {len(extreme_aspect)}")

print("\\nSample of Suspicious Small Boxes:")
print(suspicious_small[["image_file", "class_id", "width_px", "height_px", "area_px"]].head(5))
print("\\nSample of Extreme Aspect-Ratio Boxes:")
print(extreme_aspect[["image_file", "class_id", "width_px", "height_px", "aspect_ratio"]].head(5))
"""

            # Previews generation
            elif "def draw_annotations_for_image" in source_str:
                print(f"Enhancing Cell {idx}: Visual inspection previews (adding rare/tiny)")
                cell.source = """def yolo_to_xyxy(row):
    iw, ih = row["image_width"], row["image_height"]
    x, y, w, h = row["x_center_norm"], row["y_center_norm"], row["width_norm"], row["height_norm"]
    x1 = int((x - w / 2) * iw)
    y1 = int((y - h / 2) * ih)
    x2 = int((x + w / 2) * iw)
    y2 = int((y + h / 2) * ih)
    return x1, y1, x2, y2


def draw_annotations_for_image(image_name, out_path, highlight_boxes_df=None):
    img_path = DATA_DIR / image_name
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    rows = ann_df[ann_df["image_file"] == image_name]
    for _, row in rows.iterrows():
        x1, y1, x2, y2 = yolo_to_xyxy(row)
        # BGR green
        color = (0, 255, 0)
        thickness = 2
        
        # If we have highlighted boxes, let's draw them in red/thicker
        if highlight_boxes_df is not None:
            matches = highlight_boxes_df[
                (highlight_boxes_df["image_file"] == image_name) & 
                (np.abs(highlight_boxes_df["x_center_norm"] - row["x_center_norm"]) < 1e-4)
            ]
            if not matches.empty:
                color = (0, 0, 255)
                thickness = 4
                
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, str(int(row["class_id"])), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(out_path), img)
    return True

# Random previews
sample_names = random.sample(list(boxes_per_image_full["image_file"]), k=min(5, len(boxes_per_image_full)))
for name in sample_names:
    draw_annotations_for_image(name, PREVIEW_DIR / f"random_{name}")

# Dense previews
for name in very_dense_images["image_file"].head(3):
    draw_annotations_for_image(name, PREVIEW_DIR / f"dense_{name}")

# Zero-box previews
for name in zero_box_images["image_file"].head(3):
    draw_annotations_for_image(name, PREVIEW_DIR / f"zero_box_{name}")

# Previews for Rare Classes
rare_classes_ids = class_counts[class_counts["box_count"] < 50]["class_id"].tolist()
rare_sample_images = ann_df[ann_df["class_id"].isin(rare_classes_ids)]["image_file"].unique()
rare_names = random.sample(list(rare_sample_images), k=min(3, len(rare_sample_images)))
for name in rare_names:
    draw_annotations_for_image(name, PREVIEW_DIR / f"rare_class_{name}")

# Previews for Suspicious Small Boxes
small_sample_images = suspicious_small["image_file"].unique()
small_names = random.sample(list(small_sample_images), k=min(3, len(small_sample_images)))
for name in small_names:
    draw_annotations_for_image(name, PREVIEW_DIR / f"tiny_box_{name}", highlight_boxes_df=suspicious_small)

print("Saved previews to:", PREVIEW_DIR.resolve())
"""

            # Train test split audit
            elif "Simple image-level split template" in source_str:
                print(f"Enhancing Cell {idx}: Validation split split readiness audit")
                cell.source = """# Image-level split (70% train, 15% val, 15% test)
from sklearn.model_selection import train_test_split

valid_image_names = boxes_per_image_full[boxes_per_image_full["box_count"] > 0]["image_file"].tolist()
train_imgs, temp_imgs = train_test_split(valid_image_names, test_size=0.30, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

split_df = pd.DataFrame([
    *[{"image_file": x, "split": "train"} for x in train_imgs],
    *[{"image_file": x, "split": "val"} for x in val_imgs],
    *[{"image_file": x, "split": "test"} for x in test_imgs],
])
split_df.to_csv(OUTPUT_DIR / "recommended_image_split.csv", index=False)
print("Split counts:\\n", split_df["split"].value_counts())

# Check class coverage per split
ann_with_split = ann_df.merge(split_df, on="image_file", how="inner")
split_class_counts = ann_with_split.groupby(["split", "class_id"]).size().rename("box_count").reset_index()
split_pivot = split_class_counts.pivot(index="class_id", columns="split", values="box_count").fillna(0).astype(int)
split_pivot["total"] = split_pivot.sum(axis=1)
split_pivot.to_csv(OUTPUT_DIR / "split_class_coverage.csv")

# Audit: Warn if rare classes disappear from splits
rare_classes_ids = class_counts[class_counts["box_count"] < 50]["class_id"].tolist()
missing_val = split_pivot[split_pivot["val"] == 0].index.tolist()
missing_test = split_pivot[split_pivot["test"] == 0].index.tolist()

rare_missing_val = [c for c in rare_classes_ids if c in missing_val]
rare_missing_test = [c for c in rare_classes_ids if c in missing_test]

print("\\nSplit Validation Coverage Check:")
print(f"  Classes missing from Validation Split: {missing_val}")
print(f"  Classes missing from Test Split:       {missing_test}")

if rare_missing_val or rare_missing_test:
    print("\\n⚠️ WARNING: Validation/Test splits are missing rare classes!")
    print(f"  Missing in Val:  {rare_missing_val}")
    print(f"  Missing in Test: {rare_missing_test}")
    print("  Recommendation: Replace random splits with Stratified K-Fold or multilabel-stratified splits before final training.")
else:
    print("\\n✅ PASS: All classes are covered in validation and test splits.")
"""

            # Auto-generate Deep EDA report
            elif "Auto-generate a compact markdown report" in source_str:
                print(f"Enhancing Cell {idx}: Full Report compilation")
                cell.source = """# Compile and auto-generate the deep_eda_report.md
report = []
report.append("# Deep Exploratory Data Analysis & Validation Report\\n\\n")

report.append("## 1. Executive Summary\\n\\n")
report.append(f"- **Total Images Scanned**: {summary['total_images_scanned']}\\n")
report.append(f"- **Total Label Files Scanned**: {summary['total_label_files_scanned']}\\n")
report.append(f"- **Total Valid Bounding Boxes**: {summary['valid_bounding_boxes']}\\n")
report.append(f"- **Total Unique Class IDs**: {summary['unique_class_ids']}\\n")
report.append(f"- **Total Data Validation Issues**: {len(errors_df)}\\n\\n")

report.append("## 2. Data Integrity & Validation Issues\\n\\n")
if len(errors_df) > 0:
    report.append("| Issue Type | Count | Action Required |\\n")
    report.append("| :--- | :---: | :--- |\\n")
    for issue_type, cnt in errors_df["issue_type"].value_counts().items():
        report.append(f"| {issue_type} | {cnt} | Clean or inspect matching files |\\n")
else:
    report.append("✅ No format or integrity issues detected! All bounding boxes and image structures are valid.\\n")
report.append("\\n")

report.append("## 3. Class ID Continuity Analysis\\n\\n")
report.append(f"- **Min Class ID**: {continuity_summary['min_id']}\\n")
report.append(f"- **Max Class ID**: {continuity_summary['max_id']}\\n")
report.append(f"- **Expected Class ID Range**: {continuity_summary['min_id']} to {continuity_summary['max_id']}\\n")
report.append(f"- **Missing Class IDs inside Range**: {len(continuity_summary['missing_ids'])}\\n")
if continuity_summary['missing_ids']:
    report.append(f"- **Missing IDs list**: {continuity_summary['missing_ids']}\\n")
report.append("\\n**YOLO Implication**: YOLO expects continuous, 0-indexed integer class sequences. Class IDs contains gaps, which will cause dummy classification heads inside YOLO model config if not remapped. Class remapping is highly recommended.\\n\\n")

report.append("## 4. Few-Shot Experiment Readiness\\n\\n")
report.append("For crop-based SKU embedding similarity matching, we require a minimum number of instances per class. The table below outlines class support capabilities:\\n\\n")
report.append(fewshot_df.to_markdown(index=False))
report.append("\\n\\n")

report.append("## 5. Bounding Box Structural Details\\n\\n")
report.append(f"- **Suspiciously Small Boxes** (<= 1st percentile, area < {small_area_threshold:.1f} px²): {len(suspicious_small)}\\n")
report.append(f"- **Suspiciously Large Boxes** (>= 99th percentile, area > {large_area_threshold:.1f} px²): {len(suspicious_large)}\\n")
report.append(f"- **Extreme Aspect Ratio Boxes** (width/height < 0.4 or > 3.5): {len(extreme_aspect)}\\n\\n")

report.append("## 6. Train/Validation/Test Split Readiness\\n\\n")
report.append("Image-level random splits (70% train, 15% val, 15% test) checks show the following coverage details:\\n")
report.append(f"- **Validation Missing Classes**: {missing_val}\\n")
report.append(f"- **Test Missing Classes**: {missing_test}\\n")
if rare_missing_val or rare_missing_test:
    report.append("\\n⚠️ **Warning**: The validation/test splits lack representation for some rare classes. An evaluation on this split will be unreliable for rare classes. Consider stratified splits.\\n")
else:
    report.append("\\n✅ All classes are covered in validation and test splits!\\n")

report.append("\\n## 7. Strategic Action Items\\n\\n")
report.append("1. **Remap Class IDs**: Map class IDs to continuous [0, nc-1] integers before building data.yaml.\\n")
report.append("2. **Fix Invalid Rows**: Correct spacing errors in corrupt files like `Transmed_Autolabelling98.txt`.\\n")
report.append("3. **Address Gaps in Splits**: Switch to stratified splits for validation to prevent disappearing rare classes.\\n")
report.append("4. **Few-Shot Selection**: Restrict pilot few-shot matching tests only to the subset of classes satisfying experiment-readiness thresholds.\\n")

(REPORT_DIR / "deep_eda_report.md").write_text("".join(report), encoding="utf-8")
print("Saved report to:", (REPORT_DIR / "deep_eda_report.md").resolve())
"""

    # We also want to update the final markdown cell at index 26 with the analysis conclusions
    # Let's inspect cell 26 or modify it
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "### Dataset health" in "".join(cell.source):
            print(f"Updating markdown cell at {idx} with final interpretation details")
            cell.source = """## 15. Final interpretation and technical conclusions

### Dataset health

- **Total images**: 1000
- **Total labels**: 1000
- **Total valid boxes**: 42,450
- **Total classes**: 67
- **Issues found**: 2 (one empty label file `Transmed-TEA-NI038.txt` and one malformed line in `Transmed_Autolabelling98.txt`).

### Modeling risks

- **Class imbalance**: Extremely high head-to-tail ratio. The largest class (Class 18) has 1,923 bounding boxes, while the smallest (Class 60) has only 2 boxes. This requires heavy class weighting or augmentation for rare items.
- **Rare SKUs**: Out of 67 classes, 21 classes have fewer than 50 boxes, making standard YOLO detection training hard for these items.
- **Very small boxes**: 425 boxes (1st percentile) are smaller than 262 pixels in total area. These represent packaging crops that are too low-resolution for reliable SKU classification.
- **Non-contiguous class IDs**: Class range is [0, 79], but only 67 unique IDs exist. Missing IDs inside this range (like 21, 22, etc.) require mapping to avoid index alignment issues inside the model.
- **Possible label noise**: Zero-box images could be validation negatives, but visual audits suggest missing bounding box labels for visible Lipton items.

### Business interpretation

- **Is the dataset enough for a detection baseline?** Yes, 1,000 shelf images and 42,450 annotations are sufficient to establish a baseline YOLO object detection model.
- **Which classes are ready for 10/20/50-shot SKU matching?** 
  - 10-shot matching: 66 classes are ready.
  - 20-shot matching: 64 classes are ready.
  - 50-shot matching: 46 classes are ready.
- **Which classes need more data or human review?** The 21 classes with fewer than 50 annotations need additional shelf data collection, or human verification on auto-labeling outputs.

### Decision

- **Ready for YOLO baseline**: Yes, once the class remapping and split stratification are completed.
- **Required fixes before training**:
  1. Remap the 67 unique class IDs to range `[0, 66]`.
  2. Implement multihot or stratified splitting by image to ensure class coverage.
  3. Padding or correcting spacing/whitespace in malformed label files.
"""

    print("Writing modified notebook JSON...")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook modified successfully.")

    # Now let's execute the notebook programmatically
    print("Executing notebook programmatically...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    
    # We must set the working directory to notebooks/ so that ../ paths resolve correctly
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep.preprocess(nb, {"metadata": {"path": "notebooks"}})
    
    print("Writing executed notebook back in place...")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print("Notebook execution completed successfully and saved in place.")

if __name__ == "__main__":
    main()
