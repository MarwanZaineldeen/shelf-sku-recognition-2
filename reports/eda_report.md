# Dataset Exploratory Data Analysis & Validation Report

This report summarizes the findings of the automated scan and validation pipeline run on the Lipton brand retail shelf SKU recognition pilot dataset.

---

## 1. Executive Summary

- **Total issues identified**: 2
- **Status**: ⚠️ ACTION REQUIRED
- The pilot dataset consists of **1000 images** containing **42450 valid bounding boxes** across **67 unique classes**.
- On average, each image contains **42.45 annotations** (ranging from 0 to 227).

---

## 2. Dataset Overview Statistics


| Metric | Value |
| :--- | :--- |
| **Total Images Scanned** | 1000 |
| **Total Label Files Scanned** | 1000 |
| **Valid Bounding Boxes** | 42450 |
| **Unique Class IDs** | 67 |
| **Mean Boxes per Image** | 42.45 |
| **Min Boxes per Image** | 0 |
| **Max Boxes per Image** | 227 |


---

## 3. Data Integrity & Validation Auditing

Each image-annotation pair was validated against file consistency rules and YOLO format standards:
1. Missing, mismatched, or empty files.
2. OpenCV image readability.
3. Box values format (exactly 5 tokens, class_id as integer, coordinates as floats in range `[0.0, 1.0]`, and strictly positive dimensions).


| Issue Type | File Count / Row Count | Action Needed |
| :--- | :--- | :--- |
| **Corrupt Images** (unreadable by OpenCV) | 0 | Re-upload or discard image |
| **Missing Label Files** (image exists, no .txt) | 0 | Check export pipeline or label these images |
| **Labels without Images** (.txt exists, no image) | 0 | Find matching image or delete orphan labels |
| **Empty Label Files** (.txt file is empty) | 1 | Check if image indeed has 0 products or label file is corrupt |
| **Invalid Annotation Rows** (formatting/value errors) | 1 | Inspect and fix format error details |
| **Images with 0 Valid Bounding Boxes** | 2 | Verify target products are missing or label issues |


> [!NOTE]
> A detailed line-by-line list of all errors can be found in `eda_outputs/validation_errors.csv`.

---

## 4. Class Distribution Analysis

Below is the frequency mapping of boxes per class:

| Class ID | Box Count | Percentage (%) |
| :--- | :--- | :--- |
| Class 18 | 1923 | 4.53% |
| Class 30 | 1575 | 3.71% |
| Class 14 | 1526 | 3.59% |
| Class 15 | 1491 | 3.51% |
| Class 8 | 1485 | 3.50% |
| Class 31 | 1461 | 3.44% |
| Class 17 | 1353 | 3.19% |
| Class 16 | 1275 | 3.00% |
| Class 2 | 1161 | 2.73% |
| Class 54 | 1083 | 2.55% |
| Class 38 | 1063 | 2.50% |
| Class 36 | 1054 | 2.48% |
| Class 11 | 1042 | 2.45% |
| Class 24 | 1006 | 2.37% |
| Class 35 | 983 | 2.32% |
| Class 75 | 979 | 2.31% |
| Class 10 | 963 | 2.27% |
| Class 33 | 951 | 2.24% |
| Class 1 | 943 | 2.22% |
| Class 34 | 939 | 2.21% |
| Class 3 | 918 | 2.16% |
| Class 26 | 863 | 2.03% |
| Class 58 | 842 | 1.98% |
| Class 57 | 834 | 1.96% |
| Class 13 | 790 | 1.86% |
| Class 64 | 720 | 1.70% |
| Class 20 | 704 | 1.66% |
| Class 29 | 691 | 1.63% |
| Class 4 | 653 | 1.54% |
| Class 62 | 627 | 1.48% |
| Class 63 | 610 | 1.44% |
| Class 44 | 578 | 1.36% |
| Class 59 | 537 | 1.27% |
| Class 28 | 524 | 1.23% |
| Class 19 | 512 | 1.21% |
| Class 45 | 503 | 1.18% |
| Class 61 | 493 | 1.16% |
| Class 25 | 493 | 1.16% |
| Class 6 | 485 | 1.14% |
| Class 27 | 465 | 1.10% |
| Class 71 | 432 | 1.02% |
| Class 0 | 425 | 1.00% |
| Class 72 | 410 | 0.97% |
| Class 70 | 400 | 0.94% |
| Class 68 | 354 | 0.83% |
| Class 5 | 335 | 0.79% |
| Class 41 | 309 | 0.73% |
| Class 67 | 309 | 0.73% |
| Class 43 | 298 | 0.70% |
| Class 32 | 290 | 0.68% |
| Class 42 | 274 | 0.65% |
| Class 48 | 265 | 0.62% |
| Class 47 | 261 | 0.61% |
| Class 46 | 212 | 0.50% |
| Class 7 | 118 | 0.28% |
| Class 73 | 104 | 0.24% |
| Class 78 | 101 | 0.24% |
| Class 74 | 96 | 0.23% |
| Class 9 | 74 | 0.17% |
| Class 77 | 74 | 0.17% |
| Class 65 | 65 | 0.15% |
| Class 49 | 52 | 0.12% |
| Class 12 | 40 | 0.09% |
| Class 79 | 27 | 0.06% |
| Class 56 | 21 | 0.05% |
| Class 55 | 4 | 0.01% |
| Class 60 | 2 | 0.00% |

---

## 5. Bounding Box Distributions

Summary statistics of bounding box shapes (width, height, area, and aspect ratio):


| Metric | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Width (normalized)** | 0.0794 | 0.0330 | 0.0054 | 0.0560 | 0.0750 | 0.0923 | 0.3632 |
| **Height (normalized)** | 0.0571 | 0.0271 | 0.0131 | 0.0407 | 0.0536 | 0.0662 | 0.4289 |
| **Width (pixels)** | 91.58 | 48.36 | 7.72 | 57.00 | 78.00 | 117.71 | 514.00 |
| **Height (pixels)** | 86.74 | 45.82 | 17.00 | 55.00 | 73.36 | 111.18 | 537.43 |
| **Area (pixels²)** | 9421.05 | 10625.42 | 262.15 | 3045.00 | 5823.99 | 11925.72 | 201058.64 |
| **Aspect Ratio (W/H)** | 1.54 | 0.69 | 0.22 | 0.97 | 1.38 | 1.91 | 4.83 |


*Aspect ratio is computed as width / height. An aspect ratio around 1.0 indicates square boxes, > 1.0 indicates wide boxes, and < 1.0 indicates tall/narrow boxes.*

---

## 6. Generated Visual Artifacts

The following visualization plots have been generated and saved to the `eda_outputs/` folder:

- **Class Distribution Bar Chart** ([class_distribution.png](file:///D:/Marwan/ITI AI&ML/Transmid GP/eda_outputs/class_distribution.png))
- **Boxes per Image Histogram** ([boxes_per_image.png](file:///D:/Marwan/ITI AI&ML/Transmid GP/eda_outputs/boxes_per_image.png))
- **BBox Area Distribution Histogram** ([bbox_area_distribution.png](file:///D:/Marwan/ITI AI&ML/Transmid GP/eda_outputs/bbox_area_distribution.png))
- **Width vs. Height Scatter Plot** ([bbox_scatter.png](file:///D:/Marwan/ITI AI&ML/Transmid GP/eda_outputs/bbox_scatter.png))

In addition, random annotated preview images with bounding boxes drawn can be found in the `eda_outputs/previews/` directory.
