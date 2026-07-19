# Ground-Truth Crop Generation Report

This document summarizes the execution results of Milestone 3: Crop Generation Pipeline.

## 1. Executive Metrics

- **Total Crops Generated**: 42450
- **Total Skipped Boxes**: 0
- **Empty-Label Images Handled**: 2
- **Malformed-Label Rows Skipped**: 0
- **Invalid Coordinate Rows Skipped**: 0

## 2. Crops per Split

| Split | Crop Count | Percentage |
| :--- | :---: | :---: |
| **Test** | 6105 | 14.38% |
| **Train** | 30087 | 70.88% |
| **Val** | 6258 | 14.74% |

## 3. Crop Geometry & Size Statistics

- **Mean Crop Width**: 100.65 px
- **Mean Crop Height**: 95.40 px
- **Mean Crop Area**: 11386.90 px²
- **Minimum Crop Area**: 296 px²
- **Maximum Crop Area**: 243036 px²

## 4. Quality Checker Metrics

| Quality Flag | Count | Description |
| :--- | :---: | :--- |
| `extreme_aspect_ratio` | 1 | Aspect ratio exceeds maximum threshold (5.0). |
| `ok` | 42413 | Passed all size, aspect, and blur checks. |
| `too_small` | 36 | Dimension below limit (20px). |

## 5. Rare Class Audit (Classes with <= 10 crops)

| Remapped Class ID | Original Class ID | Crop Count |
| :---: | :---: | :---: |
| 45 | 55 | 4 |
| 50 | 60 | 2 |

## 6. How Crops Prepare Milestone 4 embedding-based SKU matching

Generating ground-truth product crops isolates the classification challenge from localization errors. These cropped image files serve as input for Milestone 4, where we will:
1. Compute feature embeddings using pretrained vision encoders (e.g. CLIP/DINOv2).
2. Create a lookup index in a Vector Database using these crop embeddings.
3. Evaluate KNN few-shot classifications on the val/test crop distributions to assess retrieval reliability.
