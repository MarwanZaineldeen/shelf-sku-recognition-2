# Deep Exploratory Data Analysis & Validation Report

## 1. Executive Summary

- **Total Images Scanned**: 1000
- **Total Label Files Scanned**: 1000
- **Total Valid Bounding Boxes**: 42450
- **Total Unique Class IDs**: 67
- **Total Data Validation Issues**: 2

## 2. Data Integrity & Validation Issues

| Issue Type | Count | Action Required |
| :--- | :---: | :--- |
| empty_label_file | 1 | Clean or inspect matching files |
| wrong_number_of_tokens | 1 | Clean or inspect matching files |

## 3. Class ID Continuity Analysis

- **Min Class ID**: 0
- **Max Class ID**: 79
- **Expected Class ID Range**: 0 to 79
- **Missing Class IDs inside Range**: 13
- **Missing IDs list**: [21, 22, 23, 37, 39, 40, 50, 51, 52, 53, 66, 69, 76]

**YOLO Implication**: YOLO expects continuous, 0-indexed integer class sequences. Class IDs contains gaps, which will cause dummy classification heads inside YOLO model config if not remapped. Class remapping is highly recommended.

## 4. Few-Shot Experiment Readiness

For crop-based SKU embedding similarity matching, we require a minimum number of instances per class. The table below outlines class support capabilities:

| threshold   |   classes_with_enough_boxes |   classes_with_enough_images |   classes_experiment_ready |   classes_not_ready |
|:------------|----------------------------:|-----------------------------:|---------------------------:|--------------------:|
| 10-shot     |                          65 |                           64 |                         64 |                   3 |
| 20-shot     |                          65 |                           62 |                         62 |                   5 |
| 50-shot     |                          62 |                           57 |                         57 |                  10 |

## 5. Bounding Box Structural Details

- **Suspiciously Small Boxes** (<= 1st percentile, area < 1239.8 px²): 425
- **Suspiciously Large Boxes** (>= 99th percentile, area > 52326.3 px²): 425
- **Extreme Aspect Ratio Boxes** (width/height < 0.4 or > 3.5): 157

## 6. Train/Validation/Test Split Readiness

Image-level random splits (70% train, 15% val, 15% test) checks show the following coverage details:
- **Validation Missing Classes**: [55, 60]
- **Test Missing Classes**: [65]

⚠️ **Warning**: The validation/test splits lack representation for some rare classes. An evaluation on this split will be unreliable for rare classes. Consider stratified splits.

## 7. Strategic Action Items

1. **Remap Class IDs**: Map class IDs to continuous [0, nc-1] integers before building data.yaml.
2. **Fix Invalid Rows**: Correct spacing errors in corrupt files like `Transmed_Autolabelling98.txt`.
3. **Address Gaps in Splits**: Switch to stratified splits for validation to prevent disappearing rare classes.
4. **Few-Shot Selection**: Restrict pilot few-shot matching tests only to the subset of classes satisfying experiment-readiness thresholds.
