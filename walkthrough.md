# Dataset Preparation for YOLO Walkthrough

We have successfully completed Milestone 1.5: Dataset Preparation for YOLO. The pipeline cleaned the annotations, remapped the non-contiguous class IDs into a continuous `[0, 66]` range, ran class-aware greedy stratification splitting, and generated YOLO-compatible config files.

---

## 1. Accomplishments & Files Created

- **Core Module**: [prepare_dataset.py](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/src/data/prepare_dataset.py)  
  Handles raw scanner operations, validation row cleanups, class remappings, split allocations, and data exports.
- **CLI Runner**: [prepare_dataset.py](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/scripts/prepare_dataset.py)  
  Supports CLI arguments (ratios, seeds, force overwrites) and orchestrates the dataset preparation.
- **YOLO remap directory**: [data/processed/yolo_remapped/](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/data/processed/yolo_remapped/)  
  Contains `images/` and `labels/` subfolders divided into `train/`, `val/`, and `test/` splits.
- **data.yaml configuration**: [data.yaml](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/data/processed/yolo_remapped/data.yaml)  
  Ultralytics YOLO compatible config with absolute dataset root path and 67 names entries.
- **Mapping Config files**:
  - [class_id_mapping.json](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/configs/class_id_mapping.json)
  - [class_id_mapping.csv](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/configs/class_id_mapping.csv)
- **Preparation Report**: [dataset_preparation_report.md](file:///d:/Marwan/ITI%20AI&ML/Transmid%20GP/reports/dataset_preparation_report.md)

---

## 2. Core Processing Metrics & Findings

### A. Data Cleansing
- **Malformed row skipped**: Line 1 in `Transmed_Autolabelling98.txt` containing only space characters was omitted from the processed output folder.
- **Zero-box images handled**: 
  - `Transmed-TEA-NI038.txt` (originally empty) and `Transmed_Autolabelling98.txt` (empty after filtering out the malformed line) were written as clean 0-byte `.txt` files to act as negative background training inputs in YOLO.

### B. Class ID Continuity
- **Mapped count**: 67 unique classes were successfully mapped to a continuous range from `0` to `66`.
- Mappings have been serialized in both JSON and CSV formats inside `configs/`.

### C. Split Representation
- **Split counts**:
  - **Train**: 703 images, 30,087 bounding boxes (Contains all 67 classes).
  - **Validation**: 147 images, 6,258 bounding boxes (Contains 65 classes).
  - **Test**: 150 images, 6,105 bounding boxes (Contains 65 classes).
- **Unavoidable missing classes**:
  - Classes `[55, 60]` are missing from both Validation and Test splits.
  - **Reason**: They appear in exactly 1 image each in the entire dataset. To prevent data leakage and ensure training convergence, these single-occurrence images were strictly assigned to the **Train** set, making them mathematically impossible to evaluate in Val/Test.
