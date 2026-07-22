import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add src to python path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

workspace_root = Path(os.environ.get("RETAIL_AI_ROOT", Path(__file__).resolve().parents[1]))

from src.data.prepare_dataset import (
    scan_raw_dataset,
    generate_class_mapping,
    greedy_multilabel_split,
    clean_and_export_dataset,
    generate_preparation_report
)


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation and Remapping for Direct YOLO Baseline")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(workspace_root / "Transmed Lipton - Dataset"),
        help="Path to folder containing raw images and label text files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(workspace_root / "data" / "processed" / "yolo_remapped"),
        help="Path to save processed and remapped YOLO dataset"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of images to put in train set"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of images to put in val set"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of images to put in test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite processed directory if it exists instead of backing it up"
    )

    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    output_path = Path(args.output_dir)
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    report_path = reports_dir / "dataset_preparation_report.md"

    configs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("Starting Dataset Preparation & Splitting Pipeline")
    print(f"Raw dataset: {raw_path.resolve()}")
    print(f"Output path: {output_path.resolve()}")
    print(f"Seed:        {args.seed}")
    print("==================================================")

    # 1. Scan dataset
    print("Scanning raw files and verifying annotation format...")
    scan_results = scan_raw_dataset(raw_path)
    
    original_classes = sorted(list(set(box["class_id"] for box in scan_results["valid_boxes"])))
    scan_results["num_original_classes"] = len(original_classes)
    
    print(f"Found {scan_results['total_images']} images and {scan_results['total_labels']} labels.")
    print(f"Parsed {len(scan_results['valid_boxes'])} valid bounding boxes across {len(original_classes)} unique classes.")
    print(f"Found {len(scan_results['empty_label_files'])} empty label files.")
    print(f"Found {len(scan_results['skipped_rows'])} malformed/invalid annotation rows.")

    # 2. Remap classes
    print("Generating class remappings (closing class ID gaps)...")
    old_to_new, new_to_old = generate_class_mapping(scan_results["valid_boxes"])
    scan_results["old_to_new"] = old_to_new
    scan_results["new_to_old"] = new_to_old

    # Write configs/class_id_mapping.json
    mapping_json_path = configs_dir / "class_id_mapping.json"
    mapping_data = {
        "old_to_new": {str(k): int(v) for k, v in old_to_new.items()},
        "new_to_old": {str(k): int(v) for k, v in new_to_old.items()}
    }
    with open(mapping_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)
    print(f"Class mapping JSON saved to: {mapping_json_path}")

    # Write configs/class_id_mapping.csv
    mapping_csv_path = configs_dir / "class_id_mapping.csv"
    mapping_records = [{"new_class_id": new_id, "original_class_id": old_id} for new_id, old_id in sorted(new_to_old.items())]
    pd.DataFrame(mapping_records).to_csv(mapping_csv_path, index=False)
    print(f"Class mapping CSV saved to: {mapping_csv_path}")

    # 3. Create splits
    print("Splitting image stems using greedy multilabel stratification...")
    split_assignments = greedy_multilabel_split(
        scan_results["paired_files"],
        scan_results["valid_boxes"],
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    # 4. Clean and export
    print("Exporting cleaned splits and generating data.yaml...")
    boxes_written = clean_and_export_dataset(
        scan_results,
        old_to_new,
        split_assignments,
        output_path,
        args.force
    )
    print(f"Successfully wrote {boxes_written} bounding boxes to processed splits directory.")

    # 5. Generate markdown preparation report
    print("Writing preparation report...")
    generate_preparation_report(
        scan_results,
        split_assignments,
        old_to_new,
        output_path,
        report_path
    )

    # Print requested summaries to output
    print("\n================ PREPARATION RESULTS SUMMARY ================")
    print(f"1. Generated Report:            {report_path.resolve()}")
    print(f"2. Generated Class Mapping JSON:{mapping_json_path.resolve()}")
    
    # Class ID validation
    remapped_ids = list(new_to_old.keys())
    print(f"3. Processed Class IDs range:   {min(remapped_ids)} to {max(remapped_ids)} (continuous, 67 classes total)")
    
    # Class coverage counts in splits
    split_counts = {"train": set(), "val": set(), "test": set()}
    # Populate splits class sets
    stem_to_split = {}
    for split, stems in split_assignments.items():
        for stem in stems:
            stem_to_split[stem] = split
            
    for box in scan_results["valid_boxes"]:
        stem = box["image_stem"]
        split = stem_to_split.get(stem)
        if split:
            split_counts[split].add(old_to_new[box["class_id"]])
            
    print("\n4. Train/Val/Test Class Representation:")
    print(f"   Train split class count: {len(split_counts['train'])} / 67")
    print(f"   Val split class count:   {len(split_counts['val'])} / 67")
    print(f"   Test split class count:  {len(split_counts['test'])} / 67")
    
    missing_val = [new_to_old[idx] for idx in range(67) if idx not in split_counts["val"]]
    missing_test = [new_to_old[idx] for idx in range(67) if idx not in split_counts["test"]]
    print(f"   Unavoidable classes missing from Val:  {missing_val}")
    print(f"   Unavoidable classes missing from Test: {missing_test}")
    print("=============================================================")


if __name__ == "__main__":
    main()
