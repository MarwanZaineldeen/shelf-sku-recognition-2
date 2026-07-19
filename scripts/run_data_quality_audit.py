import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.data_quality.leakage_detector import LeakageDetector, compute_md5

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Milestone 3.5: Data Quality & Leakage Audit")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/yolo_remapped",
        help="Path to preprocessed YOLO dataset"
    )
    parser.add_argument(
        "--crops-dir",
        type=str,
        default="data/processed/crops/gt",
        help="Path to generated crops output folder"
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default="configs/class_id_mapping.json",
        help="Path to class mapping config"
    )
    parser.add_argument(
        "--image-threshold",
        type=int,
        default=10,
        help="dHash Hamming distance threshold for images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/experiments/data_quality",
        help="Path to save report outputs"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing outputs"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    crops_dir = Path(args.crops_dir)
    mapping_path = Path(args.mapping_path)
    output_dir = Path(args.output_dir)

    print("==================================================")
    print("Starting Data Quality & Leakage Audit")
    print(f"Dataset Dir: {dataset_dir.resolve()}")
    print(f"Crops Dir:   {crops_dir.resolve()}")
    print(f"Output Dir:  {output_dir.resolve()}")
    print("==================================================")

    # Output dir check
    if output_dir.exists() and not args.force:
        # Check if outputs already exist
        summary_file = output_dir / "data_quality_summary.json"
        report_file = output_dir / "data_quality_report.md"
        if summary_file.exists() or report_file.exists():
            logger.error(f"Audit outputs already exist at: {output_dir}. Use --force to overwrite.")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Integrity Verification
    detector = LeakageDetector(image_threshold=args.image_threshold)
    integrity_passed, integrity_issues = detector.audit_integrity(dataset_dir, mapping_path)

    # 2. Image Leakage Detection
    logger.info("Scanning for image-level duplicate and near-duplicate leaks across splits...")
    image_leaks = detector.find_image_leaks(dataset_dir)

    # Count exact and near duplicate groups/pairs
    exact_leak_pairs = [l for l in image_leaks if l["type"] == "exact_duplicate"]
    near_leak_pairs = [l for l in image_leaks if l["type"] == "near_duplicate"]

    # 3. Crop-Level Leakage Detection
    # Fast O(N) exact duplicate crop check using MD5 of all crop files listed in metadata
    crop_metadata_csv = crops_dir / "crop_metadata.csv"
    crop_leaks = []
    
    if crop_metadata_csv.exists():
        logger.info("Reading crop metadata for split crop leakage check...")
        crop_records = []
        with open(crop_metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                crop_records.append(row)
        
        # Calculate MD5 hashes of crop files to identify exact duplicate visual signatures
        crop_hash_to_records = defaultdict(list)
        logger.info(f"Computing MD5 hashes for {len(crop_records)} crops to check cross-split leakage...")
        for r in crop_records:
            crop_abs_p = Path(__file__).resolve().parent.parent / r["crop_path"]
            if crop_abs_p.exists():
                h = compute_md5(crop_abs_p)
                if h:
                    crop_hash_to_records[h].append(r)
        
        # Check for matching visual hashes in different splits
        for crop_hash, recs in crop_hash_to_records.items():
            if len(recs) < 2:
                continue
            # Check if splits are disjoint
            splits_seen = {r["split"] for r in recs}
            if len(splits_seen) > 1:
                # Group by split to document the leak
                leak_group = defaultdict(list)
                for r in recs:
                    leak_group[r["split"]].append(r["crop_path"])
                crop_leaks.append({
                    "md5": crop_hash,
                    "splits_involved": list(splits_seen),
                    "crop_count": len(recs),
                    "locations": dict(leak_group)
                })

    # Summary calculations
    disjointness_passed = (len(image_leaks) == 0) and (len(crop_leaks) == 0)
    total_leaked_images = len({l["file_1"] for l in image_leaks} | {l["file_2"] for l in image_leaks})

    summary_data = {
        "audit_timestamp": datetime.now().isoformat(),
        "disjointness_passed": disjointness_passed,
        "exact_duplicate_groups_found": len(exact_leak_pairs),
        "near_duplicate_groups_found": len(near_leak_pairs),
        "leaked_images_count": total_leaked_images,
        "leaked_pairs": image_leaks,
        "leaked_crops_count": len(crop_leaks),
        "leaked_crops": crop_leaks,
        "integrity": {
            "passed": integrity_passed,
            "missing_label_files_count": len(integrity_issues["missing_labels"]),
            "invalid_class_ids_count": len(integrity_issues["invalid_class_ids"]),
            "mapping_mismatch": integrity_issues["mapping_mismatch"],
            "missing_label_files": integrity_issues["missing_labels"],
            "invalid_class_ids": integrity_issues["invalid_class_ids"]
        }
    }

    # Save summary JSON
    summary_path = output_dir / "data_quality_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved summary JSON report to: {summary_path}")

    # Generate Markdown Report
    report_path = output_dir / "data_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Data Quality & Leakage Audit Report\n\n")
        f.write("This document summarizes the split integrity audit for Milestone 3.5.\n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Disjointness Audit Passed**: `{'PASS' if disjointness_passed else 'FAIL'}`\n")
        f.write(f"- **Total Image Leak Pairs**: {len(image_leaks)} (Exact: {len(exact_leak_pairs)}, Near: {len(near_leak_pairs)})\n")
        f.write(f"- **Total Leaked Images**: {total_leaked_images}\n")
        f.write(f"- **Total Leaked Crops (Cross-split duplicates)**: {len(crop_leaks)}\n")
        f.write(f"- **Dataset Integrity Passed**: `{'PASS' if integrity_passed else 'FAIL'}`\n\n")

        f.write("## 2. Integrity Verification Logs\n\n")
        f.write(f"- **Missing Label Files**: {len(integrity_issues['missing_labels'])}\n")
        f.write(f"- **Invalid Remapped Class IDs**: {len(integrity_issues['invalid_class_ids'])}\n")
        f.write(f"- **Mapping Match Failures**: `{'FAIL' if integrity_issues['mapping_mismatch'] else 'PASS'}`\n\n")

        if integrity_issues["missing_labels"]:
            f.write("### Missing Labels List\n")
            for m in integrity_issues["missing_labels"]:
                f.write(f"- {m}\n")
            f.write("\n")

        if integrity_issues["invalid_class_ids"]:
            f.write("### Invalid Class IDs List\n")
            f.write("| File | Line Num | Class ID Found |\n")
            f.write("| :--- | :---: | :---: |\n")
            for m in integrity_issues["invalid_class_ids"]:
                f.write(f"| {m['file']} | {m['line']} | {m['class_id']} |\n")
            f.write("\n")

        f.write("## 3. Split Contamination & Image Leak Pairs\n\n")
        if image_leaks:
            f.write("| File 1 | Split 1 | File 2 | Split 2 | Type | Hamming Distance |\n")
            f.write("| :--- | :---: | :--- | :---: | :---: | :---: |\n")
            for leak in image_leaks:
                f.write(
                    f"| {Path(leak['file_1']).name} | {leak['split_1']} | "
                    f"{Path(leak['file_2']).name} | {leak['split_2']} | "
                    f"`{leak['type']}` | {leak['hamming_distance']} |\n"
                )
        else:
            f.write("No duplicate or near-duplicate image leaks found across split boundaries.\n")
        f.write("\n")

        f.write("## 4. Crop-Level Leakage Log\n\n")
        if crop_leaks:
            f.write("| Crop MD5 Hash | Splits Involved | Total Duplicates |\n")
            f.write("| :--- | :---: | :---: |\n")
            for cl in crop_leaks:
                f.write(f"| `{cl['md5'][:8]}...` | {', '.join(cl['splits_involved'])} | {cl['crop_count']} |\n")
        else:
            f.write("No exact duplicate crop files found leaking across splits.\n")
        f.write("\n")

        f.write("## 5. Architectural Recommendations for Milestone 4\n\n")
        if not disjointness_passed:
            f.write("> [!WARNING]\n")
            f.write("> **Split contamination was detected!** Evaluating Milestone 4 similarity classifiers ")
            f.write("on this split will result in inflated/incorrect accuracy. ")
            f.write("Before starting Milestone 4, you must rebuild dataset splits (Milestone 1.5) by ")
            f.write("removing or grouping duplicate image groups into the same splits.\n\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> **Clean Splits Verified!** The splits are completely disjoint at both the image ")
            f.write("and crop levels. Evaluations in Milestone 4 will represent true generalization performance.\n\n")

    logger.info(f"Saved Markdown report to: {report_path}")
    print("==================================================")
    print("Data Quality & Leakage Audit Completed Successfully!")
    print(f"Image leaks count: {len(image_leaks)}")
    print(f"Crop leaks count:  {len(crop_leaks)}")
    print("==================================================")


if __name__ == "__main__":
    main()
