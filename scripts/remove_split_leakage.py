import argparse
import csv
import json
import logging
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple

# Add root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_duplicate_families(leaked_pairs: list, all_images: list) -> Tuple[dict, dict]:
    """
    Finds connected components of duplicate shelf images using BFS.
    Returns:
        image_to_family: dict mapping image filename -> family_id
        families: dict mapping family_id -> list of image filenames
    """
    adj = defaultdict(set)
    for pair in leaked_pairs:
        name1 = Path(pair["file_1"]).name
        name2 = Path(pair["file_2"]).name
        adj[name1].add(name2)
        adj[name2].add(name1)

    visited = set()
    families = {}
    image_to_family = {}
    family_id_counter = 1

    # Traverse all images to build families (singletons also form a family of size 1)
    for img_name in all_images:
        if img_name in visited:
            continue
        
        # Run BFS to extract component
        component = []
        queue = deque([img_name])
        visited.add(img_name)

        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Save family details
        family_id = f"family_{family_id_counter}"
        families[family_id] = component
        for node in component:
            image_to_family[node] = family_id
        
        family_id_counter += 1

    return image_to_family, families


def main():
    parser = argparse.ArgumentParser(description="Milestone 3.5: Automatic Data Leakage Removal Preprocessing")
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
        help="Path to crops directory"
    )
    parser.add_argument(
        "--audit-summary",
        type=str,
        default="reports/experiments/data_quality/data_quality_summary.json",
        help="Path to leakage summary JSON"
    )
    parser.add_argument(
        "--clean-dataset-dir",
        type=str,
        default="data/processed/yolo_remapped_clean",
        help="Target folder for clean dataset"
    )
    parser.add_argument(
        "--clean-crops-dir",
        type=str,
        default="data/processed/crops/gt_clean",
        help="Target folder for clean crops"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing cleaned outputs"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    crops_dir = Path(args.crops_dir)
    summary_path = Path(args.audit_summary)
    clean_dataset_dir = Path(args.clean_dataset_dir)
    clean_crops_dir = Path(args.clean_crops_dir)

    print("==================================================")
    print("Automatic Leakage Removal Pipeline")
    print(f"Dataset Dir:       {dataset_dir.resolve()}")
    print(f"Audit Summary:     {summary_path.resolve()}")
    print(f"Clean Dataset Dir: {clean_dataset_dir.resolve()}")
    print(f"Clean Crops Dir:   {clean_crops_dir.resolve()}")
    print("==================================================")

    # Validate inputs
    if not dataset_dir.exists():
        logger.error(f"Input dataset folder missing: {dataset_dir}")
        sys.exit(1)
    if not summary_path.exists():
        logger.error(f"Audit summary missing at: {summary_path}. Run audit script first.")
        sys.exit(1)

    # Check target folders
    if clean_dataset_dir.exists() or clean_crops_dir.exists():
        if args.force:
            logger.info("Overwriting existing clean dataset and crops folders...")
            if clean_dataset_dir.exists():
                shutil.rmtree(clean_dataset_dir)
            if clean_crops_dir.exists():
                shutil.rmtree(clean_crops_dir)
        else:
            logger.error("Clean directories already exist. Use --force to overwrite.")
            sys.exit(1)

    clean_dataset_dir.mkdir(parents=True, exist_ok=True)
    clean_crops_dir.mkdir(parents=True, exist_ok=True)

    # Load summary JSON
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    leaked_pairs = summary.get("leaked_pairs", [])

    # Map image filename -> split
    image_to_split = {}
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    splits = ["train", "val", "test"]

    for split in splits:
        img_dir = dataset_dir / "images" / split
        if img_dir.exists():
            for p in img_dir.iterdir():
                if p.suffix.lower() in supported_exts:
                    image_to_split[p.name] = split

    all_images = sorted(list(image_to_split.keys()))

    # Build duplicate families
    logger.info(f"Grouping duplicate images into families...")
    image_to_family, families = build_duplicate_families(leaked_pairs, all_images)

    # Resolve target splits for families
    family_target_splits = {}
    migration_log = []

    for fam_id, members in families.items():
        if len(members) == 1:
            # Singleton: remains in original split
            family_target_splits[fam_id] = image_to_split[members[0]]
            continue

        # Duplicates family: determine best target split
        # We count frequency of splits in members
        split_counts = defaultdict(int)
        for m in members:
            split_counts[image_to_split[m]] += 1
        
        # Determinisitc priority order Train > Val > Test for ties
        best_split = max(splits, key=lambda s: (split_counts[s], -splits.index(s)))
        family_target_splits[fam_id] = best_split

        # Document migration
        for m in members:
            orig_s = image_to_split[m]
            migration_log.append({
                "image_name": m,
                "family_id": fam_id,
                "original_split": orig_s,
                "clean_split": best_split,
                "status": "moved" if orig_s != best_split else "retained",
                "family_members": members
            })

    # Save migration log JSON
    migration_log_path = output_dir = Path("reports/experiments/data_quality")
    migration_log_path.mkdir(parents=True, exist_ok=True)
    with open(migration_log_path / "split_migration_log.json", "w", encoding="utf-8") as f:
        json.dump(migration_log, f, indent=2)
    logger.info("Saved detailed migration log JSON.")

    # Create directories in clean dataset
    for split in splits:
        (clean_dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (clean_dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files to clean dataset folders
    logger.info("Copying split files to cleaned dataset structure...")
    for img_name, orig_split in image_to_split.items():
        fam_id = image_to_family[img_name]
        target_split = family_target_splits[fam_id]

        # Source paths
        src_img = dataset_dir / "images" / orig_split / img_name
        src_lbl = dataset_dir / "labels" / orig_split / f"{Path(img_name).stem}.txt"

        # Destination paths
        dst_img = clean_dataset_dir / "images" / target_split / img_name
        dst_lbl = clean_dataset_dir / "labels" / target_split / f"{Path(img_name).stem}.txt"

        shutil.copy2(src_img, dst_img)
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)

    # Copy and update data.yaml
    src_yaml = dataset_dir / "data.yaml"
    dst_yaml = clean_dataset_dir / "data.yaml"
    if src_yaml.exists():
        with open(src_yaml, "r", encoding="utf-8") as f:
            yaml_lines = f.readlines()
        with open(dst_yaml, "w", encoding="utf-8") as f:
            for line in yaml_lines:
                if line.startswith("path:"):
                    f.write(f"path: {clean_dataset_dir.resolve().as_posix()}\n")
                elif line.startswith("train:"):
                    f.write("train: images/train\n")
                elif line.startswith("val:"):
                    f.write("val: images/val\n")
                elif line.startswith("test:"):
                    f.write("test: images/test\n")
                else:
                    f.write(line)
        logger.info("Generated updated clean data.yaml.")

    # Migrate crop images and metadata
    crop_metadata_csv = crops_dir / "crop_metadata.csv"
    clean_crop_metadata_csv = clean_crops_dir / "crop_metadata.csv"
    
    clean_crop_records = []
    
    if crop_metadata_csv.exists():
        logger.info("Migrating crop image files and rewriting metadata...")
        with open(crop_metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["source_image_name"]
                fam_id = image_to_family[img_name]
                target_split = family_target_splits[fam_id]
                remapped_cls = row["remapped_class_id"]

                # Source crop filepath
                src_crop_path = Path(__file__).resolve().parent.parent / row["crop_path"]
                
                # Target crop filepath
                crop_filename = Path(row["crop_path"]).name
                dst_crop_dir = clean_crops_dir / target_split / f"class_{remapped_cls}"
                dst_crop_dir.mkdir(parents=True, exist_ok=True)
                dst_crop_path = dst_crop_dir / crop_filename

                # Copy crop file on disk (fast O(N) copy)
                if src_crop_path.exists():
                    shutil.copy2(src_crop_path, dst_crop_path)
                
                # Update row details
                row["split"] = target_split
                row["crop_path"] = f"data/processed/crops/gt_clean/{target_split}/class_{remapped_cls}/{crop_filename}"
                row["source_image_path"] = f"data/processed/yolo_remapped_clean/images/{target_split}/{img_name}"
                clean_crop_records.append(row)

        if clean_crop_records:
            keys = clean_crop_records[0].keys()
            with open(clean_crop_metadata_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(clean_crop_records)
            logger.info("Rewrote cleaned crops metadata catalog.")

    # Print split stats before/after
    # Before stats
    before_img_counts = defaultdict(int)
    for orig_s in image_to_split.values():
        before_img_counts[orig_s] += 1

    # After stats
    after_img_counts = defaultdict(int)
    for fam_id, target_split in family_target_splits.items():
        after_img_counts[target_split] += len(families[fam_id])

    print("================ MIGRATION SUMMARY ================")
    print("Split Image Counts:")
    print(f"  Train: Before {before_img_counts['train']} -> After {after_img_counts['train']}")
    print(f"  Val:   Before {before_img_counts['val']}   -> After {after_img_counts['val']}")
    print(f"  Test:  Before {before_img_counts['test']}  -> After {after_img_counts['test']}")
    print("====================================================")


if __name__ == "__main__":
    main()
