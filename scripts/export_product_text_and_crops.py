import os
import sys
import json
import cv2
import pandas as pd
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path("d:/Marwan/ITI AI&ML/Transmid GP")
sys.path.append(str(workspace_root))

crop_metadata_path = workspace_root / "data/processed/crops/gt_clean/crop_metadata.csv"
crops_train_dir = workspace_root / "data/processed/crops/gt_clean/train"
sku_mapping_path = workspace_root / "configs/sku_mapping.json"
ocr_gt_path = workspace_root / "configs/class_ocr_groundtruth.json"
output_catalog_dir = workspace_root / "configs/class_catalog"


def main():
    print("==================================================")
    print("Exporting 67 Product Text Files and Reference Crops")
    print("==================================================")

    # 1. Load Metadata & Catalog Files
    df = pd.read_csv(crop_metadata_path)
    df_train = df[df["split"] == "train"].copy()

    with open(sku_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)["classes"]

    with open(ocr_gt_path, "r", encoding="utf-8") as f:
        ocr_gt_data = json.load(f)

    unique_classes = sorted(df_train["remapped_class_id"].unique())
    print(f"Exporting files for {len(unique_classes)} product categories to {output_catalog_dir.name}/...\n")

    exported_count = 0

    for cid in unique_classes:
        cid_str = str(cid)
        class_info = mapping_data.get(cid_str, {})
        ocr_info = ocr_gt_data.get(cid_str, {})

        class_dir = output_catalog_dir / f"class_{cid:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        # 2. Generate text metadata file
        metadata_txt_path = class_dir / f"product_class_{cid:02d}_metadata.txt"
        
        display_name = class_info.get("display_name", f"SKU {cid}")
        project_sku_id = class_info.get("project_sku_id", f"TM_RAW_{cid:03d}")
        brand = class_info.get("brand", "")
        product_name = class_info.get("product_name", "")
        variant = class_info.get("variant", "")
        pack_count = class_info.get("pack_count", "")
        pack_type = class_info.get("pack_type", "")
        gt_ocr_text = ocr_info.get("precalculated_ocr_text", "")
        extracted_ocr = ocr_info.get("extracted_crop_ocr", "")

        meta_content = f"""==================================================
COMMERCIAL SKU METADATA CATALOG PROFILE
==================================================
Training Class ID     : {cid}
Project SKU ID        : {project_sku_id}
Commercial Display    : {display_name}
Brand                 : {brand}
Product Name          : {product_name}
Variant / Flavor      : {variant}
Pack Count / Weight   : {pack_count}
Packaging Type        : {pack_type}

==================================================
PRECALCULATED GROUND-TRUTH OCR TEXT
==================================================
Canonical Full Profile: {gt_ocr_text}
Extracted Crop Text   : {extracted_ocr}
==================================================
"""
        with open(metadata_txt_path, "w", encoding="utf-8") as f:
            f.write(meta_content)

        # 3. Find and copy top visual reference crop image
        class_crops = df_train[df_train["remapped_class_id"] == cid].sort_values(by="crop_area", ascending=False)
        crop_saved = False

        if not class_crops.empty:
            row = class_crops.iloc[0]
            rel_path = row["crop_path"]
            abs_crop_path = workspace_root / rel_path

            if not abs_crop_path.exists():
                abs_crop_path = crops_train_dir / Path(rel_path).name

            if abs_crop_path.exists():
                ref_crop_out = class_dir / f"product_class_{cid:02d}_reference_crop.jpg"
                img = cv2.imread(str(abs_crop_path))
                if img is not None:
                    cv2.imwrite(str(ref_crop_out), img)
                    crop_saved = True

        exported_count += 1
        print(f"  Class {cid:2d} ({display_name[:35]:<35}) -> {class_dir.name}/ (Meta: OK | Image: {'OK' if crop_saved else 'Missing'})")

    print("\n==================================================")
    print(f"Successfully exported {exported_count} product catalog directories to:")
    print(f"  {output_catalog_dir.resolve()}")
    print("==================================================")


if __name__ == "__main__":
    main()
