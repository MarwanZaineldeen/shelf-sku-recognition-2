import os
import sys
import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(os.environ.get("RETAIL_AI_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(workspace_root))

# Set environment cache paths BEFORE importing torch/transformers
os.environ["HF_HOME"] = str(workspace_root / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(workspace_root / ".cache" / "torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from ml.base import CropDTO, BBoxDTO
from ml.ocr.easy_ocr import EasyOCREngine

crop_metadata_path = workspace_root / "data/processed/crops/gt_clean/crop_metadata.csv"
crops_train_dir = workspace_root / "data/processed/crops/gt_clean/train"
sku_mapping_path = workspace_root / "configs/sku_mapping.json"
output_ocr_gt_path = workspace_root / "configs/class_ocr_groundtruth.json"


def resize_if_large(img, max_dim=640):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def main():
    print("==================================================", flush=True)
    print("Precalculating Ground-Truth Class OCR Text Profiles", flush=True)
    print("==================================================", flush=True)

    # 1. Load Crop Metadata CSV
    print(f"Loading crop metadata: {crop_metadata_path.name}...", flush=True)
    df = pd.read_csv(crop_metadata_path)
    df_train = df[df["split"] == "train"].copy()

    # 2. Load Commercial SKU Metadata
    with open(sku_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)["classes"]

    # 3. Initialize EasyOCR Plugin
    print("Initializing EasyOCR Engine...", flush=True)
    ocr = EasyOCREngine()
    ocr.initialize({"languages": ["en"], "gpu": False})

    ground_truth_ocr = {}

    # Get sorted unique remapped class IDs
    unique_classes = sorted(df_train["remapped_class_id"].unique())
    print(f"Found {len(unique_classes)} unique classes. Generating OCR ground-truth profiles...\n", flush=True)

    for cid in unique_classes:
        cid_str = str(cid)
        class_info = mapping_data.get(cid_str, {})
        display_name = class_info.get("display_name", f"SKU {cid}")

        # Pick the single largest reference crop for this class ID
        class_crops = df_train[df_train["remapped_class_id"] == cid].sort_values(by="crop_area", ascending=False)
        
        extracted_text = ""

        if not class_crops.empty:
            row = class_crops.iloc[0]
            rel_path = row["crop_path"]
            abs_crop_path = workspace_root / rel_path

            if not abs_crop_path.exists():
                abs_crop_path = crops_train_dir / Path(rel_path).name

            if abs_crop_path.exists():
                img = cv2.imread(str(abs_crop_path))
                if img is not None:
                    img_resized = resize_if_large(img, max_dim=480)
                    _, img_bytes_arr = cv2.imencode(".jpg", img_resized)
                    img_bytes = img_bytes_arr.tobytes()

                    crop_dto = CropDTO(
                        crop_id=f"ref_crop_{cid}",
                        image_bytes=img_bytes,
                        bbox=BBoxDTO(x1=row["x1"], y1=row["y1"], x2=row["x2"], y2=row["y2"], confidence=1.0),
                        blur_score=0.0,
                        aspect_ratio=1.0
                    )

                    ocr_res = ocr.extract_text(crop_dto, timeout_ms=800)
                    if ocr_res.text.strip():
                        extracted_text = ocr_res.text.strip()

        # Combine text extracted from top crop + canonical commercial title metadata
        canonical_text = f"{class_info.get('brand', '')} {class_info.get('product_name', '')} {class_info.get('variant', '')} {class_info.get('pack_count', '')} {class_info.get('pack_type', '')}".strip()
        full_gt_ocr = f"{canonical_text} {extracted_text}".strip()

        ground_truth_ocr[cid_str] = {
            "remapped_class_id": int(cid),
            "project_sku_id": class_info.get("project_sku_id", f"TM_RAW_{int(cid):03d}"),
            "display_name": display_name,
            "brand": class_info.get("brand", ""),
            "product_name": class_info.get("product_name", ""),
            "variant": class_info.get("variant", ""),
            "pack_count": class_info.get("pack_count", ""),
            "pack_type": class_info.get("pack_type", ""),
            "precalculated_ocr_text": full_gt_ocr,
            "extracted_crop_ocr": extracted_text,
            "canonical_title": canonical_text
        }

        print(f"  Class {cid:2d} ({display_name:<45}): `{full_gt_ocr[:55]}`", flush=True)

    # 4. Write Ground-Truth JSON Artifact
    output_ocr_gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ocr_gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_ocr, f, indent=2)

    print("\n==================================================", flush=True)
    print(f"Successfully saved 67 Class OCR Ground-Truth Profiles to:", flush=True)
    print(f"  {output_ocr_gt_path.resolve()}", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()
