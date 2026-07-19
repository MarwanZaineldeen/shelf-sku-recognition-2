import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_md5(file_path: Path) -> str:
    """Computes MD5 hash of a file for exact duplicate checking."""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute MD5 for {file_path.name}: {str(e)}")
        return ""


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    Computes a Difference Hash (dHash) for an image.
    Returns a boolean 1D numpy array of length hash_size * hash_size.
    """
    if image is None or image.size == 0:
        return np.zeros(hash_size * hash_size, dtype=bool)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize to (width = hash_size + 1, height = hash_size)
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        # Horizontal difference
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.flatten()
    except Exception as e:
        logger.warning(f"dHash computation failed, returning fallback zero hash: {str(e)}")
        return np.zeros(hash_size * hash_size, dtype=bool)


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Computes Hamming distance between two boolean dHash arrays."""
    return int(np.count_nonzero(hash1 != hash2))


class LeakageDetector:
    """Performs integrity checks and cross-split duplicate leakage checks on images and crops."""

    def __init__(self, image_threshold: int = 10, crop_threshold: int = 8):
        self.image_threshold = image_threshold
        self.crop_threshold = crop_threshold

    def audit_integrity(
        self,
        dataset_dir: Path,
        mapping_path: Path
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates that all file splits exist, labels align with images,
        and remapped classes correspond strictly to [0, 66].
        """
        dataset_dir = Path(dataset_dir)
        mapping_path = Path(mapping_path)
        
        issues = {
            "missing_images": [],
            "missing_labels": [],
            "invalid_class_ids": [],
            "mapping_mismatch": False
        }
        passed = True

        # Check mapping file
        if not mapping_path.exists():
            issues["mapping_mismatch"] = True
            passed = False
            logger.error("Class mapping config file missing on disk.")
            return passed, issues

        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            new_to_old = mapping.get("new_to_old", {})
        except Exception as e:
            issues["mapping_mismatch"] = True
            passed = False
            logger.error(f"Failed to load class mapping JSON: {str(e)}")
            return passed, issues

        # Verify directories
        splits = ["train", "val", "test"]
        for split in splits:
            img_split_dir = dataset_dir / "images" / split
            lbl_split_dir = dataset_dir / "labels" / split

            if not img_split_dir.exists():
                passed = False
                logger.error(f"Image split directory missing: {img_split_dir}")
                continue
            if not lbl_split_dir.exists():
                passed = False
                logger.error(f"Label split directory missing: {lbl_split_dir}")
                continue

            images = sorted(list(img_split_dir.iterdir()))
            for img_p in images:
                if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                
                # Check matching label file exists
                lbl_p = lbl_split_dir / f"{img_p.stem}.txt"
                if not lbl_p.exists():
                    passed = False
                    issues["missing_labels"].append(f"{split}/{lbl_p.name}")
                    continue

                # Read and check class IDs inside label
                try:
                    with open(lbl_p, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()
                    for idx, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        parts = stripped.split()
                        if len(parts) >= 1:
                            cls_id = int(parts[0])
                            if not (0 <= cls_id <= 66):
                                passed = False
                                issues["invalid_class_ids"].append({
                                    "file": f"{split}/{lbl_p.name}",
                                    "line": idx,
                                    "class_id": cls_id
                                })
                            if str(cls_id) not in new_to_old:
                                passed = False
                                issues["mapping_mismatch"] = True
                except Exception as e:
                    passed = False
                    logger.warning(f"Error parsing label file {lbl_p.name}: {str(e)}")

        return passed, issues

    def find_image_leaks(self, dataset_dir: Path) -> List[Dict[str, Any]]:
        """
        Scans split directories for exact (MD5) and near-duplicate (dHash) image leaks.
        """
        dataset_dir = Path(dataset_dir)
        splits = ["train", "val", "test"]
        image_records = []

        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Collect hashes for all images
        for split in splits:
            img_dir = dataset_dir / "images" / split
            if not img_dir.exists():
                continue
            for img_p in sorted(list(img_dir.iterdir())):
                if img_p.suffix.lower() not in supported_exts:
                    continue
                
                md5_val = compute_md5(img_p)
                
                # Load image for dHash computation
                img_cv = cv2.imread(str(img_p))
                dhash_val = compute_dhash(img_cv)
                
                image_records.append({
                    "path": img_p,
                    "rel_path": f"data/processed/yolo_remapped/images/{split}/{img_p.name}",
                    "split": split,
                    "md5": md5_val,
                    "dhash": dhash_val
                })

        leaks = []
        n = len(image_records)

        # Cross-split comparisons
        for i in range(n):
            for j in range(i + 1, n):
                rec1 = image_records[i]
                rec2 = image_records[j]

                # Only leak if splits are different
                if rec1["split"] == rec2["split"]:
                    continue

                is_exact = rec1["md5"] == rec2["md5"]
                h_dist = hamming_distance(rec1["dhash"], rec2["dhash"])
                is_near = h_dist <= self.image_threshold

                if is_exact or is_near:
                    leaks.append({
                        "file_1": rec1["rel_path"],
                        "file_2": rec2["rel_path"],
                        "split_1": rec1["split"],
                        "split_2": rec2["split"],
                        "type": "exact_duplicate" if is_exact else "near_duplicate",
                        "hamming_distance": h_dist
                    })

        return leaks
