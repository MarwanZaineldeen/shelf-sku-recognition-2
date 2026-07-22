import os
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image

from ml.base import BaseEmbedder, BaseGalleryStore, BBoxDTO, CropDTO, EmbeddingDTO
from ml.preprocessing.box_providers import YOLOLabelBoxProvider
from ml.preprocessing.crop_generator import CropGenerator
from ml.retrieval.base import VectorIndex

logger = logging.getLogger(__name__)


class SKUOnboarder:
    """Pipeline 2: Few-Shot New SKU Onboarding Engine.
    
    Supports:
    1. Ingesting ready cropped product images.
    2. Ingesting full shelf images + YOLO annotation text files (bounding box cropping).
    3. Extracting L2-normalized feature embeddings.
    4. Storing vectors and metadata into SQLite vector database & active vector retrieval index.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseGalleryStore,
        retriever: VectorIndex,
        quality_gate: Optional[Any] = None,
        detector: Optional[Any] = None
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self.quality_gate = quality_gate
        self.detector = detector

    def onboard_from_crops(
        self,
        crops_dir: Union[str, Path],
        class_id: int,
        old_class_id: int,
        family_id: str,
        source_image: str = "pre_cropped_batch",
        detector: Optional[Any] = None,
        use_yolo_crop: bool = True
    ) -> Dict[str, Any]:
        """Onboards ready cropped product images from a directory.

        Args:
            crops_dir: Folder path containing ready cropped product images.
            class_id: Remapped continuous class ID for the new SKU.
            old_class_id: Original class ID.
            family_id: Brand/Family cluster name (e.g., 'Nesquik', 'Heinz tomato ketchup').
            source_image: Optional tag for the source shelf or dataset.

        Returns:
            Dict containing onboarding summary execution stats.
        """
        crops_dir = Path(crops_dir)
        if not crops_dir.exists():
            raise FileNotFoundError(f"Crops directory does not exist: {crops_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted([
            p for p in crops_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])

        if not image_paths:
            logger.warning(f"No image files found in {crops_dir}")
            return {"status": "empty", "crops_added": 0, "rejected": 0, "image_paths": []}

        crops_added = 0
        rejected = 0
        references_to_save = []
        vectors_to_index = []
        metadata_to_index = []

        active_detector = detector or self.detector

        for img_path in image_paths:
            img_bytes = img_path.read_bytes()
            img_np = cv2.imread(str(img_path))
            if img_np is None:
                logger.warning(f"Could not read image: {img_path}")
                rejected += 1
                continue

            h, w = img_np.shape[:2]
            bbox = BBoxDTO(x1=0.0, y1=0.0, x2=float(w), y2=float(h), confidence=1.0)

            # Detect product using YOLO detector and crop tightly on the product region if localized
            if use_yolo_crop and active_detector is not None:
                try:
                    boxes = active_detector.detect(img_bytes)
                    if boxes:
                        best_box = max(boxes, key=lambda b: getattr(b, 'confidence', 1.0))
                        b_x1 = getattr(best_box, 'x1', 0.0)
                        b_y1 = getattr(best_box, 'y1', 0.0)
                        b_x2 = getattr(best_box, 'x2', float(w))
                        b_y2 = getattr(best_box, 'y2', float(h))
                        conf = float(getattr(best_box, 'confidence', 1.0))

                        x1 = max(0, int(b_x1) if b_x1 > 1.0 else int(b_x1 * w))
                        y1 = max(0, int(b_y1) if b_y1 > 1.0 else int(b_y1 * h))
                        x2 = min(w, int(b_x2) if b_x2 > 1.0 else int(b_x2 * w))
                        y2 = min(h, int(b_y2) if b_y2 > 1.0 else int(b_y2 * h))

                        if (x2 - x1) > 5 and (y2 - y1) > 5:
                            crop_np = img_np[y1:y2, x1:x2]
                            success, enc_bytes = cv2.imencode(".jpg", crop_np)
                            if success:
                                img_bytes = enc_bytes.tobytes()
                                img_np = crop_np
                                h, w = crop_np.shape[:2]
                                bbox = BBoxDTO(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=conf)
                                logger.info(f"[Pipeline 2] YOLOv8 localized product in {img_path.name} -> box [{x1}, {y1}, {x2}, {y2}] ({w}x{h}px)")
                except Exception as det_err:
                    logger.warning(f"[Pipeline 2] YOLO product detection warning for {img_path.name}: {det_err}")

            crop_dto = CropDTO(
                crop_id=f"onboard_{family_id}_{img_path.name}",
                image_bytes=img_bytes,
                bbox=bbox,
                blur_score=0.0,
                aspect_ratio=float(w) / float(max(1, h))
            )

            if self.quality_gate:
                valid, reason = self.quality_gate.is_valid(crop_dto)
                if not valid:
                    logger.warning(f"Crop {img_path.name} rejected by quality gate: {reason}")
                    rejected += 1
                    continue

            # Extract embedding
            embedding = self.embedder.extract_dto(crop_dto)
            vec = embedding.vector

            references_to_save.append((
                class_id,
                old_class_id,
                str(img_path.resolve()),
                family_id,
                source_image,
                [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                vec
            ))

            vectors_to_index.append(vec)
            metadata_to_index.append({
                "crop_path": str(img_path.resolve()),
                "remapped_class_id": class_id,
                "old_class_id": old_class_id,
                "family_id": family_id,
                "source_image_name": source_image,
                "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
            })

            crops_added += 1

        # Bulk save to SQLite database
        latest_version = 1
        if references_to_save and hasattr(self.store, "save_references_bulk"):
            latest_version = self.store.save_references_bulk(references_to_save)

        # Update in-memory retrieval index with brand clusters
        if vectors_to_index and hasattr(self.retriever, "add"):
            arr_vecs = np.array(vectors_to_index, dtype=np.float32)
            self.retriever.add(arr_vecs, metadata_to_index)

        logger.info(f"[Pipeline 2] Successfully onboarded {crops_added} crops for SKU class {class_id} ({family_id}).")

        return {
            "status": "success",
            "crops_added": crops_added,
            "rejected": rejected,
            "class_id": class_id,
            "family_id": family_id,
            "db_version": latest_version
        }

    def onboard_from_shelf_images(
        self,
        shelf_dir: Union[str, Path],
        labels_dir: Optional[Union[str, Path]] = None,
        class_id: int = 0,
        old_class_id: int = 0,
        family_id: str = "",
        padding: float = 0.05,
        detector: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Onboards product facings from shelf images using YOLO auto-detection or .txt annotation bounding box files.

        Args:
            shelf_dir: Folder path containing raw shelf images.
            labels_dir: Optional folder path containing matching .txt YOLO annotation files.
            class_id: Remapped continuous class ID for the new SKU.
            old_class_id: Original class ID.
            family_id: Brand/Family cluster name.
            padding: Bounding box padding ratio for crop extraction.
            detector: Optional YOLO detector plugin instance.

        Returns:
            Dict containing onboarding summary execution stats.
        """
        shelf_dir = Path(shelf_dir)
        if not shelf_dir.exists():
            raise FileNotFoundError(f"Shelf directory does not exist: {shelf_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        shelf_images = sorted([
            p for p in shelf_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])

        active_detector = detector or self.detector
        box_provider = YOLOLabelBoxProvider(labels_dir) if labels_dir and Path(labels_dir).exists() else None
        crop_generator = CropGenerator(padding=padding)

        crops_added = 0
        rejected = 0
        references_to_save = []
        vectors_to_index = []
        metadata_to_index = []

        for img_path in shelf_images:
            valid_boxes = []
            if box_provider:
                boxes = box_provider.get_boxes(img_path)
                valid_boxes = [b for b in boxes if b.get("status") == "ok"]
            elif active_detector:
                try:
                    img_bytes = img_path.read_bytes()
                    det_boxes = active_detector.detect(img_bytes)
                    valid_boxes = [{
                        "status": "ok",
                        "bbox": [b.x1, b.y1, b.x2, b.y2],
                        "confidence": getattr(b, "confidence", 1.0)
                    } for b in det_boxes]
                except Exception as err:
                    logger.warning(f"Failed to auto-detect products on {img_path.name}: {err}")
            
            if not valid_boxes:
                continue

            img_np = cv2.imread(str(img_path))
            if img_np is None:
                logger.warning(f"Could not read shelf image: {img_path}")
                continue

            for box_idx, box in enumerate(valid_boxes):
                crop_np, coords = crop_generator.extract_crop(img_np, box)
                if crop_np.size == 0:
                    rejected += 1
                    continue

                # Encode crop to JPEG bytes
                success, enc_bytes = cv2.imencode(".jpg", crop_np)
                if not success:
                    rejected += 1
                    continue
                crop_bytes = enc_bytes.tobytes()

                h_crop, w_crop = crop_np.shape[:2]
                bbox_dto = BBoxDTO(
                    x1=float(coords[0]), y1=float(coords[1]),
                    x2=float(coords[2]), y2=float(coords[3]),
                    confidence=1.0
                )
                crop_dto = CropDTO(
                    crop_id=f"shelf_crop_{img_path.stem}_{box_idx}",
                    image_bytes=crop_bytes,
                    bbox=bbox_dto,
                    blur_score=0.0,
                    aspect_ratio=float(w_crop) / float(max(1, h_crop))
                )

                if self.quality_gate:
                    valid, reason = self.quality_gate.is_valid(crop_dto)
                    if not valid:
                        rejected += 1
                        continue

                embedding = self.embedder.extract_dto(crop_dto)
                vec = embedding.vector

                crop_ref_name = f"{img_path.stem}_box{box_idx}.jpg"
                references_to_save.append((
                    class_id,
                    old_class_id,
                    crop_ref_name,
                    family_id,
                    img_path.name,
                    [coords[0], coords[1], coords[2], coords[3]],
                    vec
                ))

                vectors_to_index.append(vec)
                metadata_to_index.append({
                    "crop_path": crop_ref_name,
                    "remapped_class_id": class_id,
                    "old_class_id": old_class_id,
                    "family_id": family_id,
                    "source_image_name": img_path.name,
                    "bbox": list(coords)
                })

                crops_added += 1

        # Save to database
        latest_version = 1
        if references_to_save and hasattr(self.store, "save_references_bulk"):
            latest_version = self.store.save_references_bulk(references_to_save)

        # Update in-memory index
        if vectors_to_index and hasattr(self.retriever, "add"):
            arr_vecs = np.array(vectors_to_index, dtype=np.float32)
            self.retriever.add(arr_vecs, metadata_to_index)

        logger.info(f"[Pipeline 2] Onboarded {crops_added} bounding box crops from shelf images for {family_id}.")

        return {
            "status": "success",
            "crops_added": crops_added,
            "rejected": rejected,
            "class_id": class_id,
            "family_id": family_id,
            "db_version": latest_version
        }

    def validate_sku_on_shelf(
        self,
        shelf_img_bytes: bytes,
        class_id: int,
        detector: Any,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Runs automated validation benchmark on a validation shelf image for a newly onboarded SKU.

        Args:
            shelf_img_bytes: Raw image bytes of a validation shelf photo.
            class_id: Target class ID of the newly onboarded SKU.
            detector: YOLO detector plugin instance.
            top_k: Top-K retrieval candidates to evaluate.

        Returns:
            Dict containing validation audit metrics and quality recommendation.
        """
        nparr = np.frombuffer(shelf_img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            return {
                "facings_detected": 0,
                "mean_similarity": 0.0,
                "pass_validation": False,
                "recommendation": "Could not read validation shelf image."
            }

        # 1. Run YOLO bounding box detection
        boxes = detector.detect(img_np)
        if not boxes:
            return {
                "facings_detected": 0,
                "mean_similarity": 0.0,
                "pass_validation": False,
                "recommendation": "No product bounding boxes localized on validation shelf photo."
            }

        crop_generator = CropGenerator(padding=0.0)
        matched_sims = []
        facings_count = 0

        for box_dict in boxes[:50]:  # Evaluate up to 50 localized crops
            crop_np, coords = crop_generator.extract_crop(img_np, box_dict)
            if crop_np.size == 0:
                continue

            success, enc_bytes = cv2.imencode(".jpg", crop_np)
            if not success:
                continue
            crop_bytes = enc_bytes.tobytes()

            h_crop, w_crop = crop_np.shape[:2]
            bbox_dto = BBoxDTO(
                x1=float(coords[0]), y1=float(coords[1]),
                x2=float(coords[2]), y2=float(coords[3]),
                confidence=float(box_dict.get("confidence", 1.0))
            )
            crop_dto = CropDTO(
                crop_id=f"val_crop_{coords[0]}_{coords[1]}",
                image_bytes=crop_bytes,
                bbox=bbox_dto,
                blur_score=0.0,
                aspect_ratio=float(w_crop) / float(max(1, h_crop))
            )

            # Extract embedding & query retriever
            embedding = self.embedder.extract_dto(crop_dto)
            matches = self.retriever.search(embedding.vector, top_k=top_k)

            # Check if target class_id is returned in top candidates
            for match in matches:
                matched_cid = match.get("remapped_class_id", match.get("class_id"))
                if matched_cid == class_id:
                    facings_count += 1
                    sim_score = float(match.get("similarity", 0.0))
                    matched_sims.append(sim_score)
                    break

        mean_sim = float(np.mean(matched_sims)) if matched_sims else 0.0
        pass_val = facings_count > 0 and mean_sim >= 0.75

        if pass_val:
            recommendation = f"Sufficient Examples — High Recognition Quality ({facings_count} facings recognized with {mean_sim:.1%} similarity)"
        elif facings_count > 0:
            recommendation = f"Moderate Confidence ({mean_sim:.1%} similarity) — Recommend uploading 5-10 more reference crops for higher accuracy."
        else:
            recommendation = "Low Confidence — SKU not detected on shelf. Please upload 10-20 more diverse reference crops."

        return {
            "facings_detected": facings_count,
            "mean_similarity": round(mean_sim, 4),
            "pass_validation": pass_val,
            "recommendation": recommendation
        }

