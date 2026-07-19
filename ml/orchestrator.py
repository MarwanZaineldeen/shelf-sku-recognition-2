import io
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from ml.base import (
    BaseDetector,
    BaseEmbedder,
    BaseRetriever,
    BaseOCR,
    BaseCalibrator,
    BaseFusionStrategy,
    BaseDecisionPolicy,
    BaseQualityGate,
    CropDTO,
    PredictionDTO,
    CommercialSKUDTO
)


class AuditPipelineOrchestrator:
    """Master orchestrator executing complete shelf audit workflow using pluggable interfaces."""

    def __init__(
        self,
        detector: BaseDetector,
        quality_gate: BaseQualityGate,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        ocr: BaseOCR,
        calibrator: BaseCalibrator,
        fusion: BaseFusionStrategy,
        decision_policy: BaseDecisionPolicy,
        sku_mapping_path: str = "configs/sku_mapping.json"
    ):
        self.detector = detector
        self.quality_gate = quality_gate
        self.embedder = embedder
        self.retriever = retriever
        self.ocr = ocr
        self.calibrator = calibrator
        self.fusion = fusion
        self.decision_policy = decision_policy

        self.sku_mapping: Dict[str, Dict[str, Any]] = {}
        if sku_mapping_path and Path(sku_mapping_path).exists():
            with open(sku_mapping_path, "r", encoding="utf-8") as f:
                self.sku_mapping = json.load(f).get("classes", {})

    def _get_commercial_info(self, class_id: int) -> Optional[CommercialSKUDTO]:
        cid_str = str(class_id)
        if cid_str not in self.sku_mapping:
            return None

        info = self.sku_mapping[cid_str]
        return CommercialSKUDTO(
            project_sku_id=info.get("project_sku_id", f"TM_RAW_{class_id:03d}"),
            display_name=info.get("display_name", f"SKU {class_id}"),
            brand=info.get("brand", ""),
            product_name=info.get("product_name", ""),
            variant=info.get("variant", ""),
            pack_count=info.get("pack_count", ""),
            pack_type=info.get("pack_type", "")
        )

    def process_shelf(
        self,
        image_bytes: bytes,
        ocr_timeout_ms: int = 300
    ) -> Tuple[List[PredictionDTO], List[PredictionDTO]]:
        """Processes shelf image end-to-end, separating auto-annotations and HITL queue.

        Args:
            image_bytes: Raw image stream payload of the shelf.
            ocr_timeout_ms: Execution timeout limit for EasyOCR crop parsing.

        Returns:
            Tuple[List[PredictionDTO], List[PredictionDTO]]: (automated_annotations, hitl_queue).
        """
        # 1. Decode original shelf image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode input shelf image bytes.")

        h_orig, w_orig = img_bgr.shape[:2]

        # 2. Run object detection
        bboxes = self.detector.detect(image_bytes)

        annotations: List[PredictionDTO] = []
        hitl_queue: List[PredictionDTO] = []

        # 3. Crop extraction & Quality Gate filter
        valid_crops: List[CropDTO] = []
        for idx, box in enumerate(bboxes):
            x1 = max(0, int(box.x1) if box.x1 > 1.0 else int(box.x1 * w_orig))
            y1 = max(0, int(box.y1) if box.y1 > 1.0 else int(box.y1 * h_orig))
            x2 = min(w_orig, int(box.x2) if box.x2 > 1.0 else int(box.x2 * w_orig))
            y2 = min(h_orig, int(box.y2) if box.y2 > 1.0 else int(box.y2 * h_orig))

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w <= 0 or crop_h <= 0:
                continue

            crop_img = img_bgr[y1:y2, x1:x2]
            _, crop_bytes_arr = cv2.imencode(".jpg", crop_img)
            crop_bytes = crop_bytes_arr.tobytes()

            crop_dto = CropDTO(
                crop_id=f"shelf_crop_{idx}",
                image_bytes=crop_bytes,
                bbox=box,
                blur_score=0.0,
                aspect_ratio=float(crop_w) / float(crop_h) if crop_h > 0 else 1.0
            )

            valid, reject_reason = self.quality_gate.is_valid(crop_dto)
            if not valid:
                pred = PredictionDTO(
                    bbox=box,
                    predicted_class_id=-1,
                    confidence_probability=0.0,
                    automated=False,
                    reject_reason=reject_reason
                )
                hitl_queue.append(pred)
                continue

            valid_crops.append(crop_dto)

        if not valid_crops:
            return annotations, hitl_queue

        # 4. Batched DINOv2 Feature Extraction
        embeddings = self.embedder.extract_batch_dto(valid_crops)

        # 5. Retrieval, Gated OCR Fusion, Calibration & Decision Gating
        for crop_dto, embedding_dto in zip(valid_crops, embeddings):
            box = crop_dto.bbox
            matches = self.retriever.search_dto(embedding_dto, top_k=5)
            if not matches:
                pred = PredictionDTO(
                    bbox=box,
                    predicted_class_id=-1,
                    confidence_probability=0.0,
                    automated=False,
                    reject_reason="NO_MATCHING_CANDIDATES"
                )
                hitl_queue.append(pred)
                continue

            top_visual_sim = matches[0].similarity
            fused_matches = matches
            ocr_text = None

            if top_visual_sim > 0.96:
                pass
            elif top_visual_sim >= 0.85:
                ocr_result = self.ocr.extract_text(crop_dto, timeout_ms=ocr_timeout_ms)
                if ocr_result.text.strip():
                    ocr_text = ocr_result.text
                    fused_matches = self.fusion.fuse(matches, ocr_result)
            else:
                pred = PredictionDTO(
                    bbox=box,
                    predicted_class_id=matches[0].remapped_class_id,
                    confidence_probability=0.0,
                    automated=False,
                    reject_reason="LOW_VISUAL_CONFIDENCE",
                    commercial_info=self._get_commercial_info(matches[0].remapped_class_id)
                )
                hitl_queue.append(pred)
                continue

            # 8. Platt scaling calibration
            best_match = fused_matches[0]
            probability = self.calibrator.calibrate(best_match.similarity, best_match.remapped_class_id)

            # 9. Gated decision checks
            automated, reject_reason = self.decision_policy.decide(
                fused_matches, probability, best_match.remapped_class_id
            )

            prediction = PredictionDTO(
                bbox=box,
                predicted_class_id=best_match.remapped_class_id,
                confidence_probability=probability,
                automated=automated,
                reject_reason=reject_reason,
                ocr_text=ocr_text,
                commercial_info=self._get_commercial_info(best_match.remapped_class_id)
            )

            if automated:
                annotations.append(prediction)
            else:
                hitl_queue.append(prediction)

        return annotations, hitl_queue
