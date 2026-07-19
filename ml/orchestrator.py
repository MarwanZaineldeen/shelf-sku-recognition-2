import io
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ml.base import (
    BaseDetector, BaseQualityGate, BaseEmbedder, BaseRetriever,
    BaseOCR, BaseCalibrator, BaseFusionStrategy, BaseDecisionPolicy,
    BBoxDTO, CropDTO, PredictionDTO
)


class AuditPipelineOrchestrator:
    """Core domain orchestrator executing the shelf audit process."""

    def __init__(
        self,
        detector: BaseDetector,
        quality_gate: BaseQualityGate,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        ocr: BaseOCR,
        calibrator: BaseCalibrator,
        fusion: BaseFusionStrategy,
        decision_policy: BaseDecisionPolicy
    ) -> None:
        self.detector = detector
        self.quality_gate = quality_gate
        self.embedder = embedder
        self.retriever = retriever
        self.ocr = ocr
        self.calibrator = calibrator
        self.fusion = fusion
        self.decision_policy = decision_policy

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

        # 3. Process each bounding box crop
        for idx, box in enumerate(bboxes):
            # Clamp coordinates to original image bounds
            x1 = max(0, int(box.x1))
            y1 = max(0, int(box.y1))
            x2 = min(w_orig, int(box.x2))
            y2 = min(h_orig, int(box.y2))

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w <= 0 or crop_h <= 0:
                continue

            crop_img = img_bgr[y1:y2, x1:x2]
            
            # Encode crop back to bytes
            _, crop_bytes_arr = cv2.imencode(".jpg", crop_img)
            crop_bytes = crop_bytes_arr.tobytes()

            # Construct CropDTO
            blur_score = 0.0
            aspect_ratio = float(crop_w) / float(crop_h)
            crop_dto = CropDTO(
                crop_id=f"shelf_crop_{idx}",
                image_bytes=crop_bytes,
                bbox=box,
                blur_score=blur_score,
                aspect_ratio=aspect_ratio
            )

            # 4. Bbox Quality Gate check
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

            # 5. Extract DINOv2 Embedding
            embedding_dto = self.embedder.extract_dto(crop_dto)

            # 6. Query Retriever search index
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

            # 7. Gated OCR and late fusion
            top_visual_sim = matches[0].similarity
            fused_matches = matches
            ocr_text = None

            # CPU OCR gating logic
            if top_visual_sim > 0.96:
                # Fast path: bypass OCR for highly confident matches
                pass
            elif top_visual_sim >= 0.85:
                # Gated path: run OCR on uncertain matches
                ocr_result = self.ocr.extract_text(crop_dto, timeout_ms=ocr_timeout_ms)
                if ocr_result.text.strip():
                    ocr_text = ocr_result.text
                    fused_matches = self.fusion.fuse(matches, ocr_result)
            else:
                # Reject path: bypass OCR for extremely low visual similarity matches
                pred = PredictionDTO(
                    bbox=box,
                    predicted_class_id=matches[0].remapped_class_id,
                    confidence_probability=0.0,
                    automated=False,
                    reject_reason="LOW_VISUAL_CONFIDENCE"
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
                ocr_text=ocr_text
            )

            if automated:
                annotations.append(prediction)
            else:
                hitl_queue.append(prediction)

        return annotations, hitl_queue
