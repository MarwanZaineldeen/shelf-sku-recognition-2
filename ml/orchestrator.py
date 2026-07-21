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
        vlm_reranker: Optional[Any] = None,
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
        self.vlm_reranker = vlm_reranker

        self.sku_mapping: Dict[int, Dict[str, Any]] = {}
        for path in ["configs/sku_mapping_v2.json", "c:/Users/asusd/Desktop/sku_mapping_v2.json", "configs/sku_mapping.json"]:
            if Path(path).exists():
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f).get("classes", {})
                    for key, info in raw_data.items():
                        t_id = info.get("training_class_id")
                        if t_id is not None:
                            self.sku_mapping[int(t_id)] = info
                        else:
                            try:
                                self.sku_mapping[int(key)] = info
                            except ValueError:
                                pass
                break

    def _get_commercial_info(self, class_id: int) -> Optional[CommercialSKUDTO]:
        if class_id not in self.sku_mapping:
            return None

        info = self.sku_mapping[class_id]
        return CommercialSKUDTO(
            project_sku_id=info.get("project_sku_id", f"TM_RAW_{class_id:03d}"),
            display_name=info.get("display_name", f"SKU {class_id}"),
            brand=info.get("brand", "Lipton"),
            product_name=info.get("product_name", ""),
            variant=info.get("variant", ""),
            pack_count=info.get("pack_count", ""),
            pack_type=info.get("pack_type", "box")
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

            valid_crops.append(crop_dto)

        if not valid_crops:
            return annotations, hitl_queue

        # 4. Batched DINOv3 Feature Extraction
        embeddings = self.embedder.extract_batch_dto(valid_crops)

        # 5. Retrieval, Calibration & Decision Gating
        for crop_idx, (crop_dto, embedding_dto) in enumerate(zip(valid_crops, embeddings)):
            box = crop_dto.bbox
            matches = self.retriever.search_dto(embedding_dto, top_k=5)

            if not matches:
                pred = PredictionDTO(
                    crop_id=f"crop_{crop_idx+1}",
                    bbox=box,
                    predicted_class_id=-1,
                    confidence_probability=0.0,
                    automated=False,
                    reject_reason="NO_MATCHING_CANDIDATES"
                )
                hitl_queue.append(pred)
                continue

            top_visual_sim = matches[0].similarity

            import base64
            crop_b64 = base64.b64encode(crop_dto.image_bytes).decode("utf-8")
            crop_data_url = f"data:image/jpeg;base64,{crop_b64}"
            
            top5_cand_info = []
            for m in matches:
                cinfo = self._get_commercial_info(m.remapped_class_id)
                dname = cinfo.display_name if cinfo else f"SKU {m.remapped_class_id}"
                top5_cand_info.append({"class_id": m.remapped_class_id, "display_name": dname, "similarity": m.similarity})

            # ── 3-Tier Decision Gating ──────────────────────────────────
            #  Tier 1  ▸ S_vis >= 0.92  → auto-annotate directly
            #  Tier 2  ▸ 0.75 <= S_vis < 0.92  → VLM reranker picks best of Top-5
            #  Tier 3  ▸ S_vis < 0.75  → HITL queue (human must decide)

            if top_visual_sim < 0.75:
                # Tier 3: Low confidence → HITL
                pred = PredictionDTO(
                    crop_id=f"crop_{crop_idx+1}",
                    bbox=box,
                    predicted_class_id=matches[0].remapped_class_id,
                    confidence_probability=matches[0].similarity,
                    automated=False,
                    reject_reason="LOW_VISUAL_CONFIDENCE",
                    crop_bytes=crop_dto.image_bytes,
                    crop_data_url=crop_data_url,
                    top5_candidates=top5_cand_info,
                    commercial_info=self._get_commercial_info(matches[0].remapped_class_id)
                )
                hitl_queue.append(pred)
                continue

            best_match = matches[0]
            vlm_verified_name = None

            if top_visual_sim < 0.92:
                # Tier 2: Mid confidence → VLM reranker picks best of Top-5 candidates
                if self.vlm_reranker is not None and getattr(self.vlm_reranker, "is_ready", False):
                    try:
                        from PIL import Image
                        pil_crop = Image.open(io.BytesIO(crop_dto.image_bytes)).convert("RGB")
                        reranked = self.vlm_reranker.rerank_top5_candidates(pil_crop, top5_cand_info)
                        if reranked:
                            vlm_pick = reranked[0]
                            vlm_class_id = vlm_pick.get("class_id", best_match.remapped_class_id)
                            vlm_verified_name = vlm_pick.get("display_name")
                            # Override best_match class if VLM chose differently
                            for m in matches:
                                if m.remapped_class_id == vlm_class_id:
                                    best_match = m
                                    break
                            top5_cand_info = reranked  # propagate VLM-reranked order to UI
                    except Exception as vlm_err:
                        print(f"[VLM Rerank] Error: {vlm_err}")

            # 8. Platt scaling calibration
            probability = self.calibrator.calibrate(best_match.similarity, best_match.remapped_class_id)

            # 9. Gated decision checks
            automated, reject_reason = self.decision_policy.decide(
                matches, probability, best_match.remapped_class_id
            )

            # If VLM verified, force automated=True for Tier 2
            if vlm_verified_name and probability >= 0.50:
                automated = True
                reject_reason = None

            prediction = PredictionDTO(
                crop_id=f"crop_{crop_idx+1}",
                bbox=box,
                predicted_class_id=best_match.remapped_class_id,
                confidence_probability=probability,
                automated=automated,
                reject_reason=reject_reason,
                ocr_text=f"VLM: {vlm_verified_name}" if vlm_verified_name else None,
                crop_bytes=crop_dto.image_bytes,
                crop_data_url=crop_data_url,
                top5_candidates=top5_cand_info,
                commercial_info=self._get_commercial_info(best_match.remapped_class_id)
            )

            if automated:
                annotations.append(prediction)
            else:
                hitl_queue.append(prediction)

        return annotations, hitl_queue
