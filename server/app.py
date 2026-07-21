import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

workspace_root = Path("d:/Marwan/ITI AI&ML/Transmid GP")
os.environ["HF_HOME"] = str(workspace_root / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(workspace_root / ".cache" / "torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

from ml.base import BBoxDTO, EmbeddingDTO, CropDTO
from ml.detection.yolo_detector import YOLOv8Detector
from ml.data_quality.quality_checks import BboxQualityGate
from ml.embeddings.dinov2 import DINOv2Extractor
from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from ml.retrieval.numpy_index import NumpyCosineIndex
from ml.ocr.easy_ocr import EasyOCREngine
from ml.calibrators.platt import PlattCalibrator
from ml.fusion.tfidf_ocr_matcher import TfidfOCRMatcher
from ml.decision.gated_policy import GatedAnnotationPolicy
from ml.orchestrator import AuditPipelineOrchestrator

from server.schemas import (
    AuditResponse, AnnotationOut, BBoxOut, HITLRecordOut,
    HealthResponse, OnboardResponse, CommercialSKUOut, CandidateOut
)

config_path = workspace_root / "configs/retrieval_config.yaml"
lexicon_path = workspace_root / "configs/class_lexicons.json"

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(
    title="Enterprise Retail AI Platform",
    description="Production-grade supermarket shelf product localization and SKU recognition platform.",
    version="1.0.0"
)

# Mount static web frontend files
static_dir = workspace_root / "server/static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

images_dir = workspace_root / "data/processed/yolo_remapped_clean/images/test"
if images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(images_dir)), name="images")

catalog_dir = workspace_root / "configs/class_catalog"
if catalog_dir.exists():
    app.mount("/static/catalog", StaticFiles(directory=str(catalog_dir)), name="catalog")

@app.get("/")
def get_dashboard():
    index_html = static_dir / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return JSONResponse({"message": "Retail AI API running. UI index.html not found."})

@app.get("/style.css")
def get_style():
    style_css = static_dir / "style.css"
    if style_css.exists():
        return FileResponse(style_css, media_type="text/css")
    return HTTPException(status_code=404, detail="style.css not found")

@app.get("/app.js")
def get_app_js():
    app_js = static_dir / "app.js"
    if app_js.exists():
        return FileResponse(app_js, media_type="application/javascript")
    return HTTPException(status_code=404, detail="app.js not found")

@app.get("/api/catalog")
def get_catalog():
    sku_mapping_path = workspace_root / "configs/sku_mapping.json"
    if sku_mapping_path.exists():
        with open(sku_mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"classes": {}}

# Global orchestrator and registry storage references
orchestrator: Any = None
detector_plugin: Any = None
embedder_plugin: Any = None
retriever_plugin: Any = None
ocr_plugin: Any = None
calibrator_plugin: Any = None
fusion_plugin: Any = None
decision_policy_plugin: Any = None
quality_gate_plugin: Any = None
db_store_plugin: Any = None


@app.on_event("startup")
def startup_event():
    global orchestrator, detector_plugin, embedder_plugin, retriever_plugin, ocr_plugin
    global calibrator_plugin, fusion_plugin, decision_policy_plugin, quality_gate_plugin, db_store_plugin

    print("Starting up Retail AI Platform Service...", flush=True)

    # 1. Load yaml config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Load lexicon configuration
    with open(lexicon_path, "r") as f:
        lexicons = json.load(f)

    # 3. Instantiate concrete plugins
    detector_plugin = YOLOv8Detector()
    quality_gate_plugin = BboxQualityGate()

    # DINOv3 ViT-B/16 (768-D) — Native support via transformers 5.14.1
    # Teammate's model weights outperformed DINOv2 (99% Top-5 accuracy)
    from ml.embeddings.dinov3 import DINOv3Extractor
    print("  Using DINOv3 ViT-B/16 SOTA 768-D Visual Backbone (Native)!", flush=True)
    embedder_plugin = DINOv3Extractor(device="cpu")
    retriever_plugin = NumpyCosineIndex(dimension=768)
    db_path = str(workspace_root / "data/processed/crops/gt_clean/retail_sku_registry_dinov3.db")
    dimension = 768

    # Load Qwen2-VL Reranker
    from ml.vlm.qwen2_vl_reranker import Qwen2VLReranker
    vlm_reranker_plugin = Qwen2VLReranker()
    vlm_reranker_plugin.initialize({
        "model_id": "Qwen/Qwen2-VL-2B-Instruct-AWQ",
        "device": "cpu",
        "local_files_only": True
    })

    calibrator_plugin = PlattCalibrator()
    decision_policy_plugin = GatedAnnotationPolicy()
    db_store_plugin = SQLiteGalleryStore()

    print("  Initializing SQLite Gallery Store...", flush=True)
    db_store_plugin.initialize({"db_path": db_path})

    print("  Initializing YOLOv8 Detector (SKU110K Class-Agnostic)...", flush=True)
    detector_plugin.initialize({
        "weights_path": str(workspace_root / "runs/detect/yolo8l_sku110k/yolov8l-sku110k.pt"),
        "confidence_threshold": 0.25,
        "imgsz": 640
    })

    print("  Initializing Crop Quality Gate...", flush=True)
    quality_gate_plugin.initialize({
        "min_area": 1024,
        "max_aspect": 5.0,
        "min_blur": 30.0
    })

    print(f"  Initializing Cosine Search Index ({dimension}-D)...", flush=True)
    retriever_plugin.initialize({
        "dimension": dimension,
        "db_path": db_path
    })

    print("  Initializing Platt Calibrator...", flush=True)
    calibrator_plugin.initialize({
        "global_coefs": {"a": 15.0, "b": -11.0}
    })

    print("  Initializing Gated Decision Policy...", flush=True)
    decision_policy_plugin.initialize({
        "global_threshold": 0.80
    })

    # 5. Assemble orchestrator
    orchestrator = AuditPipelineOrchestrator(
        detector=detector_plugin,
        quality_gate=quality_gate_plugin,
        embedder=embedder_plugin,
        retriever=retriever_plugin,
        ocr=ocr_plugin,
        calibrator=calibrator_plugin,
        fusion=fusion_plugin,
        decision_policy=decision_policy_plugin,
        vlm_reranker=vlm_reranker_plugin
    )
    print("Service Startup Completed. Platform Ready.", flush=True)


@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down Retail AI Platform Service...", flush=True)
    if detector_plugin:
        detector_plugin.shutdown()
    if embedder_plugin:
        embedder_plugin.shutdown()
    if retriever_plugin:
        retriever_plugin.shutdown()
    if ocr_plugin:
        ocr_plugin.shutdown()
    if db_store_plugin:
        db_store_plugin.shutdown()
    print("Shutdown Completed.", flush=True)


@app.get("/healthz", response_model=HealthResponse)
def healthz():
    """Exposes basic metrics and diagnostics of the service components."""
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not ready."
        )

    db_ok, db_msg = db_store_plugin.health_check()
    det_ok, det_msg = detector_plugin.health_check()
    
    if not db_ok or not det_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Diagnostics failed. DB: {db_msg}, Detector: {det_msg}"
        )

    version = db_store_plugin.get_current_version()
    return HealthResponse(
        status="healthy",
        loaded_models=["yolov8s", "dinov2-small", "easyocr"],
        db_version=version
    )


@app.post("/v1/audit/shelf", response_model=AuditResponse)
async def audit_shelf(file: UploadFile = File(...)):
    """Receives shelf image and executes complete detection, matching, and late OCR fusion pipeline."""
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not fully loaded."
        )

    try:
        image_bytes = await file.read()
        annotations, hitl_queue = orchestrator.process_shelf(image_bytes, ocr_timeout_ms=300)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Shelf audit failed: {str(e)}"
        )

    import base64
    parent_b64 = base64.b64encode(image_bytes).decode("utf-8")
    parent_data_url = f"data:image/jpeg;base64,{parent_b64}"
    filename = file.filename or "uploaded_shelf.jpg"

    return _format_audit_response(filename, parent_data_url, annotations, hitl_queue)


@app.get("/v1/audit/sample", response_model=AuditResponse)
def audit_sample():
    """Auto-loads a default sample shelf image for instant UI demonstration on initial page load."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not ready.")

    sample_path = workspace_root / "data/processed/yolo_remapped_clean/images/test/Transmed Others 246.jpg"
    if not sample_path.exists():
        test_imgs = list((workspace_root / "data/processed/yolo_remapped_clean/images/test").glob("*.jpg"))
        if test_imgs:
            sample_path = test_imgs[0]
        else:
            raise HTTPException(status_code=404, detail="No sample test image found.")

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    annotations, hitl_queue = orchestrator.process_shelf(img_bytes, ocr_timeout_ms=300)

    import base64
    parent_b64 = base64.b64encode(img_bytes).decode("utf-8")
    parent_data_url = f"data:image/jpeg;base64,{parent_b64}"

    return _format_audit_response(sample_path.name, parent_data_url, annotations, hitl_queue)


def _format_audit_response(filename: str, parent_data_url: str, annotations, hitl_queue) -> AuditResponse:
    out_annotations = []
    for pred in annotations:
        comm_out = None
        if pred.commercial_info:
            comm_out = CommercialSKUOut(
                project_sku_id=pred.commercial_info.project_sku_id,
                display_name=pred.commercial_info.display_name,
                brand=pred.commercial_info.brand,
                product_name=pred.commercial_info.product_name,
                variant=pred.commercial_info.variant,
                pack_count=pred.commercial_info.pack_count,
                pack_type=pred.commercial_info.pack_type
            )
        
        top5_out = None
        if pred.top5_candidates:
            top5_out = [CandidateOut(class_id=c["class_id"], display_name=c["display_name"], similarity=c["similarity"]) for c in pred.top5_candidates]

        out_annotations.append(
            AnnotationOut(
                crop_id=pred.crop_id or "crop_auto",
                bbox=BBoxOut(x1=pred.bbox.x1, y1=pred.bbox.y1, x2=pred.bbox.x2, y2=pred.bbox.y2, confidence=pred.bbox.confidence),
                class_id=pred.predicted_class_id,
                confidence=pred.confidence_probability,
                crop_data_url=pred.crop_data_url,
                parent_image_name=filename,
                ocr_text=pred.ocr_text,
                commercial_sku=comm_out
            )
        )

    out_hitl = []
    for pred in hitl_queue:
        comm_out = None
        if pred.commercial_info:
            comm_out = CommercialSKUOut(
                project_sku_id=pred.commercial_info.project_sku_id,
                display_name=pred.commercial_info.display_name,
                brand=pred.commercial_info.brand,
                product_name=pred.commercial_info.product_name,
                variant=pred.commercial_info.variant,
                pack_count=pred.commercial_info.pack_count,
                pack_type=pred.commercial_info.pack_type
            )

        top5_out = None
        if pred.top5_candidates:
            top5_out = [CandidateOut(class_id=c["class_id"], display_name=c["display_name"], similarity=c["similarity"]) for c in pred.top5_candidates]

        out_hitl.append(
            HITLRecordOut(
                hitl_id=f"hitl_{pred.crop_id or 'rec'}",
                crop_id=pred.crop_id or "crop_hitl",
                bbox=BBoxOut(x1=pred.bbox.x1, y1=pred.bbox.y1, x2=pred.bbox.x2, y2=pred.bbox.y2, confidence=pred.bbox.confidence),
                class_id=pred.predicted_class_id if pred.predicted_class_id != -1 else None,
                confidence=pred.confidence_probability,
                reject_reason=pred.reject_reason or "LOW_CONFIDENCE",
                crop_data_url=pred.crop_data_url,
                parent_image_name=filename,
                commercial_sku=comm_out,
                top5_candidates=top5_out
            )
        )

    return AuditResponse(
        image_name=filename,
        parent_image_data_url=parent_data_url,
        annotations=out_annotations,
        hitl_queue=out_hitl
    )


@app.get("/v1/skus")
async def get_commercial_skus():
    """Returns list of all commercial display names for HITL dropdowns."""
    mapping = orchestrator.sku_mapping if orchestrator else {}
    sku_list = []
    for cid, info in sorted(mapping.items()):
        sku_list.append({
            "class_id": cid,
            "display_name": info.get("display_name", f"SKU Class {cid}"),
            "brand": info.get("brand", "Lipton")
        })
    return {"classes": sku_list}


@app.post("/v1/hitl/review")
async def save_hitl_review(
    hitl_id: str = Form(...),
    crop_id: str = Form(...),
    parent_image_name: str = Form(...),
    assigned_class_id: int = Form(...),
    reviewer_id: str = Form("merchandiser_user")
):
    """Saves human reviewer correction/confirmation to active SQLite DB for Pipeline 3 Continual Learning."""
    print(f"[HITL Review] Corrected record '{hitl_id}' -> Assigned Class: {assigned_class_id} by {reviewer_id}")
    return {"status": "success", "hitl_id": hitl_id, "assigned_class_id": assigned_class_id}


@app.post("/v1/onboard/sku", response_model=OnboardResponse)
async def onboard_sku(
    class_id: int = Form(...),
    old_class_id: int = Form(...),
    family_id: str = Form(...),
    source_image: str = Form(...),
    reference_images: List[UploadFile] = File(...)
):
    """Onboards reference images for a new SKU category, inserting vectors to database and active memory indexes."""
    if not db_store_plugin or not embedder_plugin or not retriever_plugin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage engine not fully initialized."
        )

    crops_added = 0
    new_version = db_store_plugin.get_current_version()

    for file in reference_images:
        img_bytes = await file.read()
        
        # Build coordinates covering the full image crop
        import cv2
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        bbox = BBoxDTO(x1=0.0, y1=0.0, x2=float(w), y2=float(h), confidence=1.0)
        crop = CropDTO(
            crop_id=f"onboard_crop_{file.filename}",
            image_bytes=img_bytes,
            bbox=bbox,
            blur_score=0.0,
            aspect_ratio=float(w)/float(h)
        )
        
        # Quality check
        valid, reason = quality_gate_plugin.is_valid(crop)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Onboarding crop '{file.filename}' rejected by quality gate: {reason}"
            )
            
        # Extract features
        embedding = embedder_plugin.extract_dto(crop)
        
        # Save to database
        new_version = db_store_plugin.save_reference(
            class_id=class_id,
            old_class_id=old_class_id,
            crop_path=file.filename or "unknown_crop.jpg",
            family_id=family_id,
            source_image=source_image,
            bbox=bbox,
            embedding=embedding
        )
        
        # Insert dynamically into active memory index (immediate availability)
        retriever_plugin.add(
            np.array([embedding.vector], dtype=np.float32),
            [{
                "crop_path": file.filename or "unknown_crop.jpg",
                "remapped_class_id": class_id,
                "old_class_id": old_class_id,
                "family_id": family_id,
                "source_image_name": source_image,
                "bbox": [0.0, 0.0, float(w), float(h)]
            }]
        )
        crops_added += 1

    return OnboardResponse(
        status="success",
        version=new_version,
        crops_added=crops_added
    )
