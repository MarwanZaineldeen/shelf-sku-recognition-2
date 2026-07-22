import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Repository root, resolved from this file so the server runs on any machine.
# Override with RETAIL_AI_ROOT if data and weights live outside the repo.
workspace_root = Path(os.environ.get("RETAIL_AI_ROOT", Path(__file__).resolve().parents[1]))
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
from ml.active_learning.hitl_store import HITLActiveLearningStore
from ml.orchestrator import AuditPipelineOrchestrator
from ml.active_learning.store import ReviewStore
from ml.active_learning.ingest import ReviewContextCache, record_review

# Pipeline 3 review storage is resolved relative to the repository, not the
# workspace_root above, so the continual learning loop stays portable across
# machines. (The remaining workspace_root paths are a separate known issue.)
repo_root = Path(__file__).resolve().parents[1]
review_db_path = repo_root / "data/processed/active_learning/reviews.db"

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

from fastapi.responses import FileResponse, JSONResponse, Response

@app.get("/favicon.ico")
def get_favicon():
    return Response(status_code=204)

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
    for mp in ["configs/sku_mapping_v2.json", "c:/Users/asusd/Desktop/sku_mapping_v2.json", "configs/sku_mapping.json"]:
        p = workspace_root / mp if not mp.startswith("c:") else Path(mp)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_classes = data.get("classes", {})
                # Key catalog cleanly by training_class_id
                by_training_id = {}
                for k, info in raw_classes.items():
                    t_id = info.get("training_class_id")
                    if t_id is not None:
                        by_training_id[str(t_id)] = info
                    else:
                        by_training_id[str(k)] = info
                return {"classes": by_training_id}
    return {"classes": {}}

# Global orchestrator and registry storage references
orchestrator: Any = None
<<<<<<< HEAD
detector_plugin: Any = None
embedder_plugin: Any = None
retriever_plugin: Any = None
ocr_plugin: Any = None
calibrator_plugin: Any = None
fusion_plugin: Any = None
decision_policy_plugin: Any = None
quality_gate_plugin: Any = None
db_store_plugin: Any = None
review_store_plugin: Any = None

# Holds audit-time context (embedding, candidate slate) between the audit
# response and the reviewer's verdict, so a reviewed crop keeps its vector
# without a second backbone pass.
review_context_cache = ReviewContextCache()
=======
# Global plugin instances
detector_plugin = None
quality_gate_plugin = None
embedder_plugin = None
retriever_plugin = None
ocr_plugin = None
calibrator_plugin = None
fusion_plugin = None
decision_policy_plugin = None
db_store_plugin = None
vlm_reranker_plugin = None
hitl_store_plugin = None
orchestrator = None
>>>>>>> 5f7b25090f9c7bc17b2c34d438731729d04d25a6


@app.on_event("startup")
def startup_event():
<<<<<<< HEAD
    global orchestrator, detector_plugin, embedder_plugin, retriever_plugin, ocr_plugin
    global calibrator_plugin, fusion_plugin, decision_policy_plugin, quality_gate_plugin, db_store_plugin
    global review_store_plugin
=======
    global detector_plugin, quality_gate_plugin, embedder_plugin, retriever_plugin
    global ocr_plugin, calibrator_plugin, fusion_plugin, decision_policy_plugin
    global db_store_plugin, vlm_reranker_plugin, hitl_store_plugin, orchestrator
>>>>>>> 5f7b25090f9c7bc17b2c34d438731729d04d25a6

    print("Starting up Retail AI Platform Service...", flush=True)

    # Instantiate plugins
    detector_plugin = YOLOv8Detector()
    quality_gate_plugin = BboxQualityGate()

    from ml.embeddings.dinov3 import DINOv3Extractor
    print("  Using DINOv3 ViT-B/16 SOTA 768-D Visual Backbone (Native)!", flush=True)
    embedder_plugin = DINOv3Extractor(device="cpu")
    retriever_plugin = NumpyCosineIndex(dimension=768)
    db_path = str(workspace_root / "data/processed/crops/gt_clean/retail_sku_registry_dinov3.db")

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
    hitl_store_plugin = HITLActiveLearningStore()

    print("  Initializing SQLite Gallery Store...", flush=True)
    db_store_plugin.initialize({"db_path": db_path})

<<<<<<< HEAD
    print("  Initializing Pipeline 3 Review Store...", flush=True)
    review_store_plugin = ReviewStore()
    review_store_plugin.initialize({"db_path": str(review_db_path)})
=======
    print("  Initializing Active Continual Learning HITL Store...", flush=True)
    hitl_store_plugin.initialize({
        "db_path": str(workspace_root / "data/processed/hitl_active_learning.db"),
        "gallery_db_path": db_path
    })
>>>>>>> 5f7b25090f9c7bc17b2c34d438731729d04d25a6

    print("  Initializing YOLOv8 Detector (SKU110K Class-Agnostic)...", flush=True)
    detector_plugin.initialize({
        "weights_path": str(workspace_root / "runs/detect/yolo8l_sku110k/yolov8l-sku110k.pt"),
        "confidence_threshold": 0.25,
        "imgsz": 640
    })

    print(f"  Initializing Cosine Search Index (768-D)...", flush=True)
    retriever_plugin.initialize({
        "dimension": 768,
        "db_path": db_path
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
    if review_store_plugin:
        review_store_plugin.shutdown()
    review_context_cache.clear()
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

    import time
    t0 = time.perf_counter()

    try:
        image_bytes = await file.read()
        annotations, hitl_queue = orchestrator.process_shelf(image_bytes, ocr_timeout_ms=300)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Shelf audit failed: {str(e)}"
        )

    proc_time_ms = (time.perf_counter() - t0) * 1000.0

    import base64
    parent_b64 = base64.b64encode(image_bytes).decode("utf-8")
    parent_data_url = f"data:image/jpeg;base64,{parent_b64}"
    filename = file.filename or "uploaded_shelf.jpg"

    return _format_audit_response(filename, parent_data_url, annotations, hitl_queue, proc_time_ms)


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

    import time
    t0 = time.perf_counter()

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    annotations, hitl_queue = orchestrator.process_shelf(img_bytes, ocr_timeout_ms=300)

    proc_time_ms = (time.perf_counter() - t0) * 1000.0

    import base64
    parent_b64 = base64.b64encode(img_bytes).decode("utf-8")
    parent_data_url = f"data:image/jpeg;base64,{parent_b64}"

    return _format_audit_response(sample_path.name, parent_data_url, annotations, hitl_queue, proc_time_ms)


<<<<<<< HEAD
def _format_audit_response(filename: str, parent_data_url: str, annotations, hitl_queue) -> AuditResponse:
    # Retain audit context for anything a human may review. Embeddings are
    # cached server-side and deliberately never enter the response schema —
    # 768 floats per crop would bloat the payload for no client benefit.
    review_context_cache.put_predictions(filename, hitl_queue)
    review_context_cache.put_predictions(filename, annotations)

=======
def _format_audit_response(filename: str, parent_data_url: str, annotations, hitl_queue, proc_time_ms: float = 0.0) -> AuditResponse:
>>>>>>> 5f7b25090f9c7bc17b2c34d438731729d04d25a6
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
            top5_out = [
                CandidateOut(
                    class_id=c["class_id"],
                    display_name=c["display_name"],
                    similarity=float(c.get("similarity", 0.0)),
                    vlm_selected=bool(c.get("qwen2_vl_verified", False)),
                    s_fused=float(c["s_fused"]) if "s_fused" in c else None,
                    exemplar_url=c.get("exemplar_url", f"/v1/exemplars/{c['class_id']}")
                )
                for c in pred.top5_candidates
            ]

        is_vlm = True if (pred.ocr_text and "VLM" in pred.ocr_text) else False

        out_annotations.append(
            AnnotationOut(
                crop_id=pred.crop_id or "crop_auto",
                bbox=BBoxOut(x1=pred.bbox.x1, y1=pred.bbox.y1, x2=pred.bbox.x2, y2=pred.bbox.y2, confidence=pred.bbox.confidence),
                class_id=pred.predicted_class_id,
                confidence=pred.confidence_probability,
                crop_data_url=pred.crop_data_url,
                parent_image_name=filename,
                ocr_text=pred.ocr_text,
                vlm_verified=is_vlm,
                vlm_reason=pred.ocr_text if is_vlm else None,
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
            top5_out = [
                CandidateOut(
                    class_id=c["class_id"],
                    display_name=c["display_name"],
                    similarity=float(c.get("similarity", 0.0)),
                    vlm_selected=bool(c.get("qwen2_vl_verified", False)),
                    s_fused=float(c["s_fused"]) if "s_fused" in c else None,
                    exemplar_url=c.get("exemplar_url", f"/v1/exemplars/{c['class_id']}")
                )
                for c in pred.top5_candidates
            ]

        is_vlm = True if (pred.ocr_text and "VLM" in pred.ocr_text) else False

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
                vlm_verified=is_vlm,
                vlm_reason=pred.ocr_text if is_vlm else None,
                commercial_sku=comm_out,
                top5_candidates=top5_out
            )
        )

    return AuditResponse(
        image_name=filename,
        parent_image_data_url=parent_data_url,
        processing_time_ms=proc_time_ms,
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
    reviewer_id: str = Form("merchandiser_user"),
    predicted_class_id: int = Form(-1),
    top1_similarity: float = Form(0.0)
):
<<<<<<< HEAD
    """Persists a human review to the Pipeline 3 review store.

    A negative assigned_class_id is the dashboard's "Unknown / Non-Catalog
    Competitor SKU" choice and is recorded as an open-set rejection.

    The audit-time embedding and candidate slate come from the server-side
    context cache. On a miss — a restart between audit and review — the
    review is still stored using the client-supplied prediction fields;
    it simply cannot be promoted into the gallery later.
    """
    if not review_store_plugin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Review store not initialized."
        )

    context = review_context_cache.get(parent_image_name, crop_id)

    try:
        review_id = record_review(
            store=review_store_plugin,
            source_image=parent_image_name,
            crop_id=crop_id,
            assigned_class_id=assigned_class_id,
            reviewer_id=reviewer_id,
            context=context,
            predicted_class_id=predicted_class_id,
            top1_similarity=top1_similarity,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    record = review_store_plugin.fetch_review(review_id)
    print(
        f"[HITL Review] {hitl_id} -> {record.decision} "
        f"(class {assigned_class_id}) by {reviewer_id}; "
        f"embedding {'captured' if record.embedding else 'unavailable'}",
        flush=True
    )

    return {
        "status": "success",
        "hitl_id": hitl_id,
        "review_id": review_id,
        "decision": record.decision,
        "assigned_class_id": assigned_class_id,
        "embedding_captured": record.embedding is not None,
        "pending_reviews": review_store_plugin.count_unconsumed(),
    }
=======
    """Saves human reviewer correction/confirmation to active SQLite DB for Pipeline 3 Continual Learning."""
    disp_name = "Class Unknown"
    if orchestrator and assigned_class_id != -1 and hasattr(orchestrator, "sku_mapping"):
        disp_name = orchestrator.sku_mapping.get(assigned_class_id, {}).get("display_name", f"Class {assigned_class_id}")
    
    if hitl_store_plugin:
        try:
            hitl_store_plugin.correct_task(
                task_id=hitl_id,
                correct_class_id=assigned_class_id,
                correct_display_name=disp_name,
                verifier_notes=f"Reviewed by {reviewer_id}"
            )
        except Exception as e:
            print(f"[HITL Store] Log note: {e}")

    print(f"[HITL Review] Corrected & logged record '{hitl_id}' -> Class: {assigned_class_id} ({disp_name}) by {reviewer_id}")
    return {"status": "success", "hitl_id": hitl_id, "assigned_class_id": assigned_class_id, "display_name": disp_name}
>>>>>>> 5f7b25090f9c7bc17b2c34d438731729d04d25a6


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
        class_id=class_id,
        version=new_version,
        crops_added=crops_added,
        message=f"Successfully onboarded {crops_added} new crop references for class {class_id}."
    )


@app.get("/v1/exemplars/{class_id}")
def get_class_exemplar(class_id: int):
    """Returns the highest quality reference crop thumbnail for a commercial class ID from Sku Preview."""
    from fastapi.responses import Response

    # Primary source: User's manually curated high-resolution Sku Preview directory
    sku_preview_dir = workspace_root / "data/processed/Sku Preview" / f"class_{class_id}"
    if sku_preview_dir.exists():
        images = sorted(list(sku_preview_dir.glob("*.jpg")) + list(sku_preview_dir.glob("*.png")))
        if images:
            with open(images[0], "rb") as f:
                return Response(content=f.read(), media_type="image/jpeg")

    # Secondary fallback: ground-truth crop directories
    for parent in ["data/processed/crops/gt", "data/processed/crops/gt_clean"]:
        for split in ["train", "test"]:
            class_dir = workspace_root / parent / split / f"class_{class_id}"
            if class_dir.exists():
                images = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
                if images:
                    with open(images[0], "rb") as f:
                        return Response(content=f.read(), media_type="image/jpeg")

    # SVG fallback icon if no crop file found
    svg_fallback = f"""<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
        <rect width="64" height="64" fill="#1e293b" rx="8"/>
        <text x="32" y="38" font-family="Outfit, sans-serif" font-size="14" font-weight="bold" fill="#38bdf8" text-anchor="middle">SKU {class_id}</text>
    </svg>"""
    return Response(content=svg_fallback, media_type="image/svg+xml")
