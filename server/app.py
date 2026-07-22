import os
from pathlib import Path
from typing import List, Dict, Any, Optional
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
from ml.retrieval.hierarchical_index import HierarchicalCosineIndex
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

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# 1. Initialize Prometheus Exporter
instrumentator = Instrumentator().instrument(app)

# 2. Custom Business & ML Metrics Definition
stage_latency_histogram = Histogram(
    "retail_ai_stage_latency_seconds",
    "Execution time per pipeline stage in seconds",
    ["stage"]  # yolo_detector, dinov3_embedder, vector_search, qwen2_vl, platt_calibrator
)

# Done
auto_annotation_ratio_gauge = Gauge(
    "retail_ai_auto_annotation_ratio",
    "Ratio of auto-approved facings vs total detected facings"
)

facings_detected_counter = Gauge(
    "retail_ai_facings_detected_total",
    "Total product facings detected across shelf scans"
)

vlm_triggers_counter = Counter(
    "retail_ai_vlm_triggers_total",
    "Total times Qwen2-VL reranker was activated for ambiguous crops"
)

# Done
hitl_reviews_counter = Counter(
    "retail_ai_hitl_reviews_total",
    "Total HITL reviews submitted by merchandisers",
    ["type"]  # confirmation vs correction
)

sku_registry_count_gauge = Gauge(
    "retail_ai_sqlite_vector_count",
    "Total 768-D DINOv3 vectors indexed in SQLite registry"
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
vlm_reranker_plugin: Any = None
hitl_store_plugin: Any = None

@app.on_event("startup")
def startup_event():
    global orchestrator, detector_plugin, embedder_plugin, retriever_plugin, ocr_plugin
    global calibrator_plugin, fusion_plugin, decision_policy_plugin, quality_gate_plugin, db_store_plugin
    global review_store_plugin, vlm_reranker_plugin, hitl_store_plugin

    instrumentator.expose(app, endpoint="/metrics")

    print("Starting up Retail AI Platform Service...", flush=True)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Load lexicon configuration
    with open(lexicon_path, "r") as f:
        lexicons = json.load(f)

    # 3. Instantiate concrete plugins
    detector_plugin = YOLOv8Detector()
    quality_gate_plugin = BboxQualityGate()

    try:
        from ml.embeddings.dinov3 import DINOv3Extractor
        embedder_plugin = DINOv3Extractor(device="cpu")
        db_path = str(workspace_root / "data/processed/crops/gt_clean/retail_sku_registry_dinov3.db")
        dimension = 768
        retriever_plugin = HierarchicalCosineIndex(dimension=768)
        print("  Using DINOv3 ViT-B/16 SOTA 768-D Visual Backbone!", flush=True)
    except Exception as e:
        print(f"  DINOv3 offline model not available ({e}). Falling back to DINOv2 (384-D)...", flush=True)
        embedder_plugin = DINOv2Extractor(model_name="facebook/dinov2-small", device="cpu")
        db_path = str(workspace_root / "data/processed/crops/gt_clean/retail_sku_registry_onboarding.db")
        dimension = 384
        retriever_plugin = HierarchicalCosineIndex(dimension=384)

    vlm_reranker_plugin = None
    try:
        from ml.vlm.qwen2_vl_reranker import Qwen2VLReranker
        vlm_reranker_plugin = Qwen2VLReranker()
        vlm_reranker_plugin.initialize({
            "model_id": "Qwen/Qwen2-VL-2B-Instruct-AWQ",
            "device": "cpu",
            "local_files_only": True
        })
    except Exception as e:
        print(f"  Qwen2-VL Reranker offline model skipped: {e}", flush=True)

    calibrator_plugin = PlattCalibrator()
    decision_policy_plugin = GatedAnnotationPolicy()
    db_store_plugin = SQLiteGalleryStore()
    hitl_store_plugin = HITLActiveLearningStore()

    print("  Initializing SQLite Gallery Store...", flush=True)
    db_store_plugin.initialize({"db_path": db_path})

    try:
        review_store_plugin = ReviewStore()
        review_store_plugin.initialize({"db_path": str(review_db_path)})
    except Exception:
        pass

    try:
        hitl_store_plugin.initialize({
            "db_path": str(workspace_root / "data/processed/hitl_active_learning.db"),
            "gallery_db_path": db_path
        })
    except Exception:
        pass

    print("  Initializing YOLOv8 Detector (SKU110K Class-Agnostic)...", flush=True)
    yolo_weights = workspace_root / "runs/detect/yolo8l_sku110k/yolov8l-sku110k.pt"
    if not yolo_weights.exists():
        alt_weights = Path("D:/ITI/Graduation Project/Project GitHub/yolov8l-sku110k.pt")
        if alt_weights.exists():
            yolo_weights = alt_weights
        else:
            yolo_weights = "yolov8n.pt"

    detector_plugin.initialize({
        "weights_path": str(yolo_weights),
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

    if retriever_plugin and hasattr(retriever_plugin, "__len__"):
        try:
            sku_registry_count_gauge.set(len(retriever_plugin))
        except Exception:
            pass

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


def _format_audit_response(filename: str, parent_data_url: str, annotations, hitl_queue, proc_time_ms: float = 0.0) -> AuditResponse:
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

    total_facings = len(out_annotations) + len(out_hitl)
    if total_facings > 0:
        facings_detected_counter.inc(total_facings)
        auto_annotation_ratio_gauge.set(len(out_annotations) / float(total_facings))

    vlm_count = sum(1 for a in out_annotations if a.vlm_verified) + sum(1 for h in out_hitl if h.vlm_verified)
    if vlm_count > 0:
        vlm_triggers_counter.inc(vlm_count)

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


def compute_next_class_id() -> int:
    """Calculates the next available auto-incremented class ID across database, mapping JSONs, and orchestrator."""
    max_id = -1
    if db_store_plugin and hasattr(db_store_plugin, "get_max_class_id"):
        max_id = max(max_id, db_store_plugin.get_max_class_id())

    for cfg_name in ["configs/sku_mapping_v2.json", "configs/sku_mapping.json"]:
        cfg_path = workspace_root / cfg_name
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cat_data = json.load(f)
                classes = cat_data.get("classes", {})
                for k, v in classes.items():
                    c_id = v.get("training_class_id", v.get("raw_class_id", k))
                    try:
                        max_id = max(max_id, int(c_id))
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass

    if orchestrator and hasattr(orchestrator, "sku_mapping"):
        for k in orchestrator.sku_mapping.keys():
            try:
                max_id = max(max_id, int(k))
            except (ValueError, TypeError):
                pass

    return max_id + 1 if max_id >= 0 else 0


@app.get("/v1/next-class-id")
def get_next_class_id():
    """Returns the next available auto-incremented class ID for new SKU onboarding."""
    return {"next_class_id": compute_next_class_id()}


@app.post("/v1/onboard/sku", response_model=OnboardResponse)
async def onboard_sku(
    class_id: int | None = Form(None),
    brand: str | None = Form(None),
    product_name: str | None = Form(None),
    family_id: str | None = Form(None),
    old_class_id: int | None = Form(None),
    variant: str | None = Form(""),
    size: str | None = Form(""),
    pack_count: str | None = Form(""),
    pack_type: str | None = Form("box"),
    display_name: str | None = Form(None),
    notes: str | None = Form(""),
    source_image: str | None = Form("web_ui_onboard"),
    folder_path: str | None = Form(None),
    reference_images: list[UploadFile] | None = File(None),
    validation_shelf_image: UploadFile | None = File(None)
):
    """Onboards reference images for a new SKU category, supporting 10-50 crops, full catalog metadata, and optional shelf validation benchmark."""
    if not db_store_plugin or not embedder_plugin or not retriever_plugin:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage engine not fully initialized."
        )

    # Auto-increment class_id if not specified (None, -1, or negative)
    if class_id is None or class_id == -1 or class_id < 0:
        class_id = compute_next_class_id()

    # Validate crop count bound (up to 50 crops)
    if reference_images is not None:
        valid_files = [f for f in reference_images if f.filename]
        if len(valid_files) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Onboarding supports up to 50 reference product crops (Received: {len(valid_files)})."
            )

    effective_brand = brand if brand and brand.strip() else (family_id if family_id and family_id.strip() else "New Brand")
    effective_product_name = product_name if product_name and product_name.strip() else f"Product (Class {class_id})"
    effective_old_class_id = old_class_id if old_class_id is not None else class_id
    effective_display_name = display_name if display_name and display_name.strip() else f"{effective_brand} {effective_product_name} {variant}".strip()
    family_cluster_id = effective_brand
    source_img_tag = source_image if source_image else "web_ui_onboard"

    crops_added = 0
    new_version = db_store_plugin.get_current_version()

    # 1. Onboard from server folder path if provided
    if folder_path and folder_path.strip():
        from ml.onboarding.onboarder import SKUOnboarder
        onboarder = SKUOnboarder(
            embedder=embedder_plugin,
            store=db_store_plugin,
            retriever=retriever_plugin,
            detector=detector_plugin
        )
        target_path = Path(folder_path.strip())
        if target_path.exists():
            res = onboarder.onboard_from_crops(
                crops_dir=target_path,
                class_id=class_id,
                old_class_id=effective_old_class_id,
                family_id=family_cluster_id,
                source_image=source_img_tag,
                detector=detector_plugin
            )
            crops_added += res.get("crops_added", 0)
            new_version = res.get("db_version", new_version)

    # 2. Onboard from uploaded reference files if provided
    if reference_images:
        for file in reference_images:
            if not file.filename:
                continue
            img_bytes = await file.read()
            if not img_bytes:
                continue
            
            import cv2
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            bbox = BBoxDTO(x1=0.0, y1=0.0, x2=float(w), y2=float(h), confidence=1.0)

            # Detect product using YOLOv8l detector plugin and crop strictly on product region
            if detector_plugin:
                try:
                    boxes = detector_plugin.detect(img_bytes)
                    if boxes:
                        best_box = max(boxes, key=lambda b: getattr(b, "confidence", 1.0))
                        b_x1 = getattr(best_box, "x1", 0.0)
                        b_y1 = getattr(best_box, "y1", 0.0)
                        b_x2 = getattr(best_box, "x2", float(w))
                        b_y2 = getattr(best_box, "y2", float(h))
                        conf = float(getattr(best_box, "confidence", 1.0))

                        x1 = max(0, int(b_x1) if b_x1 > 1.0 else int(b_x1 * w))
                        y1 = max(0, int(b_y1) if b_y1 > 1.0 else int(b_y1 * h))
                        x2 = min(w, int(b_x2) if b_x2 > 1.0 else int(b_x2 * w))
                        y2 = min(h, int(b_y2) if b_y2 > 1.0 else int(b_y2 * h))

                        if (x2 - x1) > 5 and (y2 - y1) > 5:
                            cropped_img = img[y1:y2, x1:x2]
                            success, enc_bytes = cv2.imencode(".jpg", cropped_img)
                            if success:
                                img_bytes = enc_bytes.tobytes()
                                img = cropped_img
                                h, w = cropped_img.shape[:2]
                                bbox = BBoxDTO(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=conf)
                                print(f"[Pipeline 2] YOLOv8l localized product in {file.filename} -> box [{x1}, {y1}, {x2}, {y2}] ({w}x{h}px)")
                except Exception as det_err:
                    print(f"[Pipeline 2 Warning] YOLOv8l product detection failed for {file.filename}: {det_err}")

            crop = CropDTO(
                crop_id=f"onboard_crop_{file.filename}",
                image_bytes=img_bytes,
                bbox=bbox,
                blur_score=0.0,
                aspect_ratio=float(w) / float(max(1, h))
            )
            
            # Extract embedding (bypassing Quality Gate for onboarding reference crops to maximize visual diversity)
            embedding = embedder_plugin.extract_dto(crop)
            
            new_version = db_store_plugin.save_reference(
                class_id=class_id,
                old_class_id=effective_old_class_id,
                crop_path=file.filename,
                family_id=family_cluster_id,
                source_image=source_img_tag,
                bbox=bbox,
                embedding=embedding
            )
            
            retriever_plugin.add(
                np.array([embedding.vector], dtype=np.float32),
                [{
                    "crop_path": file.filename,
                    "remapped_class_id": class_id,
                    "old_class_id": effective_old_class_id,
                    "family_id": family_cluster_id,
                    "source_image_name": source_img_tag,
                    "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
                }]
            )
            crops_added += 1

            # Sync 1st reference crop thumbnail to data/processed/Sku Preview/class_{class_id}/
            if crops_added == 1:
                try:
                    preview_dir = workspace_root / "data/processed/Sku Preview" / f"class_{class_id}"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    preview_file = preview_dir / "crop_0.jpg"
                    with open(preview_file, "wb") as pf:
                        pf.write(img_bytes)
                    print(f"[Pipeline 2] Synced catalog thumbnail to: {preview_file}")
                except Exception as p_err:
                    print(f"[Pipeline 2 Warning] Failed to sync Sku Preview thumbnail: {p_err}")

    if retriever_plugin and hasattr(retriever_plugin, "__len__"):
        try:
            sku_registry_count_gauge.set(len(retriever_plugin))
        except Exception:
            pass

    # 3. Save full metadata schema into sku_mapping.json & sku_mapping_v2.json
    sku_meta_record = {
        "raw_class_id": str(class_id),
        "training_class_id": class_id,
        "project_sku_id": f"TM_RAW_{class_id:03d}",
        "brand": effective_brand,
        "product_name": effective_product_name,
        "variant": variant or "",
        "size": size or "",
        "pack_count": pack_count or f"{crops_added} crops",
        "pack_type": pack_type or "box",
        "display_name": effective_display_name,
        "status": "verified",
        "identity_confidence": "A",
        "instance_count": crops_added,
        "source_image_count": crops_added,
        "evidence": "Onboarded via Web UI (Pipeline 2)",
        "notes": notes or "Dynamic Onboarding"
    }

    for cfg_name in ["configs/sku_mapping.json", "configs/sku_mapping_v2.json"]:
        cfg_path = workspace_root / cfg_name
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cat_data = json.load(f)
                if "classes" in cat_data:
                    cat_data["classes"][str(class_id)] = sku_meta_record
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(cat_data, f, indent=2)
            except Exception as err:
                print(f"[Warning] Failed to save {cfg_name}: {err}")

    if orchestrator and hasattr(orchestrator, "sku_mapping"):
        orchestrator.sku_mapping[class_id] = sku_meta_record

    # 4. Optional Validation Shelf Audit Benchmark
    validation_audit_res = None
    if validation_shelf_image and validation_shelf_image.filename:
        try:
            val_img_bytes = await validation_shelf_image.read()
            if val_img_bytes and len(val_img_bytes) > 0 and detector_plugin:
                from ml.onboarding.onboarder import SKUOnboarder
                onboarder = SKUOnboarder(embedder=embedder_plugin, store=db_store_plugin, retriever=retriever_plugin)
                validation_audit_res = onboarder.validate_sku_on_shelf(
                    shelf_img_bytes=val_img_bytes,
                    class_id=class_id,
                    detector=detector_plugin
                )
        except Exception as val_err:
            print(f"[Warning] Validation shelf audit error: {val_err}")
            validation_audit_res = {
                "facings_detected": 0,
                "mean_similarity": 0.0,
                "pass_validation": False,
                "recommendation": f"Shelf audit skipped or failed: {val_err}"
            }

    return OnboardResponse(
        status="success",
        class_id=class_id,
        version=new_version,
        crops_added=crops_added,
        message=f"Successfully onboarded {crops_added} new crop references for class {class_id}.",
        metadata=sku_meta_record,
        validation_audit=validation_audit_res
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
