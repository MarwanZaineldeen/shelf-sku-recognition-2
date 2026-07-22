import os
import sys
import yaml
import time
import argparse
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.base import BBoxDTO
from ml.detection.yolo_detector import YOLOv8Detector
from ml.data_quality.quality_checks import BboxQualityGate
from ml.embeddings.dinov2 import DINOv2Extractor
from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from ml.retrieval.numpy_index import NumpyCosineIndex
from ml.ocr.easy_ocr import EasyOCREngine
from ml.calibrators.platt import PlattCalibrator
from ml.fusion.lexicon_fusion import LexiconLateFusion
from ml.decision.gated_policy import GatedAnnotationPolicy
from ml.orchestrator import AuditPipelineOrchestrator

workspace_root = Path("d:/Marwan/ITI AI&ML/Transmid GP")
config_path = workspace_root / "configs/retrieval_config.yaml"
lexicon_path = workspace_root / "configs/class_lexicons.json"


def main():
    parser = argparse.ArgumentParser(description="End-to-End Supermarket Shelf Audit Processing CLI")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the shelf image to audit"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run models on ('cpu', 'cuda')"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="OCR timeout threshold in milliseconds (default: 300)"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    print("==================================================")
    print("Retail AI Platform: End-to-End Shelf Audit CLI")
    print(f"Image:  {image_path.name}")
    print(f"Device: {args.device}")
    print("==================================================")

    # 1. Load configs
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Instantiate and initialize component plugins
    db_path = str(workspace_root / "data/processed/crops/gt_clean/retail_sku_registry_dinov2.db")
    
    print("\n[1/3] Initializing platform plugins...", flush=True)
    
    detector = YOLOv8Detector()
    detector.initialize({
        "weights_path": str(workspace_root / "runs/detect/yolo8l_sku110k/yolov8l-sku110k.pt"),
        "confidence_threshold": 0.25,
        "imgsz": 640
    })

    quality_gate = BboxQualityGate()
    quality_gate.initialize({
        "min_area": 1024,
        "max_aspect": 5.0,
        "min_blur": 30.0
    })

    embedder = DINOv2Extractor(model_name="", device=args.device)
    embedder.initialize({
        "model_name": "facebook/dinov2-small",
        "device": args.device,
        "batch_size": 32
    })

    retriever = NumpyCosineIndex()
    retriever.initialize({
        "dimension": 384,
        "db_path": db_path
    })

    ocr = EasyOCREngine()
    ocr.initialize({
        "languages": ["en"],
        "gpu": (args.device == "cuda")
    })

    calibrator = PlattCalibrator()
    calibrator.initialize({
        "global_coefs": {"a": 0.7015, "b": 0.7943}
    })

    import json
    with open(lexicon_path, "r") as f:
        lexicons = json.load(f)

    fusion = LexiconLateFusion()
    fusion.initialize({
        "boost_alpha": 0.05,
        "lexicons": lexicons
    })

    decision_policy = GatedAnnotationPolicy()
    decision_policy.initialize({
        "global_threshold": 0.95
    })

    # 3. Assemble orchestrator
    orchestrator = AuditPipelineOrchestrator(
        detector, quality_gate, embedder, retriever, ocr, calibrator, fusion, decision_policy
    )
    print("Plugins initialized. Pipeline assembled successfully.")

    # 4. Read shelf image
    print("\n[2/3] Executing shelf audit pipeline...", flush=True)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    t0 = time.time()
    annotations, hitl_queue = orchestrator.process_shelf(image_bytes, ocr_timeout_ms=args.timeout)
    elapsed = time.time() - t0

    # 5. Output Audit summary stats
    print("\n[3/3] Audit Execution completed:")
    print(f"  Total processing latency: {elapsed:.3f} seconds")
    print(f"  Automated Detections (Auto-Annotations): {len(annotations)}")
    print(f"  Uncertain Detections (Routed to HITL):    {len(hitl_queue)}")
    
    total_detections = len(annotations) + len(hitl_queue)
    if total_detections > 0:
        auto_rate = (len(annotations) / total_detections) * 100
        print(f"  Calculated Automation Rate: {auto_rate:.2f}%")
    
    print("\nAutomated Detections (Classified SKUs):")
    for idx, pred in enumerate(annotations, start=1):
        print(
            f"  {idx}. Class ID: {pred.predicted_class_id} | "
            f"BBox: [{pred.bbox.x1:.1f}, {pred.bbox.y1:.1f}, {pred.bbox.x2:.1f}, {pred.bbox.y2:.1f}] | "
            f"Confidence: {pred.confidence_probability * 100:.2f}%"
        )

    print("\nHITL Queue Detections (Awaiting Verification):")
    for idx, pred in enumerate(hitl_queue, start=1):
        print(
            f"  {idx}. Reason: {pred.reject_reason} | "
            f"BBox: [{pred.bbox.x1:.1f}, {pred.bbox.y1:.1f}, {pred.bbox.x2:.1f}, {pred.bbox.y2:.1f}] | "
            f"Similarity/Confidence: {pred.confidence_probability * 100:.2f}%"
        )
    print("==================================================")


if __name__ == "__main__":
    main()
