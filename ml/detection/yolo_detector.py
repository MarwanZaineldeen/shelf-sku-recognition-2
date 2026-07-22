import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from ml.base import BaseDetector, BBoxDTO


class YOLOv8Detector(BaseDetector):
    """Ultralytics YOLOv8 object detector plugin."""

    def __init__(self) -> None:
        self.model = None
        self.weights_path = ""
        self.confidence_threshold = 0.25
        self.imgsz = 640

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes and loads weights for the YOLO model.

        Config schema:
            weights_path: str (required)
            confidence_threshold: float (default: 0.25)
            imgsz: int (default: 640)
        """
        self.weights_path = config.get("weights_path")
        if not self.weights_path:
            raise ValueError("Configuration must specify 'weights_path'.")
        
        self.confidence_threshold = config.get("confidence_threshold", self.confidence_threshold)
        self.imgsz = config.get("imgsz", self.imgsz)

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"YOLO weights not found at: {self.weights_path}")

        # Lazy import of Ultralytics YOLO
        from ultralytics import YOLO
        try:
            self.model = YOLO(self.weights_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def health_check(self) -> Tuple[bool, str]:
        if self.model is None:
            return False, "YOLO model not loaded."
        return True, "Healthy"

    def shutdown(self) -> None:
        self.model = None

    def detect(self, image_bytes: bytes) -> List[BBoxDTO]:
        """Detects shelf product packaging boxes.

        Args:
            image_bytes: Raw image byte stream payload.

        Returns:
            List[BBoxDTO]: Decoded bounding boxes.
        """
        if not self.model:
            raise RuntimeError("YOLO model not initialized.")

        # Decode image bytes via PIL
        import io
        from PIL import Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Execute YOLO predict (run on CPU by default for stability)
        results = self.model.predict(
            source=pil_img,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            device="cpu",
            verbose=False
        )

        bboxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    bboxes.append(
                        BBoxDTO(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=conf
                        )
                    )
        return bboxes
