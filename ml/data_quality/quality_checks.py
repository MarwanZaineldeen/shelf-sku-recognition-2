import io
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
from ml.base import BaseQualityGate, CropDTO


class BboxQualityGate(BaseQualityGate):
    """Quality gate validating size, aspect ratio, and blur of crops."""

    def __init__(self) -> None:
        self.min_area = 1024
        self.max_aspect = 5.0
        self.min_blur = 30.0

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes thresholds from configuration."""
        self.min_area = config.get("min_area", self.min_area)
        self.max_aspect = config.get("max_aspect", self.max_aspect)
        self.min_blur = config.get("min_blur", self.min_blur)

    def health_check(self) -> Tuple[bool, str]:
        return True, "Healthy"

    def shutdown(self) -> None:
        pass

    def is_valid(self, crop: CropDTO) -> Tuple[bool, str]:
        """Validates crop metrics using the predefined thresholds.

        Args:
            crop: The CropDTO to check.

        Returns:
            Tuple[bool, str]: (is_valid, reject_reason)
        """
        # 1. Check size / area
        nparr = np.frombuffer(crop.image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False, "CORRUPTED_IMAGE"

        h, w = img_bgr.shape[:2]
        area = w * h
        if area < self.min_area:
            return False, "LOW_RESOLUTION"

        # 2. Check aspect ratio
        aspect = float(w) / float(h)
        if aspect > self.max_aspect or (1.0 / aspect) > self.max_aspect:
            return False, "EXTREME_ASPECT_RATIO"

        # 3. Check blur (Laplacian variance)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.min_blur:
            return False, "BLURRY_IMAGE"

        return True, ""
