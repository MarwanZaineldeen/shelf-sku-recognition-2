import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CropGenerator:
    """Orchestrates crop extraction, padding, and directory exports."""

    def __init__(self, padding: float = 0.05):
        self.padding = padding

    def extract_crop(self, image: np.ndarray, box: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Converts normalized coordinates to absolute pixels, applies padding,
        clips boundaries (python slice boundaries: x2 <= w, y2 <= h),
        and extracts the crop.

        Returns:
            crop_image: extracted image numpy array
            absolute_coords: tuple of (x1, y1, x2, y2)
        """
        if image is None or image.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8), (0, 0, 0, 0)

        h, w = image.shape[:2]
        xc = box["x_center_norm"]
        yc = box["y_center_norm"]
        bw = box["width_norm"]
        bh = box["height_norm"]

        # Apply padding multiplier to normalized width and height
        pad_w = bw * (1.0 + 2.0 * self.padding)
        pad_h = bh * (1.0 + 2.0 * self.padding)

        # Convert to absolute pixel boundaries
        x1 = int((xc - pad_w / 2.0) * w)
        y1 = int((yc - pad_h / 2.0) * h)
        x2 = int((xc + pad_w / 2.0) * w)
        y2 = int((yc + pad_h / 2.0) * h)

        # Correct boundary clipping for Python slicing (exclusive end indices)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Handle crops with zero area or invalid coordinate ranges
        if x1 >= x2 or y1 >= y2:
            return np.empty((0, 0, 3), dtype=np.uint8), (x1, y1, x2, y2)

        crop = image[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)
