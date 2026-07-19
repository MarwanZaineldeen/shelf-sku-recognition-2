import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class QualityChecker:
    """Validates visual qualities and coordinates of product crops."""

    def __init__(
        self,
        min_crop_size: int = 20,
        max_aspect_ratio: float = 5.0,
        blur_threshold: float = 30.0
    ):
        self.min_crop_size = min_crop_size
        self.max_aspect_ratio = max_aspect_ratio
        self.blur_threshold = blur_threshold

    def calculate_blur_laplacian(self, crop: np.ndarray) -> float:
        """
        Computes the Laplacian variance of the crop to estimate blur.
        Higher value means sharper image.
        """
        if crop is None or crop.size == 0:
            return 0.0
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # If the crop is extremely small (e.g. less than 3x3), Laplacian might fail or be unstable.
            if gray.shape[0] < 3 or gray.shape[1] < 3:
                return 0.0
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception as e:
            logger.warning(f"Failed to calculate Laplacian variance: {str(e)}")
            return 0.0

    def check_crop(self, crop: np.ndarray) -> tuple[str, str]:
        """
        Runs size, aspect ratio, and blur checks on an image crop.

        Returns:
            quality_flag: 'ok', 'too_small', 'extreme_aspect_ratio', 'blurry', 'invalid_crop'
            quality_notes: details explaining the flag
        """
        if crop is None or crop.size == 0:
            return "invalid_crop", "Crop image data is empty or has zero area."

        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return "invalid_crop", f"Invalid crop dimensions: {w}x{h}"

        # Size check
        if h < self.min_crop_size or w < self.min_crop_size:
            return (
                "too_small",
                f"Crop size {w}x{h} is smaller than minimum size limit {self.min_crop_size}px.",
            )

        # Aspect ratio check
        aspect_ratio = max(h / w, w / h)
        if aspect_ratio > self.max_aspect_ratio:
            return (
                "extreme_aspect_ratio",
                f"Aspect ratio {aspect_ratio:.2f} exceeds limit {self.max_aspect_ratio}.",
            )

        # Blur check
        blur_score = self.calculate_blur_laplacian(crop)
        if blur_score < self.blur_threshold:
            return (
                "blurry",
                f"Laplacian variance {blur_score:.2f} is below blur threshold {self.blur_threshold}.",
            )

        return "ok", "Passed sanity checks."
