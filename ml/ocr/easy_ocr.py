import io
import cv2
import numpy as np
import concurrent.futures
from typing import Dict, Any, Tuple
from ml.base import BaseOCR, CropDTO, OCRResultDTO


class EasyOCREngine(BaseOCR):
    """EasyOCR implementation with execution timeout limits."""

    def __init__(self) -> None:
        self.reader = None
        self.languages = ["en"]
        self.gpu = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes the EasyOCR reader.

        Config schema:
            languages: ["en", "ar"] (default: ["en"])
            gpu: bool (default: False)
        """
        import easyocr
        
        self.languages = config.get("languages", self.languages)
        self.gpu = config.get("gpu", self.gpu)

        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR: {str(e)}")

    def health_check(self) -> Tuple[bool, str]:
        if self.reader is None:
            return False, "EasyOCR reader not loaded."
        return True, "Healthy"

    def shutdown(self) -> None:
        self.reader = None

    def _read_text_raw(self, img: np.ndarray) -> OCRResultDTO:
        if not self.reader:
            return OCRResultDTO(text="", confidence=0.0)

        # reader.readtext returns: [([x, y coordinates], text, confidence), ...]
        results = self.reader.readtext(img)
        if not results:
            return OCRResultDTO(text="", confidence=0.0)

        texts = []
        confidences = []
        for bbox, text, conf in results:
            if text.strip():
                texts.append(text.strip())
                confidences.append(float(conf))

        if not texts:
            return OCRResultDTO(text="", confidence=0.0)

        fused_text = " ".join(texts)
        mean_conf = float(np.mean(confidences))
        return OCRResultDTO(text=fused_text, confidence=mean_conf)

    def extract_text(self, crop: CropDTO, timeout_ms: int) -> OCRResultDTO:
        """Extracts text from crop. image_bytes must be decoded first."""
        nparr = np.frombuffer(crop.image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return OCRResultDTO(text="", confidence=0.0)

        # Enforce execution timeout limits using a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._read_text_raw, img)
            try:
                result = future.result(timeout=timeout_ms / 1000.0)
                return result
            except concurrent.futures.TimeoutError:
                # Shield thread from hanging and return empty text tag
                return OCRResultDTO(text="", confidence=0.0)
