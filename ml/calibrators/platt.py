import numpy as np
from typing import Dict, Any, Tuple, Union
from ml.base import BaseCalibrator


class PlattCalibrator(BaseCalibrator):
    """Calibrator mapping similarity scores to probabilities via Platt Scaling."""

    def __init__(self) -> None:
        self.global_a = 15.0
        self.global_b = -11.0
        self.class_coefficients: Dict[int, Tuple[float, float]] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes scaling parameters from config.

        Config schema:
            global_coefs: { "a": 15.0, "b": -11.0 }
            class_coefs: { "class_id_1": {"a": 15.0, "b": -11.0}, ... }
        """
        global_coefs = config.get("global_coefs", {})
        self.global_a = global_coefs.get("a", self.global_a)
        self.global_b = global_coefs.get("b", self.global_b)

        class_coefs = config.get("class_coefs", {})
        for cid_str, coefs in class_coefs.items():
            try:
                cid = int(cid_str)
                a = float(coefs["a"])
                b = float(coefs["b"])
                self.class_coefficients[cid] = (a, b)
            except (ValueError, KeyError):
                continue

    def health_check(self) -> Tuple[bool, str]:
        return True, "Healthy"

    def shutdown(self) -> None:
        pass

    def calibrate(self, similarity: float, class_id: int) -> float:
        """Computes calibrated probability using sigmoidal scaling.

        Args:
            similarity: Cosine similarity score.
            class_id: Remapped target class ID.

        Returns:
            float: Calibrated probability in [0.0, 1.0].
        """
        a, b = self.class_coefficients.get(class_id, (self.global_a, self.global_b))
        
        # Standard Platt Sigmoidal Logit: z = a * similarity + b
        z = a * similarity + b
        
        # Guard against overflow during exp
        z_clipped = np.clip(z, -50.0, 50.0)
        probability = 1.0 / (1.0 + np.exp(-z_clipped))
        return float(probability)
