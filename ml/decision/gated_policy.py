from typing import Dict, List, Any, Tuple, Optional
from ml.base import BaseDecisionPolicy, SearchResultDTO


class GatedAnnotationPolicy(BaseDecisionPolicy):
    """Decision engine that gates auto-annotations based on calibrated probability."""

    def __init__(self) -> None:
        self.global_threshold = 0.80
        self.class_thresholds: Dict[int, float] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes thresholds from config.

        Config schema:
            global_threshold: 0.80
            class_thresholds: { "class_id_1": 0.80, ... }
        """
        self.global_threshold = config.get("global_threshold", self.global_threshold)
        
        raw_thresholds = config.get("class_thresholds", {})
        for cid_str, val in raw_thresholds.items():
            try:
                cid = int(cid_str)
                self.class_thresholds[cid] = float(val)
            except ValueError:
                continue

    def health_check(self) -> Tuple[bool, str]:
        return True, "Healthy"

    def shutdown(self) -> None:
        pass

    def decide(
        self,
        matches: List[SearchResultDTO],
        probability: float,
        class_id: int
    ) -> Tuple[bool, Optional[str]]:
        """Determines if prediction can be safely automated.

        Args:
            matches: Fused search matching list.
            probability: Calibrated probability output of the matching.
            class_id: Remapped target class ID of the top prediction.

        Returns:
            Tuple[bool, Optional[str]]: (automated, reject_reason).
        """
        if not matches:
            return False, "NO_MATCHING_CANDIDATES"

        # Lookup class-specific calibrated probability threshold
        threshold = self.class_thresholds.get(class_id, self.global_threshold)

        if probability >= threshold:
            return True, None
        else:
            return False, "LOW_CONFIDENCE"
