import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class BoxProvider(ABC):
    """Abstract interface for bounding box data providers."""

    @abstractmethod
    def get_boxes(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Returns a list of bounding boxes for an image.
        Each box is a dict: {
            'class_id': int,
            'x_center_norm': float,
            'y_center_norm': float,
            'width_norm': float,
            'height_norm': float,
            'line_idx': int,  # original line index for debugging
            'status': str,    # 'ok', 'malformed', 'invalid_coords'
            'reason': str     # explanation if not 'ok'
        }
        """
        pass


class YOLOLabelBoxProvider(BoxProvider):
    """Reads boxes from standard YOLO .txt label files with strict validation."""

    def __init__(self, labels_dir: Path):
        self.labels_dir = Path(labels_dir)

    def get_boxes(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Loads and validates label coordinates from the corresponding label file.
        """
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes = []

        if not label_path.exists():
            logger.warning(f"Label file missing for image {image_path.name}: {label_path.name}")
            return boxes

        if label_path.stat().st_size == 0:
            logger.info(f"Label file is empty: {label_path.name}")
            return boxes

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read label file {label_path.name}: {str(e)}")
            return boxes

        for line_idx, line in enumerate(lines, 1):
            stripped = line.strip()
            # If line is empty or whitespace-only
            if not stripped:
                # Normal blank line is skipped silently.
                # If it contains spaces or tabs, it's counted as a malformed whitespace row.
                if any(c in line for c in [" ", "\t"]):
                    boxes.append({
                        "line_idx": line_idx,
                        "status": "malformed",
                        "reason": "Whitespace only row",
                        "content": line
                    })
                continue

            tokens = stripped.split()
            if len(tokens) != 5:
                logger.warning(
                    f"Malformed line at {label_path.name}:{line_idx} - Expected 5 tokens, got {len(tokens)}"
                )
                boxes.append({
                    "line_idx": line_idx,
                    "status": "malformed",
                    "reason": f"Expected 5 tokens, got {len(tokens)}",
                    "content": stripped
                })
                continue

            try:
                class_id_str, xc_str, yc_str, w_str, h_str = tokens
                class_id = int(class_id_str)
                xc, yc, w, h = map(float, [xc_str, yc_str, w_str, h_str])
            except ValueError as e:
                logger.warning(f"Failed to parse tokens at {label_path.name}:{line_idx}: {str(e)}")
                boxes.append({
                    "line_idx": line_idx,
                    "status": "malformed",
                    "reason": f"Token value parsing failure: {str(e)}",
                    "content": stripped
                })
                continue

            # Validate coordinate range [0, 1] and width/height > 0
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                logger.warning(
                    f"Invalid normalized coordinates at {label_path.name}:{line_idx} - "
                    f"[{xc}, {yc}, {w}, {h}] must be between 0 and 1"
                )
                boxes.append({
                    "line_idx": line_idx,
                    "status": "invalid_coords",
                    "reason": "Normalized coordinates must be in [0.0, 1.0]",
                    "class_id": class_id,
                    "x_center_norm": xc,
                    "y_center_norm": yc,
                    "width_norm": w,
                    "height_norm": h
                })
                continue

            if w <= 0.0 or h <= 0.0:
                logger.warning(
                    f"Invalid box size at {label_path.name}:{line_idx} - Width and height must be > 0"
                )
                boxes.append({
                    "line_idx": line_idx,
                    "status": "invalid_coords",
                    "reason": "Width and height must be greater than 0",
                    "class_id": class_id,
                    "x_center_norm": xc,
                    "y_center_norm": yc,
                    "width_norm": w,
                    "height_norm": h
                })
                continue

            boxes.append({
                "line_idx": line_idx,
                "status": "ok",
                "reason": "Passed validation",
                "class_id": class_id,
                "x_center_norm": xc,
                "y_center_norm": yc,
                "width_norm": w,
                "height_norm": h
            })

        return boxes
