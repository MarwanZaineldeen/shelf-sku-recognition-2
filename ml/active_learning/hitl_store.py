import os
import sys
import sqlite3
import json
import io
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from PIL import Image

from ml.base import IPlugin, BBoxDTO, CropDTO, PredictionDTO, EmbeddingDTO
from ml.embeddings.dinov2 import DINOv2Extractor
from ml.retrieval.sqlite_registry import SQLiteGalleryStore


class HITLActiveLearningStore(IPlugin):
    """
    SQLite Active Learning & Human-in-the-Loop (HITL) Store.
    
    Logs pending HITL shelf audit crops, tracks human verification/corrections,
    and automatically upserts verified crop embeddings into the continual gallery DB.
    """

    def __init__(self, db_path: str = "data/processed/hitl_active_learning.db"):
        self.db_path = db_path
        self.gallery_store: Optional[SQLiteGalleryStore] = None
        self.embedder: Optional[DINOv2Extractor] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes SQLite tables and connects to continual gallery store."""
        self.db_path = config.get("db_path", self.db_path)
        gallery_db_path = config.get("gallery_db_path", "data/processed/crops/gt_clean/retail_sku_registry_dinov2.db")

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

        # Connect continual gallery store
        self.gallery_store = SQLiteGalleryStore()
        self.gallery_store.initialize({"db_path": gallery_db_path})

        # Connect embedder for active continual learning
        self.embedder = DINOv2Extractor()
        self.embedder.initialize({"model_name": "facebook/dinov2-small", "device": "cpu"})

    def _init_db(self) -> None:
        """Creates hitl_tasks and verified_annotations tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hitl_tasks (
                task_id TEXT PRIMARY KEY,
                shelf_image_name TEXT,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                predicted_class_id INTEGER,
                predicted_display_name TEXT,
                similarity_score REAL,
                calibrated_probability REAL,
                reject_reason TEXT,
                crop_bytes BLOB,
                created_at TEXT,
                status TEXT DEFAULT 'PENDING'
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verified_annotations (
                annotation_id TEXT PRIMARY KEY,
                task_id TEXT,
                shelf_image_name TEXT,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                verified_class_id INTEGER,
                verified_display_name TEXT,
                verifier_notes TEXT,
                timestamp TEXT,
                FOREIGN KEY (task_id) REFERENCES hitl_tasks (task_id)
            );
        """)

        conn.commit()
        conn.close()

    def log_hitl_task(
        self,
        shelf_image_name: str,
        pred: PredictionDTO,
        crop_bytes: bytes,
        display_name: str = ""
    ) -> str:
        """Logs a new crop item to the HITL audit queue."""
        task_id = f"hitl_task_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}" if 'np' in globals() else f"hitl_task_{int(time.time() * 1000)}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO hitl_tasks (
                task_id, shelf_image_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                predicted_class_id, predicted_display_name, similarity_score,
                calibrated_probability, reject_reason, crop_bytes, created_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING');
        """, (
            task_id, shelf_image_name, pred.bbox.x1, pred.bbox.y1, pred.bbox.x2, pred.bbox.y2,
            pred.predicted_class_id, display_name, pred.bbox.confidence,
            pred.confidence_probability, pred.reject_reason or "LOW_CONFIDENCE",
            crop_bytes, timestamp
        ))

        conn.commit()
        conn.close()
        return task_id

    def approve_task(self, task_id: str, verifier_notes: str = "Approved by HITL") -> bool:
        """Approves predicted class and active-learns crop vector into continual gallery."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM hitl_tasks WHERE task_id = ?;", (task_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        shelf_name, x1, y1, x2, y2, pred_cid, display_name, crop_bytes = row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[11]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ann_id = f"ann_{int(time.time() * 1000)}"

        # 1. Update task status
        cursor.execute("UPDATE hitl_tasks SET status = 'APPROVED' WHERE task_id = ?;", (task_id,))

        # 2. Record verified annotation
        cursor.execute("""
            INSERT INTO verified_annotations (
                annotation_id, task_id, shelf_image_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                verified_class_id, verified_display_name, verifier_notes, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (ann_id, task_id, shelf_name, x1, y1, x2, y2, pred_cid, display_name, verifier_notes, timestamp))

        conn.commit()
        conn.close()

        # 3. Continual Learning: Upsert feature embedding into active gallery store
        if self.embedder and self.gallery_store and crop_bytes:
            crop_dto = CropDTO(
                crop_id=f"continual_{task_id}",
                image_bytes=crop_bytes,
                bbox=BBoxDTO(x1=x1, y1=y1, x2=x2, y2=y2, confidence=1.0),
                blur_score=0.0,
                aspect_ratio=1.0
            )
            emb_dto = self.embedder.extract_dto(crop_dto)
            self.gallery_store.insert_crop(
                crop_id=crop_dto.crop_id,
                remapped_class_id=pred_cid,
                old_class_id=pred_cid,
                family_id=display_name.split()[0] if display_name else "Lipton",
                split="continual",
                vector=emb_dto.vector,
                source_image=shelf_name
            )

        return True

    def correct_task(self, task_id: str, correct_class_id: int, correct_display_name: str, verifier_notes: str = "Corrected by HITL") -> bool:
        """Corrects misclassified crop and active-learns correct vector into continual gallery."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM hitl_tasks WHERE task_id = ?;", (task_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        shelf_name, x1, y1, x2, y2, crop_bytes = row[1], row[2], row[3], row[4], row[5], row[11]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ann_id = f"ann_{int(time.time() * 1000)}"

        # 1. Update task status
        cursor.execute("UPDATE hitl_tasks SET status = 'CORRECTED' WHERE task_id = ?;", (task_id,))

        # 2. Record verified annotation
        cursor.execute("""
            INSERT INTO verified_annotations (
                annotation_id, task_id, shelf_image_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                verified_class_id, verified_display_name, verifier_notes, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (ann_id, task_id, shelf_name, x1, y1, x2, y2, correct_class_id, correct_display_name, verifier_notes, timestamp))

        conn.commit()
        conn.close()

        # 3. Continual Learning: Upsert corrected feature embedding into active gallery store
        if self.embedder and self.gallery_store and crop_bytes:
            crop_dto = CropDTO(
                crop_id=f"continual_{task_id}",
                image_bytes=crop_bytes,
                bbox=BBoxDTO(x1=x1, y1=y1, x2=x2, y2=y2, confidence=1.0),
                blur_score=0.0,
                aspect_ratio=1.0
            )
            emb_dto = self.embedder.extract_dto(crop_dto)
            self.gallery_store.insert_crop(
                crop_id=crop_dto.crop_id,
                remapped_class_id=correct_class_id,
                old_class_id=correct_class_id,
                family_id=correct_display_name.split()[0] if correct_display_name else "Lipton",
                split="continual",
                vector=emb_dto.vector,
                source_image=shelf_name
            )

        return True

    def health_check(self) -> Tuple[bool, str]:
        if not os.path.exists(self.db_path):
            return False, "HITL DB file missing."
        return True, "HITL Active Learning Store operating normally."

    def shutdown(self) -> None:
        if self.gallery_store:
            self.gallery_store.shutdown()
