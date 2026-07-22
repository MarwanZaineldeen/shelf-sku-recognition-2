import os
import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from ml.base import BaseGalleryStore, BBoxDTO, EmbeddingDTO


class SQLiteGalleryStore(BaseGalleryStore):
    """SQLite-based vector reference repository implementation."""

    def __init__(self) -> None:
        self.db_path: Optional[str] = None
        self.conn: Optional[sqlite3.Connection] = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initializes SQLite database and creates schema if missing."""
        self.db_path = config.get("db_path")
        if not self.db_path:
            raise ValueError("Configuration must specify 'db_path'.")

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        with self.conn:
            # 1. Versioning table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS gallery_metadata (
                    version INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    active BOOLEAN DEFAULT 1
                )
            """)
            # 2. SKU crops and vector embeddings table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sku_crops (
                    id TEXT PRIMARY KEY,
                    remapped_class_id INTEGER NOT NULL,
                    old_class_id INTEGER NOT NULL,
                    crop_path TEXT NOT NULL,
                    family_id TEXT NOT NULL,
                    source_image_name TEXT NOT NULL,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    gallery_version INTEGER NOT NULL,
                    FOREIGN KEY(gallery_version) REFERENCES gallery_metadata(version)
                )
            """)

            # Ensure version 1 exists if fresh DB
            cursor = self.conn.execute("SELECT COUNT(*) FROM gallery_metadata")
            if cursor.fetchone()[0] == 0:
                self.conn.execute("INSERT INTO gallery_metadata (version, active) VALUES (1, 1)")

    def health_check(self) -> Tuple[bool, str]:
        """Verifies database availability."""
        if not self.conn:
            return False, "Database not connected."
        try:
            self.conn.execute("SELECT 1")
            return True, "Healthy"
        except Exception as e:
            return False, f"Database check failed: {str(e)}"

    def get_max_class_id(self) -> int:
        """Returns the maximum remapped_class_id present in active database, or -1 if empty."""
        if not self.conn:
            return -1
        try:
            cursor = self.conn.execute("SELECT MAX(remapped_class_id) FROM sku_crops")
            row = cursor.fetchone()
            if row and row[0] is not None:
                return int(row[0])
        except Exception:
            pass
        return -1

    def shutdown(self) -> None:
        """Safely closes connections."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def save_references_bulk(
        self,
        references: List[Tuple[int, int, str, str, str, List[float], List[float]]]
    ) -> int:
        """Persists a batch of reference crop embeddings to SQLite in a single transaction."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = MEMORY")

        with self.conn:
            cursor = self.conn.execute("INSERT INTO gallery_metadata (active) VALUES (1)")
            version = cursor.lastrowid

            insert_data = []
            for class_id, old_class_id, crop_path, family_id, source_image, bbox_coords, embedding_vector in references:
                vec_arr = np.array(embedding_vector, dtype=np.float32)
                embedding_bytes = vec_arr.tobytes()
                crop_id = f"crop_{version}_{os.path.basename(crop_path)}"
                
                insert_data.append((
                    crop_id, class_id, old_class_id, crop_path, family_id,
                    source_image, float(bbox_coords[0]), float(bbox_coords[1]),
                    float(bbox_coords[2]), float(bbox_coords[3]),
                    embedding_bytes, version
                ))

            self.conn.executemany(
                """
                INSERT INTO sku_crops (
                    id, remapped_class_id, old_class_id, crop_path, family_id, 
                    source_image_name, x1, y1, x2, y2, embedding, gallery_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_data
            )
            return version

    def save_reference(
        self,
        class_id: int,
        old_class_id: int,
        crop_path: str,
        family_id: str,
        source_image: str,
        bbox: BBoxDTO,
        embedding: EmbeddingDTO
    ) -> int:
        """Persists reference embedding vector and details to SQLite."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        with self.conn:
            # 1. Insert new version to track onboarding event
            cursor = self.conn.execute("INSERT INTO gallery_metadata (active) VALUES (1)")
            version = cursor.lastrowid

            # Serialize float vector list to raw bytes (float32 format)
            vec_arr = np.array(embedding.vector, dtype=np.float32)
            embedding_bytes = vec_arr.tobytes()

            # Unique string ID based on path or timestamp hash
            crop_id = f"crop_{version}_{os.path.basename(crop_path)}"

            self.conn.execute(
                """
                INSERT INTO sku_crops (
                    id, remapped_class_id, old_class_id, crop_path, family_id, 
                    source_image_name, x1, y1, x2, y2, embedding, gallery_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    crop_id, class_id, old_class_id, crop_path, family_id,
                    source_image, bbox.x1, bbox.y1, bbox.x2, bbox.y2,
                    embedding_bytes, version
                )
            )
            return version

    def fetch_all_references(self) -> Tuple[List[EmbeddingDTO], List[Dict[str, Any]]]:
        """Loads all registered reference vectors and metadata in memory."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        cursor = self.conn.execute(
            """
            SELECT s.* FROM sku_crops s
            INNER JOIN gallery_metadata m ON s.gallery_version = m.version
            WHERE m.active = 1
            """
        )
        rows = cursor.fetchall()

        num_rows = len(rows)
        if num_rows == 0:
            return np.empty((0, 768), dtype=np.float32), []

        # Inspect dimension from first row
        first_vec = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
        dim = len(first_vec)

        vectors = np.empty((num_rows, dim), dtype=np.float32)
        metadata: List[Dict[str, Any]] = []

        for idx, row in enumerate(rows):
            vectors[idx] = np.frombuffer(row["embedding"], dtype=np.float32)
            meta = {
                "crop_path": row["crop_path"],
                "remapped_class_id": row["remapped_class_id"],
                "old_class_id": row["old_class_id"],
                "family_id": row["family_id"],
                "source_image_name": row["source_image_name"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]]
            }
            metadata.append(meta)

        return vectors, metadata

    def delete_sku(self, class_id: int) -> int:
        """Deletes SKU references from index."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        with self.conn:
            cursor = self.conn.execute("INSERT INTO gallery_metadata (active) VALUES (1)")
            version = cursor.lastrowid

            self.conn.execute(
                "DELETE FROM sku_crops WHERE remapped_class_id = ?",
                (class_id,)
            )
            return version

    def get_current_version(self) -> int:
        """Gets current active database version."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        cursor = self.conn.execute("SELECT MAX(version) FROM gallery_metadata WHERE active = 1")
        row = cursor.fetchone()
        return row[0] if row[0] is not None else 1

    def rollback_version(self, version: int) -> None:
        """Rollbacks database states to version."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        with self.conn:
            # Deactive newer versions
            self.conn.execute(
                "UPDATE gallery_metadata SET active = 0 WHERE version > ?",
                (version,)
            )
            # Remove crops associated with rolled-back versions
            self.conn.execute(
                "DELETE FROM sku_crops WHERE gallery_version > ?",
                (version,)
            )
