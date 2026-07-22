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

        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Adds Pipeline 3 curation columns to sku_crops if absent.

        Idempotent and additive. ADD COLUMN is a metadata-only operation in
        SQLite, so this is cheap even against the 31,656-row production
        registry, and existing rows read back the column default.
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        cursor = self.conn.execute("PRAGMA table_info(sku_crops)")
        existing = {row["name"] for row in cursor.fetchall()}

        # active: soft-delete flag. Curation must never hard-DELETE, because
        # rollback_version cannot resurrect a deleted row.
        # pruned_in_version: which curation pass removed the row, so a
        # rollback can restore exactly the rows that pass took out.
        # origin: 'seed' for the original gallery, 'continual' for crops
        # promoted from HITL reviews.
        migrations = {
            "active": "ALTER TABLE sku_crops ADD COLUMN active INTEGER DEFAULT 1",
            "pruned_in_version": "ALTER TABLE sku_crops ADD COLUMN pruned_in_version INTEGER",
            "origin": "ALTER TABLE sku_crops ADD COLUMN origin TEXT DEFAULT 'seed'",
        }

        with self.conn:
            for column, statement in migrations.items():
                if column not in existing:
                    self.conn.execute(statement)

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
        references: List[Tuple[int, int, str, str, str, List[float], List[float]]],
        origin: str = "seed"
    ) -> int:
        """Persists a batch of reference crop embeddings to SQLite in a single transaction.

        Args:
            references: Tuples of (class_id, old_class_id, crop_path, family_id,
                source_image, bbox_coords, embedding_vector).
            origin: 'seed' for the original gallery, 'continual' for crops
                promoted from HITL reviews.
        """
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
                    embedding_bytes, version, origin
                ))

            self.conn.executemany(
                """
                INSERT INTO sku_crops (
                    id, remapped_class_id, old_class_id, crop_path, family_id,
                    source_image_name, x1, y1, x2, y2, embedding, gallery_version,
                    origin, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
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

    def fetch_all_references(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Loads all active reference vectors and metadata in memory.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: An (N, D) float32 array
            and N metadata dicts in matching row order.

        Note:
            Returns a raw ndarray rather than EmbeddingDTOs — NumpyCosineIndex
            feeds the array straight into its matrix search, and building
            31,656 DTOs per load would be pure overhead.
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        cursor = self.conn.execute(
            """
            SELECT s.* FROM sku_crops s
            INNER JOIN gallery_metadata m ON s.gallery_version = m.version
            WHERE m.active = 1 AND s.active = 1
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
                "id": row["id"],
                "crop_path": row["crop_path"],
                "remapped_class_id": row["remapped_class_id"],
                "old_class_id": row["old_class_id"],
                "family_id": row["family_id"],
                "source_image_name": row["source_image_name"],
                "origin": row["origin"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]]
            }
            metadata.append(meta)

        return vectors, metadata

    def delete_sku(self, class_id: int) -> int:
        """Soft-deletes all references for a SKU class.

        Marks rows inactive rather than dropping them, so rollback_version
        can restore the SKU. Rows disappear from fetch_all_references
        immediately, exactly as a hard delete would.
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        with self.conn:
            cursor = self.conn.execute("INSERT INTO gallery_metadata (active) VALUES (1)")
            version = cursor.lastrowid

            self.conn.execute(
                "UPDATE sku_crops SET active = 0, pruned_in_version = ? WHERE remapped_class_id = ? AND active = 1",
                (version, class_id)
            )
            return version

    def prune_references(self, crop_ids: List[str]) -> int:
        """Soft-deletes specific reference crops as part of a curation pass.

        Args:
            crop_ids: sku_crops.id values to deactivate.

        Returns:
            int: The gallery version stamped on the pruned rows, for rollback.
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        with self.conn:
            cursor = self.conn.execute("INSERT INTO gallery_metadata (active) VALUES (1)")
            version = cursor.lastrowid

            if crop_ids:
                self.conn.executemany(
                    "UPDATE sku_crops SET active = 0, pruned_in_version = ? WHERE id = ? AND active = 1",
                    [(version, cid) for cid in crop_ids]
                )
            return version

    def fetch_active_by_class(self, class_id: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Loads active reference vectors for a single SKU class.

        Args:
            class_id: Remapped SKU class ID.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: An (N, D) float32 array
            and N metadata dicts carrying 'id' for pruning, in matching order.
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        cursor = self.conn.execute(
            """
            SELECT s.* FROM sku_crops s
            INNER JOIN gallery_metadata m ON s.gallery_version = m.version
            WHERE m.active = 1 AND s.active = 1 AND s.remapped_class_id = ?
            ORDER BY s.rowid ASC
            """,
            (int(class_id),)
        )
        rows = cursor.fetchall()
        if not rows:
            return np.empty((0, 0), dtype=np.float32), []

        first_vec = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
        dim = len(first_vec)

        vectors = np.empty((len(rows), dim), dtype=np.float32)
        metadata: List[Dict[str, Any]] = []

        for idx, row in enumerate(rows):
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if vec.shape[0] != dim:
                raise ValueError(
                    f"Inconsistent embedding dimension for class {class_id}: "
                    f"expected {dim}, got {vec.shape[0]} in crop '{row['id']}'."
                )
            vectors[idx] = vec
            metadata.append({
                "id": row["id"],
                "crop_path": row["crop_path"],
                "remapped_class_id": row["remapped_class_id"],
                "old_class_id": row["old_class_id"],
                "family_id": row["family_id"],
                "source_image_name": row["source_image_name"],
                "origin": row["origin"],
            })

        return vectors, metadata

    def class_size_histogram(self) -> Dict[int, int]:
        """Counts active reference crops per SKU class.

        Run this before committing a curation cap — the cap only prunes
        anything if the real distribution is skewed.
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        cursor = self.conn.execute(
            """
            SELECT s.remapped_class_id AS cid, COUNT(*) AS n FROM sku_crops s
            INNER JOIN gallery_metadata m ON s.gallery_version = m.version
            WHERE m.active = 1 AND s.active = 1
            GROUP BY s.remapped_class_id
            ORDER BY s.remapped_class_id ASC
            """
        )
        return {int(row["cid"]): int(row["n"]) for row in cursor.fetchall()}

    def get_current_version(self) -> int:
        """Gets current active database version."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        cursor = self.conn.execute("SELECT MAX(version) FROM gallery_metadata WHERE active = 1")
        row = cursor.fetchone()
        return row[0] if row[0] is not None else 1

    def rollback_version(self, version: int) -> None:
        """Restores the registry to a previous version, non-destructively.

        Rows inserted after the target version are hidden by deactivating
        their gallery_metadata rows — fetch_all_references inner-joins on
        m.active, so no DELETE is needed. The previous DELETE was both
        redundant and irreversible.

        Rows soft-deleted by a later curation pass are reactivated, so
        pruning is fully undoable.
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")

        with self.conn:
            # Hide everything inserted after the target version.
            self.conn.execute(
                "UPDATE gallery_metadata SET active = 0 WHERE version > ?",
                (version,)
            )
            # Restore anything pruned after the target version.
            self.conn.execute(
                "UPDATE sku_crops SET active = 1, pruned_in_version = NULL WHERE pruned_in_version > ?",
                (version,)
            )
