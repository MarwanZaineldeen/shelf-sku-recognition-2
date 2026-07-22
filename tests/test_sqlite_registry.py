import os
import tempfile
import unittest
import numpy as np
from ml.base import BBoxDTO, EmbeddingDTO
from ml.retrieval.sqlite_registry import SQLiteGalleryStore


class TestSQLiteGalleryStore(unittest.TestCase):
    """Unit tests for the SQLite-based gallery repository store."""

    def setUp(self) -> None:
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.store = SQLiteGalleryStore()
        self.store.initialize({"db_path": self.db_path})

    def tearDown(self) -> None:
        self.store.shutdown()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_schema_creation_and_health(self) -> None:
        """Verifies database tables exist and health check passes."""
        ok, msg = self.store.health_check()
        self.assertTrue(ok)
        self.assertEqual(msg, "Healthy")
        
        # Test active version initialization
        version = self.store.get_current_version()
        self.assertEqual(version, 1)

    def test_save_and_fetch_references(self) -> None:
        """Tests inserting single reference and loading back the vectors."""
        bbox = BBoxDTO(x1=10.0, y1=15.0, x2=100.0, y2=120.0, confidence=1.0)
        embedding = EmbeddingDTO(vector=[0.1] * 384, dimension=384)
        
        # Save a single crop vector
        version = self.store.save_reference(
            class_id=5,
            old_class_id=12,
            crop_path="crops/crop_01.jpg",
            family_id="fam_01",
            source_image="shelf_01.jpg",
            bbox=bbox,
            embedding=embedding
        )
        
        self.assertEqual(version, 2)  # Initial version was 1, onboarding increments to 2
        
        # Load references back from database
        embeddings, metadata = self.store.fetch_all_references()
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(metadata), 1)
        
        # fetch_all_references returns an (N, D) ndarray, which NumpyCosineIndex
        # consumes directly — not a list of EmbeddingDTOs.
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (1, 384))
        self.assertTrue(np.allclose(embeddings[0], [0.1] * 384))
        
        self.assertEqual(metadata[0]["remapped_class_id"], 5)
        self.assertEqual(metadata[0]["old_class_id"], 12)
        self.assertEqual(metadata[0]["family_id"], "fam_01")
        self.assertEqual(metadata[0]["source_image_name"], "shelf_01.jpg")
        self.assertEqual(metadata[0]["bbox"], [10.0, 15.0, 100.0, 120.0])

    def test_save_references_bulk(self) -> None:
        """Tests optimized bulk transactions."""
        references = []
        for i in range(10):
            references.append((
                i % 3,
                10 + i,
                f"crops/crop_{i}.jpg",
                f"fam_{i}",
                f"shelf_{i}.jpg",
                [0.0, 0.0, 100.0, 100.0],
                [float(i)] * 384
            ))
            
        version = self.store.save_references_bulk(references)
        self.assertEqual(version, 2)
        
        embeddings, metadata = self.store.fetch_all_references()
        self.assertEqual(len(embeddings), 10)
        self.assertEqual(len(metadata), 10)

    def test_delete_sku(self) -> None:
        """Tests dynamic deletion of SKU references from the active gallery."""
        references = [
            (1, 10, "crop1.jpg", "fam1", "shelf1.jpg", [0, 0, 10, 10], [0.1] * 384),
            (2, 20, "crop2.jpg", "fam2", "shelf2.jpg", [0, 0, 10, 10], [0.2] * 384),
            (1, 30, "crop3.jpg", "fam3", "shelf3.jpg", [0, 0, 10, 10], [0.3] * 384)
        ]
        self.store.save_references_bulk(references)
        
        # Verify 3 items loaded
        embeddings, metadata = self.store.fetch_all_references()
        self.assertEqual(len(embeddings), 3)
        
        # Delete class_id = 1
        new_version = self.store.delete_sku(class_id=1)
        self.assertEqual(new_version, 3)
        
        # Active gallery should now only contain class_id = 2
        embeddings, metadata = self.store.fetch_all_references()
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(metadata[0]["remapped_class_id"], 2)

    def test_rollback(self) -> None:
        """Tests rolling back onboarding updates to a previous version."""
        bbox = BBoxDTO(x1=0, y1=0, x2=10, y2=10, confidence=1)
        
        # Onboard SKU 1 (version 2)
        v2 = self.store.save_reference(
            1, 10, "crop1.jpg", "fam1", "shelf1.jpg", bbox, EmbeddingDTO(vector=[0.1]*384, dimension=384)
        )
        self.assertEqual(v2, 2)
        
        # Onboard SKU 2 (version 3)
        v3 = self.store.save_reference(
            2, 20, "crop2.jpg", "fam2", "shelf2.jpg", bbox, EmbeddingDTO(vector=[0.2]*384, dimension=384)
        )
        self.assertEqual(v3, 3)
        
        # Rollback database to version 2 (removes SKU 2)
        self.store.rollback_version(2)
        
        embeddings, metadata = self.store.fetch_all_references()
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(metadata[0]["remapped_class_id"], 1)


if __name__ == "__main__":
    unittest.main()
