import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

from ml.base import EmbeddingDTO, CropDTO, BBoxDTO
from ml.embeddings.dinov2 import DINOv2Extractor
from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from ml.retrieval.hierarchical_index import HierarchicalCosineIndex
from ml.onboarding.onboarder import SKUOnboarder


class TestPipeline2Onboarding(unittest.TestCase):
    """Integration test suite for Pipeline 2 (New SKU Onboarding)."""

    def setUp(self) -> None:
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")

        # Base directories for ready cropped datasets
        self.workspace = Path(__file__).resolve().parent.parent
        self.nesquik_dir = self.workspace / "data" / "Nesquik"
        self.heinz_dir = self.workspace / "data" / "Heinz tomato ketchup"

        # Initialize embedder, store, and index
        self.embedder = DINOv2Extractor(model_name="facebook/dinov2-small", device="cpu")
        self.store = SQLiteGalleryStore()
        self.store.initialize({"db_path": self.db_path})

        self.index = HierarchicalCosineIndex(dimension=self.embedder.dimension)
        self.index.initialize({"dimension": self.embedder.dimension, "db_path": self.db_path})

        self.onboarder = SKUOnboarder(
            embedder=self.embedder,
            store=self.store,
            retriever=self.index
        )

    def tearDown(self) -> None:
        self.store.shutdown()
        self.index.shutdown()
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_onboard_nesquik_and_heinz_ready_crops(self) -> None:
        """Verifies onboarding ready crops from Nesquik and Heinz datasets into vector DB & clusters."""
        # 1. Onboard Nesquik dataset (Class ID: 70)
        res_nesquik = self.onboarder.onboard_from_crops(
            crops_dir=self.nesquik_dir,
            class_id=70,
            old_class_id=700,
            family_id="Nesquik",
            augment=False
        )

        self.assertEqual(res_nesquik["status"], "success")
        self.assertEqual(res_nesquik["crops_added"], 15)

        # 2. Onboard Heinz dataset with augmentation (Class ID: 71)
        res_heinz = self.onboarder.onboard_from_crops(
            crops_dir=self.heinz_dir,
            class_id=71,
            old_class_id=710,
            family_id="Heinz tomato ketchup",
            augment=False
        )

        self.assertEqual(res_heinz["status"], "success")
        self.assertEqual(res_heinz["crops_added"], 147)

        # 3. Verify total registered crops in SQLite DB (15 + 147 = 162)
        embeddings, metadata = self.store.fetch_all_references()
        total_crops = len(embeddings)
        self.assertEqual(total_crops, 162)

        # 4. Verify vector dimensions
        self.assertEqual(embeddings.shape[1], 384)

        # 5. Verify brand clusters in metadata
        nesquik_meta = [m for m in metadata if m["family_id"] == "Nesquik"]
        heinz_meta = [m for m in metadata if m["family_id"] == "Heinz tomato ketchup"]
        self.assertEqual(len(nesquik_meta), 15)
        self.assertEqual(len(heinz_meta), 147)

    def test_retrieval_query_on_onboarded_skus(self) -> None:
        """Verifies visual vector similarity retrieval queries correctly match the onboarded SKUs."""
        self.onboarder.onboard_from_crops(self.nesquik_dir, class_id=70, old_class_id=700, family_id="Nesquik")
        self.onboarder.onboard_from_crops(self.heinz_dir, class_id=71, old_class_id=710, family_id="Heinz tomato ketchup")

        # Query using a Nesquik crop embedding DTO
        nesquik_sample_path = sorted(list(self.nesquik_dir.glob("*.jpg")))[0]
        crop_dto = CropDTO(
            crop_id="query_nesquik",
            image_bytes=nesquik_sample_path.read_bytes(),
            bbox=BBoxDTO(x1=0.0, y1=0.0, x2=100.0, y2=100.0, confidence=1.0),
            blur_score=0.0,
            aspect_ratio=1.0
        )
        query_emb = self.embedder.extract_dto(crop_dto)

        # Retrieve candidates from Hierarchical Index
        results = self.index.search_dto(query_emb, top_k=5, family_id="Nesquik")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].remapped_class_id, 70)
        self.assertGreaterEqual(results[0].similarity, 0.85)

    def test_onboard_with_yolo_detector(self) -> None:
        """Verifies YOLO product detection and tight cropping during Pipeline 2 onboarding."""
        class MockBox:
            def __init__(self):
                self.x1 = 5.0
                self.y1 = 5.0
                self.x2 = 50.0
                self.y2 = 50.0
                self.confidence = 0.95

        class MockDetector:
            def detect(self, image_bytes):
                return [MockBox()]

        mock_detector = MockDetector()
        onboarder = SKUOnboarder(
            embedder=self.embedder,
            store=self.store,
            retriever=self.index,
            detector=mock_detector
        )

        res = onboarder.onboard_from_crops(
            crops_dir=self.nesquik_dir,
            class_id=88,
            old_class_id=880,
            family_id="MockNesquik",
            use_yolo_crop=True
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["crops_added"], 15)

        # Check metadata stored in DB
        embeddings, metadata = self.store.fetch_all_references()
        mock_meta = [m for m in metadata if m["family_id"] == "MockNesquik"]
        self.assertEqual(len(mock_meta), 15)
        # Verify bounding box was cropped to detected coordinates
        self.assertEqual(mock_meta[0]["bbox"], [5.0, 5.0, 50.0, 50.0])


if __name__ == "__main__":
    unittest.main()
