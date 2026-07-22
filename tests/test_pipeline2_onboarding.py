import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

from ml.base import EmbeddingDTO
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
            family_id="Nesquik"
        )

        self.assertEqual(res_nesquik["status"], "success")
        self.assertEqual(res_nesquik["crops_added"], 15)

        # 2. Onboard Heinz dataset (Class ID: 71)
        res_heinz = self.onboarder.onboard_from_crops(
            crops_dir=self.heinz_dir,
            class_id=71,
            old_class_id=710,
            family_id="Heinz tomato ketchup"
        )

        self.assertEqual(res_heinz["status"], "success")
        self.assertEqual(res_heinz["crops_added"], 49)

        # 3. Verify total registered crops in SQLite DB
        embeddings, metadata = self.store.fetch_all_references()
        total_crops = len(embeddings)
        self.assertEqual(total_crops, 64)

        # 4. Verify vector dimensions
        self.assertEqual(embeddings.shape[1], 384)

        # 5. Verify brand clusters in metadata
        nesquik_meta = [m for m in metadata if m["family_id"] == "Nesquik"]
        heinz_meta = [m for m in metadata if m["family_id"] == "Heinz tomato ketchup"]
        self.assertEqual(len(nesquik_meta), 15)
        self.assertEqual(len(heinz_meta), 49)

    def test_retrieval_query_on_onboarded_skus(self) -> None:
        """Verifies visual vector similarity retrieval queries correctly match the onboarded SKUs."""
        # Onboard both datasets
        self.onboarder.onboard_from_crops(self.nesquik_dir, class_id=70, old_class_id=700, family_id="Nesquik")
        self.onboarder.onboard_from_crops(self.heinz_dir, class_id=71, old_class_id=710, family_id="Heinz tomato ketchup")

        # Take 1 sample crop from Nesquik dataset as query
        nesquik_sample = list(self.nesquik_dir.glob("*.jpg"))[0]
        pil_nesquik = Image.open(nesquik_sample).convert("RGB")
        vec_nesquik = self.embedder.extract([pil_nesquik])[0]
        dto_nesquik = EmbeddingDTO(vector=vec_nesquik.tolist(), dimension=384)

        # Search top 3 candidates
        results_nesquik = self.index.search_dto(dto_nesquik, top_k=3)
        self.assertGreaterEqual(len(results_nesquik), 1)
        self.assertEqual(results_nesquik[0].remapped_class_id, 70)
        self.assertEqual(results_nesquik[0].metadata["family_id"], "Nesquik")
        self.assertGreater(results_nesquik[0].similarity, 0.85)

        # Take 1 sample crop from Heinz dataset as query
        heinz_sample = list(self.heinz_dir.glob("*.jpg"))[0]
        pil_heinz = Image.open(heinz_sample).convert("RGB")
        vec_heinz = self.embedder.extract([pil_heinz])[0]
        dto_heinz = EmbeddingDTO(vector=vec_heinz.tolist(), dimension=384)

        # Search top 3 candidates
        results_heinz = self.index.search_dto(dto_heinz, top_k=3)
        self.assertGreaterEqual(len(results_heinz), 1)
        self.assertEqual(results_heinz[0].remapped_class_id, 71)
        self.assertEqual(results_heinz[0].metadata["family_id"], "Heinz tomato ketchup")
        self.assertGreater(results_heinz[0].similarity, 0.85)

    def test_onboard_with_yolo_detector(self) -> None:
        """Verifies YOLO product detection and tight cropping during Pipeline 2 onboarding."""
        from ml.base import BBoxDTO

        class DummyYOLODetector:
            def detect(self, image_bytes: bytes):
                # Simulate detecting a product box in the image
                return [BBoxDTO(x1=5.0, y1=5.0, x2=50.0, y2=50.0, confidence=0.92)]

        mock_detector = DummyYOLODetector()
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
