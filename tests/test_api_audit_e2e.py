import os
from pathlib import Path

workspace_root = Path(os.environ.get("RETAIL_AI_ROOT", Path(__file__).resolve().parents[1]))
os.environ["HF_HOME"] = str(workspace_root / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(workspace_root / ".cache" / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(workspace_root / ".cache" / "torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import io
import cv2
import unittest
import numpy as np
from fastapi.testclient import TestClient
from server.app import app


class TestAPIAuditE2E(unittest.TestCase):
    """End-to-End API Integration test suite."""

    def setUp(self) -> None:
        from server.app import startup_event
        startup_event()
        self.client = TestClient(app)
        
        # Create a mock 100x100 image with BGR channels (JPEG encoded bytes)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw some details so it is not completely empty
        cv2.putText(img, "SKU", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, img_arr = cv2.imencode(".jpg", img)
        self.mock_image_bytes = img_arr.tobytes()

    def tearDown(self) -> None:
        from server.app import shutdown_event
        shutdown_event()

    def test_healthz_endpoint(self) -> None:
        """Verifies health check status is 200 and schema matches."""
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("yolov8s", data["loaded_models"])
        self.assertGreaterEqual(data["db_version"], 1)

    def test_audit_shelf_endpoint(self) -> None:
        """Verifies daily shelf audit runs and responds with Annotation/HITL schemas."""
        files = {"file": ("mock_shelf.jpg", self.mock_image_bytes, "image/jpeg")}
        response = self.client.post("/v1/audit/shelf", files=files)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["image_name"], "mock_shelf.jpg")
        self.assertIn("annotations", data)
        self.assertIn("hitl_queue", data)

    def test_onboard_sku_endpoint(self) -> None:
        """Verifies new SKU dynamic onboarding upserts reference crop to DB and active index."""
        # 1. Fetch current DB version prior to onboarding
        health_resp = self.client.get("/healthz")
        version_before = health_resp.json()["db_version"]
        
        # 2. Execute onboarding POST request
        payload = {
            "class_id": 99,
            "old_class_id": 999,
            "family_id": "test_family",
            "source_image": "source_shelf.jpg"
        }
        files = [
            ("reference_images", ("ref_01.jpg", self.mock_image_bytes, "image/jpeg"))
        ]
        
        response = self.client.post("/v1/onboard/sku", data=payload, files=files)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["crops_added"], 3)
        self.assertGreater(data["version"], version_before)


if __name__ == "__main__":
    unittest.main()
