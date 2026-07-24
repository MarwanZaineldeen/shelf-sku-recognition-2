import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ml.base import BBoxDTO, EmbeddingDTO
from ml.retrieval.numpy_index import NumpyCosineIndex
from ml.retrieval.sqlite_registry import SQLiteGalleryStore
from server.catalog_service import delete_and_reindex_catalog


def catalog_document():
    return {
        "classes": {
            "10": {"raw_class_id": "10", "training_class_id": 0, "display_name": "A"},
            "99": {"raw_class_id": "99", "training_class_id": 1, "display_name": "B"},
            # Its raw ID collides with B's runtime ID. Deleting runtime class 1
            # must not delete this class.
            "1": {"raw_class_id": "1", "training_class_id": 2, "display_name": "C"},
        }
    }


class TestCoordinatedCatalogDeletion(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs" / "sku_mapping.json").write_text(
            json.dumps(catalog_document()), encoding="utf-8"
        )
        (self.root / "configs" / "class_id_mapping.json").write_text("{}", encoding="utf-8")
        (self.root / "configs" / "class_id_mapping.csv").write_text(
            "old_class_id,new_class_id\n", encoding="utf-8"
        )
        for class_id in range(3):
            directory = self.root / "configs" / "class_catalog" / f"class_{class_id:02d}"
            directory.mkdir(parents=True)
            (directory / "marker.txt").write_text(str(class_id), encoding="utf-8")

        self.store = SQLiteGalleryStore()
        self.store.initialize({"db_path": str(self.root / "gallery.db")})
        box = BBoxDTO(x1=0, y1=0, x2=1, y2=1, confidence=1)
        rows = (
            (0, 10, "a.jpg"),
            (1, 99, "b.jpg"),
            (2, 1, "c.jpg"),
        )
        for runtime_id, raw_id, path in rows:
            self.store.save_reference(
                class_id=runtime_id,
                old_class_id=raw_id,
                crop_path=path,
                family_id="test",
                source_image="shelf.jpg",
                bbox=box,
                embedding=EmbeddingDTO(vector=[1.0, 0.0, 0.0, 0.0], dimension=4),
            )

        vectors, metadata = self.store.fetch_all_references()
        self.index = NumpyCosineIndex(dimension=4)
        self.index.add(vectors, metadata)
        self.orchestrator = SimpleNamespace(
            sku_mapping={0: {}, 1: {}, 2: {}}
        )

    def tearDown(self):
        self.store.shutdown()
        self.temp.cleanup()

    def test_delete_compacts_every_runtime_consumer_without_raw_id_collision(self):
        result = delete_and_reindex_catalog(
            self.root, self.store, self.index, self.orchestrator, [1]
        )

        self.assertEqual(result["deleted_vectors_count"], 1)
        self.assertEqual(result["id_remap"], {"0": 0, "2": 1})
        self.assertEqual(self.store.class_size_histogram(), {0: 1, 1: 1})
        row = self.store.conn.execute(
            "SELECT remapped_class_id, old_class_id FROM sku_crops "
            "WHERE crop_path='c.jpg'"
        ).fetchone()
        self.assertEqual((row[0], row[1]), (1, 1))
        self.assertEqual(set(self.orchestrator.sku_mapping), {0, 1})
        self.assertEqual(self.index.gallery_vectors.shape, (2, 4))
        self.assertEqual(
            [meta["remapped_class_id"] for meta in self.index.metadata], [0, 1]
        )

        catalog = json.loads(
            (self.root / "configs" / "sku_mapping.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("99", catalog["classes"])
        self.assertEqual(catalog["classes"]["1"]["training_class_id"], 1)
        self.assertFalse((self.root / "configs" / "class_catalog" / "class_02").exists())
        self.assertEqual(
            (self.root / "configs" / "class_catalog" / "class_01" / "marker.txt")
            .read_text(encoding="utf-8"),
            "2",
        )
        self.assertTrue(Path(result["backup_path"]).exists())
