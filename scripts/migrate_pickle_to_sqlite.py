import os
import pickle
import numpy as np
from pathlib import Path
from ml.retrieval.sqlite_registry import SQLiteGalleryStore

workspace_root = Path(os.environ.get("RETAIL_AI_ROOT", Path(__file__).resolve().parents[1]))
data_dir = workspace_root / "data/processed/crops/gt_clean"

def migrate_pickle(pkl_name: str, db_name: str) -> None:
    pkl_path = data_dir / pkl_name
    db_path = data_dir / db_name
    
    if not pkl_path.exists():
        print(f"Skipping migration: {pkl_path} does not exist.")
        return
        
    print(f"Loading pickle embedding cache: {pkl_path}")
    with open(pkl_path, "rb") as f:
        db_data = pickle.load(f)
        
    embeddings = db_data["embeddings"]
    records = db_data["crop_records"]
    
    print(f"Loaded {len(records)} reference crops from pickle.")
    
    # Initialize SQLite Store
    store = SQLiteGalleryStore()
    store.initialize({"db_path": str(db_path)})
    
    print(f"Migrating to SQLite DB: {db_path}...")
    bulk_data = []
    for idx, record in enumerate(records):
        bbox_coords = record.get("bbox", [0.0, 0.0, 0.0, 0.0])
        vec = embeddings[idx].tolist()
        
        bulk_data.append((
            int(record["remapped_class_id"]),
            int(record.get("old_class_id", -1)),
            record["crop_path"],
            record.get("family_id", "unknown_family"),
            record.get("source_image_name", "unknown_source"),
            [float(c) for c in bbox_coords],
            vec
        ))
        
    store.save_references_bulk(bulk_data)
    success_count = len(records)
    store.shutdown()
    print(f"Successfully migrated {success_count} / {len(records)} records into SQLite database.")


if __name__ == "__main__":
    # Migrate DINOv2 Train Gallery
    migrate_pickle("embeddings_dinov2_train.pkl", "retail_sku_registry_dinov2.db")
    # Migrate CLIP Train Gallery
    migrate_pickle("embeddings_clip_train.pkl", "retail_sku_registry_clip.db")
