"""Script to inspect SQLite schemas and completely purge Class 99 and Class 100 across all database tables and catalog JSONs.
"""

import json
import sqlite3
import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
target_classes = [99, 100]
target_str_classes = [str(c) for c in target_classes]

print("=" * 70)
print(f"REMOVING CLASSES {target_classes} FROM DATABASE & CATALOG METADATA")
print("=" * 70)

# 1. Clean SQLite Databases
db_files = list(repo_root.glob("**/*.db"))
for db_path in db_files:
    if ".cache" in str(db_path):
        continue
    print(f"\n[Database] Processing: {db_path.relative_to(repo_root)}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        
        deleted_count = 0
        for tbl in tables:
            # Inspect column names
            cursor.execute(f"PRAGMA table_info({tbl});")
            cols = [col[1] for col in cursor.fetchall()]
            
            for class_col in ["remapped_class_id", "training_class_id", "class_id", "old_class_id"]:
                if class_col in cols:
                    query = f"DELETE FROM {tbl} WHERE {class_col} IN ({','.join(target_str_classes)})"
                    cursor.execute(query)
                    deleted_count += cursor.rowcount

        conn.commit()
        conn.close()
        print(f"  -> Successfully deleted {deleted_count} records from {db_path.name}")
    except Exception as e:
        print(f"  -> Error cleaning {db_path.name}: {e}")

# 2. Clean Catalog Metadata JSONs
catalog_json_paths = [
    repo_root / "configs" / "sku_mapping_v2.json",
    repo_root / "configs" / "sku_mapping.json"
]

for cat_path in catalog_json_paths:
    if not cat_path.exists():
        continue
    print(f"\n[Catalog JSON] Processing: {cat_path.relative_to(repo_root)}")
    try:
        with open(cat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        classes_dict = data.get("classes", {})
        removed_keys = []
        for key in list(classes_dict.keys()):
            val = classes_dict[key]
            t_id = val.get("training_class_id")
            r_id = val.get("raw_class_id")
            if key in target_str_classes or int(key) in target_classes or t_id in target_classes or (r_id and int(r_id) in target_classes):
                del classes_dict[key]
                removed_keys.append(key)
        
        with open(cat_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"  -> Removed keys {removed_keys} from {cat_path.name}")
    except Exception as e:
        print(f"  -> Error updating {cat_path.name}: {e}")

# 3. Clean Sku Preview Thumbnails
preview_base = repo_root / "data" / "processed" / "Sku Preview"
for cid in target_classes:
    p_dir = preview_base / f"class_{cid}"
    if p_dir.exists():
        shutil.rmtree(p_dir)
        print(f"\n[Thumbnail] Deleted Sku Preview directory: {p_dir}")

print("\n" + "=" * 70)
print("CLEANUP COMPLETE: Class 99 and Class 100 have been completely purged.")
print("=" * 70)
