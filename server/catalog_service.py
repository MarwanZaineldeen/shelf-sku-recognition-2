"""Coordinated catalog mutations shared by the API and maintenance scripts.

Runtime recognition uses ``training_class_id``. ``raw_class_id`` is immutable
dataset provenance and must never be used as a second deletion predicate:
different SKUs can legitimately have a runtime ID equal to another SKU's raw
ID.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np


CATALOG_FILE = "sku_mapping.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, payload)


def load_catalog_documents(workspace_root: Path) -> Dict[Path, Dict[str, Any]]:
    documents: Dict[Path, Dict[str, Any]] = {}
    path = workspace_root / "configs" / CATALOG_FILE
    if not path.exists():
        raise FileNotFoundError(f"Required catalog file is missing: {path}")
    document = _load_json(path)
    if not isinstance(document.get("classes"), dict):
        raise ValueError(f"Catalog file has no classes object: {path}")
    documents[path] = document
    return documents


def catalog_by_training_id(document: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for key, raw_info in document.get("classes", {}).items():
        info = dict(raw_info)
        class_id = int(info.get("training_class_id", key))
        by_id[str(class_id)] = info
    return dict(sorted(by_id.items(), key=lambda item: int(item[0])))


def build_compacted_catalog(
    document: Mapping[str, Any],
    deleted_ids: Iterable[int],
) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """Removes runtime IDs and compacts all remaining runtime IDs.

    JSON keys and ``raw_class_id`` remain stable. Only
    ``training_class_id`` is compacted.
    """

    deleted = {int(class_id) for class_id in deleted_ids}
    source_classes = document.get("classes", {})
    remaining: List[Tuple[int, str, Dict[str, Any]]] = []
    for key, raw_info in source_classes.items():
        info = dict(raw_info)
        runtime_id = int(info.get("training_class_id", key))
        if runtime_id not in deleted:
            remaining.append((runtime_id, str(key), info))

    remaining.sort(key=lambda item: item[0])
    id_remap = {old_id: new_id for new_id, (old_id, _, _) in enumerate(remaining)}
    compacted_classes: Dict[str, Dict[str, Any]] = {}
    for old_id, key, info in remaining:
        info["training_class_id"] = id_remap[old_id]
        compacted_classes[key] = info

    result = dict(document)
    result["classes"] = compacted_classes
    return result, id_remap


def _class_id_mapping_payload(document: Mapping[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    old_to_new: Dict[str, int] = {}
    new_to_old: Dict[str, int] = {}
    rows: List[Tuple[int, int]] = []
    for key, info in document.get("classes", {}).items():
        raw_value = info.get("raw_class_id", key)
        try:
            raw_id = int(raw_value)
        except (TypeError, ValueError):
            # Onboarded non-numeric provenance cannot be represented in the
            # legacy integer map, but it remains present in the SKU catalog.
            continue
        runtime_id = int(info.get("training_class_id", key))
        old_to_new[str(raw_id)] = runtime_id
        new_to_old[str(runtime_id)] = raw_id
        rows.append((runtime_id, raw_id))

    mapping_json = {
        "old_to_new": dict(sorted(old_to_new.items(), key=lambda item: int(item[0]))),
        "new_to_old": dict(sorted(new_to_old.items(), key=lambda item: int(item[0]))),
    }
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["new_class_id", "original_class_id"])
    for runtime_id, raw_id in sorted(rows):
        writer.writerow([runtime_id, raw_id])
    return mapping_json, output.getvalue().encode("utf-8")


def write_catalog_documents(
    workspace_root: Path,
    documents: Mapping[Path, Mapping[str, Any]],
) -> None:
    for path, document in documents.items():
        atomic_write_json(path, document)

    authoritative = documents[workspace_root / "configs" / CATALOG_FILE]
    mapping_json, mapping_csv = _class_id_mapping_payload(authoritative)
    atomic_write_json(workspace_root / "configs" / "class_id_mapping.json", mapping_json)
    _atomic_write_bytes(workspace_root / "configs" / "class_id_mapping.csv", mapping_csv)


def _backup_sqlite_connection(connection: sqlite3.Connection, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = sqlite3.connect(str(destination))
    try:
        connection.backup(backup)
    finally:
        backup.close()


def _directory_class_id(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith("class_"):
        return None
    try:
        return int(name.removeprefix("class_"))
    except ValueError:
        return None


def _stage_class_directories(
    bases: Iterable[Path],
    deleted_ids: set[int],
    id_remap: Mapping[int, int],
    backup_dir: Path,
) -> List[Tuple[Path, Path]]:
    """Renames class directories in two phases and returns reversible moves."""

    completed_moves: List[Tuple[Path, Path]] = []
    token = uuid.uuid4().hex
    for base in bases:
        if not base.exists():
            continue
        staged: List[Tuple[int, Path, Path]] = []
        for child in list(base.iterdir()):
            if not child.is_dir():
                continue
            old_id = _directory_class_id(child)
            should_delete = old_id in deleted_ids
            should_move = old_id in id_remap and id_remap[old_id] != old_id
            if old_id is None or (not should_delete and not should_move):
                continue
            stage = base / f".catalog-reindex-{token}-{old_id}"
            child.rename(stage)
            completed_moves.append((child, stage))
            staged.append((old_id, child, stage))

        for old_id, original, stage in staged:
            if old_id in deleted_ids:
                relative = base.as_posix().replace(":", "").strip("/").replace("/", "__")
                destination = backup_dir / "removed_directories" / relative / original.name
            else:
                destination = base / f"class_{id_remap[old_id]:02d}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(
                    f"Cannot reindex {original}: destination already exists: {destination}"
                )
            stage.rename(destination)
            completed_moves.append((stage, destination))
    return completed_moves


def _reverse_moves(moves: Iterable[Tuple[Path, Path]]) -> None:
    for source, destination in reversed(list(moves)):
        if destination.exists() and not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(source)


def _prepare_index_snapshot(
    store: Any,
    retriever: Any,
) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    if retriever is None:
        return None, []
    vectors, metadata = store.fetch_all_references()
    dimension = int(getattr(retriever, "dimension", vectors.shape[1] if vectors.size else 0))
    if vectors.shape[0] and vectors.shape[1] != dimension:
        raise ValueError(
            f"Cannot rebuild runtime index: database has {vectors.shape[1]}-D vectors "
            f"but the active index expects {dimension}-D."
        )
    # Normalize on a copy now so no fallible work remains after commit.
    prepared = vectors.astype(np.float32, copy=True)
    if prepared.shape[0]:
        norms = np.linalg.norm(prepared, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        prepared /= norms
    return prepared, metadata


def delete_and_reindex_catalog(
    workspace_root: Path,
    store: Any,
    retriever: Any,
    orchestrator: Any,
    class_ids: Iterable[int],
) -> Dict[str, Any]:
    """Deletes classes and commits one coordinated catalog state transition."""

    requested = sorted({int(class_id) for class_id in class_ids})
    if not requested or any(class_id < 0 for class_id in requested):
        raise ValueError("class_ids must contain at least one non-negative class ID.")
    if store is None or getattr(store, "conn", None) is None:
        raise RuntimeError("Gallery database is not initialized.")

    documents = load_catalog_documents(workspace_root)
    authoritative_path = workspace_root / "configs" / CATALOG_FILE
    authoritative = documents[authoritative_path]
    known_ids = {
        int(info.get("training_class_id", key))
        for key, info in authoritative.get("classes", {}).items()
    }
    database_ids = set(store.class_size_histogram())
    unreconciled = sorted(database_ids - known_ids - set(requested))
    if unreconciled:
        raise ValueError(
            "Catalog mutation refused because the database contains unmapped "
            f"runtime classes {unreconciled}. Reconcile or explicitly delete "
            "those classes first."
        )

    prospective: Dict[Path, Dict[str, Any]] = {}
    canonical_remap: Optional[Dict[int, int]] = None
    for path, document in documents.items():
        compacted, remap = build_compacted_catalog(document, requested)
        prospective[path] = compacted
        if path == authoritative_path:
            canonical_remap = remap
    assert canonical_remap is not None

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = (
        workspace_root
        / "data"
        / "processed"
        / "backups"
        / "catalog_mutations"
        / f"{stamp}-{uuid.uuid4().hex[:8]}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)

    files_to_backup = [
        *documents.keys(),
        workspace_root / "configs" / "class_id_mapping.json",
        workspace_root / "configs" / "class_id_mapping.csv",
    ]
    original_bytes: Dict[Path, bytes] = {}
    for path in files_to_backup:
        if path.exists():
            payload = path.read_bytes()
            original_bytes[path] = payload
            relative = path.relative_to(workspace_root)
            destination = backup_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)

    db_backup = backup_dir / Path(store.db_path).name
    _backup_sqlite_connection(store.conn, db_backup)

    connection: sqlite3.Connection = store.conn
    moves: List[Tuple[Path, Path]] = []
    deleted_vectors = 0
    try:
        connection.execute("BEGIN IMMEDIATE")
        placeholders = ",".join("?" for _ in requested)
        cursor = connection.execute(
            f"DELETE FROM sku_crops WHERE remapped_class_id IN ({placeholders})",
            requested,
        )
        deleted_vectors = max(0, int(cursor.rowcount))

        changed = {old: new for old, new in canonical_remap.items() if old != new}
        if changed:
            cases = " ".join("WHEN ? THEN ?" for _ in changed)
            params: List[int] = []
            for old_id, new_id in sorted(changed.items()):
                params.extend([old_id, new_id])
            old_ids = sorted(changed)
            old_placeholders = ",".join("?" for _ in old_ids)
            connection.execute(
                f"UPDATE sku_crops SET remapped_class_id = CASE remapped_class_id "
                f"{cases} ELSE remapped_class_id END "
                f"WHERE remapped_class_id IN ({old_placeholders})",
                params + old_ids,
            )

        write_catalog_documents(workspace_root, prospective)
        moves = _stage_class_directories(
            bases=[
                workspace_root / "data" / "processed" / "Sku Preview",
                workspace_root / "configs" / "class_catalog",
            ],
            deleted_ids=set(requested),
            id_remap=canonical_remap,
            backup_dir=backup_dir,
        )
        prepared_vectors, prepared_metadata = _prepare_index_snapshot(store, retriever)
        prepared_sku_mapping = {
            int(class_id): dict(info)
            for class_id, info in catalog_by_training_id(
                prospective[authoritative_path]
            ).items()
        }
        connection.commit()

        if retriever is not None:
            retriever.gallery_vectors = prepared_vectors if prepared_vectors is not None and len(prepared_vectors) else None
            retriever.metadata = list(prepared_metadata)
        if orchestrator is not None:
            orchestrator.sku_mapping = prepared_sku_mapping
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        _reverse_moves(moves)
        for path in files_to_backup:
            if path in original_bytes:
                _atomic_write_bytes(path, original_bytes[path])
            elif path.exists():
                path.unlink()
        raise

    final_catalog = catalog_by_training_id(prospective[authoritative_path])
    return {
        "status": "success",
        "deleted_class_ids": requested,
        "missing_catalog_class_ids": sorted(set(requested) - known_ids),
        "deleted_vectors_count": deleted_vectors,
        "id_remap": {str(old): new for old, new in sorted(canonical_remap.items())},
        "next_class_id": len(final_catalog),
        "catalog": {"classes": final_catalog},
        "backup_path": str(backup_dir),
    }


def choose_onboarding_raw_id(
    documents: Mapping[Path, Mapping[str, Any]],
    requested_raw_id: int,
) -> int:
    """Returns a provenance ID that cannot overwrite an existing JSON key."""

    used: set[int] = set()
    for document in documents.values():
        for key, info in document.get("classes", {}).items():
            raw_value = info.get("raw_class_id", key)
            try:
                used.add(int(raw_value))
            except (TypeError, ValueError):
                continue
    if requested_raw_id not in used:
        return requested_raw_id
    return (max(used) + 1) if used else 0
