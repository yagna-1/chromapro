"""Migration and maintenance CLI helpers."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
from typing import Any

from .client import PersistentClient


def _resolve_sqlite_source(path: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(expanded):
        candidate = os.path.join(expanded, "chroma.sqlite3")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"No chroma.sqlite3 found in directory: {expanded}")
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Source SQLite path not found: {expanded}")
    return expanded


def _find_table(conn: sqlite3.Connection, names: list[str]) -> str | None:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    existing = {row[0] for row in rows}
    for name in names:
        if name in existing:
            return name
    return None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


def _parse_embedding(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return [float(x) for x in json.loads(value)]
    if isinstance(value, (bytes, bytearray)):
        return [float(x) for x in json.loads(value.decode("utf-8"))]
    if isinstance(value, list):
        return [float(x) for x in value]
    raise ValueError(f"Unsupported embedding payload type: {type(value)}")


def migrate_sqlite_to_chromapro(
    source: str,
    target_path: str,
    report_path: str | None = None,
    batch_size: int = 1000,
) -> dict[str, Any]:
    sqlite_path = _resolve_sqlite_source(source)

    report: dict[str, Any] = {
        "source": sqlite_path,
        "target": os.path.abspath(os.path.expanduser(target_path)),
        "collections_found": 0,
        "records_read": 0,
        "records_written": 0,
        "records_skipped": 0,
        "errors": [],
    }

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    with PersistentClient(path=target_path) as client:
        collections_table = _find_table(conn, ["collections", "collection"])
        embeddings_table = _find_table(conn, ["embeddings", "embedding", "embeddings_queue"])

        if embeddings_table is None:
            raise RuntimeError("No embeddings table found in source SQLite database")

        collection_map: dict[str, Any] = {}

        if collections_table is not None:
            col_columns = _table_columns(conn, collections_table)
            id_col = "id" if "id" in col_columns else None
            name_col = "name" if "name" in col_columns else None
            metadata_col = "metadata" if "metadata" in col_columns else None

            if id_col and name_col:
                rows = conn.execute(
                    f"SELECT {id_col} as id, {name_col} as name"
                    + (f", {metadata_col} as metadata" if metadata_col else "")
                    + f" FROM {collections_table}"
                ).fetchall()
                report["collections_found"] = len(rows)
                for row in rows:
                    metadata = None
                    if metadata_col and row["metadata"]:
                        try:
                            metadata = json.loads(row["metadata"])
                        except Exception:
                            metadata = {"raw_metadata": str(row["metadata"])}
                    collection_map[str(row["id"])] = client.get_or_create_collection(
                        name=str(row["name"]),
                        metadata=metadata,
                    )

        emb_columns = _table_columns(conn, embeddings_table)
        if "id" not in emb_columns or "embedding" not in emb_columns:
            raise RuntimeError(f"Missing required columns in {embeddings_table}")

        collection_ref_col = None
        for candidate in ["collection_id", "segment_id", "collection"]:
            if candidate in emb_columns:
                collection_ref_col = candidate
                break

        document_col = "document" if "document" in emb_columns else None
        metadata_col = "metadata" if "metadata" in emb_columns else None

        select_cols = ["id", "embedding"]
        if collection_ref_col:
            select_cols.append(collection_ref_col)
        if document_col:
            select_cols.append(document_col)
        if metadata_col:
            select_cols.append(metadata_col)

        cursor = conn.execute(f"SELECT {', '.join(select_cols)} FROM {embeddings_table}")

        buffers: dict[str, dict[str, list[Any]]] = {}

        def flush_buffer(coll_key: str) -> None:
            buf = buffers[coll_key]
            if not buf["ids"]:
                return
            collection = collection_map.get(coll_key)
            if collection is None:
                collection = client.get_or_create_collection(name=f"collection_{coll_key}")
                collection_map[coll_key] = collection
            collection.add(
                ids=buf["ids"],
                embeddings=buf["embeddings"],
                documents=buf["documents"],
                metadatas=buf["metadatas"],
            )
            report["records_written"] += len(buf["ids"])
            buf["ids"].clear()
            buf["embeddings"].clear()
            buf["documents"].clear()
            buf["metadatas"].clear()

        for row in cursor:
            report["records_read"] += 1
            try:
                coll_key = str(row[collection_ref_col]) if collection_ref_col else "default"
                if coll_key not in buffers:
                    buffers[coll_key] = {
                        "ids": [],
                        "embeddings": [],
                        "documents": [],
                        "metadatas": [],
                    }

                metadata = {}
                if metadata_col and row[metadata_col]:
                    try:
                        metadata = json.loads(row[metadata_col])
                    except Exception:
                        metadata = {"raw_metadata": str(row[metadata_col])}

                buffers[coll_key]["ids"].append(str(row["id"]))
                buffers[coll_key]["embeddings"].append(_parse_embedding(row["embedding"]))
                buffers[coll_key]["documents"].append(str(row[document_col]) if document_col else "")
                buffers[coll_key]["metadatas"].append(metadata)

                if len(buffers[coll_key]["ids"]) >= batch_size:
                    flush_buffer(coll_key)
            except Exception as exc:  # noqa: BLE001
                report["records_skipped"] += 1
                if len(report["errors"]) < 50:
                    report["errors"].append(str(exc))

        for coll_key in list(buffers):
            flush_buffer(coll_key)

    conn.close()

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def rebuild_index(path: str, collection_name: str) -> dict[str, Any]:
    with PersistentClient(path=path) as client:
        collection = client.get_collection(collection_name)
        before = collection.count()
        collection.rebuild_index()
        after = collection.count()
        return {"collection": collection_name, "before": before, "after": after}


def health(path: str) -> dict[str, Any]:
    with PersistentClient(path=path) as client:
        hb = client.heartbeat()
        names = [c.name for c in client.list_collections()]
        return {"heartbeat": hb, "collections": names, "status": "ok"}


def stats(path: str, collection_name: str | None = None) -> dict[str, Any]:
    with PersistentClient(path=path) as client:
        if collection_name:
            collection = client.get_collection(collection_name)
            deleted = len(getattr(collection, "_deleted_ids", set()))
            total = collection._index.get_current_count()  # pylint: disable=protected-access
            active = collection.count()
            ratio = (deleted / total) if total else 0.0
            return {
                "collection": collection_name,
                "total_index_entries": total,
                "active_entries": active,
                "tombstones": deleted,
                "tombstone_ratio": ratio,
            }

        return {
            "collections": [
                {
                    "name": c.name,
                    "active_entries": c.count(),
                    "tombstones": len(getattr(c, "_deleted_ids", set())),
                }
                for c in client.list_collections()
            ]
        }


def _collect_integrity_snapshot(collection) -> dict[str, Any]:
    id_map = dict(getattr(collection, "_id_map", {}))
    reverse_map = dict(getattr(collection, "_reverse_map", {}))
    deleted = set(getattr(collection, "_deleted_ids", set()))
    storage_ids = set(collection._client._storage.list_embedding_ids(collection.id))
    mapping_ids = set(id_map.keys())
    active_mapping_ids = {id_ for id_ in mapping_ids if id_ not in deleted}
    reverse_ids = set(reverse_map.values())

    missing_reverse = sorted(mapping_ids - reverse_ids)
    dangling_reverse = sorted(reverse_ids - mapping_ids)
    stale_mapping_ids = sorted(active_mapping_ids - storage_ids)
    unmapped_storage_ids = sorted(storage_ids - mapping_ids)
    deleted_ids_present_in_storage = sorted(deleted & storage_ids)

    reverse_pair_mismatch = 0
    for str_id, label in id_map.items():
        if reverse_map.get(int(label)) != str_id:
            reverse_pair_mismatch += 1

    counter = int(getattr(collection, "_counter", 0))
    max_label = max(reverse_map.keys()) if reverse_map else -1
    counter_regression = counter < (max_label + 1)
    index_count = int(collection._index.get_current_count())
    index_count_mismatch = index_count != len(reverse_map)

    dimension_mismatch_ids: list[str] = []
    for str_id in sorted(storage_ids):
        rows = collection._client._storage.get_embeddings(collection.id, [str_id])
        if not rows or not rows[0][0]:
            continue
        vector = pickle.loads(bytes(rows[0][0]))
        if len(vector) != int(collection._dimension):
            dimension_mismatch_ids.append(str_id)
            if len(dimension_mismatch_ids) >= 20:
                break

    active_ids = sorted(id_ for id_ in mapping_ids if id_ not in deleted)
    sample = active_ids[:1000]
    missing = sorted(id_ for id_ in sample if id_ not in storage_ids)

    issues: list[str] = []
    if missing_reverse:
        issues.append("missing_reverse_entries")
    if dangling_reverse:
        issues.append("dangling_reverse_entries")
    if stale_mapping_ids:
        issues.append("mapping_ids_without_storage")
    if unmapped_storage_ids:
        issues.append("storage_ids_without_mapping")
    if deleted_ids_present_in_storage:
        issues.append("deleted_ids_present_in_storage")
    if reverse_pair_mismatch:
        issues.append("reverse_pair_mismatch")
    if counter_regression:
        issues.append("counter_regression")
    if index_count_mismatch:
        issues.append("index_count_mismatch")
    if dimension_mismatch_ids:
        issues.append("dimension_mismatch")
    if missing:
        issues.append("sample_missing_payloads")

    return {
        "checked": len(sample),
        "missing": missing,
        "missing_reverse": missing_reverse,
        "dangling_reverse": dangling_reverse,
        "stale_mapping_ids": stale_mapping_ids,
        "unmapped_storage_ids": unmapped_storage_ids,
        "deleted_ids_present_in_storage": deleted_ids_present_in_storage,
        "reverse_pair_mismatch": reverse_pair_mismatch,
        "counter": counter,
        "max_label": max_label,
        "counter_regression": counter_regression,
        "index_count": index_count,
        "reverse_count": len(reverse_map),
        "index_count_mismatch": index_count_mismatch,
        "dimension_mismatch_ids": dimension_mismatch_ids,
        "issues": issues,
        "ok": len(issues) == 0,
    }


def _repair_collection_integrity(collection, snapshot: dict[str, Any]) -> list[str]:
    repairs: list[str] = []

    deleted_in_storage = snapshot.get("deleted_ids_present_in_storage", [])
    if deleted_in_storage:
        collection._client._storage.delete_embeddings(collection.id, deleted_in_storage)
        repairs.append("deleted_payloads_removed")

    if snapshot.get("issues"):
        collection._rebuild_from_storage_locked(clear_tombstones=False)
        repairs.append("index_and_mapping_rebuilt")

    return repairs


def verify(path: str, collection_name: str, repair: bool = False) -> dict[str, Any]:
    with PersistentClient(path=path) as client:
        collection = client.get_collection(collection_name)
        collection._acquire_lock(shared=not repair)
        try:
            with collection._write_lock:
                collection._refresh_from_disk_locked()
                before = _collect_integrity_snapshot(collection)

                repairs_applied: list[str] = []
                after = None
                if repair and not before["ok"]:
                    repairs_applied = _repair_collection_integrity(collection, before)
                    collection._refresh_from_disk_locked()
                    after = _collect_integrity_snapshot(collection)
        finally:
            collection._release_lock()

        return {
            "collection": collection_name,
            "checked": before["checked"],
            "missing": before["missing"],
            "ok": before["ok"],
            "issues": before["issues"],
            "details": before,
            "repair_requested": repair,
            "repair_applied": len(repairs_applied) > 0 if repair else False,
            "repairs": repairs_applied if repair else [],
            "post_repair_ok": after["ok"] if after is not None else None,
            "post_repair_issues": after["issues"] if after is not None else None,
            "post_repair_details": after,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chromapro")
    sub = parser.add_subparsers(dest="command", required=True)

    migrate_cmd = sub.add_parser("migrate", help="Migrate SQLite data into ChromaPro")
    migrate_cmd.add_argument("source", help="Source SQLite file or directory")
    migrate_cmd.add_argument("target", help="Target ChromaPro data directory")
    migrate_cmd.add_argument("--report", required=False, help="Output JSON report path")
    migrate_cmd.add_argument("--batch-size", type=int, default=1000)

    rebuild_cmd = sub.add_parser("rebuild-index", help="Rebuild collection HNSW index")
    rebuild_cmd.add_argument("path", help="ChromaPro data directory")
    rebuild_cmd.add_argument("collection_name", help="Collection name")

    health_cmd = sub.add_parser("health", help="Run health check")
    health_cmd.add_argument("path", help="ChromaPro data directory")

    stats_cmd = sub.add_parser("stats", help="Show tombstone/index stats")
    stats_cmd.add_argument("path", help="ChromaPro data directory")
    stats_cmd.add_argument("collection_name", nargs="?", default=None)

    verify_cmd = sub.add_parser("verify", help="Verify collection consistency")
    verify_cmd.add_argument("path", help="ChromaPro data directory")
    verify_cmd.add_argument("collection_name", help="Collection name")
    verify_cmd.add_argument("--repair", action="store_true", help="Attempt automatic repair for detected issues")

    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "migrate":
        result = migrate_sqlite_to_chromapro(
            source=args.source,
            target_path=args.target,
            report_path=args.report,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "rebuild-index":
        result = rebuild_index(args.path, args.collection_name)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "health":
        result = health(args.path)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "stats":
        result = stats(args.path, args.collection_name)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "verify":
        result = verify(args.path, args.collection_name, repair=args.repair)
        print(json.dumps(result, indent=2))
        return 0

    parser.print_help()
    return 1
