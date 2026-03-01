"""
Production-ready collection with:
- Bounded tombstone deletion
- Immediate index persistence
- Thread-safe operations
- File-based process locking
"""

from __future__ import annotations

import errno
import json
import os
import pickle
import threading
import time
from typing import Any

import hnswlib
import numpy as np

if os.name != "nt":
    import fcntl


class Collection:
    """
    ChromaDB-compatible collection with production-grade correctness.
    """

    def __init__(
        self,
        client,
        name: str,
        id: str,
        dimension: int = 384,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
    ):
        self._client = client
        self.name = name
        self.id = id
        self.metadata = metadata or {}
        self._embedding_function = embedding_function
        self._dimension = int(self.metadata.get("dimension", dimension))
        self._index_version = int(getattr(client, "_storage_meta", {}).get("index_version", 1))
        self._mapping_version = int(getattr(client, "_storage_meta", {}).get("mapping_version", 1))

        self._write_lock = threading.Lock()

        self._index = hnswlib.Index(space="l2", dim=self._dimension)
        self._index_path = os.path.join(client.path, f"{id}.hnsw")

        self._id_map: dict[str, int] = {}
        self._reverse_map: dict[int, str] = {}
        self._counter = 0

        self._deleted_ids: set[str] = set()
        self._deleted_path = os.path.join(client.path, f"{id}.deleted")

        self._lock_path = os.path.join(client.path, f"{id}.lock")
        self._lock_file = None

        if os.path.exists(self._index_path):
            try:
                self._load_index()
            except (json.JSONDecodeError, RuntimeError, OSError):
                self._recover_from_corrupt_index_state()
        else:
            # .hnsw is absent — check if RocksDB has data (surviving index deletion / crash)
            # If yes, rebuild from RocksDB (the ground truth).
            # If no, start fresh (truly new collection).
            try:
                existing_ids = client._storage.list_embedding_ids(id)
            except Exception:
                existing_ids = []
            if existing_ids:
                # RocksDB has data but HNSW is missing — rebuild
                self._reset_index(self._dimension)
                self._recover_from_corrupt_index_state()
            else:
                self._reset_index(self._dimension)

    def _reset_index(self, dimension: int) -> None:
        self._dimension = int(dimension)
        self._index = hnswlib.Index(space="l2", dim=self._dimension)
        self._index.init_index(max_elements=1_000_000, ef_construction=200, M=16)
        self._index.set_ef(64)

    def _acquire_lock(self, shared: bool = False) -> None:
        self._lock_file = open(self._lock_path, "a+", encoding="utf-8")
        if os.name == "nt":
            # msvcrt provides only exclusive region locking; shared reads are
            # upgraded to exclusive lock on Windows for correctness.
            self._lock_file.seek(0)
            marker = self._lock_file.read(1)
            if marker == "":
                self._lock_file.seek(0)
                self._lock_file.write("\0")
                self._lock_file.flush()
            self._acquire_windows_lock()
            return

        lock_type = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(self._lock_file.fileno(), lock_type)

    def _release_lock(self) -> None:
        if self._lock_file:
            if os.name == "nt":
                self._release_windows_lock()
            else:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def _acquire_windows_lock(self) -> None:
        import msvcrt

        while True:
            try:
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError as exc:
                if exc.errno in (getattr(errno, "EACCES", 13), getattr(errno, "EDEADLK", 36), 13, 36):
                    time.sleep(0.05)
                    continue
                raise

    def _release_windows_lock(self) -> None:
        import msvcrt

        self._lock_file.seek(0)
        msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    def add(
        self,
        ids: list[str] | str,
        embeddings: list[list[float]] | np.ndarray | None = None,
        documents: list[str] | str | None = None,
        metadatas: list[dict] | dict | None = None,
    ) -> None:
        if isinstance(ids, str):
            ids = [ids]
        if documents and isinstance(documents, str):
            documents = [documents]
        if metadatas and isinstance(metadatas, dict):
            metadatas = [metadatas]

        if embeddings is None:
            if documents is None:
                raise ValueError("Either embeddings or documents required")
            if self._embedding_function is None:
                raise ValueError("No embedding function provided")
            embeddings = self._embedding_function(documents)

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            embeddings = embeddings.astype(np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self._dimension:
            if self._index.get_current_count() == 0 and self._counter == 0:
                self._reset_index(int(embeddings.shape[1]))
            else:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._dimension}, got {embeddings.shape[1]}"
                )

        documents = documents or [""] * len(ids)
        metadatas = metadatas or [{}] * len(ids)

        if len(ids) != len(documents) or len(ids) != len(metadatas) or len(ids) != len(embeddings):
            raise ValueError("ids, embeddings, documents, and metadatas must have same length")

        self._acquire_lock(shared=False)
        try:
            with self._write_lock:
                # Pull latest on-disk state to avoid clobbering other processes.
                self._refresh_from_disk_locked()

                numeric_ids: list[int] = []
                index_vectors: list[np.ndarray] = []
                resurrected_ids: set[str] = set()

                for i, id_str in enumerate(ids):
                    if id_str not in self._id_map:
                        numeric_id = self._counter
                        self._id_map[id_str] = numeric_id
                        self._reverse_map[numeric_id] = id_str
                        self._counter += 1
                    else:
                        numeric_id = self._id_map[id_str]

                    numeric_ids.append(numeric_id)
                    index_vectors.append(embeddings[i])
                    if id_str in self._deleted_ids:
                        resurrected_ids.add(id_str)

                # hnswlib insert/update first
                for i, label in enumerate(numeric_ids):
                    vec = np.asarray(index_vectors[i], dtype=np.float32).reshape(1, -1)
                    try:
                        self._index.add_items(vec, [label])
                    except RuntimeError:
                        # Duplicate labels can fail in some hnswlib builds.
                        # Allocate fresh label and update mappings.
                        fresh = self._counter
                        self._counter += 1
                        id_str = ids[i]
                        self._id_map[id_str] = fresh
                        self._reverse_map[fresh] = id_str
                        self._index.add_items(vec, [fresh])

                # Immediate index persistence for crash consistency.
                self._save_index()

                # Then persist payload to RocksDB.
                serialized_embeddings = [pickle.dumps(emb.tolist()) for emb in embeddings]
                serialized_metadatas = [json.dumps(meta) for meta in metadatas]

                try:
                    self._client._storage.batch_insert(
                        collection_id=self.id,
                        ids=ids,
                        embeddings=serialized_embeddings,
                        documents=documents,
                        metadatas=serialized_metadatas,
                    )
                except Exception:
                    # If RocksDB write fails after index save, repair index from storage
                    # so orphan entries do not accumulate.
                    self._rebuild_from_storage_locked(clear_tombstones=False)
                    raise

                if resurrected_ids:
                    self._deleted_ids.difference_update(resurrected_ids)
                    self._persist_deleted_ids_locked()
        finally:
            self._release_lock()

    def query(
        self,
        query_embeddings: list[list[float]] | np.ndarray | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] = ["documents", "metadatas", "distances"],
    ) -> dict[str, Any]:
        if query_embeddings is None:
            if query_texts is None:
                raise ValueError("Either query_embeddings or query_texts required")
            if self._embedding_function is None:
                raise ValueError("No embedding function provided")
            query_embeddings = self._embedding_function(query_texts)

        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.array(query_embeddings, dtype=np.float32)
        else:
            query_embeddings = query_embeddings.astype(np.float32)

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        if query_embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dimension}, got {query_embeddings.shape[1]}"
            )

        self._acquire_lock(shared=True)
        try:
            with self._write_lock:
                self._refresh_from_disk_locked()

                current_count = self._index.get_current_count()
                if current_count == 0 or n_results <= 0:
                    return {
                        "ids": [[] for _ in range(len(query_embeddings))],
                        "distances": [[] for _ in range(len(query_embeddings))] if "distances" in include else None,
                        "documents": [[] for _ in range(len(query_embeddings))] if "documents" in include else None,
                        "metadatas": [[] for _ in range(len(query_embeddings))] if "metadatas" in include else None,
                    }

                deleted_count = len(self._deleted_ids)
                if where is not None or where_document is not None:
                    # For filter queries, fetch full candidate set and apply filters in Python.
                    k = current_count
                elif deleted_count == 0:
                    k = n_results
                else:
                    deletion_ratio = deleted_count / max(current_count, 1)
                    if deletion_ratio < 0.1:
                        k = n_results + 10
                    elif deletion_ratio < 0.3:
                        k = n_results * 2
                    else:
                        k = n_results * 3
                    k = min(k, current_count)

                k = max(1, min(k, current_count))
                labels, distances = self._index.knn_query(query_embeddings, k=k)

                filtered_rows: list[tuple[list[str], list[float]]] = []
                for i, label_row in enumerate(labels):
                    valid_ids: list[str] = []
                    valid_distances: list[float] = []

                    for j, label in enumerate(label_row):
                        str_id = self._reverse_map.get(int(label), "")
                        if str_id and str_id not in self._deleted_ids:
                            valid_ids.append(str_id)
                            valid_distances.append(float(distances[i][j]))
                        if where is None and where_document is None and len(valid_ids) >= n_results:
                            break

                    filtered_rows.append((valid_ids, valid_distances))
        finally:
            self._release_lock()

        all_results: list[dict[str, Any]] = []
        for ids_row, dist_row in filtered_rows:
            if ids_row:
                rows = self._client._storage.get_embeddings(self.id, ids_row)

                kept_ids: list[str] = []
                kept_dists: list[float] = []
                kept_rows: list[tuple[bytes, str, str]] = []
                for idx, row in enumerate(rows):
                    if not row[0]:
                        continue
                    metadata = self._parse_metadata(row[2])
                    if not self._matches_where(metadata, where):
                        continue
                    if not self._matches_where_document(row[1], where_document):
                        continue
                    kept_ids.append(ids_row[idx])
                    kept_dists.append(dist_row[idx])
                    kept_rows.append(row)
                    if len(kept_ids) >= n_results:
                        break

                all_results.append(
                    {
                        "ids": kept_ids,
                        "distances": kept_dists,
                        "rows": kept_rows,
                    }
                )
            else:
                all_results.append({"ids": [], "distances": [], "rows": []})

        return {
            "ids": [r["ids"] for r in all_results],
            "distances": [r["distances"] for r in all_results] if "distances" in include else None,
            "documents": [[res[1] for res in r["rows"]] for r in all_results] if "documents" in include else None,
            "metadatas": [[self._parse_metadata(res[2]) for res in r["rows"]] for r in all_results]
            if "metadatas" in include
            else None,
            "embeddings": [
                [pickle.loads(self._as_bytes(res[0])) for res in r["rows"]]
                for r in all_results
            ]
            if "embeddings" in include
            else None,
        }

    def delete(self, ids: list[str] | str) -> None:
        if isinstance(ids, str):
            ids = [ids]

        self._acquire_lock(shared=False)
        try:
            with self._write_lock:
                self._refresh_from_disk_locked()
                self._client._storage.delete_embeddings(self.id, ids)
                self._deleted_ids.update(ids)
                self._persist_deleted_ids_locked()
        finally:
            self._release_lock()

    def update(
        self,
        ids: list[str] | str,
        embeddings: list[list[float]] | np.ndarray | None = None,
        documents: list[str] | str | None = None,
        metadatas: list[dict] | dict | None = None,
    ) -> None:
        """Update existing records. Raises ValueError if any id does not exist."""
        if isinstance(ids, str):
            ids = [ids]
        if documents and isinstance(documents, str):
            documents = [documents]
        if metadatas and isinstance(metadatas, dict):
            metadatas = [metadatas]

        # Verify all IDs exist before making any changes
        existing = set(self._client._storage.list_embedding_ids(self.id)) - self._deleted_ids
        missing = [id_ for id_ in ids if id_ not in existing]
        if missing:
            raise ValueError(f"IDs not found (cannot update): {missing}")

        # upsert path handles all the write logic correctly
        self.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def upsert(
        self,
        ids: list[str] | str,
        embeddings: list[list[float]] | np.ndarray | None = None,
        documents: list[str] | str | None = None,
        metadatas: list[dict] | dict | None = None,
    ) -> None:
        """Insert or update records. Equivalent to add() with overwrite semantics."""
        self.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] = ["embeddings", "documents", "metadatas"],
    ) -> dict[str, Any]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        if offset is not None and offset < 0:
            raise ValueError("offset must be >= 0")

        self._acquire_lock(shared=True)
        try:
            with self._write_lock:
                self._refresh_from_disk_locked()
                if ids is None:
                    base_ids = self._client._storage.list_embedding_ids(self.id)
                else:
                    base_ids = [str(id_) for id_ in ids]
                base_ids = [id_ for id_ in base_ids if id_ not in self._deleted_ids]
        finally:
            self._release_lock()

        results = self._client._storage.get_embeddings(self.id, base_ids)
        matched: list[tuple[str, tuple[bytes, str, str]]] = []
        for idx, row in enumerate(results):
            if not row[0]:
                continue
            metadata = self._parse_metadata(row[2])
            if not self._matches_where(metadata, where):
                continue
            if not self._matches_where_document(row[1], where_document):
                continue
            matched.append((base_ids[idx], row))

        start = offset or 0
        end = None if limit is None else start + limit
        sliced = matched[start:end]

        found_ids: list[str] = [item[0] for item in sliced]
        found_rows: list[tuple[bytes, str, str]] = [item[1] for item in sliced]

        return {
            "ids": found_ids,
            "embeddings": [pickle.loads(self._as_bytes(r[0])) for r in found_rows] if "embeddings" in include else None,
            "documents": [r[1] for r in found_rows] if "documents" in include else None,
            "metadatas": [self._parse_metadata(r[2]) for r in found_rows] if "metadatas" in include else None,
        }

    def peek(
        self,
        limit: int = 10,
        include: list[str] = ["embeddings", "documents", "metadatas"],
    ) -> dict[str, Any]:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        return self.get(ids=None, limit=limit, offset=0, include=include)

    def count(self) -> int:
        self._acquire_lock(shared=True)
        try:
            with self._write_lock:
                self._refresh_from_disk_locked()
                return max(0, self._index.get_current_count() - len(self._deleted_ids))
        finally:
            self._release_lock()

    def rebuild_index(self) -> None:
        self._acquire_lock(shared=False)
        try:
            with self._write_lock:
                self._refresh_from_disk_locked()
                self._rebuild_from_storage_locked(clear_tombstones=True)
        finally:
            self._release_lock()

    def _recover_from_corrupt_index_state(self) -> None:
        self._acquire_lock(shared=False)
        try:
            with self._write_lock:
                if os.path.exists(self._deleted_path):
                    try:
                        with open(self._deleted_path, encoding="utf-8") as f:
                            self._deleted_ids = set(json.load(f))
                    except Exception:
                        self._deleted_ids = set()
                else:
                    self._deleted_ids = set()
                self._rebuild_from_storage_locked(clear_tombstones=False)
        finally:
            self._release_lock()

    def _rebuild_from_storage_locked(self, clear_tombstones: bool) -> None:
        ids = self._client._storage.list_embedding_ids(self.id)
        if clear_tombstones:
            self._deleted_ids.clear()
        active_ids = [id_ for id_ in ids if id_ not in self._deleted_ids]

        new_index = hnswlib.Index(space="l2", dim=self._dimension)
        new_index.init_index(max_elements=max(1_000_000, max(len(active_ids), 1)), ef_construction=200, M=16)
        new_index.set_ef(64)

        new_id_map: dict[str, int] = {}
        new_reverse_map: dict[int, str] = {}
        new_counter = 0

        if active_ids:
            rows = self._client._storage.get_embeddings(self.id, active_ids)
            for idx, row in enumerate(rows):
                emb_bytes = row[0]
                if not emb_bytes:
                    continue
                vector = np.array(
                    pickle.loads(self._as_bytes(emb_bytes)),
                    dtype=np.float32,
                ).reshape(1, -1)
                label = new_counter
                new_counter += 1
                str_id = active_ids[idx]
                new_id_map[str_id] = label
                new_reverse_map[label] = str_id
                new_index.add_items(vector, [label])

        self._index = new_index
        self._id_map = new_id_map
        self._reverse_map = new_reverse_map
        self._counter = new_counter
        self._save_index()
        self._persist_deleted_ids_locked()

    def _save_index(self) -> None:
        tmp_index = self._tmp_path(self._index_path)
        try:
            self._index.save_index(tmp_index)
            os.replace(tmp_index, self._index_path)
        finally:
            if os.path.exists(tmp_index):
                os.remove(tmp_index)

        mapping_payload = {
            "mapping_version": self._mapping_version,
            "index_version": self._index_version,
            "id_map": self._id_map,
            "reverse_map": {str(k): v for k, v in self._reverse_map.items()},
            "counter": self._counter,
            "dimension": self._dimension,
        }
        self._atomic_write_json(self._mapping_path(), mapping_payload)

    def _load_index(self) -> None:
        self._refresh_from_disk_locked()

    def persist(self) -> None:
        # Writes are persisted immediately in mutating operations.
        # Refresh here to avoid stale process state being written back.
        self._acquire_lock(shared=True)
        try:
            with self._write_lock:
                self._refresh_from_disk_locked()
        finally:
            self._release_lock()

    def _mapping_path(self) -> str:
        return os.path.join(self._client.path, f"{self.id}.mapping")

    def _refresh_from_disk_locked(self) -> None:
        data: dict[str, Any] = {}
        mapping_path = self._mapping_path()
        if os.path.exists(mapping_path):
            with open(mapping_path, encoding="utf-8") as f:
                data = json.load(f)
            mapping_version = int(data.get("mapping_version", 1))
            if mapping_version != self._mapping_version:
                raise ValueError(
                    f"Unsupported mapping format version for collection '{self.name}': "
                    f"{mapping_version} (expected {self._mapping_version})"
                )

        dim = int(data.get("dimension", self._dimension))
        if os.path.exists(self._index_path):
            fresh_index = hnswlib.Index(space="l2", dim=dim)
            fresh_index.load_index(self._index_path)
            fresh_index.set_ef(64)
            self._index = fresh_index
        else:
            self._reset_index(dim)

        self._dimension = dim
        self._id_map = {str(k): int(v) for k, v in data.get("id_map", {}).items()}

        reverse_map = data.get("reverse_map", {})
        if reverse_map:
            self._reverse_map = {int(k): str(v) for k, v in reverse_map.items()}
        else:
            self._reverse_map = {v: k for k, v in self._id_map.items()}

        self._counter = int(data.get("counter", len(self._id_map)))

        if os.path.exists(self._deleted_path):
            with open(self._deleted_path, encoding="utf-8") as f:
                self._deleted_ids = set(json.load(f))
        else:
            self._deleted_ids = set()

    @staticmethod
    def _as_bytes(blob: bytes | bytearray | list[int]) -> bytes:
        if isinstance(blob, (bytes, bytearray)):
            return bytes(blob)
        return bytes(blob)

    @staticmethod
    def _tmp_path(path: str) -> str:
        return f"{path}.tmp.{os.getpid()}.{time.time_ns()}"

    @staticmethod
    def _atomic_write_json(path: str, payload: Any) -> None:
        directory = os.path.dirname(path) or "."
        tmp_path = Collection._tmp_path(path)

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        try:
            dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    def _persist_deleted_ids_locked(self) -> None:
        if self._deleted_ids:
            self._atomic_write_json(self._deleted_path, sorted(self._deleted_ids))
            return
        if os.path.exists(self._deleted_path):
            os.remove(self._deleted_path)

    @staticmethod
    def _parse_metadata(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except Exception:
            return {"raw_metadata": str(raw)}

    @staticmethod
    def _compare_scalar(value: Any, op: str, expected: Any) -> bool:
        if op == "$eq":
            return value == expected
        if op == "$ne":
            return value != expected

        if op in ("$gt", "$gte", "$lt", "$lte"):
            if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
                return False
            if op == "$gt":
                return value > expected
            if op == "$gte":
                return value >= expected
            if op == "$lt":
                return value < expected
            return value <= expected

        if op == "$in":
            if not isinstance(expected, list):
                return False
            return value in expected

        if op == "$nin":
            if not isinstance(expected, list):
                return False
            return value not in expected

        raise ValueError(f"Unsupported where operator: {op}")

    @classmethod
    def _matches_where(cls, metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if where is None:
            return True
        if not isinstance(where, dict):
            raise ValueError("where must be a dict")

        if "$and" in where:
            clauses = where["$and"]
            if not isinstance(clauses, list):
                raise ValueError("$and must be a list")
            return all(cls._matches_where(metadata, clause) for clause in clauses)

        if "$or" in where:
            clauses = where["$or"]
            if not isinstance(clauses, list):
                raise ValueError("$or must be a list")
            return any(cls._matches_where(metadata, clause) for clause in clauses)

        for key, condition in where.items():
            if key.startswith("$"):
                raise ValueError(f"Unsupported where clause key: {key}")
            value = metadata.get(key)
            if isinstance(condition, dict):
                for op, expected in condition.items():
                    if not cls._compare_scalar(value, op, expected):
                        return False
            else:
                if value != condition:
                    return False
        return True

    @classmethod
    def _matches_where_document(cls, document: str, where_document: dict[str, Any] | None) -> bool:
        if where_document is None:
            return True
        if not isinstance(where_document, dict):
            raise ValueError("where_document must be a dict")

        if "$and" in where_document:
            clauses = where_document["$and"]
            if not isinstance(clauses, list):
                raise ValueError("where_document $and must be a list")
            return all(cls._matches_where_document(document, clause) for clause in clauses)

        if "$or" in where_document:
            clauses = where_document["$or"]
            if not isinstance(clauses, list):
                raise ValueError("where_document $or must be a list")
            return any(cls._matches_where_document(document, clause) for clause in clauses)

        contains = where_document.get("$contains")
        if contains is not None:
            return str(contains) in document

        not_contains = where_document.get("$not_contains")
        if not_contains is not None:
            return str(not_contains) not in document

        raise ValueError("where_document supports $contains, $not_contains, $and, $or")
