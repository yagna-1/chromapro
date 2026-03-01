"""Production-ready client with proper resource management."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

from chromapro_core import ChromaProStorage

from .collection import Collection

STORAGE_META_FILENAME = "chromapro_meta.json"
STORAGE_FORMAT_VERSION = 1
INDEX_FORMAT_VERSION = 1
MAPPING_FORMAT_VERSION = 1
COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]$")


class PersistentClient:
    """
    Drop-in replacement for chromadb.PersistentClient.

    Key improvements:
    - Explicit close() (not __del__)
    - Context manager support
    - Crash-consistent persistence
    - Multi-process safe
    """

    def __init__(
        self,
        path: str = "./chroma_data",
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(self.path, exist_ok=True)

        self._settings = settings or {}
        self._storage_meta = self._load_or_initialize_storage_meta()
        self._storage = self._open_storage_with_retry()
        self._collections: dict[str, Collection] = {}

    def __enter__(self) -> PersistentClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if hasattr(self, "_collections"):
            for collection in self._collections.values():
                collection.persist()

        if hasattr(self, "_storage"):
            self._storage.close()

    def create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
        get_or_create: bool = False,
    ) -> Collection:
        self._validate_collection_name(name)
        if get_or_create:
            try:
                return self.get_collection(name, embedding_function=embedding_function)
            except ValueError:
                pass

        collection_id = str(uuid.uuid4())

        meta = metadata or {}
        self._storage.put_collection(
            collection_id=collection_id,
            name=name,
            metadata=json.dumps(meta),
        )

        collection = Collection(
            client=self,
            name=name,
            id=collection_id,
            metadata=meta,
            embedding_function=embedding_function,
        )

        self._collections[name] = collection
        return collection

    def get_collection(
        self,
        name: str,
        embedding_function: Any | None = None,
    ) -> Collection:
        self._validate_collection_name(name)
        if name in self._collections:
            return self._collections[name]

        for coll_data in self._storage.list_collections():
            if coll_data.get("name") != name:
                continue

            metadata = self._normalize_metadata(coll_data.get("metadata", {}))

            collection = Collection(
                client=self,
                name=coll_data.get("name", name),
                id=coll_data.get("id", ""),
                metadata=metadata,
                embedding_function=embedding_function,
            )
            self._collections[name] = collection
            return collection

        raise ValueError(f"Collection '{name}' not found")

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> Collection:
        return self.create_collection(
            name=name,
            metadata=metadata,
            embedding_function=embedding_function,
            get_or_create=True,
        )

    def list_collections(self) -> list[Collection]:
        result: list[Collection] = []

        for coll_data in self._storage.list_collections():
            name = coll_data.get("name", "")
            if not name:
                continue

            if name not in self._collections:
                metadata = self._normalize_metadata(coll_data.get("metadata", {}))

                self._collections[name] = Collection(
                    client=self,
                    name=name,
                    id=coll_data.get("id", ""),
                    metadata=metadata,
                )

            result.append(self._collections[name])

        return result

    def delete_collection(self, name: str) -> None:
        self._validate_collection_name(name)
        collection = None

        if name in self._collections:
            collection = self._collections[name]
        else:
            for coll_data in self._storage.list_collections():
                if coll_data.get("name") == name:
                    collection = Collection(
                        client=self,
                        name=name,
                        id=coll_data.get("id", ""),
                        metadata={},
                    )
                    break

        if collection is None:
            return

        self._storage.delete_collection_data(collection.id)

        for ext in [".hnsw", ".mapping", ".deleted", ".lock"]:
            file_path = os.path.join(self.path, f"{collection.id}{ext}")
            if os.path.exists(file_path):
                os.remove(file_path)

        self._collections.pop(name, None)

    def heartbeat(self) -> int:
        self._storage.ping()
        return int(time.time() * 1000)

    def _open_storage_with_retry(self) -> ChromaProStorage:
        timeout_s = float(self._settings.get("open_retry_timeout_s", 30.0))
        interval_s = float(self._settings.get("open_retry_interval_s", 0.1))
        deadline = time.time() + max(0.0, timeout_s)
        last_error: Exception | None = None

        while True:
            try:
                return ChromaProStorage(self.path)
            except OSError as exc:
                msg = str(exc).lower()
                lock_related = "lock" in msg or "resource temporarily unavailable" in msg
                if not lock_related:
                    raise
                last_error = exc
                if time.time() >= deadline:
                    raise OSError(
                        f"Timed out opening RocksDB at {self.path} after {timeout_s:.1f}s; "
                        f"another process may still hold the DB lock"
                    ) from last_error
                time.sleep(max(0.01, interval_s))

    def _load_or_initialize_storage_meta(self) -> dict[str, Any]:
        expected = {
            "version": STORAGE_FORMAT_VERSION,
            "index_version": INDEX_FORMAT_VERSION,
            "mapping_version": MAPPING_FORMAT_VERSION,
        }

        path = self._storage_meta_path()
        if not os.path.exists(path):
            self._atomic_write_json(path, expected)
            return expected

        with open(path, encoding="utf-8") as f:
            current = json.load(f)

        for key, expected_value in expected.items():
            current_value = int(current.get(key, -1))
            if current_value != expected_value:
                raise ValueError(
                    f"Unsupported on-disk format in {path}: {key}={current_value}, expected {expected_value}"
                )

        return current

    def _storage_meta_path(self) -> str:
        return os.path.join(self.path, STORAGE_META_FILENAME)

    @staticmethod
    def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
        directory = os.path.dirname(path) or "."
        tmp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp.{os.getpid()}.{time.time_ns()}")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Best-effort parent directory fsync for durability of rename metadata.
        try:
            dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    @staticmethod
    def _normalize_metadata(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            if not value:
                return {}
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except Exception:
                return {"raw_metadata": value}
        return {"raw_metadata": str(value)}

    @staticmethod
    def _validate_collection_name(name: str) -> None:
        if not isinstance(name, str) or len(name) < 3 or len(name) > 512 or not COLLECTION_NAME_RE.fullmatch(name):
            raise ValueError(
                "Expected a name containing 3-512 characters from [a-zA-Z0-9._-], "
                "starting and ending with a character in [a-zA-Z0-9]."
            )


Client = PersistentClient
