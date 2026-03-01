import multiprocessing as mp

import pytest

from chromapro import PersistentClient


def test_deleted_ids_are_filtered_from_query(tmp_path):
    with PersistentClient(path=str(tmp_path / "data")) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(
            ids=["1", "2", "3"],
            embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            documents=["d1", "d2", "d3"],
        )

        col.delete(["2"])

        out = col.query(query_embeddings=[[1.0, 0.0]], n_results=3)
        assert "2" not in out["ids"][0]


def test_rebuild_index_clears_tombstones(tmp_path):
    with PersistentClient(path=str(tmp_path / "data")) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(
            ids=["1", "2", "3", "4"],
            embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.0, 0.9]],
        )

        col.delete(["2", "4"])
        before = col.count()
        assert before == 2
        assert len(col._deleted_ids) == 2

        col.rebuild_index()

        after = col.count()
        assert after == before
        assert len(col._deleted_ids) == 0


def _mp_writer(data_path: str, collection_name: str, prefix: str) -> None:
    with PersistentClient(path=data_path) as client:
        col = client.get_collection(collection_name)
        ids = [f"{prefix}_{i}" for i in range(100)]
        vectors = [[float(i), 1.0] for i in range(100)]
        docs = [f"doc-{prefix}-{i}" for i in range(100)]
        col.add(ids=ids, embeddings=vectors, documents=docs)


def test_multi_process_writes_do_not_clobber(tmp_path):
    data_path = str(tmp_path / "shared")

    with PersistentClient(path=data_path) as client:
        client.create_collection("shared", metadata={"dimension": 2})

    ctx = mp.get_context("spawn")
    p1 = ctx.Process(target=_mp_writer, args=(data_path, "shared", "p1"))
    p2 = ctx.Process(target=_mp_writer, args=(data_path, "shared", "p2"))
    p1.start()
    p2.start()
    p1.join(timeout=60)
    p2.join(timeout=60)

    assert p1.exitcode == 0
    assert p2.exitcode == 0

    with PersistentClient(path=data_path) as client:
        col = client.get_collection("shared")
        assert col.count() == 200


def test_crash_window_hnsw_saved_but_rocksdb_missing(tmp_path):
    data_path = str(tmp_path / "crash")
    with PersistentClient(path=data_path) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})

        class FailingStorage:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def batch_insert(self, *args, **kwargs):
                raise RuntimeError("simulated RocksDB write failure")

        real_storage = col._client._storage
        col._client._storage = FailingStorage(real_storage)
        with pytest.raises(RuntimeError, match="simulated RocksDB write failure"):
            col.add(
                ids=["x1"],
                embeddings=[[1.0, 0.0]],
                documents=["crash"],
                metadatas=[{}],
            )
        col._client._storage = real_storage

        out = col.query(query_embeddings=[[1.0, 0.0]], n_results=1)
        assert out["ids"][0] == []
        assert col.count() == 0
        assert "x1" not in col._id_map


def test_readd_clears_tombstone_persistently(tmp_path):
    data_path = str(tmp_path / "readd")

    with PersistentClient(path=data_path) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(ids=["1"], embeddings=[[1.0, 0.0]], documents=["v1"])
        col.delete(["1"])
        assert col.count() == 0
        col.add(ids=["1"], embeddings=[[0.0, 1.0]], documents=["v2"])
        assert col.count() == 1

    with PersistentClient(path=data_path) as client:
        col = client.get_collection("docs")
        assert col.count() == 1
        out = col.query(query_embeddings=[[0.0, 1.0]], n_results=1)
        assert out["ids"][0] == ["1"]
        got = col.get(ids=["1"], include=["documents"])
        assert got["documents"] == ["v2"]
