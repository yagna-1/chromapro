import json
import pickle

from chromapro import PersistentClient
from chromapro.migrate import verify


def test_verify_detects_mapping_drift_and_repairs(tmp_path):
    data_path = str(tmp_path / "data")

    with PersistentClient(path=data_path) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            documents=["da", "db"],
            metadatas=[{"k": "a"}, {"k": "b"}],
        )
        mapping_path = tmp_path / "data" / f"{col.id}.mapping"

    with mapping_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["id_map"]["ghost"] = 999
    payload["reverse_map"]["999"] = "ghost"
    payload["counter"] = 1000
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    before = verify(data_path, "docs")
    assert before["ok"] is False
    assert "mapping_ids_without_storage" in before["issues"]

    after = verify(data_path, "docs", repair=True)
    assert after["repair_applied"] is True
    assert after["post_repair_ok"] is True

    with PersistentClient(path=data_path) as client:
        col = client.get_collection("docs")
        assert col.count() == 2
        assert "ghost" not in col._id_map


def test_verify_repair_removes_deleted_payloads_left_in_storage(tmp_path):
    data_path = str(tmp_path / "data")

    with PersistentClient(path=data_path) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(ids=["1", "2"], embeddings=[[1.0, 0.0], [0.0, 1.0]], documents=["d1", "d2"])
        col.delete(["2"])

        # Simulate drift where a deleted payload exists in RocksDB.
        col._client._storage.batch_insert(
            collection_id=col.id,
            ids=["2"],
            embeddings=[pickle.dumps([0.0, 1.0])],
            documents=["d2-restored"],
            metadatas=[json.dumps({"restored": True})],
        )

    before = verify(data_path, "docs")
    assert before["ok"] is False
    assert "deleted_ids_present_in_storage" in before["issues"]

    after = verify(data_path, "docs", repair=True)
    assert after["post_repair_ok"] is True

    with PersistentClient(path=data_path) as client:
        col = client.get_collection("docs")
        got = col.get(ids=["2"], include=["documents"])
        assert got["ids"] == []
