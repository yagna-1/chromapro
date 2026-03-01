import json

import pytest

from chromapro import PersistentClient


def test_storage_meta_file_created_with_expected_versions(tmp_path):
    data_path = tmp_path / "data"

    with PersistentClient(path=str(data_path)):
        pass

    meta_path = data_path / "chromapro_meta.json"
    assert meta_path.exists()

    with meta_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["version"] == 1
    assert payload["index_version"] == 1
    assert payload["mapping_version"] == 1


def test_storage_meta_version_mismatch_raises_on_open(tmp_path):
    data_path = tmp_path / "data"

    with PersistentClient(path=str(data_path)):
        pass

    meta_path = data_path / "chromapro_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["mapping_version"] = 999
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(ValueError, match="Unsupported on-disk format"):
        PersistentClient(path=str(data_path))


def test_mapping_version_mismatch_raises_on_collection_load(tmp_path):
    data_path = tmp_path / "data"

    with PersistentClient(path=str(data_path)) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(ids=["1"], embeddings=[[1.0, 0.0]])
        mapping_path = data_path / f"{col.id}.mapping"

    with mapping_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["mapping_version"] = 999
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    with PersistentClient(path=str(data_path)) as client:
        with pytest.raises(ValueError, match="Unsupported mapping format version"):
            client.get_collection("docs")


def test_atomic_saves_leave_no_temporary_files(tmp_path):
    data_path = tmp_path / "data"

    with PersistentClient(path=str(data_path)) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(ids=["1", "2"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
        assert not list(data_path.glob(f"{col.id}.hnsw.tmp.*"))
        assert not list(data_path.glob(f"{col.id}.mapping.tmp.*"))
