import pytest

from chromapro import PersistentClient


def test_collection_name_validation_matches_chromadb_rules(tmp_path):
    with PersistentClient(path=str(tmp_path / "data")) as client:
        with pytest.raises(ValueError, match="3-512 characters"):
            client.create_collection("x")

        with pytest.raises(ValueError, match="3-512 characters"):
            client.create_collection("-bad")

        ok = client.create_collection("abc_123")
        assert ok.name == "abc_123"
