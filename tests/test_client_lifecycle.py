from chromapro import PersistentClient


def test_context_manager_roundtrip(tmp_path):
    db_path = tmp_path / "data"

    with PersistentClient(path=str(db_path)) as client:
        c = client.create_collection("docs")
        c.add(ids=["1"], embeddings=[[1.0, 2.0]], documents=["hello"], metadatas=[{"a": 1}])
        assert c.count() == 1

    with PersistentClient(path=str(db_path)) as client2:
        c2 = client2.get_collection("docs")
        result = c2.get(ids=["1"])
        assert result["ids"] == ["1"]
        assert result["documents"] == ["hello"]
        q = c2.query(query_embeddings=[[1.0, 2.0]], n_results=1)
        assert q["ids"][0] == ["1"]


def test_list_collections_returns_typed_metadata(tmp_path):
    db_path = tmp_path / "data"

    with PersistentClient(path=str(db_path)) as client:
        client.create_collection("docs", metadata={"team": "infra", "ver": 1})

    with PersistentClient(path=str(db_path)) as client2:
        cols = client2.list_collections()
        assert len(cols) == 1
        assert isinstance(cols[0].metadata, dict)
        assert cols[0].metadata["team"] == "infra"
        assert cols[0].metadata["ver"] == 1
