from chromapro import PersistentClient


def test_corrupt_mapping_auto_recovers_on_load(tmp_path):
    data_path = tmp_path / "data"

    with PersistentClient(path=str(data_path)) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(
            ids=["1", "2"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            documents=["d1", "d2"],
        )
        mapping_path = data_path / f"{col.id}.mapping"

    mapping_path.write_text("{bad-json", encoding="utf-8")

    with PersistentClient(path=str(data_path)) as client:
        col = client.get_collection("docs")
        assert col.count() == 2
        out = col.query(query_embeddings=[[1.0, 0.0]], n_results=2)
        assert set(out["ids"][0]) == {"1", "2"}
