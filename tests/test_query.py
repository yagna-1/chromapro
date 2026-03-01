from chromapro import PersistentClient


def test_query_returns_nearest_neighbors(tmp_path):
    client = PersistentClient(path=str(tmp_path / "data"))
    col = client.create_collection("vectors")

    col.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        documents=["doc-a", "doc-b", "doc-c"],
        metadatas=[{"tag": "a"}, {"tag": "b"}, {"tag": "c"}],
    )

    result = col.query(query_embeddings=[[1.0, 0.0]], n_results=2)
    assert result["ids"][0][0] == "a"
    assert len(result["ids"][0]) == 2
    assert result["distances"][0][0] <= result["distances"][0][1]

    client.close()
