from chromapro import PersistentClient


def test_add_get_delete(tmp_path):
    client = PersistentClient(path=str(tmp_path / "data"))
    col = client.create_collection("items")

    col.add(
        ids=["a", "b"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        documents=["doc-a", "doc-b"],
        metadatas=[{"k": "v1"}, {"k": "v2"}],
    )
    assert col.count() == 2

    got = col.get(ids=["a", "b"])
    assert got["ids"] == ["a", "b"]

    col.delete(ids=["a"])
    assert col.count() == 1

    client.close()
