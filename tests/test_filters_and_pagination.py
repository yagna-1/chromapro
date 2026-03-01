from chromapro import PersistentClient


def test_query_supports_where_and_where_document(tmp_path):
    with PersistentClient(path=str(tmp_path / "data")) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
            documents=["alpha launch", "beta update", "alpha deep dive"],
            metadatas=[
                {"tag": "news", "score": 5},
                {"tag": "blog", "score": 2},
                {"tag": "news", "score": 9},
            ],
        )

        out = col.query(
            query_embeddings=[[1.0, 0.0]],
            n_results=3,
            where={"tag": "news"},
            where_document={"$contains": "alpha"},
        )
        assert out["ids"][0] == ["a", "c"]

        out2 = col.query(
            query_embeddings=[[1.0, 0.0]],
            n_results=3,
            where={"score": {"$gte": 8}},
        )
        assert out2["ids"][0] == ["c"]


def test_get_supports_pagination_and_filters(tmp_path):
    with PersistentClient(path=str(tmp_path / "data")) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(
            ids=["a", "b", "c", "d"],
            embeddings=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]],
            documents=["x needle", "y needle", "z needle", "q nope"],
            metadatas=[
                {"group": "g1", "rank": 1},
                {"group": "g1", "rank": 2},
                {"group": "g1", "rank": 3},
                {"group": "g2", "rank": 4},
            ],
        )

        out = col.get(ids=None, where={"group": "g1"}, where_document={"$contains": "needle"}, offset=1, limit=1)
        assert out["ids"] == ["b"]
        assert out["documents"] == ["y needle"]
        assert out["metadatas"] == [{"group": "g1", "rank": 2}]

        peek = col.peek(limit=2, include=["documents"])
        assert len(peek["ids"]) == 2
        assert peek["documents"] == ["x needle", "y needle"]
        assert peek["embeddings"] is None


def test_invalid_filter_operators_raise(tmp_path):
    with PersistentClient(path=str(tmp_path / "data")) as client:
        col = client.create_collection("docs", metadata={"dimension": 2})
        col.add(ids=["a"], embeddings=[[1.0, 0.0]], documents=["alpha"], metadatas=[{"tag": "news"}])

        try:
            col.get(ids=["a"], where={"tag": {"$unknown": 1}})
            raised = False
        except ValueError:
            raised = True
        assert raised

        try:
            col.query(
                query_embeddings=[[1.0, 0.0]],
                where_document={"$bad": "x"},
            )
            raised = False
        except ValueError:
            raised = True
        assert raised
