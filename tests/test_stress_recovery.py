import random

from chromapro import PersistentClient
from chromapro.migrate import verify


def _vec(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random(), rng.random(), rng.random(), rng.random()]


def test_randomized_stress_reopen_and_verify(tmp_path):
    data_path = str(tmp_path / "stress")
    rng = random.Random(1337)
    expected: dict[str, tuple[list[float], str, dict]] = {}

    client = PersistentClient(path=data_path)
    try:
        col = client.create_collection("stress", metadata={"dimension": 4})

        for step in range(250):
            op = rng.choice(["add", "delete", "query", "get"])

            if op == "add":
                id_ = f"id-{rng.randint(0, 79)}"
                emb = _vec(step * 17 + rng.randint(0, 10000))
                doc = f"doc-{step}"
                meta = {"step": step, "bucket": step % 5}
                col.add(ids=[id_], embeddings=[emb], documents=[doc], metadatas=[meta])
                expected[id_] = (emb, doc, meta)

            elif op == "delete":
                if expected:
                    ids = list(expected.keys())
                    rng.shuffle(ids)
                    to_delete = ids[: rng.randint(1, min(3, len(ids)))]
                    col.delete(to_delete)
                    for id_ in to_delete:
                        expected.pop(id_, None)

            elif op == "query":
                if expected:
                    probe = next(iter(expected.values()))[0]
                    out = col.query(query_embeddings=[probe], n_results=5)
                    assert set(out["ids"][0]).issubset(set(expected.keys()))

            elif op == "get":
                if expected:
                    ids = list(expected.keys())
                    rng.shuffle(ids)
                    subset = ids[: min(5, len(ids))]
                    got = col.get(ids=subset, include=["documents", "metadatas"])
                    assert set(got["ids"]).issubset(set(expected.keys()))

            if step % 50 == 0 and step > 0:
                client.close()
                snapshot = verify(data_path, "stress")
                assert snapshot["ok"] is True
                client = PersistentClient(path=data_path)
                col = client.get_collection("stress")

        assert col.count() == len(expected)
    finally:
        client.close()

    final = verify(data_path, "stress")
    assert final["ok"] is True
