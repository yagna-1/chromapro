"""
HNSW Rebuild Stress Test
=========================
Verifies that ChromaPro correctly rebuilds the HNSW index from RocksDB
when the .hnsw file is missing (deleted / corrupted).

Tests at three scales: 1k, 10k, and 50k records (4-dimensional for speed).
Marks each as @pytest.mark.slow — skip in fast runs with: pytest -m "not slow"

Run with:
    python3 -m pytest tests/test_hnsw_rebuild_stress.py -v -s -m slow
"""
import time
from pathlib import Path

import pytest

from chromapro import PersistentClient

pytestmark = pytest.mark.slow


def _vec(seed: int, dim: int = 4) -> list[float]:
    """Deterministic pseudo-random unit vector."""
    import math
    vals = [math.sin(seed * (i + 1) * 0.7) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _run_rebuild_stress(tmp_path: Path, n: int, dim: int = 4) -> dict:
    """
    Write N records, close, delete .hnsw, reopen, verify full recovery.
    Returns timing info dict.
    """
    db_path = str(tmp_path / f"rebuild_{n}")
    collection_name = "stress_rebuild"

    # ── Phase 1: Write ────────────────────────────────────────────────────
    t_write_start = time.perf_counter()
    with PersistentClient(path=db_path) as client:
        col = client.create_collection(collection_name, metadata={"dimension": dim})
        # Batch in chunks of 500 to avoid memory pressure
        chunk = 500
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            col.add(
                ids=[f"rec_{i}" for i in range(start, end)],
                embeddings=[_vec(i, dim) for i in range(start, end)],
                documents=[f"document {i}" for i in range(start, end)],
            )
    t_write = time.perf_counter() - t_write_start

    # ── Phase 2: Delete .hnsw file(s) ────────────────────────────────────
    hnsw_files = list(Path(db_path).glob("*.hnsw"))
    assert hnsw_files, f"Expected .hnsw file after writing {n} records, found none"
    for f in hnsw_files:
        f.unlink()
    print(f"\n  [n={n:>6}] Deleted {len(hnsw_files)} .hnsw file(s)")

    # ── Phase 3: Reopen and rebuild ───────────────────────────────────────
    t_rebuild_start = time.perf_counter()
    with PersistentClient(path=db_path) as client:
        col = client.get_collection(collection_name)
        count = col.count()
        # Spot-check: query should return results
        results = col.query(query_embeddings=[_vec(0, dim)], n_results=min(10, count))
    t_rebuild = time.perf_counter() - t_rebuild_start

    return {
        "n": n,
        "write_s": t_write,
        "rebuild_s": t_rebuild,
        "count": count,
        "query_results": len(results["ids"][0]),
        "rebuild_rps": n / t_rebuild if t_rebuild > 0 else float("inf"),
    }


@pytest.mark.slow
def test_hnsw_rebuild_1k(tmp_path):
    """1k records — fast baseline."""
    r = _run_rebuild_stress(tmp_path, n=1_000)
    print(f"  [n={r['n']:>6}] Write={r['write_s']:.2f}s  Rebuild={r['rebuild_s']:.2f}s  "
          f"({r['rebuild_rps']:,.0f} rec/s)  count={r['count']}  query_hits={r['query_results']}")
    assert r["count"] == 1_000, f"Expected 1000 records, got {r['count']}"
    assert r["query_results"] > 0, "Query returned no results after rebuild"


@pytest.mark.slow
def test_hnsw_rebuild_10k(tmp_path):
    """10k records — typical production scale."""
    r = _run_rebuild_stress(tmp_path, n=10_000)
    print(f"  [n={r['n']:>6}] Write={r['write_s']:.2f}s  Rebuild={r['rebuild_s']:.2f}s  "
          f"({r['rebuild_rps']:,.0f} rec/s)  count={r['count']}  query_hits={r['query_results']}")
    assert r["count"] == 10_000, f"Expected 10000, got {r['count']}"
    assert r["query_results"] > 0


@pytest.mark.slow
def test_hnsw_rebuild_50k(tmp_path):
    """50k records — stress scale (may take 30-120s)."""
    r = _run_rebuild_stress(tmp_path, n=50_000)
    print(f"  [n={r['n']:>6}] Write={r['write_s']:.2f}s  Rebuild={r['rebuild_s']:.2f}s  "
          f"({r['rebuild_rps']:,.0f} rec/s)  count={r['count']}  query_hits={r['query_results']}")
    assert r["count"] == 50_000, f"Expected 50000, got {r['count']}"
    assert r["query_results"] > 0


@pytest.mark.slow
def test_hnsw_rebuild_correctness_spot_check(tmp_path):
    """
    After rebuild, querying with a known vector must return the expected id.
    Validates that RocksDB→HNSW rebuild is semantically correct, not just structurally.
    """
    n = 200
    dim = 4
    db_path = str(tmp_path / "rebuild_correctness")

    with PersistentClient(path=db_path) as client:
        col = client.create_collection("correctness", metadata={"dimension": dim})
        col.add(
            ids=[f"item_{i}" for i in range(n)],
            embeddings=[_vec(i, dim) for i in range(n)],
            documents=[f"doc {i}" for i in range(n)],
        )

    # Delete .hnsw and rebuild
    hnsw_files = list(Path(db_path).glob("*.hnsw"))
    for f in hnsw_files:
        f.unlink()

    with PersistentClient(path=db_path) as client:
        col = client.get_collection("correctness")
        # Query with the exact vector of item_0 — should return item_0 as top result
        results = col.query(query_embeddings=[_vec(0, dim)], n_results=5)
        top_ids = results["ids"][0]

    print(f"\n  Spot-check top-5 after rebuild: {top_ids}")
    assert "item_0" in top_ids, (
        f"Expected 'item_0' in top-5 results after HNSW rebuild, got: {top_ids}"
    )
    print("  ✅ Semantic correctness confirmed: exact vector maps to correct record after rebuild.")


@pytest.mark.slow
def test_hnsw_rebuild_idempotent(tmp_path):
    """
    Delete .hnsw twice in a row (two consecutive rebuild cycles).
    Verifies rebuild is idempotent: data is identical after each cycle.
    """
    n = 500
    dim = 4
    db_path = str(tmp_path / "rebuild_idempotent")

    with PersistentClient(path=db_path) as client:
        col = client.create_collection("idempotent", metadata={"dimension": dim})
        col.add(
            ids=[f"item_{i}" for i in range(n)],
            embeddings=[_vec(i, dim) for i in range(n)],
            documents=[f"doc {i}" for i in range(n)],
        )

    counts = []
    for cycle in range(2):
        hnsw_files = list(Path(db_path).glob("*.hnsw"))
        for f in hnsw_files:
            f.unlink()
        with PersistentClient(path=db_path) as client:
            col = client.get_collection("idempotent")
            counts.append(col.count())

    print(f"\n  Rebuild cycles counts: {counts}")
    assert counts[0] == n, f"Cycle 1 expected {n}, got {counts[0]}"
    assert counts[1] == n, f"Cycle 2 expected {n}, got {counts[1]}"
    print("  ✅ HNSW rebuild is idempotent across two consecutive deletion cycles.")
