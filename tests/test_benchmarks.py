"""
Benchmark suite: ChromaPro vs ChromaDB
=======================================
Measures write latency, bulk write, query throughput, and cold-open time.
No assertions on timing — prints a summary table for documentation.

Run with:
    python3 -m pytest tests/test_benchmarks.py -v -s -m benchmark
"""
import time

import numpy as np
import pytest

from chromapro import PersistentClient

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

DIMENSION = 384
pytestmark = pytest.mark.benchmark


def rand_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(DIMENSION).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _fmt(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"


def _print_row(label: str, chromapro_t: float, chromadb_t: float | None) -> None:
    chromadb_str = _fmt(chromadb_t) if chromadb_t is not None else "N/A (not installed)"
    ratio = f"{chromadb_t / chromapro_t:.1f}×faster" if chromadb_t else ""
    indicator = "🟢" if chromadb_t and chromadb_t >= chromapro_t else "🔴"
    print(f"  {indicator} {label:<40} ChromaPro={_fmt(chromapro_t):<14}  ChromaDB={chromadb_str:<14}  {ratio}")


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed — skipping comparative rows")
class TestBenchmarkWriteLatency:
    """Single-record add() latency: N individual write calls."""

    N = 500

    def test_single_record_write_latency(self, tmp_path):
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Single-record add() × {self.N}  (dim={DIMENSION})")
        print(f"{'='*70}")

        # ── ChromaPro ──────────────────────────────────────────────────────
        cp_path = str(tmp_path / "chromapro")
        with PersistentClient(path=cp_path) as client:
            col = client.create_collection("bench", metadata={"dimension": DIMENSION})
            # warm up
            col.add(ids=["warmup"], embeddings=[rand_vec(0)], documents=["w"])

            t0 = time.perf_counter()
            for i in range(self.N):
                col.add(ids=[f"rec_{i}"], embeddings=[rand_vec(i + 1)], documents=[f"doc {i}"])
            chromapro_t = (time.perf_counter() - t0) / self.N

        # ── ChromaDB ─────────────────────────────────────────────────────
        cdb_path = str(tmp_path / "chromadb")
        c = chromadb.PersistentClient(path=cdb_path)
        col2 = c.create_collection("bench")
        col2.add(ids=["warmup"], embeddings=[rand_vec(0)])

        t0 = time.perf_counter()
        for i in range(self.N):
            col2.add(ids=[f"rec_{i}"], embeddings=[rand_vec(i + 1)])
        chromadb_t = (time.perf_counter() - t0) / self.N

        _print_row(f"Add latency / record (N={self.N})", chromapro_t, chromadb_t)
        print("\n  ChromaPro note: each add() is fsync'd — durability vs. latency tradeoff.")


class TestBenchmarkBulkWrite:
    """Bulk add(): one call with N records."""

    N = 10_000

    def test_bulk_add(self, tmp_path):
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Bulk add() — {self.N} records in one call (dim={DIMENSION})")
        print(f"{'='*70}")

        ids = [f"bulk_{i}" for i in range(self.N)]
        embeddings = [rand_vec(i) for i in range(self.N)]
        documents = [f"document number {i}" for i in range(self.N)]

        # ── ChromaPro ──────────────────────────────────────────────────────
        cp_path = str(tmp_path / "chromapro_bulk")
        with PersistentClient(path=cp_path) as client:
            col = client.create_collection("bulk", metadata={"dimension": DIMENSION})
            t0 = time.perf_counter()
            col.add(ids=ids, embeddings=embeddings, documents=documents)
            chromapro_t = time.perf_counter() - t0

        chromadb_t = None
        if CHROMADB_AVAILABLE:
            import chromadb as cdb
            CHUNK = 5000  # ChromaDB Rust backend max batch size
            cdb_path = str(tmp_path / "chromadb_bulk")
            c = cdb.PersistentClient(path=cdb_path)
            col2 = c.create_collection("bulk")
            t0 = time.perf_counter()
            for start in range(0, self.N, CHUNK):
                end = min(start + CHUNK, self.N)
                col2.add(ids=ids[start:end], embeddings=embeddings[start:end])
            chromadb_t = time.perf_counter() - t0

        _print_row(f"Bulk add ({self.N} records)", chromapro_t, chromadb_t)
        rps_cp = self.N / chromapro_t
        print(f"  ChromaPro throughput: {rps_cp:,.0f} records/s")
        if chromadb_t:
            rps_cdb = self.N / chromadb_t
            print(f"  ChromaDB throughput:  {rps_cdb:,.0f} records/s")


class TestBenchmarkQueryThroughput:
    """ANN query() throughput: N queries on a 10k-record collection."""

    N_RECORDS = 10_000
    N_QUERIES = 200

    def test_query_throughput(self, tmp_path):
        print(f"\n{'='*70}")
        print(f"BENCHMARK: query() × {self.N_QUERIES} on {self.N_RECORDS} records (dim={DIMENSION})")
        print(f"{'='*70}")

        ids = [f"q_{i}" for i in range(self.N_RECORDS)]
        embeddings = [rand_vec(i) for i in range(self.N_RECORDS)]
        probes = [rand_vec(self.N_RECORDS + i) for i in range(self.N_QUERIES)]

        # ── ChromaPro ──────────────────────────────────────────────────────
        cp_path = str(tmp_path / "chromapro_query")
        with PersistentClient(path=cp_path) as client:
            col = client.create_collection("query_bench", metadata={"dimension": DIMENSION})
            col.add(ids=ids, embeddings=embeddings, documents=ids)
            # warm up
            col.query(query_embeddings=[probes[0]], n_results=10)
            t0 = time.perf_counter()
            for probe in probes:
                col.query(query_embeddings=[probe], n_results=10)
            chromapro_t = (time.perf_counter() - t0) / self.N_QUERIES

        chromadb_t = None
        if CHROMADB_AVAILABLE:
            import chromadb as cdb
            CHUNK = 5000  # ChromaDB Rust backend max batch size
            cdb_path = str(tmp_path / "chromadb_query")
            c = cdb.PersistentClient(path=cdb_path)
            col2 = c.create_collection("query_bench")
            for start in range(0, self.N_RECORDS, CHUNK):
                end = min(start + CHUNK, self.N_RECORDS)
                col2.add(ids=ids[start:end], embeddings=embeddings[start:end])
            col2.query(query_embeddings=[probes[0]], n_results=10)
            t0 = time.perf_counter()
            for probe in probes:
                col2.query(query_embeddings=[probe], n_results=10)
            chromadb_t = (time.perf_counter() - t0) / self.N_QUERIES

        _print_row(f"Query latency / query (N={self.N_QUERIES})", chromapro_t, chromadb_t)


class TestBenchmarkColdOpen:
    """Cold open: fresh PersistentClient + get_collection() with pre-loaded N records."""

    N_RECORDS = 10_000

    def test_cold_open_time(self, tmp_path):
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Cold open with {self.N_RECORDS} pre-loaded records (dim={DIMENSION})")
        print(f"{'='*70}")

        ids = [f"open_{i}" for i in range(self.N_RECORDS)]
        embeddings = [rand_vec(i) for i in range(self.N_RECORDS)]

        # ── ChromaPro — write, close, reopen ──────────────────────────────
        cp_path = str(tmp_path / "chromapro_cold")
        with PersistentClient(path=cp_path) as client:
            col = client.create_collection("cold", metadata={"dimension": DIMENSION})
            col.add(ids=ids, embeddings=embeddings, documents=ids)

        t0 = time.perf_counter()
        with PersistentClient(path=cp_path) as client:
            col = client.get_collection("cold")
            _ = col.count()
        chromapro_t = time.perf_counter() - t0

        chromadb_t = None
        if CHROMADB_AVAILABLE:
            import chromadb as cdb
            CHUNK = 5000  # ChromaDB Rust backend max batch size
            cdb_path = str(tmp_path / "chromadb_cold")
            c = cdb.PersistentClient(path=cdb_path)
            col2 = c.create_collection("cold")
            for start in range(0, self.N_RECORDS, CHUNK):
                end = min(start + CHUNK, self.N_RECORDS)
                col2.add(ids=ids[start:end], embeddings=embeddings[start:end])
            del c

            t0 = time.perf_counter()
            c2 = cdb.PersistentClient(path=cdb_path)
            col3 = c2.get_collection("cold")
            _ = col3.count()
            chromadb_t = time.perf_counter() - t0

        _print_row(f"Cold open + get_collection ({self.N_RECORDS} records)", chromapro_t, chromadb_t)
        print(f"\n{'='*70}")
        print("END BENCHMARK SUITE")
        print(f"{'='*70}\n")
