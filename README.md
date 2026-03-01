# ChromaPro

[![CI](https://github.com/yagna-1/chromapro/actions/workflows/ci.yml/badge.svg)](https://github.com/yagna-1/chromapro/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Production-safe persistence layer for local vector collections.**  
Chroma-compatible API backed by RocksDB + hnswlib, built for crash safety, multi-process correctness, and operational durability.

---

## Why ChromaPro?

| Feature | ChromaDB | ChromaPro |
|---|---|---|
| Storage engine | SQLite (WAL) — single-file SPOF | **RocksDB** — multi-file SSTs + CRC32c checksums |
| Crash/SIGKILL durability | Async WAL flush — data loss possible | **Synchronous fsync** per write — guaranteed durable |
| Cross-process safety | No file locks on HNSW binary | **`fcntl.LOCK_EX`** per collection — serializes writers |
| Context manager | ❌ Not supported | ✅ `with PersistentClient(...) as client:` |
| Version metadata | ❌ None | ✅ `chromapro_meta.json` — detects format mismatches |
| HNSW recovery | Fails on cold restart after deletion | ✅ Auto-rebuilds from RocksDB ground truth |
| Corruption detection | Single SQLite page corruption = silent data loss | ✅ RocksDB CRC32c detects corrupted blocks |
| Explicit `rebuild_index()` | ❌ Not available | ✅ Compacts HNSW + clears tombstones from RocksDB |

**Trade-off:** ChromaPro is ~1.5–3× slower on raw write throughput vs ChromaDB due to synchronous `fsync`. If you never experience crashes or multi-process access, ChromaDB is fine. If you need hard durability guarantees, use ChromaPro.

---

## Install

### Local development (recommended)

```bash
pip install maturin chroma-hnswlib numpy
maturin develop -m chromapro_core/Cargo.toml
```

### From a built wheel

```bash
pip install chromapro-0.1.0-*.whl
```

### Build the wheel yourself

```bash
maturin build --release -m chromapro_core/Cargo.toml
pip install target/wheels/chromapro-*.whl
```

---

## Quick Start

```python
import chromapro

# Context manager guarantees flush + close
with chromapro.PersistentClient(path="./my_db") as client:
    col = client.create_collection("docs", metadata={"dimension": 384})

    col.add(
        ids=["doc1", "doc2", "doc3"],
        embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
        documents=["Hello world", "Vector search", "RocksDB durability"],
        metadatas=[{"source": "wiki"}, {"source": "arXiv"}, {"source": "blog"}],
    )

    results = col.query(
        query_embeddings=[[0.15] * 384],
        n_results=2,
        where={"source": {"$eq": "wiki"}},
    )
    print(results["ids"][0])  # ['doc1']
```

---

## API Reference

### `PersistentClient(path: str)`

Opens (or creates) a ChromaPro database at the given path.

```python
client = chromapro.PersistentClient(path="./db")
# or as a context manager:
with chromapro.PersistentClient(path="./db") as client:
    ...
```

| Method | Description |
|---|---|
| `create_collection(name, metadata={"dimension": N})` | Create a new collection. `dimension` is required. |
| `get_collection(name)` | Open an existing collection. |
| `get_or_create_collection(name, metadata=...)` | Get if exists, otherwise create. |
| `delete_collection(name)` | Delete collection + all data. |
| `list_collections()` | Returns list of `Collection` objects for all collections in this database. |
| `close()` | Flush and release all locks. Called automatically by context manager. |

### `Collection`

| Method | Signature | Description |
|---|---|---|
| `add` | `(ids, embeddings, documents=None, metadatas=None)` | Insert records. ids must be unique. |
| `get` | `(ids=None, where=None, where_document=None, include=["documents","metadatas","embeddings"], limit=None, offset=None)` | Retrieve records with optional filtering. |
| `query` | `(query_embeddings, n_results=10, where=None, where_document=None, include=["documents","metadatas","distances"])` | ANN nearest-neighbour search. |
| `delete` | `(ids)` | Delete records by id. Changes are persisted immediately. |
| `update` | `(ids, embeddings=None, documents=None, metadatas=None)` | Update existing records. Raises `ValueError` if any id does not exist. |
| `upsert` | `(ids, embeddings, documents=None, metadatas=None)` | Insert or update — never raises on existing ids. |
| `count` | `()` | Return number of active (non-deleted) records. |
| `peek` | `(limit=10)` | Return first N records. |
| `rebuild_index` | `()` | Rebuild HNSW from RocksDB, compacting tombstones. Use after heavy deletions. |

---

## Filter Operators

### `where` — metadata filters

```python
# Equality
col.get(where={"source": {"$eq": "wiki"}})

# Comparison
col.get(where={"score": {"$gte": 0.8}})

# Set membership
col.get(where={"category": {"$in": ["A", "B"]}})

# Logical
col.get(where={"$and": [{"score": {"$gt": 0.5}}, {"source": {"$ne": "spam"}}]})
col.get(where={"$or": [{"tag": {"$eq": "news"}}, {"tag": {"$eq": "blog"}}]})
```

| Operator | Meaning |
|---|---|
| `$eq` | Equal |
| `$ne` | Not equal |
| `$gt` / `$gte` | Greater than / greater than or equal |
| `$lt` / `$lte` | Less than / less than or equal |
| `$in` | In list |
| `$nin` | Not in list |
| `$and` | Logical AND (list of conditions) |
| `$or` | Logical OR (list of conditions) |

### `where_document` — text filters

```python
col.query(
    query_embeddings=[my_vec],
    where_document={"$contains": "neural network"},
)
col.query(
    query_embeddings=[my_vec],
    where_document={"$not_contains": "deprecated"},
)
```

---

## Pagination

```python
# Get records 100–199
page = col.get(limit=100, offset=100, include=["ids", "documents"])

# Peek at first 5
col.peek(limit=5)
```

---

## CLI Reference

```bash
# Migrate from ChromaDB SQLite → ChromaPro
chromapro migrate ./chroma.sqlite3 ./chromapro_data

# Migrate with a report and custom batch size
chromapro migrate ./chroma.sqlite3 ./chromapro_data --report ./report.json --batch-size 1000

# Verify data integrity (count + spot query)
chromapro verify ./chromapro_data my_collection

# Auto-repair corrupt HNSW (rebuilds from RocksDB)
chromapro verify ./chromapro_data my_collection --repair

# Rebuild HNSW index (after heavy deletions)
chromapro rebuild-index ./chromapro_data my_collection

# Overall database health check
chromapro health ./chromapro_data

# Collection statistics
chromapro stats ./chromapro_data my_collection
```

---

## Architecture

```
chromapro/
├── PersistentClient            Python API + lifecycle management
│   └── Collection              Per-collection index + lock
│       ├── hnswlib.Index       ANN search (derived — rebuilt from RocksDB)
│       ├── <uuid>.hnsw         Persisted HNSW binary (auto-rebuilt if missing)
│       ├── <uuid>.deleted      Tombstone set (JSON, survives restart)
│       └── <uuid>.lock         fcntl file lock (cross-process exclusion)
└── chromapro_core (Rust/PyO3)
    └── RocksDB                 Ground truth for all embedding/document/metadata
        ├── CF: embeddings      float32 arrays
        ├── CF: documents       text strings
        └── CF: metadata        JSON blobs
```

**Durability contract:**
1. Every `add()` / `delete()` / `update()` calls RocksDB `WriteBatch` (atomic) then `flush()` + `set_use_fsync(true)`.
2. HNSW is a derived index. If it is missing or corrupt on open, ChromaPro rebuilds it from RocksDB automatically.
3. `close()` (or context manager `__exit__`) saves the HNSW index atomically via `os.replace()`.

---

## Benchmark Results

> Measured on Ubuntu 22.04, Python 3.12, Intel i7 (single-core), 16 GB RAM.  
> ChromaPro v0.1.0 vs ChromaDB v0.6.x.  
> Run yourself: `python3 -m pytest tests/test_benchmarks.py -v -s -m benchmark`

### Write Latency — Single Record (384-dim, N=500)

Each call to `add()` persists one record then `fsync`s to disk.

| | Latency / record | Notes |
|---|---|---|
| **ChromaPro** | ~34 ms | fsync on every write — crash-safe guarantee |
| **ChromaDB** | ~14 ms | async WAL — may lose data on SIGKILL |
| Difference | **~2.4× slower** | Cost of the synchronous durability guarantee |

### Bulk Write — 10,000 Records in One Call (384-dim)

| | Total time | Throughput | Notes |
|---|---|---|---|
| **ChromaPro** | ~11.8 s | ~850 rec/s | Single fsync after batch — far better than per-record |
| **ChromaDB** | ~2.9 s | ~3,400 rec/s | SQLite WAL batching |
| Difference | **~4× slower** | | Use bulk `add()` to minimise the gap |

> **Tip:** Always prefer one large `add()` call over many small ones. ChromaPro flushes once per `add()`, so a 10k-record batch is ~40× more efficient than 10k single-record calls.

### ANN Query Throughput — 10,000 Records, N=200 Queries

| | Latency / query | Notes |
|---|---|---|
| **ChromaPro** | ~47 ms | Includes lock-acquire + disk refresh per query |
| **ChromaDB** | ~2.4 ms | In-memory HNSW, no disk touch |
| Difference | **~20× slower** | Lock + `_refresh_from_disk_locked()` overhead dominates |

> The underlying `hnswlib` ANN performance is identical. The overhead is ChromaPro's per-query cross-process safety check. For read-heavy workloads, consider batching queries.

### Cold Open — Fresh Client + `get_collection()` (10,000 records pre-loaded)

| | Time | Notes |
|---|---|---|
| **ChromaPro** | ~400 ms | Loads HNSW from `.hnsw` file on disk |
| **ChromaDB** | ~7 ms | SQLite + in-memory index reconstruction |
| Difference | **~57× slower** | HNSW binary load is O(N); typically a one-time startup cost |

### HNSW Rebuild After Index Deletion

When the `.hnsw` binary is deleted (accidental or crash), ChromaPro rebuilds it from RocksDB automatically on the next open:

| Scale | Rebuild time | Throughput |
|---|---|---|
| 1,000 records | ~0.2 s | ~5,100 rec/s |
| 10,000 records | ~2–4 s | ~3,000–5,000 rec/s |
| 50,000 records | ~15–40 s | ~1,500–3,000 rec/s |

ChromaDB has **no rebuild path** — if the HNSW binary is deleted and the process restarts without the binary, queries fail permanently with no recovery mechanism.

---

## Persistence Guarantees & Adversarial Testing

ChromaPro ships a comprehensive adversarial test suite (`tests/test_chromadb_vs_chromapro.py`, 39 tests) that directly compares behaviour under failure conditions. These are the key findings:

### 1. HNSW Index Recovery

| Scenario | ChromaDB | ChromaPro |
|---|---|---|
| `.hnsw` deleted, same process | Query survives (in-memory cache) | ✅ Rebuilds from RocksDB |
| `.hnsw` deleted, fresh process restart | **Queries fail permanently** | ✅ Auto-rebuilds from RocksDB |
| No integrity check between SQLite and HNSW | ❌ Confirmed — no cross-validation | ✅ RocksDB is the ground truth |

```
[ChromaDB] Critical: there is NO integrity check between SQLite and HNSW files
[ChromaPro] ✅ HNSW deletion auto-recovered from RocksDB. count=30/30
```

### 2. Cross-Process File Locking

ChromaDB's Rust backend creates **no lock files** on its HNSW binary data. Two separate OS processes can open and write the same binary simultaneously, corrupting it silently.

| | ChromaDB | ChromaPro |
|---|---|---|
| Lock file exists | ❌ None | ✅ `<collection-uuid>.lock` |
| External `flock(LOCK_EX)` on HNSW data | **Succeeds** (Chrome holds no lock) | **Blocks** — lock is held exclusively |
| Concurrent cross-process writers | Silent binary corruption possible | ✅ Serialized, 100/100 records correct |

```
[ChromaDB] Lock files created: []   ← confirmed: no cross-process locking
[ChromaDB] ❌ CONFIRMED: No cross-process locking → concurrent writers risk binary corruption.
[ChromaPro] Second writer BLOCKED (could not get lock): True
[ChromaPro] ✅ Cross-process exclusive lock confirmed. 100/100 records correct.
```

### 3. Explicit Close Guarantee

ChromoDB's Rust backend currently survives GC cycles, but this is an **implementation detail** — there is no contractual guarantee. ChromaPro makes it a contract:

| | ChromaDB | ChromaPro |
|---|---|---|
| `with client:` context manager | ❌ Not supported | ✅ Fully supported |
| Durability guaranteed without explicit `close()` | ❌ Depends on GC timing | ✅ `close()` = explicit fsync |

### 4. Corruption Detection

| Scenario | ChromaDB | ChromaPro |
|---|---|---|
| Architecture | Single `chroma.sqlite3` — **SPOF** | Multi-file SSTs — **distributed** |
| SST/block corruption | N/A | ✅ CRC32c checksum detects it |
| Behaviour on detection | Silent wrong data possible | ✅ Error raised — never silent |
| SQLite header corruption | Fatal on next cold open | N/A (no SQLite) |

```
[ChromaPro] ✅ RocksDB CRC32c checksum CAUGHT the corruption — no wrong data returned silently.
```

### 5. SIGKILL Durability

ChromaPro uses `set_use_fsync(true)` in RocksDB and calls `flush()` after every `WriteBatch`. Even if the process is `SIGKILL`ed immediately after `add()` returns, the data is on disk. ChromaDB's async WAL means data written between WAL flushes may be silently lost on kill.

---

## Maintenance

```bash
# Run the full test suite
python3 -m pytest tests/ -v

# Run only the adversarial ChromaDB comparison suite
python3 -m pytest tests/test_chromadb_vs_chromapro.py -v -s

# Run benchmarks (prints a side-by-side table, no assertions on speed)
python3 -m pytest tests/test_benchmarks.py -v -s -m benchmark

# Run HNSW rebuild stress tests (may take ~2 minutes for 50k scale)
python3 -m pytest tests/test_hnsw_rebuild_stress.py -v -s -m slow

# Run the hardening + corruption recovery suite
bash scripts/hardening_suite.sh

# Build and validate a release wheel
bash scripts/release_smoke.sh
```

---

## Contributing

1. `maturin develop -m chromapro_core/Cargo.toml` to build the Rust extension
2. `python3 -m pytest tests/ -v` to run all tests
3. `ruff check chromapro tests` for linting
4. `cargo check --manifest-path chromapro_core/Cargo.toml` for Rust type-check
