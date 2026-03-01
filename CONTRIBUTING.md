# Contributing to ChromaPro

Thank you for considering a contribution! ChromaPro is a durability-first persistence layer — quality and correctness are the primary constraints.

---

## Philosophy

Before opening a PR, please understand the core tradeoffs ChromaPro is intentionally making:

- **Durability > throughput** — synchronous `fsync` on every write is by design
- **Correctness > convenience** — no silent failure modes
- **Simplicity > features** — resist scope creep

---

## Setup

```bash
# 1. Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install Python dependencies
pip install maturin chroma-hnswlib numpy pytest ruff chromadb

# 3. Build the Rust extension
maturin develop -m chromapro_core/Cargo.toml

# 4. Verify everything works
python3 -m pytest tests/ -v
```

---

## Before Submitting a PR

- [ ] All tests pass: `python3 -m pytest tests/ -v`
- [ ] Ruff linting is clean: `ruff check chromapro tests`
- [ ] Rust type check passes: `cargo check --manifest-path chromapro_core/Cargo.toml`
- [ ] New features have tests in `tests/`
- [ ] README updated if public API changed

---

## Running Specific Test Suites

```bash
# Core suite (fast, ~90s)
python3 -m pytest tests/ -v --ignore=tests/test_benchmarks.py --ignore=tests/test_hnsw_rebuild_stress.py --ignore=tests/test_chromadb_vs_chromapro.py

# Adversarial ChromaDB comparison (requires chromadb installed)
python3 -m pytest tests/test_chromadb_vs_chromapro.py -v -s

# Benchmarks (prints timing table, no assertions on speed)
python3 -m pytest tests/test_benchmarks.py -v -s -m benchmark

# HNSW rebuild stress tests (~2 minutes for 50k scale)
python3 -m pytest tests/test_hnsw_rebuild_stress.py -v -s -m slow

# Full hardening suite
bash scripts/hardening_suite.sh
```

---

## Project Structure

```
chromapro/               Python package (public API)
├── client.py            PersistentClient — lifecycle + collection management
├── collection.py        Collection — HNSW + tombstones + locking
├── migrate.py           CLI commands + SQLite migration
└── cli.py               CLI entrypoint

chromapro_core/          Rust extension (PyO3 + RocksDB)
└── src/
    ├── lib.rs           PyO3 bindings
    └── db.rs            ChromaProStorage — 3 column families

tests/                   Test suite (61 tests)
scripts/                 Development utilities
docker/                  Dockerfile + docker-compose
```

---

## What We Won't Accept

- Features that compromise crash consistency
- Changes that remove the `fsync` guarantee
- Breaking changes to the on-disk format without version bump
- Dependencies that don't work on Linux + macOS

---

## Reporting Issues

Please include:
1. OS + Python version
2. How ChromaPro was installed (`maturin develop` vs wheel)
3. Minimal reproduction script
4. Whether the issue is a crash, data loss, or wrong result

Data loss bugs are the highest priority.
