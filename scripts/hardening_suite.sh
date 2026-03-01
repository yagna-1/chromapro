#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m pytest -q \
  tests/test_correctness.py \
  tests/test_verify_hardening.py \
  tests/test_corruption_recovery.py \
  tests/test_stress_recovery.py \
  tests/test_hnsw_rebuild_stress.py \
  tests/test_chromadb_vs_chromapro.py
