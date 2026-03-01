#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip maturin pytest ruff chroma-hnswlib numpy
maturin develop -m chromapro_core/Cargo.toml
ruff check chromapro tests
python3 -m pytest
