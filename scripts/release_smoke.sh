#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v maturin >/dev/null 2>&1; then
  python3 -m pip install --user --break-system-packages maturin
  export PATH="$HOME/.local/bin:$PATH"
fi

maturin build --release -m chromapro_core/Cargo.toml

wheel_path="$(ls -1t chromapro_core/target/wheels/chromapro-*.whl | head -n1)"
if [[ -z "${wheel_path:-}" ]]; then
  echo "release smoke failed: no wheel found under chromapro_core/target/wheels"
  exit 1
fi

smoke_site="$(mktemp -d)"
python3 -m pip install --target "$smoke_site" "$wheel_path"

PYTHONPATH="$smoke_site" python3 - <<'PY'
import tempfile

from chromapro import PersistentClient

tmpdir = tempfile.mkdtemp(prefix="chromapro_release_smoke_")

with PersistentClient(path=tmpdir) as client:
    col = client.get_or_create_collection("smoke", metadata={"dimension": 2})
    col.add(
        ids=["s1", "s2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["doc-1", "doc-2"],
        metadatas=[{"idx": 1}, {"idx": 2}],
    )
    out = col.query(query_embeddings=[[1.0, 0.0]], n_results=1)
    assert out["ids"][0] == ["s1"], out
    assert out["documents"][0] == ["doc-1"], out

print("release smoke CRUD/query passed")
PY

tmp_cli="$(mktemp -d)"
PYTHONPATH="$smoke_site" python3 -m chromapro.cli health "$tmp_cli"
rm -rf "$tmp_cli"

rm -rf "$smoke_site"

echo "release smoke passed"
