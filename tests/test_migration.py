import json
import sqlite3

from chromapro import PersistentClient
from chromapro.migrate import migrate_sqlite_to_chromapro


def test_sqlite_migration_creates_data_and_report(tmp_path):
    source = tmp_path / "legacy.sqlite3"
    target = tmp_path / "new_data"
    report = tmp_path / "report.json"

    conn = sqlite3.connect(source)
    conn.execute(
        "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, metadata TEXT)"
    )
    conn.execute(
        "CREATE TABLE embeddings (id TEXT PRIMARY KEY, collection_id TEXT, embedding TEXT, document TEXT, metadata TEXT)"
    )
    conn.execute(
        "INSERT INTO collections (id, name, metadata) VALUES (?, ?, ?)",
        ("c1", "migrated", json.dumps({"env": "test"})),
    )
    conn.execute(
        "INSERT INTO embeddings (id, collection_id, embedding, document, metadata) VALUES (?, ?, ?, ?, ?)",
        ("r1", "c1", json.dumps([1.0, 2.0]), "hello", json.dumps({"k": "v"})),
    )
    conn.execute(
        "INSERT INTO embeddings (id, collection_id, embedding, document, metadata) VALUES (?, ?, ?, ?, ?)",
        ("r2", "c1", json.dumps([2.0, 1.0]), "world", json.dumps({"k": "w"})),
    )
    conn.commit()
    conn.close()

    summary = migrate_sqlite_to_chromapro(
        source=str(source),
        target_path=str(target),
        report_path=str(report),
        batch_size=1,
    )

    assert summary["records_read"] == 2
    assert summary["records_written"] == 2
    assert report.exists()

    with PersistentClient(path=str(target)) as client:
        col = client.get_collection("migrated")
        got = col.get(ids=["r1", "r2"])
        assert got["ids"] == ["r1", "r2"]
        assert got["documents"] == ["hello", "world"]
