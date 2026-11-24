"""Resumable query/passage score caches for concurrent reranking workers."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import OrderedDict

import pyarrow as pa
import pyarrow.parquet as pq

from .util import digest, json_dumps


class QueryPairCache:
    """Share immutable score fragments on scratch. Hold at most one query lock."""

    def __init__(self, root):
        self.root = root
        self.path = None
        self.lock = None
        self.existing = {}
        self.pending = {}

    def activate(self, identity):
        import fcntl

        path = self.root / digest(identity)
        if path == self.path:
            return
        self.close()
        path.mkdir(parents=True, exist_ok=True)
        self.lock = (path / "scores.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX)
            self.path = path
            for fragment in sorted(path.glob("*.parquet")):
                for row in pq.ParquetFile(fragment).read().to_pylist():
                    self.existing[row["score_key"]] = row["result_json"]
        except BaseException:
            self.close()
            raise

    def get(self, key, identity):
        if key in self.pending:
            return self.pending[key]
        value = self.existing.get(key)
        return json.loads(value) if value is not None else None

    def put_batch(self, records):
        self.pending.update(records)
        return dict(records)

    def close(self):
        if self.lock is None:
            return
        try:
            if self.pending:
                name = uuid.uuid4().hex
                temp = self.path / (name + ".tmp")
                table = pa.table(
                    {
                        "score_key": list(self.pending),
                        "result_json": [json_dumps(value) for value in self.pending.values()],
                    }
                )
                pq.write_table(table, temp, compression="zstd")
                temp.replace(self.path / (name + ".parquet"))
        finally:
            self.lock.close()
            self.lock = self.path = None
            self.pending.clear()
            self.existing.clear()


class SQLitePairCache:
    def __init__(self, root):
        self.root = root
        self.connections = OrderedDict()

    def _connection(self, identity):
        query_identity = {
            key: value
            for key, value in identity.items()
            if key not in {"document", "document_language"}
        }
        key = digest(query_identity)
        if key in self.connections:
            self.connections.move_to_end(key)
            return self.connections[key]
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.root / f"{key}.sqlite3", timeout=120)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, result TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS concurrent_attempts (key TEXT, result TEXT NOT NULL)"
        )
        connection.commit()
        self.connections[key] = connection
        if len(self.connections) > 16:
            _, old = self.connections.popitem(last=False)
            old.close()
        return connection

    def get(self, key, identity):
        row = (
            self._connection(identity)
            .execute("SELECT result FROM pairs WHERE key = ?", (key,))
            .fetchone()
        )
        return json.loads(row[0]) if row else None

    def put_batch(self, records):
        """Publish a batch; retain the first measurement and audit any duplicate work."""
        grouped = {}
        for key, result in records:
            connection = self._connection(result["identity"])
            grouped.setdefault(connection, []).append((key, result))
        stored = {}
        for connection, members in grouped.items():
            with connection:
                for key, result in members:
                    payload = json_dumps(result)
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO pairs VALUES (?, ?)", (key, payload)
                    ).rowcount
                    if not inserted:
                        connection.execute(
                            "INSERT INTO concurrent_attempts VALUES (?, ?)", (key, payload)
                        )
                    stored[key] = json.loads(
                        connection.execute(
                            "SELECT result FROM pairs WHERE key = ?", (key,)
                        ).fetchone()[0]
                    )
        return stored

    def close(self):
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()
