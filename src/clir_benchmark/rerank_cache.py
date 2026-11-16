"""Compact resumable document-reranking cache, shared by concurrent retrievers.

One SQLite file per scoring configuration and original query avoids millions of
small files. Short transactions use SQLite's rollback journal (not WAL) and file
locking. Completed runs retain immutable Parquet copies of their pair records.
"""

from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict

from .util import digest, json_dumps


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
