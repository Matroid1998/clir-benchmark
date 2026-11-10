from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .errors import ValidationError
from .types import Chunk, Query, SourceUnit
from .util import json_dumps, object_field

SOURCE_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "id",
            "corpus",
            "source_id",
            "parent_id",
            "language",
            "text",
            "unit_type",
            "title",
            "metadata_json",
        )
    ]
)
CHUNK_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "id",
            "source_unit_id",
            "source_id",
            "parent_id",
            "corpus",
            "language",
            "text",
            "chunking_hash",
        )
    ]
    + [
        (name, pa.int64())
        for name in ("start", "end", "token_count", "index", "overlap_chars", "overlap_tokens")
    ]
)
QUERY_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "id",
            "corpus",
            "question",
            "question_language",
            "context_language",
            "answer",
            "evidence",
            "source_unit_id",
            "generator_model_id",
            "persona",
            "task_id",
            "evidence_all_languages_json",
            "evidence_alignment_status_json",
            "evidence_alignment_eligibility",
            "evidence_alignment_contract",
            "metadata_json",
        )
    ]
    + [
        ("involved_source_unit_ids", pa.list_(pa.string())),
        ("is_best", pa.bool_()),
        ("candidate_rank", pa.int64()),
        ("total_score", pa.float64()),
        ("unresolved_sources", pa.bool_()),
    ]
)
QREL_SCHEMA = pa.schema(
    [(name, pa.string()) for name in ("setting_id", "query_id", "chunk_id", "required_source_id")]
    + [("relevance", pa.int64())]
)
DOCUMENT_QREL_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in ("setting_id", "query_id", "document_id", "required_source_id")
    ]
    + [("relevance", pa.int64())]
)
MATCH_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "setting_id",
            "query_id",
            "chunk_id",
            "required_source_id",
            "source_unit_id",
            "field",
            "span",
            "annotation_json",
        )
    ]
    + [("start", pa.int64()), ("end", pa.int64())]
)
ELIGIBILITY_SCHEMA = pa.schema(
    [(name, pa.string()) for name in ("setting_id", "query_id", "status", "reason", "detail_json")]
    + [("required_source_ids", pa.list_(pa.string()))]
)
RUN_SCHEMA = pa.schema(
    [(name, pa.string()) for name in ("setting_id", "query_id", "chunk_id")]
    + [("rank", pa.int64()), ("score", pa.float64())]
)
DOCUMENT_RUN_SCHEMA = pa.schema(
    [(name, pa.string()) for name in ("setting_id", "query_id", "document_id", "winning_chunk_id")]
    + [("rank", pa.int64()), ("score", pa.float64())]
)
EMBED_ROW_SCHEMA = pa.schema(
    [(name, pa.string()) for name in ("id", "language", "source_unit_id")]
    + [
        ("source_start", pa.int64()),
        ("input_tokens", pa.int64()),
        ("encoded_tokens", pa.int64()),
        ("truncated", pa.bool_()),
        ("visible_ranges_json", pa.string()),
    ]
)


class ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 512):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema = schema
        self.batch_size = batch_size
        self.buffer: list[dict] = []
        self.count = 0

    def append(self, row: dict) -> None:
        self.buffer.append(row)
        self.count += 1
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def __enter__(self) -> ParquetSink:
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            self.flush()
        finally:
            self.writer.close()


def rows(path: Path, columns: list[str] | None = None) -> Iterator[dict]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
        yield from batch.to_pylist()


def source_record(unit: SourceUnit) -> dict:
    row = asdict(unit)
    row["metadata_json"] = json_dumps(row.pop("metadata"))
    return row


def query_record(query: Query) -> dict:
    row = asdict(query)
    for name in ("metadata", "evidence_all_languages", "evidence_alignment_status"):
        row[name + "_json"] = json_dumps(row.pop(name))
    return row


def read_queries(path: Path) -> list[Query]:
    result = []
    for row in rows(path):
        for name in ("metadata", "evidence_all_languages", "evidence_alignment_status"):
            row[name] = object_field(row.pop(name + "_json", None), name)
        row["involved_source_unit_ids"] = tuple(row["involved_source_unit_ids"])
        result.append(Query(**row))
    return result


class CorpusStore:
    """Disk-backed lookup; source text need not fit in memory during labeling."""

    def __init__(self, path: Path, *, create: bool = False):
        if create:
            self.db = sqlite3.connect(path)
            self.db.executescript("""
                CREATE TABLE sources (id TEXT PRIMARY KEY, corpus TEXT NOT NULL,
                    source_id TEXT NOT NULL, language TEXT NOT NULL, record TEXT NOT NULL);
                CREATE INDEX source_versions ON sources(corpus,source_id,language);
                CREATE TABLE chunks (id TEXT PRIMARY KEY, source_unit_id TEXT NOT NULL,
                    start INTEGER NOT NULL, end INTEGER NOT NULL, record TEXT NOT NULL);
                CREATE INDEX source_chunks ON chunks(source_unit_id,start);
                CREATE TABLE direct_pairs (left_id TEXT NOT NULL,right_id TEXT NOT NULL,
                    PRIMARY KEY(left_id,right_id));
            """)
        else:
            if not path.is_file():
                raise ValidationError(f"Corpus is not prepared: {path}. Run prepare-corpus first.")
            self.db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)

    def __enter__(self) -> CorpusStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.db.close()

    def add_source(self, unit: SourceUnit) -> bool:
        previous = self.get_source(unit.id)
        if previous:
            if (previous.corpus, previous.source_id, previous.language, previous.text) != (
                unit.corpus,
                unit.source_id,
                unit.language,
                unit.text,
            ):
                raise ValidationError(f"Conflicting duplicate source unit: {unit.id}")
            return False
        self.db.execute(
            "INSERT INTO sources VALUES(?,?,?,?,?)",
            (unit.id, unit.corpus, unit.source_id, unit.language, json_dumps(asdict(unit))),
        )
        return True

    def get_source(self, source_unit_id: str) -> SourceUnit | None:
        found = self.db.execute(
            "SELECT record FROM sources WHERE id=?", (source_unit_id,)
        ).fetchone()
        return SourceUnit(**object_field(found[0], "source")) if found else None

    def variants(self, unit: SourceUnit) -> list[str]:
        if unit.corpus == "chemistry":
            return [unit.id] + [
                row[0]
                for row in self.db.execute(
                    "SELECT right_id FROM direct_pairs WHERE left_id=? ORDER BY right_id",
                    (unit.id,),
                )
            ]
        return [
            row[0]
            for row in self.db.execute(
                "SELECT id FROM sources WHERE corpus=? AND source_id=? ORDER BY id",
                (unit.corpus, unit.source_id),
            )
        ]

    def add_pairs(self, pairs: Iterable[tuple[str, str]]) -> None:
        for left, right in pairs:
            if not self.get_source(left) or not self.get_source(right):
                raise ValidationError(
                    f"Translation pair references missing source: {left}, {right}"
                )
            self.db.executemany(
                "INSERT OR IGNORE INTO direct_pairs VALUES(?,?)", [(left, right), (right, left)]
            )

    def add_chunk(self, chunk: Chunk) -> None:
        record = asdict(chunk)
        record.pop("text")
        self.db.execute(
            "INSERT INTO chunks VALUES(?,?,?,?,?)",
            (chunk.id, chunk.source_unit_id, chunk.start, chunk.end, json_dumps(record)),
        )

    def chunks(self, unit: SourceUnit) -> list[Chunk]:
        return [
            Chunk(**object_field(row[0], "chunk"), text=unit.text[row[1] : row[2]])
            for row in self.db.execute(
                "SELECT record,start,end FROM chunks WHERE source_unit_id=? ORDER BY start,id",
                (unit.id,),
            )
        ]

    def has_chunks(self, source_unit_id: str) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM chunks WHERE source_unit_id=? LIMIT 1", (source_unit_id,)
            ).fetchone()
            is not None
        )

    def chunk(self, chunk_id: str) -> dict | None:
        found = self.db.execute("SELECT record FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return object_field(found[0], "chunk") if found else None

    def languages(self) -> list[str]:
        return [
            row[0]
            for row in self.db.execute("SELECT DISTINCT language FROM sources ORDER BY language")
        ]
