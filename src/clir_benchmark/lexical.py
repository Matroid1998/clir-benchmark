"""Disk-backed BM25 over the unchanged, complete chunk corpus."""

from __future__ import annotations

import math
import sqlite3
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from .artifacts import rows
from .config import BM25Config, Config
from .errors import ValidationError
from .util import digest, new_directory, read_json, versions, write_json


def lexical_tokens(text: str) -> list[str]:
    """NFC + casefold, letter/number/mark words, individual Han and kana.

    Punctuation separates tokens. No stop list, stemming, translation or model
    tokenizer is used. These transformations never modify stored text or qrels.
    """
    text = unicodedata.normalize("NFC", unicodedata.normalize("NFC", text).casefold())
    result, word = [], []
    for char in text:
        code = ord(char)
        cjk = (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0x20000 <= code <= 0x323AF
            or 0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
        )
        category = unicodedata.category(char)[0]
        if cjk and category == "L":
            if word:
                result.append("".join(word))
                word.clear()
            result.append(char)
        elif category in {"L", "N"} or (category == "M" and word):
            word.append(char)
        else:
            if word:
                result.append("".join(word))
                word.clear()
    if word:
        result.append("".join(word))
    return result


def analyzer_identity(config: BM25Config) -> dict:
    return {
        "name": config.analyzer,
        "unicode_version": unicodedata.unidata_version,
        "normalization": "NFC-casefold-NFC",
        "stopwords": False,
        "stemming": False,
        "translation": False,
    }


def bm25_path(config: Config, corpus: str) -> Path:
    manifest_path = config.corpus_dir(corpus) / "manifest.json"
    if not manifest_path.is_file():
        raise ValidationError(
            f"Missing corpus artifacts: {manifest_path.parent}. Run prepare-corpus first."
        )
    fingerprint = read_json(manifest_path)["corpus_fingerprint"]
    identity = {
        "schema": 1,
        "corpus_fingerprint": fingerprint,
        "analyzer": analyzer_identity(config.bm25),
        "storage": "sqlite-inverted-index-v1",
    }
    return config.output_dir / "lexical" / corpus / digest(identity)


def build_index(
    destination: Path, records: Iterable[dict], fingerprint: str, config: BM25Config
) -> Path:
    if destination.exists():
        manifest = read_json(destination / "manifest.json")
        if manifest["corpus_fingerprint"] != fingerprint or manifest[
            "analyzer"
        ] != analyzer_identity(config):
            raise ValidationError(f"BM25 cache identity mismatch: {destination}")
        with BM25Index(destination, config):
            pass
        return destination
    started = time.perf_counter()
    with new_directory(destination) as temporary:
        db = sqlite3.connect(temporary / "index.sqlite")
        try:
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA cache_size=-32768")
            db.executescript("""
                CREATE TABLE documents (
                    doc INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE,
                    language TEXT NOT NULL, length INTEGER NOT NULL
                );
                CREATE TABLE postings (
                    term TEXT NOT NULL, doc INTEGER NOT NULL, tf INTEGER NOT NULL,
                    PRIMARY KEY (term, doc)
                ) WITHOUT ROWID;
            """)
            count = 0
            for count, row in enumerate(records, 1):
                if (
                    not all(isinstance(row.get(key), str) for key in ("id", "language", "text"))
                    or not row["id"]
                    or not row["language"]
                ):
                    raise ValidationError(
                        "BM25 input needs a nonempty chunk ID/language and string text"
                    )
                terms = Counter(lexical_tokens(row["text"]))
                db.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?)",
                    (count, row["id"], row["language"], sum(terms.values())),
                )
                db.executemany(
                    "INSERT INTO postings VALUES (?, ?, ?)",
                    ((term, count, tf) for term, tf in sorted(terms.items())),
                )
                if count % 512 == 0:
                    db.commit()
            db.executescript("""
                CREATE INDEX documents_language ON documents(language);
                CREATE TABLE language_stats AS
                    SELECT language, COUNT(*) AS documents, SUM(length) AS tokens
                    FROM documents GROUP BY language;
                CREATE TABLE term_stats (
                    term TEXT NOT NULL, language TEXT NOT NULL, df INTEGER NOT NULL,
                    PRIMARY KEY (term, language)
                ) WITHOUT ROWID;
                INSERT INTO term_stats
                    SELECT p.term, d.language, COUNT(*)
                    FROM postings p JOIN documents d ON d.doc=p.doc
                    GROUP BY p.term, d.language;
            """)
            languages = {
                language: {"documents": documents, "tokens": tokens}
                for language, documents, tokens in db.execute(
                    "SELECT * FROM language_stats ORDER BY language"
                )
            }
            db.commit()
        except sqlite3.Error as exc:
            raise ValidationError(
                f"Could not build BM25 index (check duplicate chunk IDs): {exc}"
            ) from exc
        finally:
            db.close()
        write_json(
            temporary / "manifest.json",
            {
                "schema": 1,
                "corpus_fingerprint": fingerprint,
                "chunk_count": count,
                "analyzer": analyzer_identity(config),
                "languages": languages,
                "storage": "sqlite-inverted-index-v1",
                "sqlite_version": sqlite3.sqlite_version,
                "seconds": time.perf_counter() - started,
                "versions": versions(),
            },
        )
    return destination


def prepare_bm25(config: Config, corpus: str) -> Path:
    source = config.corpus_dir(corpus)
    destination = bm25_path(config, corpus)
    return build_index(
        destination,
        rows(source / "chunks.parquet", ["id", "language", "text"]),
        read_json(source / "manifest.json")["corpus_fingerprint"],
        config.bm25,
    )


class BM25Index:
    """Score posting matches with language-pool statistics and disk-backed SQL.

    SQLite spills grouping/sorting to disk. Only a chunk's terms during indexing
    and one query's terms/top-k during retrieval are held in Python memory.
    """

    def __init__(self, path: Path, config: BM25Config | None = None):
        config = config or BM25Config()
        if not (path / "manifest.json").is_file() or not (path / "index.sqlite").is_file():
            raise ValidationError(f"BM25 index is not prepared: {path}. Run index-bm25 first.")
        self.manifest = read_json(path / "manifest.json")
        if self.manifest["analyzer"] != analyzer_identity(config):
            raise ValidationError(f"BM25 analyzer mismatch: {path}. Rebuild the index.")
        self.config = config
        self.db = sqlite3.connect((path / "index.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA cache_size=-32768")
        self.db.execute(
            "CREATE TEMP TABLE query_terms (term TEXT PRIMARY KEY, idf REAL NOT NULL) WITHOUT ROWID"
        )

    def __enter__(self) -> BM25Index:
        return self

    def __exit__(self, *args) -> None:
        self.db.close()

    def search(
        self, queries: Iterable[str], target_languages: tuple[str, ...], *, k: int = 100
    ) -> Iterator[list[tuple[str, float]]]:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValidationError("BM25 retrieval k must be a positive integer")
        languages = tuple(sorted(set(target_languages) & self.manifest["languages"].keys()))
        stats = [self.manifest["languages"][language] for language in languages]
        n = sum(item["documents"] for item in stats)
        total_tokens = sum(item["tokens"] for item in stats)
        marks = ",".join("?" for _ in languages)
        for query in queries:
            if not isinstance(query, str):
                raise ValidationError("BM25 queries must be strings")
            self.db.execute("DELETE FROM query_terms")
            if not n or not total_tokens:
                yield []
                continue
            weights = []
            for term in sorted(set(lexical_tokens(query))):
                df = self.db.execute(
                    f"SELECT COALESCE(SUM(df), 0) FROM term_stats WHERE term=? AND language IN ({marks})",
                    (term, *languages),
                ).fetchone()[0]
                if df:
                    weights.append((term, math.log1p((n - df + 0.5) / (df + 0.5))))
            if not weights:
                yield []
                continue
            self.db.executemany("INSERT INTO query_terms VALUES (?, ?)", weights)
            # CROSS JOIN fixes the term-first traversal. Values (including text)
            # are parameters; the only interpolated SQL is placeholder counts.
            ranking = self.db.execute(
                f"""
                SELECT d.id, SUM(q.idf * p.tf * (? + 1.0) /
                    (p.tf + ? * (1.0 - ? + ? * d.length / ?))) AS score
                FROM query_terms q CROSS JOIN postings p ON p.term=q.term
                JOIN documents d ON d.doc=p.doc
                WHERE d.language IN ({marks})
                GROUP BY d.doc
                ORDER BY score DESC, d.id COLLATE BINARY ASC LIMIT ?
            """,
                (
                    self.config.k1,
                    self.config.k1,
                    self.config.b,
                    self.config.b,
                    total_tokens / n,
                    *languages,
                    k,
                ),
            ).fetchall()
            yield [(chunk_id, float(score)) for chunk_id, score in ranking]


def bm25_scoring(config: BM25Config) -> dict:
    return {
        **asdict(config),
        "algorithm": "bm25",
        "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
        "tf": "tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))",
        "statistics": "all_chunks_in_target_language_pool_including_empty",
        "query_terms": "unique",
        "zero_score_documents": "omitted",
        "precision": "float64",
        "sqlite_version": sqlite3.sqlite_version,
    }
