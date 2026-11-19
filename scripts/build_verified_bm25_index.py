"""Build a verified-release chunk index or copy an identical prior scratch index."""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from clir_benchmark.artifacts import rows
from clir_benchmark.config import BM25Config, load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.lexical import BM25Index, bm25_path, build_index
from clir_benchmark.util import new_directory, read_json


def check_index(path: Path, bm25: BM25Config, fingerprint: str, chunks: int) -> None:
    with BM25Index(path, bm25) as index:
        if (
            index.manifest["storage"] != "sqlite-inverted-index-v1"
            or index.manifest["corpus_fingerprint"] != fingerprint
            or index.manifest["chunk_count"] != chunks
        ):
            raise ValidationError(f"BM25 manifest disagrees with the prepared corpus: {path}")
        count = index.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count != chunks:
            raise ValidationError(f"BM25 has {count} indexed chunks, expected {chunks}: {path}")
        for table in ("postings", "language_stats", "term_stats"):
            index.db.execute(f"SELECT 1 FROM {table} LIMIT 1")
        if index.db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValidationError(f"SQLite quick_check failed: {path}")


def publish(source: Path, index: Path, bm25: BM25Config, fingerprint: str, chunks: int) -> None:
    with new_directory(index) as staging:
        for name in ("manifest.json", "index.sqlite"):
            shutil.copyfile(source / name, staging / name)
        check_index(staging, bm25, fingerprint, chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", choices=("chemistry", "eurlex", "un"), required=True)
    parser.add_argument("--prior-artifacts", type=Path, required=True)
    parser.add_argument("--build-temp", type=Path, required=True, help="Node-local build directory")
    args = parser.parse_args()
    config = load_config(args.config)
    corpus = config.corpus_dir(args.corpus)
    manifest = read_json(corpus / "manifest.json")
    fingerprint = manifest["corpus_fingerprint"]
    chunks = pq.read_metadata(corpus / "chunks.parquet").num_rows
    if chunks <= 0 or manifest["chunk_count"] != chunks:
        raise ValidationError(f"Prepared chunk count disagrees with corpus manifest: {corpus}")
    index = bm25_path(config, args.corpus)
    previous = args.prior_artifacts / "lexical" / args.corpus / index.name

    if index.exists():
        check_index(index, config.bm25, fingerprint, chunks)
        print(f"Reused {args.corpus}: {chunks} prepared and indexed chunks", file=sys.stderr)
        print(index)
        return
    if previous.is_dir():
        try:
            publish(previous, index, config.bm25, fingerprint, chunks)
        except (sqlite3.DatabaseError, ValueError, KeyError, FileNotFoundError) as exc:
            print(f"Discarding incompatible prior index {previous}: {exc}", file=sys.stderr)
        else:
            print(f"Copied and verified identical prior index {previous}", file=sys.stderr)
            print(f"{args.corpus}: {chunks} prepared and indexed chunks", file=sys.stderr)
            print(index)
            return

    print(f"Building {args.corpus} BM25 over {chunks} chunks on {args.build_temp}", file=sys.stderr)
    with TemporaryDirectory(dir=args.build_temp, prefix=f"bm25-{args.corpus}-") as work:
        records = rows(corpus / "chunks.parquet", ["id", "language", "text"])
        fresh = build_index(Path(work) / "index", records, fingerprint, config.bm25)
        publish(fresh, index, config.bm25, fingerprint, chunks)
    print(f"Built {args.corpus}: {chunks} prepared and indexed chunks", file=sys.stderr)
    print(index)


if __name__ == "__main__":
    main()
