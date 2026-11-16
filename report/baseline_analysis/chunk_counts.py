#!/usr/bin/env python3
"""Question 10: count prepared chunks and chunks in the configured search languages."""

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    root = Path(__file__).resolve().parent
    plans = json.loads((root / "question_04_corpus_sizes/corpus_sizes.json").read_text())
    result = []
    for plan in plans:
        path = Path(plan["source_manifest"])
        manifest = json.loads(path.read_text())
        if manifest["corpus_fingerprint"] != plan["corpus_fingerprint"]:
            raise ValueError("Corpus fingerprint differs from the selected experiment")
        chunks_path = path.parent / "chunks.parquet"
        prepared = pq.ParquetFile(chunks_path).metadata.num_rows
        if prepared != manifest["chunk_count"]:
            raise ValueError("Chunk table size differs from the corpus manifest")
        searched = pq.read_table(
            chunks_path,
            columns=["language"],
            filters=[("language", "in", plan["configured_search_languages"])],
        ).num_rows
        result.append(
            {
                "corpus": plan["corpus"],
                "prepared_chunks": prepared,
                "searched_chunks": searched,
                "excluded_language_chunks": prepared - searched,
            }
        )
    output = root / "question_10_chunk_counts"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "chunk_counts.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (output / "chunk_counts.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
