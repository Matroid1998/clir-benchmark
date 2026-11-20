"""Measure text and prior UN chunk counts in the edition order used by the workers."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MIRRORS = {
    "chemistry": "patents-verified-20261118/release-with-abstracts",
    "eurlex": "verified-legal-20261118/eurlex",
    "un": "verified-legal-20261118/un",
}


def records(path, columns):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=columns):
        yield from batch.to_pylist()


def measure(corpus, directory, early, later, minutes, prior_un):
    fields = ["text", "available_languages"]
    previous, chunks = {}, Counter()
    if corpus == "un":
        fields += ["source_unit_ids", "_id", "parent_document_id"]
        for row in records(
            prior_un / "sources.parquet", ["id", "source_id", "parent_id", "language", "text"]
        ):
            previous[row["id"]] = (
                row["source_id"],
                row["parent_id"],
                row["language"],
                hashlib.sha256(row["text"].encode()).digest(),
            )
        for batch in pq.ParquetFile(prior_un / "chunks.parquet").iter_batches(
            columns=["source_unit_id"]
        ):
            chunks.update(batch.column(0).to_pylist())

    lengths, chunk_counts = [], []
    for table in ["corpus", "annexes"] if corpus == "eurlex" else ["corpus"]:
        for shard in sorted((directory / "data" / table).glob("*.parquet")):
            for row in records(shard, fields):
                for lang in row["available_languages"]:
                    text = row["text"][lang]
                    lengths.append(len(text))
                    if corpus == "un":
                        unit = row["source_unit_ids"][lang]
                        identity = (
                            row["_id"],
                            row["parent_document_id"],
                            lang,
                            hashlib.sha256(text.encode()).digest(),
                        )
                        if previous.pop(unit) != identity or unit not in chunks:
                            raise ValueError(f"UN chunk inputs changed for {unit}")
                        chunk_counts.append(chunks.pop(unit))
    if previous or chunks or len(lengths) != later["total_editions"]:
        raise ValueError(f"Incomplete source coverage for {corpus}")
    first, second = (s["estimated_editions_written"] for s in (early, later))

    def interval(start, end, duration=None):
        values = lengths[start:end]
        result = {
            "editions": len(values),
            "unicode_characters": sum(values),
            "mean_characters": round(sum(values) / len(values), 1),
            "median_characters": median(values),
        }
        if corpus == "un":
            result["chunks"] = sum(chunk_counts[start:end])
        if duration is not None:
            result["elapsed_minutes"] = round(duration, 2)
            result["editions_per_hour"] = round(len(values) * 60 / duration)
        return result

    return {
        "all": interval(0, len(lengths)),
        "start_to_first_sample": interval(0, first, early["minutes_writing_editions"]),
        "between_samples": interval(first, second, minutes),
        "after_second_sample": interval(second, len(lengths)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exports", type=Path, default=ROOT.parent / "clir-dataset-creation/data/hf_export"
    )
    args = parser.parse_args()
    a, b = (
        json.loads((HERE / f"progress_20261119_{stamp}.json").read_text())
        for stamp in ["1532", "1953"]
    )
    elapsed = (
        datetime.fromisoformat(b["sampled_at"]) - datetime.fromisoformat(a["sampled_at"])
    ).total_seconds() / 60
    (prior_un,) = (ROOT / "artifacts/fir-20261109-all-questions/artifacts/corpus/un").glob(
        "*/manifest.json"
    )
    result = {
        "sampled_at": [a["sampled_at"], b["sampled_at"]],
        "interval_boundaries": "Approximate editions written at each sample, rounded to 1,000 by the job progress counter.",
        "un_chunk_counts": "Previous completed run, matched on every edition ID, canonical and parent IDs, language, and SHA-256 of text.",
        "corpora": {
            c: measure(
                c, args.exports / p, a["corpora"][c], b["corpora"][c], elapsed, prior_un.parent
            )
            for c, p in MIRRORS.items()
        },
    }
    print(json.dumps(result, indent=2))
