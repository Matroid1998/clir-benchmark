"""Summarize the highest possible chunk Recall@10 from the verified judgments."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pyarrow.parquet as pq


def summarize(directory):
    settings = json.loads((directory / "settings.json").read_text())
    positives = defaultdict(set)
    for batch in pq.ParquetFile(directory / "qrels.parquet").iter_batches(
        columns=["setting_id", "query_id", "chunk_id", "relevance"]
    ):
        for row in batch.to_pylist():
            if row["relevance"] > 0:
                positives[row["setting_id"], row["query_id"]].add(row["chunk_id"])
    counts = defaultdict(list)
    for (setting_id, _), chunks in positives.items():
        setting = settings[setting_id]
        counts[setting["kind"]].append(len(chunks))
    return {
        regime: {
            "query_setting_pairs": len(values),
            "more_than_ten_relevant_chunks": sum(n > 10 for n in values),
            "max_relevant_chunks": max(values),
            "mean_highest_possible_recall_at_10": mean(min(10, n) / n for n in values),
        }
        for regime, values in counts.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", required=True, type=Path)
    args = parser.parse_args()
    data = {}
    for corpus in ("chemistry", "eurlex", "un"):
        receipt = json.loads((args.receipts / f"check-{corpus}.json").read_text())
        data[corpus] = summarize(Path(receipt["relevance_directory"]))
    print(json.dumps(data, indent=2))
