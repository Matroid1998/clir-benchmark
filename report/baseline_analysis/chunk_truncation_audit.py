"""Summarize document-input truncation from the earlier embedding manifests."""

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_ARTIFACTS = (
    Path(__file__).resolve().parents[2] / "artifacts/fir-20261109-all-questions/artifacts"
)


def audit(root):
    by_fingerprint = {
        json.loads(path.read_text())["corpus_fingerprint"]: path.parent.parent.name
        for path in (root / "corpus").glob("*/*/manifest.json")
    }
    summaries = {}
    for path in sorted((root / "embeddings/document").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        corpus = by_fingerprint.get(manifest["input_fingerprint"])
        if corpus is None:
            continue
        model = manifest["model"]
        truncation = manifest["truncation"]["all"]
        key = (corpus, model["alias"])
        stats = (model["max_length"], truncation["inputs"], truncation["truncated"])
        if key in summaries and summaries[key] != stats:
            raise ValueError(f"Conflicting truncation summaries for {key}")
        summaries[key] = stats
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["corpus", "model", "max_input_tokens", "chunks", "truncated", "percent"])
    for (corpus, model), (limit, count, truncated) in sorted(summaries.items()):
        writer.writerow([corpus, model, limit, count, truncated, f"{100 * truncated / count:.2f}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    audit(parser.parse_args().artifacts)
