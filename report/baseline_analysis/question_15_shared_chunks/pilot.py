"""Measure exact word windows with the eight native document tokenizers on Narval."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from clir_benchmark.adapters import DatasetReader
from clir_benchmark.config import load_config
from clir_benchmark.shared_chunks import SharedSentenceChunker, TextLayout
from clir_benchmark.util import file_digest, read_json, write_json
from clir_benchmark.verified_corpora import iter_verified_units


def measure(root, config_path):
    selection = read_json(root / "inputs/selection.json")
    for filename, checksum in selection["sha256"].items():
        if file_digest(root / "inputs" / filename) != checksum:
            raise ValueError(f"Sample file changed: {filename}")
    config = load_config(config_path)
    counter = SharedSentenceChunker.load(config.chunking, config.model_registry)
    out = root / "results"
    out.mkdir(exist_ok=True)
    with (
        (out / "word_windows.csv").open("w", newline="") as window_file,
        (out / "editions.csv").open("w", newline="") as edition_file,
    ):
        windows = csv.writer(window_file)
        windows.writerow(["corpus", "edition", "language", "start", "end", *counter.budgets])
        editions = csv.writer(edition_file)
        editions.writerow(["corpus", "source_id", "edition", "language", "words"])
        for corpus, spec in config.datasets.items():
            source = selection["sources"][corpus]
            if (spec.corpus.repo_id, spec.corpus.revision) != (
                source["source"],
                source["revision"],
            ):
                raise ValueError(f"Different source revision for {corpus}")
            reader = DatasetReader(replace(spec.corpus, local_dir=str(root / "inputs" / corpus)))
            for unit in iter_verified_units(reader, corpus):
                layout = TextLayout(unit.text, unit.language)
                editions.writerow(
                    [corpus, unit.source_id, unit.id, unit.language, len(layout.word_starts)]
                )
                for start, end in layout.word_probes(config.chunking.max_words):
                    counts = counter.counts(unit.text[start:end])
                    windows.writerow(
                        [
                            corpus,
                            unit.id,
                            unit.language,
                            start,
                            end,
                            *(counts[m] for m in counter.budgets),
                        ]
                    )
    write_json(out / "tokenizer_policy.json", counter.manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("tokenizers.yaml"))
    args = parser.parse_args()
    measure(args.root.resolve(), args.config)
