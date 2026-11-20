"""Count formatted inputs before token-based shortening for two word-overlap policies."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

from clir_benchmark.adapters import DatasetReader
from clir_benchmark.config import load_config
from clir_benchmark.shared_chunks import SharedSentenceChunker, TextLayout
from clir_benchmark.util import file_digest, read_json, write_json
from clir_benchmark.verified_corpora import iter_verified_units


class WordOnlySentenceChunker(SharedSentenceChunker):
    def fits(self, layout, start, end):
        # The experiment must expose any token-limit overflow in the proposed word policy.
        return layout.words(start, end) <= self.config.max_words


def fixed_windows(layout, size, overlap):
    words = layout.word_starts
    for i in range(0, max(1, len(words)), size - overlap):
        start = words[i] if i else 0
        if i + size >= len(words):
            yield start, len(layout.text)
            break
        yield start, words[i + size]


def measure(root, config_path, overlap):
    config = load_config(config_path)
    policy = replace(config.chunking, overlap_words=overlap)
    tokenizer = WordOnlySentenceChunker.load(policy, config.model_registry)
    native = {m: b["native_limit"] for m, b in tokenizer.budgets.items()}
    selection = read_json(root / "inputs/selection.json")
    for filename, expected in selection["sha256"].items():
        if file_digest(root / "inputs" / filename) != expected:
            raise ValueError(f"Input hash mismatch: {filename}")
    summaries = {
        mode: {
            "chunks": 0,
            "full_word_windows": 0,
            "over_any_native_limit": 0,
            "over_common_cap": 0,
            "max_tokens": dict.fromkeys(native, 0),
        }
        for mode in ("fixed_windows", "sentence_first")
    }
    documents, editions = set(), 0
    for corpus, data in config.datasets.items():
        source = selection["sources"][corpus]
        if (source["source"], source["revision"]) != (data.corpus.repo_id, data.corpus.revision):
            raise ValueError(f"Different release for {corpus}")
        reader = DatasetReader(replace(data.corpus, local_dir=str(root / "inputs" / corpus)))
        for unit in iter_verified_units(reader, corpus):
            documents.add((corpus, unit.source_id))
            editions += 1
            layout = TextLayout(unit.text, unit.language)
            for mode, windows in (
                ("fixed_windows", fixed_windows(layout, policy.max_words, overlap)),
                ("sentence_first", ((c.start, c.end) for c in tokenizer.split(unit))),
            ):
                summary = summaries[mode]
                previous_end = 0
                for start, end in windows:
                    nwords = layout.words(start, end)
                    if (
                        nwords > policy.max_words
                        or start > previous_end
                        or layout.words(start, previous_end) > overlap
                    ):
                        raise ValueError(f"Invalid window in {unit.id}")
                    counts = tokenizer.counts(unit.text[start:end])
                    summary["chunks"] += 1
                    summary["full_word_windows"] += int(nwords == policy.max_words)
                    summary["over_any_native_limit"] += int(
                        any(n > native[m] for m, n in counts.items())
                    )
                    summary["over_common_cap"] += int(max(counts.values()) > policy.max_tokens)
                    for m, n in counts.items():
                        summary["max_tokens"][m] = max(summary["max_tokens"][m], n)
                    previous_end = end
                if previous_end != len(unit.text):
                    raise ValueError(f"Incomplete edition {unit.id}")
    return {
        "documents": len(documents),
        "editions": editions,
        "max_words": policy.max_words,
        "overlap_words": overlap,
        "native_limits": native,
        "common_token_cap": policy.max_tokens,
        "includes_short_final_chunks": True,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "selection_sha256": file_digest(root / "inputs/selection.json"),
        "policies": summaries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, measure(args.root.resolve(), args.config, args.overlap_words))
