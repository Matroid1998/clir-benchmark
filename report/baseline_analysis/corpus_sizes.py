#!/usr/bin/env python3
"""Question 4: count prepared and searched units for all three saved corpora."""

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    with (root / "question_01_recall_at_10/tables/report_catalog.csv").open() as handle:
        catalog = list(csv.DictReader(handle))
    results, language_rows = [], []
    for corpus in ["chemistry", "eurlex", "un"]:
        reports = [
            r
            for r in catalog
            if r["cohort"] == "selected_documents"
            and r["corpus"] == corpus
            and r["baseline"] == "dense"
        ]
        runs = [json.loads(r["resolved_run_parameters_json"]) for r in reports]
        fingerprints = {run["corpus_fingerprint"] for run in runs}
        targets = {
            tuple(
                sorted(
                    {
                        lang
                        for setting in run["settings"].values()
                        for lang in setting["target_languages"]
                    }
                )
            )
            for run in runs
        }
        if len(fingerprints) != 1 or len(targets) != 1:
            raise ValueError(f"Runs differ in corpus or target languages: {corpus}")
        languages = set(next(iter(targets)))
        manifests = [
            (p, json.loads(p.read_text()))
            for p in (repo / "artifacts/fir-20261109-all-questions/artifacts/corpus" / corpus).glob(
                "*/manifest.json"
            )
        ]
        path, manifest = next(
            (p, m) for p, m in manifests if m["corpus_fingerprint"] in fingerprints
        )
        table_path = path.parent / "sources.parquet"
        units = pq.read_table(
            table_path, columns=["source_id", "parent_id", "language", "unit_type"]
        ).to_pylist()
        counts = Counter(unit["language"] for unit in units)
        if counts != manifest["languages"] or len(units) != manifest["source_count"]:
            raise ValueError(f"Manifest/source count mismatch: {corpus}")
        searched = [u for u in units if u["language"] in languages]
        results.append(
            {
                "corpus": corpus,
                "prepared_language_versions": len(units),
                "prepared_distinct_source_ids": len({u["source_id"] for u in units}),
                "prepared_distinct_parent_ids": len({u["parent_id"] for u in units}),
                "prepared_chunks": manifest["chunk_count"],
                "searched_language_versions": len(searched),
                "searched_distinct_source_ids": len({u["source_id"] for u in searched}),
                "unit_types": dict(Counter(u["unit_type"] for u in units)),
                "configured_search_languages": sorted(languages),
                "available_search_languages": sorted(set(counts) & languages),
                "selected_evaluation_questions": int(reports[0]["unique_query_count"]),
                "language_version_counts": dict(sorted(counts.items())),
                "source_manifest": str(path),
                "source_table": str(table_path),
                "corpus_fingerprint": manifest["corpus_fingerprint"],
            }
        )
        language_rows.extend(
            (corpus, lang, n, lang in languages) for lang, n in sorted(counts.items())
        )
    output = root / "question_04_corpus_sizes"
    output.mkdir(parents=True, exist_ok=True)
    (output / "corpus_sizes.json").write_text(json.dumps(results, indent=2) + "\n")
    with (output / "language_counts.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corpus", "language", "unit_language_versions", "searched"])
        writer.writerows(language_rows)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
