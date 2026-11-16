#!/usr/bin/env python3
"""Question 3: count prepared and searched chemistry units from saved artifacts."""

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    catalog = root / "question_01_recall_at_10/tables/report_catalog.csv"
    with catalog.open() as handle:
        reports = [
            r
            for r in csv.DictReader(handle)
            if r["cohort"] == "selected_documents"
            and r["corpus"] == "chemistry"
            and r["baseline"] == "dense"
        ]
    runs = [json.loads(r["resolved_run_parameters_json"]) for r in reports]
    fingerprints = {run["corpus_fingerprint"] for run in runs}
    target_sets = {
        tuple(sorted({lang for s in run["settings"].values() for lang in s["target_languages"]}))
        for run in runs
    }
    if len(fingerprints) != 1 or len(target_sets) != 1:
        raise ValueError("Chemistry runs do not share one corpus and target-language set")
    target_languages = set(next(iter(target_sets)))
    manifests = [
        (p, json.loads(p.read_text()))
        for p in (repo / "artifacts/fir-20261109-all-questions/artifacts/corpus/chemistry").glob(
            "*/manifest.json"
        )
    ]
    path, manifest = next((p, m) for p, m in manifests if m["corpus_fingerprint"] in fingerprints)
    sources_path = path.parent / "sources.parquet"
    sources = pq.read_table(sources_path, columns=["source_id", "language"]).to_pylist()
    counts = Counter(r["language"] for r in sources)
    if counts != manifest["languages"] or len(sources) != manifest["source_count"]:
        raise ValueError("Source counts differ from the saved manifest")
    searched = [r for r in sources if r["language"] in target_languages]
    result = {
        "prepared_abstract_language_versions": len(sources),
        "prepared_canonical_patent_ids": len({r["source_id"] for r in sources}),
        "prepared_chunks": manifest["chunk_count"],
        "searched_languages": sorted(target_languages),
        "searched_abstract_language_versions": len(searched),
        "searched_canonical_patent_ids": len({r["source_id"] for r in searched}),
        "selected_evaluation_questions": int(reports[0]["unique_query_count"]),
        "language_version_counts": dict(sorted(counts.items())),
        "source_manifest": str(path),
        "source_table": str(sources_path),
        "corpus_fingerprint": manifest["corpus_fingerprint"],
    }
    output = root / "question_03_chemistry_corpus_size"
    output.mkdir(parents=True, exist_ok=True)
    (output / "corpus_size.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "language_counts.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "abstract_language_versions", "searched"])
        writer.writerows((lang, n, lang in target_languages) for lang, n in sorted(counts.items()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
