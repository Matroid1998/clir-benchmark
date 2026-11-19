#!/usr/bin/env python3
"""Question 11: count chemistry publications after a year using saved date metadata."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-year", type=int, default=2020)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cutoff = date(args.after_year + 1, 1, 1)
    plan_path = root / "question_04_corpus_sizes/corpus_sizes.json"
    plan = next(p for p in json.loads(plan_path.read_text()) if p["corpus"] == "chemistry")
    manifest_path = Path(plan["source_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["corpus_fingerprint"] != plan["corpus_fingerprint"]:
        raise ValueError("Source corpus differs from the saved experiment")
    sources_path = manifest_path.parent / "sources.parquet"
    units = pq.read_table(
        sources_path, columns=["id", "source_id", "language", "metadata_json"]
    ).to_pylist()
    dates = defaultdict(set)
    raw_date_problems = []
    for unit in units:
        value = json.loads(unit["metadata_json"]).get("publication_date")
        try:
            if not re.fullmatch(r"\d{8}", str(value)):
                raise ValueError("Expected YYYYMMDD")
            parsed = datetime.strptime(str(value), "%Y%m%d").date()
        except ValueError:
            raw_date_problems.append({"unit_id": unit["id"], "value": value})
            continue
        dates[unit["source_id"]].add(parsed)
    conflicts = {pub: sorted(map(str, values)) for pub, values in dates.items() if len(values) > 1}
    if conflicts:
        raise ValueError(f"Conflicting publication dates across language versions: {conflicts}")
    publication_dates = {pub: next(iter(values)) for pub, values in dates.items()}
    publications = {unit["source_id"] for unit in units}
    selected_ids = {pub for pub, published in publication_dates.items() if published >= cutoff}
    selected_units = [u for u in units if u["source_id"] in selected_ids]
    search_languages = set(plan["configured_search_languages"])
    searched = [u for u in selected_units if u["language"] in search_languages]
    by_year = Counter(published.year for published in publication_dates.values())
    if sum(n for year, n in by_year.items() if year > args.after_year) != len(selected_ids):
        raise ValueError("Year totals disagree with the publication-date cutoff")
    catalog_path = root / "question_01_recall_at_10/tables/report_catalog.csv"
    with catalog_path.open() as handle:
        entry = next(
            r
            for r in csv.DictReader(handle)
            if r["cohort"] == "selected_documents"
            and r["corpus"] == "chemistry"
            and r["model"] == "embeddinggemma"
            and r["baseline"] == "dense"
        )
    run = json.loads(entry["resolved_run_parameters_json"])
    if run["corpus_fingerprint"] != manifest["corpus_fingerprint"]:
        raise ValueError("Question cohort is associated with another corpus")
    relevance_dir = manifest_path.parents[3] / "relevance/chemistry" / run["relevance_hash"]
    queries_path = relevance_dir / "queries.parquet"
    queries = pq.read_table(queries_path, columns=["id", "source_unit_id"]).to_pylist()
    unit_to_publication = {unit["id"]: unit["source_id"] for unit in units}
    selected_queries = [
        q for q in queries if unit_to_publication[q["source_unit_id"]] in selected_ids
    ]
    output = (
        args.output or root / "question_11_chemistry_publication_dates" / f"after_{args.after_year}"
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "publications_after_cutoff.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["publication_id", "publication_date"])
        writer.writerows((pub, publication_dates[pub].isoformat()) for pub in sorted(selected_ids))
    with (output / "publication_counts_by_year.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["publication_year", "distinct_publications", "after_cutoff"])
        writer.writerows((year, n, year > args.after_year) for year, n in sorted(by_year.items()))
    result = {
        "date_field": "sources.metadata_json.publication_date (upstream pub_date)",
        "cutoff_inclusive": cutoff.isoformat(),
        "after_year": args.after_year,
        "counting_unit": "Distinct source_id / publication number, not patent families or language versions",
        "total_distinct_publications": len(publications),
        "distinct_publications_after_cutoff": len(selected_ids),
        "percent_of_all_publications": 100 * len(selected_ids) / len(publications),
        "abstract_language_versions_after_cutoff": len(selected_units),
        "searched_abstract_language_versions_after_cutoff": len(searched),
        "searched_distinct_publications_after_cutoff": len({u["source_id"] for u in searched}),
        "language_version_counts_after_cutoff": dict(
            sorted(Counter(u["language"] for u in selected_units).items())
        ),
        "distinct_publications_by_year_after_cutoff": {
            year: n for year, n in sorted(by_year.items()) if year > args.after_year
        },
        "selected_evaluation_questions_after_cutoff": len(selected_queries),
        "total_selected_evaluation_questions": len(queries),
        "publications_without_a_valid_date": len(publications - publication_dates.keys()),
        "language_versions_with_missing_or_invalid_date": len(raw_date_problems),
        "date_conflicts": conflicts,
        "observed_publication_date_range": [
            min(publication_dates.values()).isoformat(),
            max(publication_dates.values()).isoformat(),
        ],
        "source_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                plan_path,
                manifest_path,
                sources_path,
                catalog_path,
                queries_path,
                Path(__file__),
            ]
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "source_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
