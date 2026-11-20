"""Read-only verification of the November 20 retrieval campaign before archiving."""

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from clir_benchmark.artifacts import rows
from clir_benchmark.config import load_config
from clir_benchmark.fingerprints import cached_file_digest
from clir_benchmark.pipeline import bm25_path, embedding_path, model_spec, run_path
from clir_benchmark.util import read_json, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_file_inventory(root, entries, *, size_only):
    errors = {}
    for name, (size, checksum) in sorted(entries.items()):
        try:
            path = root / name
            require(path.is_file(), "missing file")
            require(path.stat().st_size == size, "size differs")
            if not size_only:
                require(cached_file_digest(path) == checksum, "SHA-256 differs")
        except (OSError, ValueError) as exc:
            errors[name] = str(exc)
    return errors


def merge_inventory(entries, files):
    names = set()
    for row in files:
        name = row["path"]
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Invalid path: {name}")
        item = row["bytes"], row["sha256"]
        require(item == entries.setdefault(name, item), f"Conflicting inventory entries: {name}")
        names.add(name)
    return names


def check_run(config, corpus, alias, baseline, out, entries, eligible, raw_qa):
    root = config.output_dir
    spec = model_spec(config, alias) if alias else None
    run = run_path(config, corpus, spec, baseline=baseline)
    report = root / "reports" / corpus / run.name
    out.update(run=run.relative_to(root).as_posix(), report=report.relative_to(root).as_posix())
    analysis = read_json(report / "analysis_manifest.json")
    names = merge_inventory(entries, analysis["files"])
    manifest = read_json(run / "manifest.json")
    summary = read_json(report / "report.json")
    rel = config.relevance_dir(corpus)
    require(
        analysis["run_id"] == run.name
        and analysis["baseline"] == baseline
        and manifest["baseline"] == baseline
        and manifest["corpus"] == corpus
        and manifest["model"] == (asdict(spec) if spec else None)
        and manifest["relevance_hash"] == rel.name
        and manifest["top_k"] == 100
        and summary["run"] == manifest,
        "Manifest differs from pinned run configuration",
    )
    mandatory = [
        run / "manifest.json",
        run / "run.parquet",
        run / "timings.parquet",
        report / "analysis/rankings.parquet",
        report / "analysis/manifest.json",
        report / "report.json",
        report / "per_query.json",
        report / "settings.csv",
        root / analysis["mteb_result"],
    ]
    for path, filenames in (
        (
            rel,
            (
                "manifest.json",
                "queries.parquet",
                "qrels.parquet",
                "matches.parquet",
                "eligibility.parquet",
                "settings.json",
            ),
        ),
        (config.corpus_dir(corpus), ("chunks.parquet", "sources.parquet", "manifest.json")),
    ):
        mandatory.extend(path / f for f in filenames)
    if spec:
        for kind in ("document", "query"):
            embedding = embedding_path(config, corpus, spec, kind)
            mandatory.extend(
                embedding / f for f in ("manifest.json", "rows.parquet", "vectors.npy")
            )
    if baseline in ("bm25", "hybrid"):
        mandatory.append(bm25_path(config, corpus) / "index.sqlite")
    if baseline == "hybrid":
        for component, component_spec in (("bm25", None), ("dense", spec)):
            component_run = run_path(config, corpus, component_spec, baseline=component)
            require(
                manifest["component_runs"][component] == str(component_run),
                f"Unexpected {component} candidate run",
            )
            mandatory.append(component_run / "candidates.parquet")
    else:
        candidates = pq.ParquetFile(run / "candidates.parquet")
        require(
            candidates.metadata.num_rows == len(eligible)
            and "chunk_ids" in candidates.schema_arrow.names,
            "Candidate cohort/schema differs",
        )
        mandatory.append(run / "candidates.parquet")
    mixed = {key for key, value in manifest["settings"].items() if value["kind"] == "mixed"}
    require(bool(mixed), "No mixed setting recorded")
    mandatory.extend(run / f"{key}.trec" for key in mixed)
    missing = [str(p.relative_to(root)) for p in mandatory if str(p.relative_to(root)) not in names]
    require(not missing, f"Missing artifact inventory entries: {missing}")
    # The QA snapshot keeps all raw fields and the published multilingual evidence.
    snapshot = {
        (name.split("/", 2)[-1], entries[name][1])
        for name in names
        if name.startswith("input_snapshots/")
    }
    require(raw_qa and raw_qa <= snapshot, "Published QA/evidence snapshot missing")
    timings = pq.ParquetFile(run / "timings.parquet")
    cols = {"retrieval_seconds_amortized"}
    if baseline == "hybrid":
        cols.update(
            ("dense_seconds_amortized", "bm25_seconds_amortized", "fusion_seconds_amortized")
        )
    require(cols <= set(timings.schema_arrow.names), "Missing measured stage times")
    keys = {
        (r["setting_id"], r["query_id"])
        for r in rows(run / "timings.parquet", ["setting_id", "query_id"])
    }
    require(keys == eligible and timings.metadata.num_rows == len(keys), "Timing cohort differs")
    ranking = pq.ParquetFile(run / "run.parquet")
    detail = pq.ParquetFile(report / "analysis/rankings.parquet")
    mixed_rows = 0
    for batch in ranking.iter_batches(columns=["setting_id"]):
        values, counts = batch.column(0).value_counts().flatten()
        mixed_rows += sum(
            n
            for key, n in zip(values.to_pylist(), counts.to_pylist(), strict=False)
            if key in mixed
        )
    require(
        mixed_rows > 0 and ranking.metadata.num_rows == detail.metadata.num_rows,
        "Missing mixed ranks or incomplete annotated export",
    )
    require(
        {
            "chunk_id",
            "document_language",
            "source_unit_id",
            "start",
            "end",
            "text",
            "rank",
            "score",
            "query_id",
            "setting_id",
        }
        <= set(detail.schema_arrow.names),
        "Missing annotated ranking fields",
    )
    out.update(
        ranked_rows=ranking.metadata.num_rows,
        mixed_ranked_rows=mixed_rows,
        eligible_query_settings=len(keys),
        inventoried_files=len(names),
    )


def verify(campaign, root, result, *, size_only):
    config_path = campaign / "project/configs/verified-20261120-retrieval.yaml"
    config = load_config(config_path)
    require(config.output_dir == root, "Pinned output directory differs from --artifacts")
    require(
        set(config.datasets) == {"chemistry", "eurlex", "un"}
        and len(set(config.models)) == 8
        and set(config.baselines) == {"bm25", "dense", "hybrid"}
        and config.cache_hybrid_candidates,
        "Unexpected campaign configuration",
    )
    state = campaign / "job_status"
    fingerprint = cached_file_digest(config_path)[:16]
    result["config_sha256"] = cached_file_digest(config_path)
    submitted = defaultdict(set)
    with (state / "submissions.tsv").open() as handle:
        for stage, job, _, fp in csv.reader(handle, delimiter="\t"):
            if fp == fingerprint:
                submitted[stage].add(job)
    entries = {}
    for corpus in config.datasets:
        stages = [f"prepare-{corpus}", f"bm25-{corpus}"]
        stages.extend(
            f"{s}-{corpus}-{m}" for m in config.models for s in ("embed", "dense", "hybrid")
        )
        for stage in stages:
            try:
                job = (state / f"{stage}-{fingerprint}.done").read_text().strip()
                require(job.isdigit() and job in submitted[stage], "Unrecognized successful job ID")
                for suffix in ("out", "err"):
                    require(
                        (campaign / f"logs/{stage}-{job}.{suffix}").is_file(),
                        f"Missing .{suffix} log",
                    )
                result["jobs"][stage] = job
            except (ValueError, OSError) as exc:
                result["errors"][stage + ":receipt"] = str(exc)
        try:
            rel = config.relevance_dir(corpus)
            eligible = {
                (r["setting_id"], r["query_id"])
                for r in rows(rel / "eligibility.parquet")
                if r["status"] == "eligible"
            }
            raw_qa = set(config.datasets[corpus].qa.identity()["files"].items())
        except (OSError, ValueError, KeyError) as exc:
            result["errors"][corpus] = str(exc)
            continue
        for baseline in ("bm25", "dense", "hybrid"):
            for alias in (None,) if baseline == "bm25" else config.models:
                key = "-".join(filter(None, (baseline, corpus, alias)))
                record = result["runs"][key] = {}
                try:
                    check_run(config, corpus, alias, baseline, record, entries, eligible, raw_qa)
                except (ValueError, OSError, KeyError, TypeError, pa.ArrowException) as exc:
                    result["errors"][key] = str(exc)
    result["file_errors"] = check_file_inventory(root, entries, size_only=size_only)
    result["unique_files"] = len(entries)
    result["inventoried_bytes"] = sum(size for size, _ in entries.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--size-only", action="store_true", help="Skip the SHA-256 reread of inventoried files"
    )
    args = parser.parse_args()
    result = {
        "schema": 1,
        "jobs": {},
        "runs": {},
        "errors": {},
        "file_errors": {},
        "verification": "size" if args.size_only else "size_and_sha256",
        "expected_jobs": 78,
        "expected_runs": 51,
    }
    try:
        verify(args.campaign.resolve(), args.artifacts.resolve(), result, size_only=args.size_only)
    except Exception as exc:
        result["errors"]["campaign"] = f"{type(exc).__name__}: {exc}"
    result["complete"] = (
        not result["errors"]
        and not result["file_errors"]
        and len(result["jobs"]) == 78
        and len(result["runs"]) == 51
    )
    result["verified_at_utc"] = datetime.now(UTC).isoformat()
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    write_json(tmp, result)
    tmp.replace(args.output)
    print(
        f"Verified {len(result['jobs'])}/78 receipts. Complete: {result['complete']}. See {args.output}"
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
