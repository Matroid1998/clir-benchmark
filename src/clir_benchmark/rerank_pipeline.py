"""Explicit reranking of saved retrieval candidates, with offline evaluation."""

from __future__ import annotations

import csv
import math
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa

from .artifacts import RUN_SCHEMA, CorpusStore, ParquetSink, rows
from .config import RERANK_BASELINES, Config
from .errors import ValidationError
from .generation import nullable_sum
from .metrics import aggregate, score_query
from .reranking import Qwen3Reranker, scoring_identity
from .util import (
    digest,
    file_digest,
    hf_token,
    json_dumps,
    new_directory,
    read_json,
    versions,
    write_json,
)

CANDIDATE_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in ("setting_id", "query_id", "chunk_id", "score_key", "audit_json")
    ]
    + [(name, pa.int64()) for name in ("parent_rank", "rank")]
    + [(name, pa.float64()) for name in ("parent_score", "score", "seconds_amortized")]
)

TIMING_NOTES = (
    "Reranking time is measured over each uncached inference batch, including tokenization and "
    "device synchronization, then amortized over its pairs; original measurements are retained on reuse. "
    "Model loading is recorded separately per invocation. Parent component timings retain their original "
    "measurement methods. Component sums are estimates, not measured end-to-end latency. "
    "Local reranking compute has no configured monetary price and remains unknown."
)


def _base(baseline: str) -> str:
    if baseline not in RERANK_BASELINES:
        raise ValidationError(f"Unknown reranking baseline: {baseline}")
    return baseline.removeprefix("rerank-")


def _parent(config, corpus, spec, baseline, translation_language, hyde_language):
    from .pipeline import run_path

    base = _base(baseline)
    path = run_path(
        config,
        corpus,
        spec,
        baseline=base,
        translation_language=translation_language,
        hyde_language=hyde_language,
    )
    if not (path / "manifest.json").is_file() or not (path / "run.parquet").is_file():
        command = "run-adore" if base.startswith("adore-") else "retrieve"
        raise ValidationError(
            f"Parent retrieval run is not prepared: {path}. Run {command} explicitly with --baseline {base} first."
        )
    if base.startswith("adore-"):
        from .adore_pipeline import load_adore_run

        manifest = load_adore_run(path)
    else:
        manifest = read_json(path / "manifest.json")
    expected = {
        "baseline": base,
        "model": asdict(spec) if spec else None,
        "relevance_hash": config.relevance_key(corpus),
        "corpus_fingerprint": read_json(config.corpus_dir(corpus) / "manifest.json")[
            "corpus_fingerprint"
        ],
        "query_fingerprint": read_json(config.relevance_dir(corpus) / "manifest.json")[
            "query_fingerprint"
        ],
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValidationError(
                f"Parent retrieval {name} does not match this reranking experiment"
            )
    hashes = {
        name: file_digest(path / name)
        for name in ("manifest.json", "run.parquet", "timings.parquet")
        if (path / name).is_file()
    }
    return path, manifest, hashes


def _score_identity(config: Config) -> dict:
    settings = asdict(config.reranker)
    settings.pop("candidate_k")  # A score is reusable at any candidate depth.
    return {
        "schema": 1,
        "configuration": settings,
        "scoring": scoring_identity(),
        "versions": versions(),
    }


def rerank_run_path(
    config: Config,
    corpus: str,
    spec=None,
    *,
    baseline="rerank-dense",
    translation_language=None,
    hyde_language=None,
) -> Path:
    parent, _, hashes = _parent(config, corpus, spec, baseline, translation_language, hyde_language)
    identity = {
        "schema": 1,
        "baseline": baseline,
        "parent": str(parent),
        "parent_files": hashes,
        "relevance_hash": config.relevance_key(corpus),
        "reranker": asdict(config.reranker),
        "scoring": _score_identity(config),
        "query_policy": "original_question",
        "translation_language": translation_language,
        "hyde_language": hyde_language,
        "tie_policy": "parent_rank_then_chunk_id",
    }
    return config.output_dir / "runs" / corpus / digest(identity)


def load_rerank_run(path: Path) -> dict:
    if not (path / "manifest.json").is_file():
        raise ValidationError(
            f"Reranking run is not prepared: {path}. Run rerank explicitly first."
        )
    manifest = read_json(path / "manifest.json")
    for name, expected in manifest["files"].items():
        if not (path / name).is_file() or file_digest(path / name) != expected:
            raise ValidationError(f"Corrupt reranking run artifact: {path / name}")
    return manifest


def _rankings(path: Path) -> dict:
    result = defaultdict(list)
    for row in rows(path / "run.parquet"):
        result[(row["setting_id"], row["query_id"])].append(row)
    for key, ranking in result.items():
        ranking.sort(key=lambda row: row["rank"])
        if (
            [row["rank"] for row in ranking] != list(range(1, len(ranking) + 1))
            or len({row["chunk_id"] for row in ranking}) != len(ranking)
            or any(not math.isfinite(row["score"]) for row in ranking)
        ):
            raise ValidationError(f"Invalid retrieval ranks, duplicates, or scores for {key}")
    return result


def _eligible(config, corpus):
    root = config.relevance_dir(corpus)
    settings = read_json(root / "settings.json")
    eligible = {
        (row["setting_id"], row["query_id"]): row
        for row in rows(root / "eligibility.parquet")
        if row["status"] == "eligible" and settings[row["setting_id"]]["status"] == "ready"
    }
    return settings, eligible


def _pair_result(path, identity):
    result = read_json(path / "result.json")
    if (
        result.get("identity") != identity
        or result.get("result_digest")
        != digest({key: value for key, value in result.items() if key != "result_digest"})
        or isinstance(result.get("score"), bool)
        or not isinstance(result.get("score"), (int, float))
        or not math.isfinite(result["score"])
        or not 0 <= result["score"] <= 1
    ):
        raise ValidationError(f"Corrupt reranking pair artifact: {path}")
    return result


def _summary(results):
    values = list(results)
    return {
        "unique_pair_count": len(values),
        "reranking_seconds_amortized": nullable_sum(row["seconds_amortized"] for row in values),
        "input_tokens": nullable_sum(row["audit"].get("input_tokens") for row in values),
        "encoded_tokens": nullable_sum(row["audit"].get("encoded_tokens") for row in values),
        "truncated_pairs": sum(bool(row["audit"].get("truncated")) for row in values),
        "cost_usd": None,
    }


def rerank(
    config: Config,
    corpus: str,
    alias: str | None = None,
    *,
    baseline="rerank-dense",
    translation_language=None,
    hyde_language=None,
    reranker_factory=Qwen3Reranker,
) -> Path:
    from .pipeline import retrieval_spec

    spec = retrieval_spec(config, alias, baseline)
    parent, parent_manifest, parent_hashes = _parent(
        config, corpus, spec, baseline, translation_language, hyde_language
    )
    destination = rerank_run_path(
        config,
        corpus,
        spec,
        baseline=baseline,
        translation_language=translation_language,
        hyde_language=hyde_language,
    )
    invocation = {
        "schema": 1,
        "baseline": baseline,
        "corpus": corpus,
        "model": alias,
        "run": str(destination),
        "status": "failed",
        "artifact_reused": False,
        "new_pairs": 0,
        "cached_pairs": 0,
        "new_reranking_seconds": 0.0,
        "model_load_seconds": 0.0,
        "reranking_cost_usd": None,
    }
    started = time.perf_counter()
    model = None
    used = {}
    try:
        if destination.exists():
            manifest = load_rerank_run(destination)
            invocation.update(
                status="success",
                artifact_reused=True,
                cached_pairs=manifest["reranking_summary"]["unique_pair_count"],
            )
            return destination
        settings, eligible = _eligible(config, corpus)
        rankings = _rankings(parent)
        if set(rankings) - set(eligible):
            raise ValidationError("Parent run contains query/settings outside the eligible cohort")
        queries = {
            row["id"]: row
            for row in rows(
                config.relevance_dir(corpus) / "queries.parquet",
                ["id", "question", "question_language"],
            )
        }
        score_identity = _score_identity(config)
        traces = []
        with (
            CorpusStore(config.corpus_dir(corpus) / "lookup.sqlite") as store,
            new_directory(destination) as temporary,
        ):
            with (
                ParquetSink(temporary / "run.parquet", RUN_SCHEMA) as sink,
                ParquetSink(temporary / "candidates.parquet", CANDIDATE_SCHEMA) as candidates,
            ):
                handles = {}
                try:
                    for setting_id, setting in settings.items():
                        if setting["status"] == "ready":
                            handles[setting_id] = (temporary / f"{setting_id}.trec").open(
                                "w", encoding="utf-8"
                            )
                    for key in sorted(eligible):
                        setting_id, query_id = key
                        query = queries[query_id]
                        selected = rankings[key][: config.reranker.candidate_k]
                        pair_keys, missing = [], {}
                        for candidate in selected:
                            chunk = store.chunk(candidate["chunk_id"])
                            if (
                                chunk is None
                                or chunk["language"] not in settings[setting_id]["target_languages"]
                            ):
                                raise ValidationError(
                                    "Parent candidate does not belong to the configured corpus/language pool"
                                )
                            source = store.get_source(chunk["source_unit_id"])
                            if source is None:
                                raise ValidationError(f"Missing source for chunk {chunk['id']}")
                            document = source.text[chunk["start"] : chunk["end"]]
                            identity = {
                                "scoring": score_identity,
                                "query": query["question"],
                                "query_language": query["question_language"],
                                "document": document,
                                "document_language": chunk["language"],
                            }
                            pair_key = digest(identity)
                            pair_keys.append(pair_key)
                            if pair_key in used:
                                invocation["cached_pairs"] += 1
                                continue
                            path = config.output_dir / "reranking_pairs" / pair_key
                            if path.exists():
                                used[pair_key] = _pair_result(path, identity)
                                invocation["cached_pairs"] += 1
                            else:
                                missing[pair_key] = identity
                        pending = list(missing.items())
                        for offset in range(0, len(pending), config.reranker.batch_size):
                            batch = pending[offset : offset + config.reranker.batch_size]
                            if model is None:
                                load_started = time.perf_counter()
                                try:
                                    model = reranker_factory(
                                        config.reranker, token=hf_token(config.env_file)
                                    )
                                    if hasattr(model, "synchronize"):
                                        model.synchronize()
                                finally:
                                    invocation["model_load_seconds"] = (
                                        time.perf_counter() - load_started
                                    )
                            if hasattr(model, "synchronize"):
                                model.synchronize()
                            batch_started = time.perf_counter()
                            try:
                                scored = model.score(
                                    [
                                        (identity["query"], identity["document"])
                                        for _, identity in batch
                                    ]
                                )
                                if hasattr(model, "synchronize"):
                                    model.synchronize()
                            finally:
                                elapsed = time.perf_counter() - batch_started
                                invocation["new_reranking_seconds"] += elapsed
                            if len(scored.scores) != len(batch) or len(scored.audits) != len(batch):
                                raise ValidationError(
                                    "Reranker returned an inconsistent score/audit batch"
                                )
                            for (pair_key, identity), score, audit in zip(
                                batch, scored.scores, scored.audits, strict=True
                            ):
                                if (
                                    isinstance(score, bool)
                                    or not isinstance(score, (int, float))
                                    or not math.isfinite(score)
                                    or not 0 <= score <= 1
                                ):
                                    raise ValidationError(
                                        "Reranker returned an invalid yes/no probability"
                                    )
                                result = {
                                    "identity": identity,
                                    "score": float(score),
                                    "audit": audit,
                                    "seconds_amortized": elapsed / len(batch),
                                    "batch_size": len(batch),
                                    "model_metadata": model.metadata,
                                }
                                result["result_digest"] = digest(result)
                                with new_directory(
                                    config.output_dir / "reranking_pairs" / pair_key
                                ) as pair_tmp:
                                    write_json(pair_tmp / "result.json", result)
                                used[pair_key] = result
                                invocation["new_pairs"] += 1
                        reranked = sorted(
                            zip(selected, pair_keys, strict=True),
                            key=lambda item: (
                                -used[item[1]]["score"],
                                item[0]["rank"],
                                item[0]["chunk_id"],
                            ),
                        )
                        for rank, (candidate, pair_key) in enumerate(reranked, 1):
                            score = used[pair_key]
                            row = {
                                "setting_id": setting_id,
                                "query_id": query_id,
                                "chunk_id": candidate["chunk_id"],
                                "rank": rank,
                                "score": score["score"],
                            }
                            sink.append(row)
                            candidates.append(
                                {
                                    **row,
                                    "parent_rank": candidate["rank"],
                                    "parent_score": candidate["score"],
                                    "score_key": pair_key,
                                    "seconds_amortized": score["seconds_amortized"],
                                    "audit_json": json_dumps(score["audit"]),
                                }
                            )
                            handles[setting_id].write(
                                f"{query_id}\tQ0\t{candidate['chunk_id']}\t{rank}\t{score['score']:.17g}\t{baseline}\n"
                            )
                        traces.append(
                            {
                                "setting_id": setting_id,
                                "query_id": query_id,
                                "query_language": query["question_language"],
                                "candidate_count": len(selected),
                                "score_keys": pair_keys,
                                "reranking_seconds_amortized": nullable_sum(
                                    used[pair_key]["seconds_amortized"]
                                    for pair_key in sorted(set(pair_keys))
                                ),
                                "reranking_cost_usd": None,
                            }
                        )
                finally:
                    for handle in handles.values():
                        handle.close()
            write_json(temporary / "traces.json", traces)
            summary = {
                **_summary(used.values()),
                "pair_count": sum(row["candidate_count"] for row in traces),
            }
            write_json(
                temporary / "manifest.json",
                {
                    "schema": 1,
                    "baseline": baseline,
                    "parent_baseline": _base(baseline),
                    "parent_run": str(parent),
                    "parent_files": parent_hashes,
                    "parent_manifest": parent_manifest,
                    "selected_baselines": list(config.baselines),
                    "corpus": corpus,
                    "model": asdict(spec) if spec else None,
                    "reranker": asdict(config.reranker),
                    "scoring": scoring_identity(),
                    "query_policy": "original_question",
                    "translation_language": translation_language,
                    "hyde_language": hyde_language,
                    "relevance_hash": config.relevance_key(corpus),
                    "corpus_fingerprint": parent_manifest["corpus_fingerprint"],
                    "query_fingerprint": parent_manifest["query_fingerprint"],
                    "settings": settings,
                    "candidate_k": config.reranker.candidate_k,
                    "tie_policy": "parent_rank_then_chunk_id",
                    "reranking_summary": summary,
                    "score_keys": sorted(used),
                    "timing_notes": TIMING_NOTES,
                    "model_metadata": model.metadata
                    if model
                    else next(iter(used.values()))["model_metadata"]
                    if used
                    else None,
                    "seconds": time.perf_counter() - started,
                    "versions": versions(),
                    "files": {
                        path.name: file_digest(path)
                        for path in temporary.iterdir()
                        if path.is_file()
                    },
                },
            )
        invocation["status"] = "success"
        return destination
    except BaseException as exc:
        invocation["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        invocation["error_type"] = type(exc).__name__
        raise
    finally:
        invocation["seconds"] = time.perf_counter() - started
        write_json(
            config.output_dir / "reranking_invocations" / f"{uuid.uuid4().hex}.json", invocation
        )


def _parent_components(row, baseline):
    if baseline.startswith("adore-"):
        return row.get("component_seconds"), row.get("generation_cost_usd")
    generation_seconds = row.get("generation_seconds") if baseline == "hyde-dense" else 0.0
    translation_seconds = (
        row.get("translation_seconds") if baseline.startswith("translate-") else 0.0
    )
    cost = (
        row.get("generation_cost_usd")
        if baseline == "hyde-dense"
        else row.get("translation_cost_usd")
        if baseline.startswith("translate-")
        else 0.0
    )
    return nullable_sum(
        [
            generation_seconds,
            translation_seconds,
            row.get("query_encoding_seconds_amortized"),
            row.get("retrieval_seconds_amortized"),
        ]
    ), cost


def _csv(path, records):
    flattened = []
    for record in records:
        flat = {}
        for key, value in record.items():
            if isinstance(value, dict):
                flat.update({f"{key}.{name}": item for name, item in value.items()})
            else:
                flat[key] = value
        flattened.append(flat)
    fields = list(dict.fromkeys(key for record in flattened for key in record)) or ["setting_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def evaluate_rerank(
    config: Config,
    corpus: str,
    alias=None,
    *,
    baseline="rerank-dense",
    translation_language=None,
    hyde_language=None,
) -> Path:
    from .pipeline import evaluate, retrieval_spec

    spec = retrieval_spec(config, alias, baseline)
    run_dir = rerank_run_path(
        config,
        corpus,
        spec,
        baseline=baseline,
        translation_language=translation_language,
        hyde_language=hyde_language,
    )
    manifest = load_rerank_run(run_dir)
    destination = config.output_dir / "reports" / corpus / run_dir.name
    base = _base(baseline)
    parent_report_dir = evaluate(
        config,
        corpus,
        alias,
        baseline=base,
        translation_language=translation_language,
        hyde_language=hyde_language,
    )
    parent_report = read_json(parent_report_dir / "report.json")
    if parent_report.get("run") != manifest["parent_manifest"]:
        raise ValidationError("Parent evaluation report is stale: its run manifest differs")
    parent_rows = read_json(parent_report_dir / "per_query.json")
    originals = {(row["setting_id"], row["query_id"]): row for row in parent_rows}
    if len(originals) != len(parent_rows):
        raise ValidationError("Duplicate query/setting in parent evaluation report")
    if parent_report.get("metrics") != aggregate(parent_rows):
        raise ValidationError("Parent evaluation report is stale: aggregate metrics differ")
    settings, eligible = _eligible(config, corpus)
    queries = {
        row["id"]: row
        for row in rows(
            config.relevance_dir(corpus) / "queries.parquet",
            ["id", "question_language"],
        )
    }
    traces = {
        (row["setting_id"], row["query_id"]): row for row in read_json(run_dir / "traces.json")
    }
    if originals.keys() != eligible.keys() or traces.keys() != eligible.keys():
        raise ValidationError("Reranking and parent evaluation cohorts differ")
    parent_report_files = {
        name: file_digest(parent_report_dir / name) for name in ("report.json", "per_query.json")
    }
    if destination.exists():
        cached_report = read_json(destination / "report.json")
        if cached_report.get("parent_report_files") != parent_report_files:
            raise ValidationError(
                "Parent evaluation report changed after reranking evaluation; cached comparison is stale"
            )
        if cached_report.get("per_query_fingerprint") != digest(
            read_json(destination / "per_query.json")
        ):
            raise ValidationError("Corrupt cached reranking evaluation rows")
        return destination
    parent_rankings, rankings = _rankings(Path(manifest["parent_run"])), _rankings(run_dir)
    qrels = defaultdict(lambda: defaultdict(set))
    for row in rows(config.relevance_dir(corpus) / "qrels.parquet"):
        qrels[(row["setting_id"], row["query_id"])][row["chunk_id"]].add(row["required_source_id"])
    records = []
    for key in sorted(eligible):
        original, trace = originals[key], traces[key]
        setting = settings[key[0]]
        expected = {
            "baseline": base,
            "model": alias,
            "corpus": corpus,
            "query_language": queries[key[1]]["question_language"],
            "target_languages": ",".join(setting["target_languages"]),
            "source_count": len(eligible[key]["required_source_ids"]),
            "regime": setting["kind"],
        }
        for name, value in expected.items():
            if original.get(name) != value:
                raise ValidationError(f"Parent evaluation {name} differs for {key}")
        candidates = [
            row["chunk_id"] for row in parent_rankings[key][: config.reranker.candidate_k]
        ]
        reranked = [row["chunk_id"] for row in rankings[key]]
        if set(candidates) != set(reranked) or len(candidates) != len(reranked):
            raise ValidationError(f"Reranked candidate set differs from parent for {key}")
        required = set(eligible[key]["required_source_ids"])
        full_parent_metrics = score_query(
            [row["chunk_id"] for row in parent_rankings[key]], qrels[key], required
        )
        if original["metrics"] != full_parent_metrics:
            raise ValidationError(f"Parent evaluation report is stale: metrics differ for {key}")
        parent_seconds, parent_cost = _parent_components(original, base)
        candidate_metrics = score_query(candidates, qrels[key], required)
        metrics = score_query(reranked, qrels[key], required)
        records.append(
            {
                **original,
                "baseline": baseline,
                "parent_baseline": base,
                "reranker_model": config.reranker.repo_id,
                "reranker_revision": config.reranker.revision,
                "reranking_query_policy": "original_question",
                "candidate_count": len(candidates),
                "reranking_seconds_amortized": trace["reranking_seconds_amortized"],
                "parent_component_seconds": parent_seconds,
                "component_seconds": nullable_sum(
                    [parent_seconds, trace["reranking_seconds_amortized"]]
                ),
                "end_to_end_seconds": None,
                "parent_api_cost_usd": parent_cost,
                "reranking_cost_usd": None,
                "total_cost_usd": None,
                "metrics": metrics,
                "candidate_control_metrics": candidate_metrics,
                "full_parent_metrics": full_parent_metrics,
                "metric_deltas": {
                    name: value - candidate_metrics[name] for name, value in metrics.items()
                },
                "full_parent_metric_deltas": {
                    name: value - original["metrics"][name] for name, value in metrics.items()
                },
            }
        )
    report = {
        "schema": 1,
        "corpus": corpus,
        "baseline": baseline,
        "model": alias,
        "parent_baseline": base,
        "parent_report": str(parent_report_dir),
        "parent_report_files": parent_report_files,
        "per_query_fingerprint": digest(records),
        "reranker": asdict(config.reranker),
        "run": manifest,
        "settings": settings,
        "candidate_selection": parent_report["candidate_selection"],
        "metrics": aggregate(records),
        "candidate_control_metrics": aggregate(
            [{**row, "metrics": row["candidate_control_metrics"]} for row in records]
        ),
        "full_parent_metrics": parent_report["metrics"],
        "reranking_summary": manifest["reranking_summary"],
        "parent_accounting": {
            name: parent_report[name]
            for name in (
                "translation_summary",
                "hyde_summary",
                "adore_summary",
                "component_summary",
            )
            if name in parent_report
        },
        "truncation": {
            "parent": parent_report.get("truncation"),
            "reranker": manifest["reranking_summary"],
            "pair_audits": str(run_dir / "candidates.parquet"),
        },
        "timing_notes": TIMING_NOTES,
        "versions": versions(),
    }
    with new_directory(destination) as temporary:
        write_json(temporary / "report.json", report)
        write_json(temporary / "per_query.json", records)
        _csv(temporary / "per_query.csv", records)
        _csv(temporary / "settings.csv", report["metrics"]["setting"])
    return destination


def compare_rerank(
    config: Config,
    corpus: str,
    alias=None,
    *,
    baseline="rerank-dense",
    translation_language=None,
    hyde_language=None,
) -> Path:
    directory = evaluate_rerank(
        config,
        corpus,
        alias,
        baseline=baseline,
        translation_language=translation_language,
        hyde_language=hyde_language,
    )
    report = read_json(directory / "report.json")
    records = read_json(directory / "per_query.json")
    grouped = defaultdict(list)
    comparisons = []
    for record in records:
        row = {
            name: record[name]
            for name in (
                "setting_id",
                "query_id",
                "query_language",
                "target_languages",
                "candidate_count",
                "baseline",
                "parent_baseline",
                "model",
                "reranker_model",
                "reranker_revision",
                "metric_deltas",
                "full_parent_metric_deltas",
                "parent_component_seconds",
                "component_seconds",
                "parent_api_cost_usd",
                "reranking_cost_usd",
                "total_cost_usd",
            )
        }
        row.update(
            {
                "reranked_metrics": record["metrics"],
                "control_metrics": record["candidate_control_metrics"],
                "full_parent_metrics": record["full_parent_metrics"],
                "added_component_seconds": record["reranking_seconds_amortized"],
            }
        )
        comparisons.append(row)
        grouped[row["setting_id"]].append(row)
    summaries = []
    for _setting_id, members in sorted(grouped.items()):
        summary = {
            name: members[0][name]
            for name in (
                "setting_id",
                "baseline",
                "parent_baseline",
                "model",
                "reranker_model",
                "query_language",
                "target_languages",
            )
        }
        summary["query_count"] = len(members)
        for name in (
            "parent_component_seconds",
            "component_seconds",
            "added_component_seconds",
            "parent_api_cost_usd",
            "reranking_cost_usd",
            "total_cost_usd",
        ):
            total = nullable_sum(row[name] for row in members)
            summary[f"mean_{name}"] = total / len(members) if total is not None else None
        for name in (
            "reranked_metrics",
            "control_metrics",
            "full_parent_metrics",
            "metric_deltas",
            "full_parent_metric_deltas",
        ):
            for metric in members[0][name]:
                summary[f"{name}.{metric}"] = sum(row[name][metric] for row in members) / len(
                    members
                )
        summaries.append(summary)
    comparison_report = {
        "schema": 1,
        "corpus": corpus,
        "baseline": baseline,
        "parent_baseline": _base(baseline),
        "model": alias,
        "reranker": asdict(config.reranker),
        "reranked_report": str(directory),
        "parent_report": report["parent_report"],
        "reranking_summary": report["reranking_summary"],
        "parent_accounting": report["parent_accounting"],
        "settings": summaries,
        "notes": [
            "metric_deltas compare the same candidate IDs before and after reranking. Control ordering is the parent's top candidate_k. full_parent_metric_deltas separately include the effect of candidate truncation.",
            "Reranking always scores the original question, including when translation, HyDE, or ADORE generated the retrieval candidates. Document languages and cohort do not change.",
            "Parent API costs and original timings are retained. Reranking compute and pipeline monetary totals remain unknown; local work is not treated as free.",
            "Pair summaries count each compatible cached query/document score once; do not sum shared work across settings or parent runs. Invocation audits distinguish current work from cached original measurements.",
            TIMING_NOTES,
        ],
    }
    destination = (
        config.output_dir
        / "comparisons"
        / corpus
        / digest({"report": comparison_report, "queries": comparisons})
    )
    if destination.exists():
        return destination
    with new_directory(destination) as temporary:
        write_json(temporary / "report.json", comparison_report)
        write_json(temporary / "per_query.json", comparisons)
        _csv(temporary / "settings.csv", summaries)
        _csv(temporary / "per_query.csv", comparisons)
    return destination
