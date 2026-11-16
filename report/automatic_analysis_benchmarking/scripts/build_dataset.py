#!/usr/bin/env python3
"""Extract immutable, matched document-level results for offline paper analysis.

This script reads completed artifacts only. It never invokes benchmark stages,
loads retrieval models, or calls APIs. Run from the repository with:
    .venv/bin/python report/acl_analysis_20261115/scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LANGUAGES = ("de", "en", "es", "fr", "zh")
METHODS = {"dense": "original", "translate-dense": "translate", "hyde-dense": "hyde"}
METRICS = {"nDCG@10": "ndcg10", "MRR@10": "mrr10"}
for original, alias in (
    ("Recall", "recall"),
    ("Hit", "hit"),
    ("SourceRecall", "source_recall"),
    ("AllSourceSuccess", "all_source_success"),
):
    METRICS.update({f"{original}@{k}": f"{alias}{k}" for k in (1, 5, 10, 20, 100)})

STRING_COLUMNS = (
    "corpus",
    "model",
    "baseline",
    "method",
    "output_language",
    "variant",
    "query_id",
    "canonical_source_id",
    "source_unit_id",
    "query_language",
    "context_language",
    "setting_id",
    "target_language",
    "target_languages_json",
    "regime",
    "generator",
    "persona",
    "quality_audit_status",
    "retrieval_unit",
    "report_id",
    "report_path",
    "translation_language",
    "translation_status",
    "generation_model",
    "generation_request_key",
    "generation_artifact_path",
)
INT_COLUMNS = (
    "source_count",
    "relevant_document_count",
    "required_source_count",
    "question_characters",
    "question_whitespace_tokens",
    "generation_attempts",
    "generation_input_tokens",
    "generation_cached_input_tokens",
    "generation_output_tokens",
    "generated_characters",
    "generated_whitespace_tokens",
)
FLOAT_COLUMNS = (
    "total_score",
    "query_encoding_seconds_amortized",
    "retrieval_seconds_amortized",
    "translation_cost_usd",
    "translation_seconds",
    "generation_cost_usd",
    "generation_seconds",
    "added_api_cost_usd",
    "added_generation_seconds",
    *METRICS.values(),
)
SCHEMA = pa.schema(
    [(c, pa.string()) for c in STRING_COLUMNS]
    + [(c, pa.int64()) for c in INT_COLUMNS]
    + [(c, pa.float64()) for c in FLOAT_COLUMNS]
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_frame(rows, output, name, *, csv=False):
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / f"{name}.parquet", index=False)
    if csv:
        frame.to_csv(output / f"{name}.csv", index=False)
    return frame


def validate_dataset(output):
    """Independently reconstruct the saved report aggregates from exported rows."""
    frame = pd.read_parquet(output / "per_query.parquet")
    metric_names = list(METRICS.values())
    keys = ["corpus", "model", "method", "output_language", "setting_id"]
    assert not frame.duplicated(keys + ["query_id"]).any()
    values = frame[metric_names].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    means = frame.groupby(keys, observed=True)[metric_names].mean().sort_index()
    reported = (
        pd.read_parquet(output / "reported_setting_metrics.parquet")
        .set_index(keys)[metric_names]
        .sort_index()
    )
    assert means.index.equals(reported.index)
    setting_error = float(np.abs(means.to_numpy() - reported.to_numpy()).max())
    assert setting_error < 1e-12
    setting_frame = pd.read_parquet(output / "reported_setting_metrics.parquet")
    macro_keys = ["corpus", "model", "method", "output_language", "regime"]
    macros = setting_frame.groupby(macro_keys, observed=True)[metric_names].mean().sort_index()
    reported_macros = (
        pd.read_parquet(output / "reported_macro_metrics.parquet")
        .set_index(macro_keys)[metric_names]
        .sort_index()
    )
    assert macros.index.equals(reported_macros.index)
    macro_error = float(np.abs(macros.to_numpy() - reported_macros.to_numpy()).max())
    assert macro_error < 1e-12
    mixed = frame[frame["regime"] == "mixed"]
    assert mixed["query_id"].nunique() == 638
    assert frame["required_source_count"].eq(1).all()
    source_hit_error = max(
        float(np.abs(frame[f"source_recall{k}"] - frame[f"hit{k}"]).max())
        for k in (1, 5, 10, 20, 100)
    )
    assert source_hit_error == 0
    generated = pd.read_parquet(output / "generation.parquet")
    identities = generated[
        (generated["method"] == "translate")
        & (generated["query_language"] == generated["output_language"])
    ]
    assert len(identities) == 638 and identities["status"].eq("identity").all()
    assert identities["text"].eq(identities["original_text"]).all()
    assert identities["estimated_cost_usd"].eq(0).all()
    result = {
        "status": "passed",
        "per_query_rows": len(frame),
        "unique_queries": int(frame["query_id"].nunique()),
        "paired_row_keys_unique": True,
        "metric_values_finite_and_in_unit_interval": True,
        "reconstructed_setting_rows": len(means),
        "max_setting_metric_error": setting_error,
        "reconstructed_macro_rows": len(macros),
        "max_macro_metric_error": macro_error,
        "source_recall_equals_hit_max_error": source_hit_error,
        "identity_translation_text_pairs_exact": len(identities),
        "identity_metric_tolerance": 1e-10,
        "identity_metric_exceptions": len(read_json(output / "identity_exceptions.json")),
        "identity_metric_note": "Saved model outputs exhibit a small number of identity-ranking differences. These are retained and explicitly audited, not silently corrected.",
        "request_keys_shared_across_distinct_query_records": int(
            generated["request_key"].dropna().duplicated().sum()
        ),
    }
    write_json(output / "validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output or repo / "report/acl_analysis_20261115/data"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = repo / "artifacts/fir-20261109-all-questions/artifacts"
    campaign = repo / "artifacts/narval-20261111-selected-document"
    remote_campaign = Path("/scratch/matroid/clir-benchmark/runs/narval-20261111-selected-document")
    remote_artifacts = Path(
        "/scratch/matroid/clir-benchmark/runs/fir-20261109-all-questions/artifacts"
    )

    def relocate(path):
        path = Path(path)
        for remote, local in ((remote_campaign, campaign), (remote_artifacts, artifacts)):
            if path.is_relative_to(remote):
                return local / path.relative_to(remote)
        raise ValueError(f"Unrecognized source artifact path: {path}")

    status_path = campaign / ".local-monitor/latest-attempts-final-status-normalized-20261115.json"
    status = read_json(status_path)
    assert status["all_succeeded"] and len(status["jobs"]) == 60
    reports = []
    source_files = {str(status_path.relative_to(repo)): digest(status_path)}
    for job in status["jobs"]:
        if job["kind"] != "variant-score":
            continue
        assert job["state"] == "COMPLETED"
        candidates = [
            (relocate(j["path"]), read_json(relocate(j["path"]))) for j in job["stage_journals"]
        ]
        journal_path, journal = max(
            candidates, key=lambda item: (item[1].get("started_at", ""), item[0].name)
        )
        assert journal["status"] == "success"
        source_files[str(journal_path.relative_to(repo))] = digest(journal_path)
        chosen = [s for s in journal["stages"] if s["stage"].startswith("evaluate-")]
        assert len(chosen) == 11 and all(s["status"] == "success" for s in chosen)
        reports.extend(relocate(s["result"]) for s in chosen)
    assert len(reports) == len(set(reports)) == 264

    # Query and relevance data are shared across every retriever and variant.
    queries = {}
    selections = {}
    settings_by_corpus = {}
    qrel_counts = {}
    selected_rows, selection_rows, qrel_rows, eligibility_rows, base_settings = [], [], [], [], []
    expected_queries = {"chemistry": 197, "eurlex": 194, "un": 247}
    for corpus in expected_queries:
        paths = [
            p
            for p in (artifacts / "relevance" / corpus).glob("*/manifest.json")
            if read_json(p).get("candidate_mode") == "best-per-document"
        ]
        assert len(paths) == 1, (corpus, paths)
        directory = paths[0].parent
        manifest = read_json(paths[0])
        assert manifest["retrieval_unit"] == "document" and not manifest["blocked_settings"]
        selection = read_json(directory / "selection.json")
        for group in selection:
            assert group["status"] == "selected" and len(group["selected_ids"]) == 1
            query_id = group["selected_ids"][0]
            assert query_id not in selections
            selections[query_id] = group
            selection_rows.append(
                {
                    "corpus": corpus,
                    "query_id": query_id,
                    "canonical_source_id": group["canonical_source_id"],
                    "candidate_count": group["candidate_count"],
                    "scored_candidate_count": group["scored_candidate_count"],
                    "excluded_missing_score_count": group["excluded_missing_score_count"],
                    "maximum_score": group["maximum_score"],
                    "maximum_tie_count": len(group["maximum_score_ids"]),
                    "preferred_tie_count": len(group["preferred_tied_ids"]),
                    "random_pool_count": len(group["random_pool_ids"]),
                    "selected_generator": group["selected_generator"],
                    "selected_language": group["selected_language"],
                    "generator_counts_json": json.dumps(group["generator_counts"], sort_keys=True),
                    "language_counts_json": json.dumps(group["language_counts"], sort_keys=True),
                    "selection_seed": group["selection_seed"],
                }
            )
        query_rows = pq.read_table(directory / "queries.parquet").to_pylist()
        assert len(query_rows) == expected_queries[corpus]
        for q in query_rows:
            assert q["id"] in selections and q["id"] not in queries
            q["query_id"] = q["id"]
            q["query_language"] = q["question_language"]
            q["generator"] = q["generator_model_id"]
            q["canonical_source_id"] = selections[q["id"]]["canonical_source_id"]
            q["question_characters"] = len(q["question"])
            q["question_whitespace_tokens"] = len(q["question"].split())
            queries[q["id"]] = q
            selected_rows.append(q)
        settings = read_json(directory / "settings.json")
        settings_by_corpus[corpus] = settings
        for sid, setting in sorted(settings.items()):
            target = setting["target_languages"]
            base_settings.append(
                {
                    "corpus": corpus,
                    "setting_id": sid,
                    "query_language": setting["query_language"],
                    "target_language": "mixed" if len(target) > 1 else target[0],
                    "target_languages_json": json.dumps(target),
                    "regime": setting["kind"],
                    "status": setting["status"],
                    "eligible_queries": setting["counts"].get("eligible", 0),
                    "dropped_queries": setting["counts"].get("dropped", 0),
                    "counts_json": json.dumps(setting["counts"], sort_keys=True),
                    "reasons_json": json.dumps(setting["reasons"], sort_keys=True),
                }
            )
        grouped_qrels = defaultdict(lambda: {"documents": set(), "sources": set()})
        for row in pq.read_table(directory / "qrels.parquet").to_pylist():
            qrel_rows.append({"corpus": corpus, **row})
            key = (row["setting_id"], row["query_id"])
            grouped_qrels[key]["documents"].add(row["document_id"])
            grouped_qrels[key]["sources"].add(row["required_source_id"])
        qrel_counts.update(
            {key: (len(v["documents"]), len(v["sources"])) for key, v in grouped_qrels.items()}
        )
        eligibility_rows.extend(
            {"corpus": corpus, **row}
            for row in pq.read_table(directory / "eligibility.parquet").to_pylist()
        )
        for filename in (
            "manifest.json",
            "queries.parquet",
            "selection.json",
            "settings.json",
            "qrels.parquet",
            "eligibility.parquet",
        ):
            source_files[str((directory / filename).relative_to(repo))] = digest(
                directory / filename
            )
    assert len(queries) == len(selections) == 638

    generation_cache = {}
    generation_rows = {}
    generation_catalog = []

    def generation(report, method, language):
        if method == "original":
            return {}, None
        corpus = report["corpus"]
        key = (corpus, method, language)
        manifest = report["run"]["translation" if method == "translate" else "hyde"]
        if key in generation_cache:
            cached, path = generation_cache[key]
            assert (
                generation_catalog[[r["key"] for r in generation_catalog].index(key)]["summary"]
                == manifest["summary"]
            )
            return cached, path
        tree = "translations" if method == "translate" else "hyde"
        fingerprint = "query_fingerprint" if method == "translate" else "passage_fingerprint"
        matches = []
        for p in (artifacts / tree / corpus / language).glob("*/manifest.json"):
            candidate = read_json(p)
            if (
                candidate.get(fingerprint) == manifest[fingerprint]
                and candidate.get("configuration") == manifest["configuration"]
                and candidate.get("cache_reuse_max_completion_tokens")
                == manifest.get("cache_reuse_max_completion_tokens")
                and candidate.get("summary") == manifest["summary"]
            ):
                matches.append(p)
        assert len(matches) == 1, (key, matches)
        path = matches[0].parent
        name = "queries.parquet" if method == "translate" else "passages.parquet"
        rows = pq.read_table(path / name).to_pylist()
        assert {r["id"] for r in rows} == {q["id"] for q in selected_rows if q["corpus"] == corpus}
        cached = {r["id"]: r for r in rows}
        generation_cache[key] = (cached, path)
        generation_catalog.append(
            {
                "key": key,
                "summary": manifest["summary"],
                "artifact_path": str(path.relative_to(repo)),
                "configuration": manifest["configuration"],
                "manifest_sha256": digest(path / "manifest.json"),
            }
        )
        source_files[str((path / "manifest.json").relative_to(repo))] = digest(
            path / "manifest.json"
        )
        source_files[str((path / name).relative_to(repo))] = digest(path / name)
        for row in rows:
            q = queries[row["id"]]
            generation_rows[(corpus, method, language, row["id"])] = {
                "corpus": corpus,
                "method": method,
                "output_language": language,
                "query_id": row["id"],
                "canonical_source_id": q["canonical_source_id"],
                "query_language": row["source_language"],
                "text": row["text"],
                "original_text": row["original_text"],
                "request_key": row["request_key"],
                "status": row.get("status", "generated"),
                "attempts": row["attempts"],
                "input_tokens": row["input_tokens"],
                "cached_input_tokens": row["cached_input_tokens"],
                "output_tokens": row["output_tokens"],
                "estimated_cost_usd": row["estimated_cost_usd"],
                "generation_seconds": row[
                    "translation_seconds" if method == "translate" else "generation_seconds"
                ],
                "generated_characters": len(row["text"]),
                "generated_whitespace_tokens": len(row["text"].split()),
                "artifact_path": str(path.relative_to(repo)),
                "generation_model": manifest["configuration"]["model"],
            }
        return cached, path

    catalogue, coverage, setting_metrics, regime_metrics, truncation = [], [], [], [], []
    config_keys = set()
    cohorts = {}
    seen_variants = defaultdict(set)
    dense_metrics = {}
    identity_candidates = []
    total_rows = 0
    writer = pq.ParquetWriter(output / "per_query.parquet", SCHEMA, compression="zstd")
    for directory in sorted(reports):
        report = read_json(directory / "report.json")
        assert report["retrieval_unit"] == "document"
        assert report["candidate_selection"]["candidate_mode"] == "best-per-document"
        method = METHODS[report["baseline"]]
        language = report.get("output_language") or report.get("translation_language") or "original"
        assert language == "original" if method == "original" else language in LANGUAGES
        corpus, model = report["corpus"], report["model"]
        config_key = (corpus, model, method, language)
        assert config_key not in config_keys
        config_keys.add(config_key)
        seen_variants[(corpus, model)].add((method, language))
        variant = "original" if method == "original" else f"{method}_{language}"
        rows = read_json(directory / "per_query.json")
        report_path = str(directory.relative_to(repo))
        gen, gen_path = generation(report, method, language)
        flat, key_set = [], set()
        grouped = Counter()
        for row in rows:
            q = queries[row["query_id"]]
            setting = report["settings"][row["setting_id"]]
            assert q["corpus"] == corpus and q["question_language"] == row["query_language"]
            key = (row["setting_id"], row["query_id"])
            assert key not in key_set
            key_set.add(key)
            grouped[row["setting_id"]] += 1
            targets = setting["target_languages"]
            relevant_count, required_count = qrel_counts[key]
            record = {
                **{c: row.get(c) for c in STRING_COLUMNS + INT_COLUMNS + FLOAT_COLUMNS},
                "corpus": corpus,
                "model": model,
                "method": method,
                "output_language": language,
                "variant": variant,
                "canonical_source_id": q["canonical_source_id"],
                "source_unit_id": q["source_unit_id"],
                "context_language": q["context_language"],
                "target_language": "mixed" if len(targets) > 1 else targets[0],
                "target_languages_json": json.dumps(targets),
                "total_score": q["total_score"],
                "question_characters": q["question_characters"],
                "question_whitespace_tokens": q["question_whitespace_tokens"],
                "relevant_document_count": relevant_count,
                "required_source_count": required_count,
                "report_id": directory.name,
                "report_path": report_path,
                **{new: row["metrics"][old] for old, new in METRICS.items()},
            }
            if method != "original":
                generated = gen[row["query_id"]]
                seconds_key = (
                    "translation_seconds" if method == "translate" else "generation_seconds"
                )
                record.update(
                    generation_model=report["run"][
                        "translation" if method == "translate" else "hyde"
                    ]["configuration"]["model"],
                    generation_request_key=generated["request_key"],
                    generation_artifact_path=str(gen_path.relative_to(repo)),
                    generation_attempts=generated["attempts"],
                    generation_input_tokens=generated["input_tokens"],
                    generation_cached_input_tokens=generated["cached_input_tokens"],
                    generation_output_tokens=generated["output_tokens"],
                    generated_characters=len(generated["text"]),
                    generated_whitespace_tokens=len(generated["text"].split()),
                    added_api_cost_usd=generated["estimated_cost_usd"],
                    added_generation_seconds=generated[seconds_key],
                )
            else:
                # This baseline explicitly makes no paid generation calls. Local
                # compute is unpriced, and no end-to-end timing is invented.
                record.update(added_api_cost_usd=0.0, added_generation_seconds=0.0)
            flat.append(record)
            pair_key = (corpus, model, *key)
            if method == "original":
                dense_metrics[pair_key] = {name: record[name] for name in METRICS.values()}
            elif method == "translate" and language == q["question_language"]:
                identity_candidates.append(
                    (pair_key, {name: record[name] for name in METRICS.values()})
                )
        cohorts[config_key] = key_set
        writer.write_table(pa.Table.from_pylist(flat, schema=SCHEMA))
        total_rows += len(flat)
        for sid, setting in sorted(report["settings"].items()):
            expected = setting["counts"].get("eligible", 0)
            assert grouped[sid] == expected, (config_key, sid, grouped[sid], expected)
            coverage.append(
                {
                    "corpus": corpus,
                    "model": model,
                    "method": method,
                    "output_language": language,
                    "variant": variant,
                    "setting_id": sid,
                    "regime": setting["kind"],
                    "query_language": setting["query_language"],
                    "target_language": "mixed"
                    if len(setting["target_languages"]) > 1
                    else setting["target_languages"][0],
                    "eligible_queries": expected,
                    "observed_queries": grouped[sid],
                    "dropped_queries": setting["counts"].get("dropped", 0),
                }
            )
        for metrics in report["metrics"]["setting"]:
            setting_metrics.append(
                {
                    "corpus": corpus,
                    "model": model,
                    "method": method,
                    "output_language": language,
                    "variant": variant,
                    **{k: v for k, v in metrics.items() if k != "metrics"},
                    **{new: metrics["metrics"][old] for old, new in METRICS.items()},
                }
            )
        for metrics in report["metrics"]["macro_by_regime"]:
            regime_metrics.append(
                {
                    "corpus": corpus,
                    "model": model,
                    "method": method,
                    "output_language": language,
                    "variant": variant,
                    "regime": metrics["regime"],
                    "setting_count": metrics["setting_count"],
                    **{new: metrics["metrics"][old] for old, new in METRICS.items()},
                }
            )
        for scope, values in report.get("truncation", {}).items():
            if not isinstance(values, dict):
                continue
            for lang, stats in values.items():
                if isinstance(stats, dict) and "inputs" in stats:
                    truncation.append(
                        {
                            "corpus": corpus,
                            "model": model,
                            "method": method,
                            "output_language": language,
                            "variant": variant,
                            "encoding_scope": scope,
                            "text_language": lang,
                            **stats,
                        }
                    )
        run = report["run"]
        specification = run["model"]
        summary = report.get("translation_summary") or report.get("hyde_summary") or {}
        catalogue.append(
            {
                "corpus": corpus,
                "model": model,
                "baseline": report["baseline"],
                "method": method,
                "output_language": language,
                "variant": variant,
                "report_id": directory.name,
                "report_path": report_path,
                "per_query_rows": len(rows),
                "unique_queries": len({row["query_id"] for row in rows}),
                "nonempty_settings": len(grouped),
                "retrieval_unit": report["retrieval_unit"],
                "relevance_hash": run["relevance_hash"],
                "query_fingerprint": run["query_fingerprint"],
                "corpus_fingerprint": run["corpus_fingerprint"],
                "similarity": run["similarity"],
                "top_k": run["top_k"],
                "model_repo": specification["repo_id"],
                "model_revision": specification["revision"],
                "model_dimension": specification["dimension"],
                "model_max_length": specification["max_length"],
                "model_specification_json": json.dumps(specification, sort_keys=True),
                "runtime_json": json.dumps(run["runtime"], sort_keys=True),
                "retrieval_run_seconds": run["seconds"],
                "generation_model": report.get("generation_model")
                or (run.get("translation") or {}).get("configuration", {}).get("model"),
                "generation_summary_json": json.dumps(summary, sort_keys=True),
                "generation_estimated_cost_usd": summary.get("estimated_cost_usd"),
                "generation_input_tokens": summary.get("input_tokens"),
                "generation_cached_input_tokens": summary.get("cached_input_tokens"),
                "generation_output_tokens": summary.get("output_tokens"),
                "generation_api_attempts": summary.get("api_attempts"),
                "generation_seconds": summary.get(
                    "generation_seconds", summary.get("translation_seconds")
                ),
                "timing_notes": report.get("timing_notes"),
                "generation_timing_notes": report.get("generation_timing_notes"),
                "timing_contract_json": json.dumps(run.get("timing_contract"), sort_keys=True),
                "report_sha256": digest(directory / "report.json"),
                "per_query_sha256": digest(directory / "per_query.json"),
            }
        )
        source_files[str((directory / "report.json").relative_to(repo))] = catalogue[-1][
            "report_sha256"
        ]
        source_files[str((directory / "per_query.json").relative_to(repo))] = catalogue[-1][
            "per_query_sha256"
        ]
    writer.close()

    # Match cohorts exactly; zero-query settings remain explicit in coverage.
    expected_variants = {("original", "original")} | {
        (method, lang) for method in ("translate", "hyde") for lang in LANGUAGES
    }
    exceptions = []
    for (corpus, model), variants in seen_variants.items():
        assert variants == expected_variants, (corpus, model, variants)
        reference = cohorts[(corpus, model, "original", "original")]
        for method, lang in variants:
            actual = cohorts[(corpus, model, method, lang)]
            if actual != reference:
                exceptions.append(
                    {
                        "corpus": corpus,
                        "model": model,
                        "method": method,
                        "output_language": lang,
                        "missing_keys": sorted(reference - actual),
                        "extra_keys": sorted(actual - reference),
                    }
                )
    for corpus in expected_queries:
        originals = [
            v for (c, _, method, _), v in cohorts.items() if c == corpus and method == "original"
        ]
        assert len(originals) == 8 and all(v == originals[0] for v in originals)
    write_json(output / "cohort_exceptions.json", exceptions)
    assert not exceptions
    assert len(config_keys) == 264 and len(seen_variants) == 24 and len(generation_rows) == 6380

    identity = []
    for key, translated in identity_candidates:
        original = dense_metrics[key]
        differences = {name: translated[name] - original[name] for name in METRICS.values()}
        identity.append(
            {
                "corpus": key[0],
                "model": key[1],
                "setting_id": key[2],
                "query_id": key[3],
                "query_language": queries[key[3]]["question_language"],
                "max_absolute_metric_delta": max(abs(v) for v in differences.values()),
                "matches_within_1e10": all(abs(v) <= 1e-10 for v in differences.values()),
                **{f"delta_{name}": value for name, value in differences.items()},
            }
        )

    save_frame(selected_rows, output, "selected_queries")
    save_frame(selection_rows, output, "selection_audit", csv=True)
    save_frame(qrel_rows, output, "qrels")
    save_frame(eligibility_rows, output, "eligibility")
    save_frame(base_settings, output, "settings", csv=True)
    save_frame(catalogue, output, "report_catalog", csv=True)
    save_frame(coverage, output, "cohort_coverage", csv=True)
    save_frame(setting_metrics, output, "reported_setting_metrics")
    save_frame(regime_metrics, output, "reported_macro_metrics", csv=True)
    save_frame(truncation, output, "truncation", csv=True)
    save_frame(list(generation_rows.values()), output, "generation")
    save_frame(identity, output, "identity_control", csv=True)
    write_json(output / "generation_catalog.json", generation_catalog)
    identity_failures = [r for r in identity if not r["matches_within_1e10"]]
    write_json(output / "identity_exceptions.json", identity_failures)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "script": str(Path(__file__).relative_to(repo)),
        "script_sha256": digest(__file__),
        "source_status": str(status_path.relative_to(repo)),
        "scope": "264 selected document-level reports only; excludes legacy chunk/all-question baselines",
        "reports": len(reports),
        "per_query_rows": total_rows,
        "selected_questions": len(queries),
        "questions_by_corpus": expected_queries,
        "variants_per_corpus_model": 11,
        "generation_query_language_records": len(generation_rows),
        "cohort_exceptions": len(exceptions),
        "identity_control_pairs": len(identity),
        "identity_control_exceptions": len(identity_failures),
        "identity_max_absolute_metric_delta": max(r["max_absolute_metric_delta"] for r in identity),
        "metric_aliases": METRICS,
        "per_query_schema": str(SCHEMA),
        "source_files_sha256": source_files,
        "analysis_notes": [
            "A query is repeated across models, variants and target settings; rows are not independent samples.",
            "Question/source-document IDs are the resampling clusters. All 638 winners remain represented in mixed retrieval.",
            "Regime is defined by original query language versus search collection language; generated output language does not redefine it.",
            "Per-report saved macro metrics give equal weight to eligible settings, not equal weight to questions.",
            "Empty target-language cohorts are recorded explicitly in settings and cohort_coverage, never scored as zero.",
            "generation.parquet counts each query/method/output once, independent of models/settings. Do not sum per-query-row costs to estimate spend.",
            "Costs are configured API estimates; local compute is unpriced. Recorded amortized timings are not measured end-to-end latency.",
            "Null measurements remain null. Original mode added API cost/time are known zero because that stage is disabled.",
            "Translation token/request fields are joined from the exact generation artifact matching report fingerprint/configuration/summary.",
            "Saved selection includes original evidence/answers for audit; these were not inputs to translation/HyDE generation.",
        ],
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__},
    }
    manifest["validation"] = validate_dataset(output)
    write_json(output / "dataset_manifest.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "reports",
                    "per_query_rows",
                    "selected_questions",
                    "generation_query_language_records",
                    "cohort_exceptions",
                    "identity_control_pairs",
                    "identity_control_exceptions",
                    "identity_max_absolute_metric_delta",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
