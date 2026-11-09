"""Explicit, bounded integration runs; importing this module performs no work."""

from __future__ import annotations

import os
import platform
import re
import sys
import time
import traceback
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from .adore_pipeline import adore_run_path, run_adore
from .comparison import compare_adore, compare_hyde, compare_translations
from .config import BASELINES, Config, language_tag
from .errors import ValidationError
from .hyde import hyde_languages as default_hyde_languages
from .hyde import hyde_path, prepare_hyde
from .lexical import bm25_path, prepare_bm25
from .pipeline import (
    embed,
    embedding_path,
    evaluate,
    model_spec,
    prepare_corpus,
    prepare_relevance,
    retrieve,
    run_path,
)
from .rerank_pipeline import compare_rerank, rerank, rerank_run_path
from .translation import prepare_translations, translation_path
from .translation import translation_languages as default_translation_languages
from .util import read_json, versions, write_json


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _safe_error(value: str) -> str:
    # Provider/HTTP exceptions may contain headers or token-shaped strings. Never
    # persist credentials in the orchestration journal, even in a traceback.
    value = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", value)
    value = re.sub(r"\b(?:sk-|hf_)[A-Za-z0-9_-]+", "[REDACTED]", value)
    return re.sub(
        r"(?i)((?:api[_-]?key|authorization|access[_-]?token)\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        value,
    )


def _backend(baseline):
    return baseline.removeprefix("rerank-").removeprefix("translate-").removeprefix("adore-")


def run_smoke(
    config: Config,
    corpus: str,
    alias: str | None = None,
    *,
    translation_languages: tuple[str, ...] | None = None,
    hyde_languages: tuple[str, ...] | None = None,
    retry_failed: bool = False,
    max_queries: int = 5,
    max_sources: int = 40,
    max_chunks: int = 200,
) -> Path:
    """Run selected modes on a prepared local subset and journal every outcome.

    This is an execution command: it may download models and make paid generation
    calls. Required rerank parent runs are explicit dependencies. Other omitted
    controls are never executed merely to make a comparison available. Failures
    block dependent stages while independent baselines continue.
    """
    baselines = tuple(dict.fromkeys(config.baselines))
    if not baselines or set(baselines) - set(BASELINES):
        raise ValidationError("Smoke execution requires a nonempty valid baseline selection")
    if corpus not in config.datasets:
        raise ValidationError(f"Corpus {corpus} is not configured")
    for name, value in (
        ("max_queries", max_queries),
        ("max_sources", max_sources),
        ("max_chunks", max_chunks),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"{name} must be a positive integer")
    for language in (*(translation_languages or ()), *(hyde_languages or ())):
        language_tag(language)
    config = replace(config, baselines=baselines)
    destination = config.output_dir / "smoke_invocations" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    started = time.perf_counter()
    selected_and_parents = tuple(
        dict.fromkeys(
            (
                *baselines,
                *(
                    value.removeprefix("rerank-")
                    for value in baselines
                    if value.startswith("rerank-")
                ),
            )
        )
    )
    journal = {
        "schema": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "execution": "live",
        "corpus": corpus,
        "model": alias,
        "selected_baselines": list(baselines),
        "required_parent_baselines": [
            value for value in selected_and_parents if value not in baselines
        ],
        "configuration": _jsonable(asdict(config)),
        "limits": {
            "max_queries": max_queries,
            "max_sources": max_sources,
            "max_chunks": max_chunks,
        },
        "retry_failed": retry_failed,
        "versions": versions(),
        "hardware_and_runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "logical_cpu_count": os.cpu_count(),
            "configured_embedding_device": config.runtime.device,
            "configured_embedding_dtype": config.runtime.dtype,
            "configured_reranker_device": config.reranker.device
            if any(value.startswith("rerank-") for value in baselines)
            else None,
            "configured_reranker_dtype": config.reranker.dtype
            if any(value.startswith("rerank-") for value in baselines)
            else None,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "TOKENIZERS_PARALLELISM",
                )
            },
            "device_measurement": "Configured devices only; actual model device/dtype are retained in model artifact metadata. No accelerator/model library is imported for hardware inspection.",
        },
        "stages": [],
        "baselines": [],
        "measurement_methodology": {
            "stage_seconds": "Measured current invocation wall time, including cache validation; not historical model latency.",
            "original_costs_and_timings": "Retained in each linked stage artifact/report and request cache, including previous attempts.",
            "invocation_artifacts": "New per-stage invocation records written during this smoke invocation; shared stage execution is memoized.",
            "total_seconds": "Measured orchestration end-to-end wall time including preparation, loading, execution and reporting.",
            "purpose": "Functional smoke test on an intentionally restricted corpus; not a full benchmark quality estimate.",
        },
    }
    stages = {}
    baseline_records = {}
    invocation_names = (
        "translation_invocations",
        "hyde_invocations",
        "adore_invocations",
        "reranking_invocations",
    )
    prior_invocations = {
        str(path) for name in invocation_names for path in (config.output_dir / name).glob("*.json")
    }

    def save():
        journal["seconds"] = time.perf_counter() - started
        temporary = destination / ".status.json.tmp"
        write_json(temporary, journal)
        temporary.replace(destination / "status.json")

    def stage(key, operation, *, dependencies=(), expected=None):
        if key in stages:
            return stages[key]
        record = {
            "id": key,
            "dependencies": list(dependencies),
            "status": "pending",
            "seconds": 0.0,
        }
        stages[key] = record
        journal["stages"].append(record)
        unavailable = [
            dependency for dependency in dependencies if stages[dependency]["status"] != "success"
        ]
        if unavailable:
            record.update(status="blocked", blocked_by=unavailable)
            save()
            return record
        stage_started = time.perf_counter()
        record["status"] = "running"
        print(f"smoke: {key}", file=sys.stderr, flush=True)
        save()
        try:
            paths = expected() if callable(expected) else expected
            if paths is not None:
                paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
                record["artifacts_existed_before"] = {
                    str(path): Path(path).exists() for path in paths
                }
            record["result"] = _jsonable(operation())
            record["status"] = "success"
        except Exception as exc:
            record.update(
                status="failed",
                error_type=type(exc).__name__,
                error=_safe_error(str(exc)),
                traceback=_safe_error(traceback.format_exc()),
            )
        except BaseException as exc:
            record.update(status="interrupted", error_type=type(exc).__name__)
            raise
        finally:
            record["seconds"] = time.perf_counter() - stage_started
            save()
        return record

    def ensure_subset():
        dataset = config.datasets[corpus]
        if not dataset.qa.local_dir or not dataset.corpus.local_dir:
            raise ValidationError(
                "Smoke execution requires explicit local QA and corpus subsets; run prepare-subset first"
            )
        # Bound the source input before tokenizer/model initialization. Chemistry
        # pair rows can contain two sources, so the prepared count is checked too.
        from .adapters import DatasetReader

        reader = DatasetReader(dataset.corpus)
        for index, _ in enumerate(reader.read("default" if corpus == "chemistry" else "corpus"), 1):
            if index > max_sources:
                raise ValidationError(f"Smoke source input exceeds max_sources={max_sources}")
        return {"local_subset": True}

    def check_cohort():
        source_manifest = read_json(config.corpus_dir(corpus) / "manifest.json")
        relevance_manifest = read_json(config.relevance_dir(corpus) / "manifest.json")
        actual = {
            "queries": relevance_manifest["selected_query_count"],
            "sources": source_manifest["source_count"],
            "chunks": source_manifest["chunk_count"],
        }
        for name, limit in (
            ("queries", max_queries),
            ("sources", max_sources),
            ("chunks", max_chunks),
        ):
            if not 0 < actual[name] <= limit:
                raise ValidationError(f"Smoke {name} count {actual[name]} is outside 1..{limit}")
        if relevance_manifest["blocked_settings"]:
            raise ValidationError(
                "Smoke cohort contains blocked relevance settings; inspect eligibility diagnostics before execution"
            )
        settings = read_json(config.relevance_dir(corpus) / "settings.json")
        if not any(value["counts"].get("eligible", 0) for value in settings.values()):
            raise ValidationError("Smoke cohort has no eligible questions")
        journal["cohort"] = {
            **actual,
            "corpus_fingerprint": source_manifest["corpus_fingerprint"],
            "query_fingerprint": relevance_manifest["query_fingerprint"],
            "settings": settings,
        }
        return journal["cohort"]

    def languages_for(baseline):
        parent = baseline.removeprefix("rerank-")
        if parent.startswith("translate-"):
            values = translation_languages or config.target_languages
            if not values and stages.get("prepare-corpus", {}).get("status") == "success":
                values = default_translation_languages(config, corpus)
            return tuple(dict.fromkeys(values)) or (None,)
        if parent == "hyde-dense":
            values = hyde_languages or config.hyde.output_languages or config.target_languages
            if not values and stages.get("prepare-corpus", {}).get("status") == "success":
                values = default_hyde_languages(config, corpus)
            return tuple(dict.fromkeys(values)) or (None,)
        return (None,)

    def execute_baseline(baseline, language):
        key = f"{baseline}:{language or 'original'}"
        if key in baseline_records:
            return baseline_records[key]
        parent = baseline.removeprefix("rerank-")
        selected = baseline in baselines
        model = None if _backend(baseline) == "bm25" else alias
        kwargs = (
            {"translation_language": language}
            if parent.startswith("translate-")
            else {"hyde_language": language}
            if parent == "hyde-dense"
            else {}
        )
        record = {
            "baseline": baseline,
            "model": model,
            "language": language,
            "selected": selected,
            "role": "selected" if selected else "required_rerank_parent",
            "status": "pending",
            "stages": [],
            "comparison_status": "not_requested",
        }
        baseline_records[key] = record
        journal["baselines"].append(record)
        dependencies = ["check-cohort"]
        if baseline.startswith("rerank-"):
            execute_baseline(parent, language)
            dependencies = [f"evaluate:{parent}:{language or 'original'}"]
        else:
            if parent.startswith("translate-"):
                generation_key = f"translate:{language}"
                stage(
                    generation_key,
                    lambda: prepare_translations(
                        config, corpus, language, retry_failed=retry_failed
                    ),
                    dependencies=dependencies,
                    expected=lambda: translation_path(config, corpus, language),
                )
                dependencies.append(generation_key)
            elif parent == "hyde-dense":
                generation_key = f"generate-hyde:{language}"
                stage(
                    generation_key,
                    lambda: prepare_hyde(config, corpus, language, retry_failed=retry_failed),
                    dependencies=dependencies,
                    expected=lambda: hyde_path(config, corpus, language),
                )
                dependencies.append(generation_key)
            if _backend(parent) in {"bm25", "hybrid"}:
                stage(
                    "index-bm25",
                    lambda: prepare_bm25(config, corpus),
                    dependencies=("check-cohort",),
                    expected=lambda: bm25_path(config, corpus),
                )
                dependencies.append("index-bm25")
            if _backend(parent) != "bm25":

                def encode(kind, options):
                    if model is None:
                        raise ValidationError(
                            "Select an embedding model alias for dense/hybrid smoke baselines"
                        )
                    return embed(config, corpus, model, kind=kind, **options)

                stage(
                    "embed:document",
                    lambda: encode("document", {}),
                    dependencies=("check-cohort",),
                    expected=lambda: embedding_path(
                        config, corpus, model_spec(config, model), "document"
                    ),
                )
                dependencies.append("embed:document")
                if not parent.startswith("adore-"):
                    query_key = f"embed:query:{'translation' if parent.startswith('translate-') else 'hyde' if parent == 'hyde-dense' else 'original'}:{language or 'original'}"
                    stage(
                        query_key,
                        lambda: encode("query", kwargs),
                        dependencies=dependencies,
                        expected=lambda: embedding_path(
                            config, corpus, model_spec(config, model), "query", **kwargs
                        ),
                    )
                    dependencies.append(query_key)
        run_key = f"run:{key}"
        if baseline.startswith("rerank-"):

            def operation():
                return rerank(config, corpus, model, baseline=baseline, **kwargs)

            def expected():
                return rerank_run_path(
                    config,
                    corpus,
                    model_spec(config, model) if model else None,
                    baseline=baseline,
                    **kwargs,
                )
        elif parent.startswith("adore-"):

            def operation():
                return run_adore(
                    config, corpus, model, baseline=baseline, retry_failed=retry_failed
                )

            def expected():
                return adore_run_path(
                    config, corpus, model_spec(config, model) if model else None, baseline=baseline
                )
        else:

            def operation():
                return retrieve(config, corpus, model, baseline=baseline, **kwargs)

            def expected():
                return run_path(
                    config,
                    corpus,
                    model_spec(config, model) if model else None,
                    baseline=baseline,
                    **kwargs,
                )

        run_record = stage(run_key, operation, dependencies=dependencies, expected=expected)
        evaluation_key = f"evaluate:{key}"
        evaluation = stage(
            evaluation_key,
            lambda: evaluate(config, corpus, model, baseline=baseline, **kwargs),
            dependencies=(run_key,),
        )
        record.update(
            status=evaluation["status"],
            stages=[*dependencies, run_key, evaluation_key],
            run=run_record.get("result"),
            report=evaluation.get("result"),
        )
        if evaluation["status"] != "success":
            record["blocked_by"] = [
                name for name in record["stages"] if stages[name]["status"] != "success"
            ]
        save()
        return record

    try:
        stage("check-local-subset", ensure_subset)
        stage(
            "prepare-corpus",
            lambda: prepare_corpus(config, corpus),
            dependencies=("check-local-subset",),
            expected=lambda: config.corpus_dir(corpus),
        )
        stage(
            "prepare-relevance",
            lambda: prepare_relevance(config, corpus),
            dependencies=("prepare-corpus",),
            expected=lambda: config.relevance_dir(corpus),
        )
        stage("check-cohort", check_cohort, dependencies=("prepare-relevance",))
        for baseline in selected_and_parents:
            for language in languages_for(baseline):
                execute_baseline(baseline, language)
        # Compare only modes executed above, without activating omitted controls.
        for record in journal["baselines"]:
            baseline, language, model = record["baseline"], record["language"], record["model"]
            control = None
            if baseline.startswith("rerank-"):
                control = baseline.removeprefix("rerank-")
                parent = control
                kwargs = (
                    {"translation_language": language}
                    if parent.startswith("translate-")
                    else {"hyde_language": language}
                    if parent == "hyde-dense"
                    else {}
                )

                def compare(b=baseline, m=model, kw=kwargs):
                    return compare_rerank(config, corpus, m, baseline=b, **kw)

                control_language = language
            elif baseline.startswith("translate-"):
                control = baseline.removeprefix("translate-")

                def compare(b=control, m=model, lang=language):
                    return compare_translations(config, corpus, m, baseline=b, languages=(lang,))

                control_language = None
            elif baseline == "hyde-dense":
                control = "dense"

                def compare(m=model, lang=language):
                    return compare_hyde(config, corpus, m, languages=(lang,), controls=("dense",))

                control_language = None
            elif baseline.startswith("adore-"):
                control = baseline.removeprefix("adore-")

                def compare(b=baseline, m=model):
                    return compare_adore(config, corpus, m, baseline=b)

                control_language = None
            if control is None:
                continue
            control_key = f"{control}:{control_language or 'original'}"
            if control_key not in baseline_records:
                record["comparison_reason"] = (
                    "Control was not selected or required; it was not executed"
                )
                continue
            comparison = stage(
                f"compare:{baseline}:{language or 'original'}",
                compare,
                dependencies=(
                    f"evaluate:{baseline}:{language or 'original'}",
                    f"evaluate:{control_key}",
                ),
            )
            record.update(
                comparison_status=comparison["status"], comparison=comparison.get("result")
            )
        successful = [
            record
            for record in journal["baselines"]
            if record["selected"] and record["status"] == "success"
        ]
        all_selected = [record for record in journal["baselines"] if record["selected"]]
        journal["status"] = (
            "success"
            if len(successful) == len(all_selected)
            and not any(stage["status"] == "failed" for stage in journal["stages"])
            else "partial"
            if successful
            else "failed"
        )
    except BaseException:
        journal["status"] = "interrupted"
        raise
    finally:
        journal["finished_at"] = datetime.now(UTC).isoformat()
        journal["invocation_artifacts"] = sorted(
            str(path)
            for name in invocation_names
            for path in (config.output_dir / name).glob("*.json")
            if str(path) not in prior_invocations
        )
        save()
    return destination
