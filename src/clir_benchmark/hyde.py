from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import pyarrow as pa

from .artifacts import ParquetSink, rows
from .config import QUESTION_LANGUAGE, Config, HyDEConfig, language_tag
from .errors import ValidationError
from .generation import (
    GENERATION_AUDIT_FIELDS,
    ChatCompletionsClient,
    RequestCache,
    assemble_generation_shards,
    attempt_accounting,
    completion_budget_candidates,
    completion_payload,
    generation_audit,
    generation_breakdown,
    generation_inputs,
    generation_partition,
    generation_provenance,
    nullable_sum,
    output_model,
    parse_generated_text,
    planned_generation,
    provider_refused,
    response_text,
    verify_generation_inputs,
)
from .util import digest, new_directory, read_json, write_json

HYDE_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "id",
            "text",
            "language",
            "source_language",
            "original_text",
            "request_key",
        )
    ]
    + [
        (name, pa.int64())
        for name in ("attempts", "input_tokens", "cached_input_tokens", "output_tokens")
    ]
    + [(name, pa.float64()) for name in ("generation_seconds", "estimated_cost_usd")]
)


def output_schema(audit, record_refusals):
    fields = list(HYDE_SCHEMA)
    if audit:
        fields.extend(GENERATION_AUDIT_FIELDS)
    if record_refusals:
        fields.append(("status", pa.string()))
    return pa.schema(fields)


def prompts(corpus: str) -> tuple[str, str]:
    if corpus not in {"chemistry", "eurlex", "un"}:
        raise ValidationError(f"Unknown HyDE collection: {corpus}")
    root = files("clir_benchmark").joinpath("prompts")
    domain = "chemistry" if corpus == "chemistry" else "law"
    return (
        root.joinpath(f"hyde_{domain}.txt").read_text(encoding="utf-8").rstrip("\n"),
        root.joinpath("hyde_query.txt").read_text(encoding="utf-8").rstrip("\n"),
    )


def hyde_languages(config: Config, corpus: str) -> tuple[str, ...]:
    if config.query_generation.hyde_language_policy == QUESTION_LANGUAGE:
        return (QUESTION_LANGUAGE,)
    languages = config.hyde.output_languages or config.query_generation.corpus_languages.get(
        corpus, config.target_languages
    )
    if not languages:
        languages = read_json(config.corpus_dir(corpus) / "manifest.json")["languages"]
    return tuple(sorted(set(languages)))


def request_payload(config: HyDEConfig, corpus: str, row: dict, output_language: str) -> dict:
    language_tag(output_language, "HyDE output language")
    system, template = prompts(corpus)
    user = template.format(
        collection=corpus,
        source_language=row["question_language"],
        output_language=output_language,
        query=row["question"],
    )
    return completion_payload(
        config, system, user, key="hypothetical_document", schema_name="hypothetical_document"
    )


def hyde_inputs(config: Config, corpus: str) -> list[dict]:
    return generation_inputs(config, corpus, action="generating HyDE passages")


def _hyde_job(config: Config, corpus: str, output_language: str, shard=None):
    native = config.query_generation.hyde_language_policy == QUESTION_LANGUAGE
    if native:
        if output_language != QUESTION_LANGUAGE:
            raise ValidationError("Select question-language for native-language HyDE")
    else:
        language_tag(output_language, "HyDE output language")
    inputs = hyde_inputs(config, corpus)
    languages = (
        sorted({row["question_language"] for row in inputs}) if native else [output_language]
    )
    routes = {language: output_model(config, language, config.hyde) for language in languages}
    if native:
        resolved = {
            **asdict(config.hyde),
            "model": None,
            "language_policy": QUESTION_LANGUAGE,
            "models_by_output_language": {key: asdict(route) for key, route in routes.items()},
        }
    else:
        resolved = asdict(routes[output_language])
    provenance = generation_provenance(config, corpus, inputs, prompts(corpus))
    identity = {
        "schema": 1,
        "corpus": corpus,
        "output_language": output_language,
        "inputs": inputs,
        "prompts": prompts(corpus),
        "hyde": resolved,
        **provenance,
        **(
            {
                "cache_reuse_max_completion_tokens": config.generation_cache_reuse.hyde_max_completion_tokens
            }
            if config.generation_cache_reuse.hyde_max_completion_tokens
            else {}
        ),
    }
    path, inputs, provenance = generation_partition(
        config.output_dir / "hyde" / corpus / output_language / digest(identity),
        inputs,
        provenance,
        shard,
    )
    return path, inputs, routes, resolved, provenance


def hyde_path(config: Config, corpus: str, output_language: str) -> Path:
    return _hyde_job(config, corpus, output_language)[0]


def hyde_plan(config: Config, corpus: str, output_language: str, *, shard=None) -> dict:
    path, inputs, routes, _, provenance = _hyde_job(config, corpus, output_language, shard)
    counts = Counter(
        row["question_language"] if output_language == QUESTION_LANGUAGE else output_language
        for row in inputs
    )
    return {
        "corpus": corpus,
        "artifact": str(path),
        "output_language_policy": output_language,
        **provenance,
        "outputs": [
            planned_generation(routes[language], language, count)
            for language, count in sorted(counts.items())
        ],
    }


def passage_fingerprint(records: list[dict]) -> str:
    return digest([(row["id"], row["language"], row["text"]) for row in records])


def load_hyde(
    config: Config, corpus: str, output_language: str, *, shard=None
) -> tuple[Path, list[dict], dict]:
    path, inputs, _, _, provenance = _hyde_job(config, corpus, output_language, shard)
    return _load_hyde(path, output_language, inputs, provenance)


def _load_hyde(
    path: Path, output_language: str, inputs: list[dict], provenance: dict
) -> tuple[Path, list[dict], dict]:
    if not (path / "manifest.json").is_file():
        raise ValidationError(
            f"HyDE passages are not prepared for {output_language}. Run generate-hyde first."
        )
    manifest = read_json(path / "manifest.json")
    records = list(rows(path / "passages.parquet"))
    if (
        passage_fingerprint(records) != manifest["passage_fingerprint"]
        or len(records) != manifest["query_count"]
        or len({row["id"] for row in records}) != len(records)
    ):
        raise ValidationError(f"Corrupt HyDE artifact: {path}")
    if provenance or output_language == QUESTION_LANGUAGE:
        if any(manifest.get(key) != value for key, value in provenance.items()):
            raise ValidationError(f"HyDE provenance disagrees with the pinned questions: {path}")
        record_refusals = provenance.get("record_refusals", False)
        verify_generation_inputs(
            records,
            inputs,
            None if output_language == QUESTION_LANGUAGE else output_language,
            path,
            record_refusals=record_refusals,
        )
        if record_refusals and (
            any(row.get("status") not in {"generated", "provider_refused"} for row in records)
            or manifest["summary"].get("provider_refused_count")
            != sum(row["status"] == "provider_refused" for row in records)
        ):
            raise ValidationError(f"HyDE outcome counts disagree with the records: {path}")
    return path, records, manifest


def parse_hypothetical_document(content: str) -> str:
    return parse_generated_text(content, key="hypothetical_document", label="HyDE")


def response_hypothetical_document(response: dict) -> str:
    return parse_hypothetical_document(response_text(response, label="HyDE"))


class ChatCompletionsHyDE(ChatCompletionsClient):
    label = "HyDE"
    command = "generate-hyde"


def hyde_summary(records: list[dict]) -> dict:
    return {
        "query_count": len(records),
        "provider_refused_count": sum(row.get("status") == "provider_refused" for row in records),
        "api_attempts": sum(row["attempts"] for row in records),
        **{
            name: nullable_sum(row[name] for row in records)
            for name in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "estimated_cost_usd",
                "generation_seconds",
            )
        },
    }


def prepare_hyde(
    config: Config,
    corpus: str,
    output_language: str,
    *,
    generator_factory=ChatCompletionsHyDE,
    retry_failed: bool = False,
    retry_transient: bool = False,
    shard=None,
) -> Path:
    destination, inputs, routes, resolved, provenance = _hyde_job(
        config, corpus, output_language, shard
    )
    caches = {
        language: RequestCache(
            config.output_dir / "hyde_requests",
            route,
            config.env_file,
            client_factory=generator_factory,
            label="HyDE",
            command="generate-hyde",
            retry_transient=retry_transient,
            record_refusals=config.query_generation.record_refusals,
        )
        for language, route in routes.items()
    }
    audit = bool(provenance) or output_language == QUESTION_LANGUAGE
    invocation = {
        "schema": 1,
        "corpus": corpus,
        "output_language": output_language,
        "artifact": str(destination),
        "artifact_reused": destination.exists(),
        "retry_transient": retry_transient,
        "configuration": resolved,
        **provenance,
        **(
            {
                "cache_reuse_max_completion_tokens": config.generation_cache_reuse.hyde_max_completion_tokens
            }
            if config.generation_cache_reuse.hyde_max_completion_tokens
            else {}
        ),
        "status": "interrupted",
    }
    started = time.perf_counter()
    try:
        if invocation["artifact_reused"]:
            _load_hyde(destination, output_language, inputs, provenance)
        else:
            _prepare_hyde(
                config,
                corpus,
                output_language,
                destination,
                inputs,
                resolved,
                provenance,
                caches,
                audit,
                retry_failed=retry_failed,
            )
        invocation["status"] = "success"
        return destination
    except Exception:
        invocation["status"] = "failed"
        raise
    finally:
        work = []
        keys = []
        for language, cache in caches.items():
            for item in cache.new_attempts:
                attempt = {"attempts": [item["attempt"]]}
                work.append(
                    {
                        **attempt_accounting(
                            attempt, cache.config, seconds_key="generation_seconds"
                        ),
                        **(generation_audit(attempt, cache.config) if audit else {}),
                        "language": language,
                        "status": "provider_refused"
                        if provider_refused(item["attempt"].get("response"))
                        else item["attempt"]["status"],
                    }
                )
                keys.append(item["request_key"])
        invocation.update(
            new_work=hyde_summary(work),
            cached_requests=sum(cache.cache_hits for cache in caches.values()),
            new_request_keys=keys,
            seconds=time.perf_counter() - started,
        )
        if audit:
            invocation["new_work_by_language_and_model"] = generation_breakdown(
                work, seconds_key="generation_seconds"
            )
        write_json(config.output_dir / "hyde_invocations" / f"{uuid.uuid4().hex}.json", invocation)


def assemble_hyde(config: Config, corpus: str, output_language: str, shard_size: int) -> Path:
    path, inputs, _, resolved, provenance = _hyde_job(config, corpus, output_language)
    audit = bool(provenance) or output_language == QUESTION_LANGUAGE
    schema = output_schema(audit, config.query_generation.record_refusals)
    return assemble_generation_shards(
        path,
        inputs,
        provenance,
        shard_size,
        output_language,
        loader=_load_hyde,
        configuration=resolved,
        schema=schema,
        filename="passages.parquet",
        fingerprint_key="passage_fingerprint",
        summarize=hyde_summary,
        seconds_key="generation_seconds",
    )


def _prepare_hyde(
    config,
    corpus,
    output_language,
    destination,
    inputs,
    resolved,
    provenance,
    caches,
    audit,
    *,
    retry_failed,
) -> Path:
    if not inputs:
        raise ValidationError("No eligible queries in ready settings to generate HyDE passages")
    started = time.perf_counter()
    records, new_records = [], []
    for row in inputs:
        language = (
            row["question_language"] if output_language == QUESTION_LANGUAGE else output_language
        )
        cache = caches[language]
        route = cache.config
        payload = request_payload(route, corpus, row, language)

        def key_for_payload(value, query_id=row["id"], endpoint=route.endpoint):
            return digest(
                {
                    "schema": 1,
                    "corpus": corpus,
                    "query_id": query_id,
                    "endpoint": endpoint,
                    "payload": value,
                }
            )

        request_key = key_for_payload(payload)
        attempts_before = len(cache.new_attempts)
        state, is_new = cache.complete(
            request_key,
            payload,
            parse_response=response_hypothetical_document,
            retry_failed=retry_failed,
            description=f"{row['id']} -> {language}",
            compatible_requests=completion_budget_candidates(
                payload,
                config.generation_cache_reuse.hyde_max_completion_tokens,
                key_for_payload,
            ),
        )
        record = {
            "id": row["id"],
            "text": state["text"],
            "language": language,
            "source_language": row["question_language"],
            "original_text": row["question"],
            "request_key": state["request_key"],
            "status": "provider_refused" if state["status"] == "provider_refused" else "generated",
            **attempt_accounting(state, route, seconds_key="generation_seconds"),
            **(generation_audit(state, route) if audit else {}),
        }
        records.append(record)
        if is_new:
            attempts_count = len(cache.new_attempts) - attempts_before
            new_records.append(
                {
                    **record,
                    **attempt_accounting(
                        {"attempts": state["attempts"][-attempts_count:]},
                        route,
                        seconds_key="generation_seconds",
                    ),
                }
            )
    schema = output_schema(audit, config.query_generation.record_refusals)
    with new_directory(destination) as temporary:
        with ParquetSink(temporary / "passages.parquet", schema) as sink:
            for record in records:
                sink.append(record)
        write_json(
            temporary / "manifest.json",
            {
                "schema": 1,
                "corpus": corpus,
                "output_language": output_language,
                "configuration": resolved,
                **(
                    {
                        "cache_reuse_max_completion_tokens": config.generation_cache_reuse.hyde_max_completion_tokens
                    }
                    if config.generation_cache_reuse.hyde_max_completion_tokens
                    else {}
                ),
                "system_prompt": prompts(corpus)[0],
                "user_template": prompts(corpus)[1],
                "query_count": len(records),
                "passage_fingerprint": passage_fingerprint(records),
                "summary": hyde_summary(records),
                "new_api_attempts": sum(row["attempts"] for row in new_records),
                "new_work": hyde_summary(new_records),
                "cached_requests": len(records) - len(new_records),
                "seconds": time.perf_counter() - started,
                **provenance,
                **(
                    {
                        "by_language_and_model": generation_breakdown(
                            records, seconds_key="generation_seconds"
                        )
                    }
                    if audit
                    else {}
                ),
            },
        )
    return destination
