from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import pyarrow as pa

from .artifacts import ParquetSink, rows
from .config import Config, TranslationConfig, language_tag
from .errors import ValidationError
from .generation import (
    GENERATION_AUDIT_FIELDS,
    IDENTITY_AUDIT,
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
    output_model,
    parse_generated_text,
    planned_generation,
    provider_refused,
    response_text,
    verify_generation_inputs,
)
from .generation import estimated_cost as estimated_cost
from .generation import nullable_sum as nullable_sum
from .generation import token_usage as token_usage
from .util import digest, new_directory, read_json, write_json

TRANSLATION_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "id",
            "text",
            "language",
            "source_language",
            "original_text",
            "request_key",
            "status",
        )
    ]
    + [
        (name, pa.int64())
        for name in ("attempts", "input_tokens", "cached_input_tokens", "output_tokens")
    ]
    + [(name, pa.float64()) for name in ("translation_seconds", "estimated_cost_usd")]
)


def prompts(corpus: str) -> tuple[str, str]:
    if corpus not in {"chemistry", "eurlex", "un"}:
        raise ValidationError(f"Unknown translation domain: {corpus}")
    root = files("clir_benchmark").joinpath("prompts")
    domain = "chemistry" if corpus == "chemistry" else "law"
    return (
        root.joinpath(f"{domain}.txt").read_text(encoding="utf-8").rstrip("\n"),
        root.joinpath("query.txt").read_text(encoding="utf-8").rstrip("\n"),
    )


def translation_languages(config: Config, corpus: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(config.query_generation.corpus_languages.get(corpus, config.target_languages))
            or set(read_json(config.corpus_dir(corpus) / "manifest.json")["languages"])
        )
    )


def translation_inputs(config: Config, corpus: str) -> list[dict]:
    return generation_inputs(config, corpus, action="translating queries")


def request_payload(config: TranslationConfig, corpus: str, row: dict, target: str) -> dict:
    system, template = prompts(corpus)
    user = template.format(
        source_language=row["question_language"], target_language=target, query=row["question"]
    )
    return completion_payload(
        config, system, user, key="translation", schema_name="query_translation"
    )


def _translation_job(config: Config, corpus: str, target: str, shard=None):
    language_tag(target, "translation language")
    if target not in translation_languages(config, corpus):
        raise ValidationError(
            f"Translation language {target} is outside the configured target language set"
        )
    inputs = translation_inputs(config, corpus)
    translated_count = sum(
        row["question_language"].casefold() != target.casefold() for row in inputs
    )
    route = (
        output_model(config, target, config.translation) if translated_count else config.translation
    )
    provenance = generation_provenance(config, corpus, inputs, prompts(corpus))
    identity = {
        "schema": 1,
        "corpus": corpus,
        "target": target,
        "inputs": inputs,
        "prompts": prompts(corpus),
        "translation": asdict(route),
        **provenance,
        **(
            {
                "cache_reuse_max_completion_tokens": config.generation_cache_reuse.translation_max_completion_tokens
            }
            if config.generation_cache_reuse.translation_max_completion_tokens
            else {}
        ),
    }
    path, inputs, provenance = generation_partition(
        config.output_dir / "translations" / corpus / target / digest(identity),
        inputs,
        provenance,
        shard,
    )
    if shard is not None:
        translated_count = sum(
            row["question_language"].casefold() != target.casefold() for row in inputs
        )
    return path, inputs, route, provenance, translated_count


def translation_path(config: Config, corpus: str, target: str) -> Path:
    return _translation_job(config, corpus, target)[0]


def translation_plan(config: Config, corpus: str, target: str, *, shard=None) -> dict:
    path, inputs, route, provenance, paid = _translation_job(config, corpus, target, shard)
    return {
        "corpus": corpus,
        "artifact": str(path),
        **provenance,
        **planned_generation(route, target, paid, len(inputs) - paid),
    }


def load_translations(
    config: Config, corpus: str, target: str, *, shard=None
) -> tuple[Path, list[dict], dict]:
    path, inputs, _, provenance, _ = _translation_job(config, corpus, target, shard)
    return _load_translations(path, target, inputs, provenance)


def _load_translations(
    path: Path, target: str, inputs: list[dict], provenance: dict
) -> tuple[Path, list[dict], dict]:
    if not (path / "manifest.json").is_file():
        raise ValidationError(f"Translations are not prepared for {target}. Run translate first.")
    manifest = read_json(path / "manifest.json")
    records = list(rows(path / "queries.parquet"))
    fingerprint = digest([(row["id"], row["language"], row["text"]) for row in records])
    if fingerprint != manifest["query_fingerprint"] or len(records) != manifest["query_count"]:
        raise ValidationError(f"Corrupt translation artifact: {path}")
    if provenance:
        if any(manifest.get(key) != value for key, value in provenance.items()):
            raise ValidationError(
                f"Translation provenance disagrees with the pinned questions: {path}"
            )
        record_refusals = provenance.get("record_refusals", False)
        verify_generation_inputs(records, inputs, target, path, record_refusals=record_refusals)
        if record_refusals and manifest["summary"].get("provider_refused_count") != sum(
            row["status"] == "provider_refused" for row in records
        ):
            raise ValidationError(f"Translation refusal count disagrees with the records: {path}")
        for row in records:
            identity = row["source_language"].casefold() == target.casefold()
            if identity and (
                row["status"] != "identity"
                or row["text"] != row["original_text"]
                or row["attempts"]
                or row["request_key"] is not None
            ):
                raise ValidationError(
                    f"Same-language translation must preserve the original: {path}"
                )
            if not identity and (
                row["status"] not in {"translated", "provider_refused"} or not row["request_key"]
            ):
                raise ValidationError(f"Missing generated translation: {path}")
    return path, records, manifest


def parse_translation(content: str) -> str:
    return parse_generated_text(content, key="translation", label="Translation")


def response_translation(response: dict) -> str:
    return parse_translation(response_text(response, label="Translation"))


class ChatCompletionsTranslator(ChatCompletionsClient):
    label = "Translation"
    command = "translate"


def translation_summary(records: list[dict]) -> dict:
    return {
        "query_count": len(records),
        "translated_query_count": sum(row["status"] == "translated" for row in records),
        "identity_query_count": sum(row["status"] == "identity" for row in records),
        "provider_refused_count": sum(row["status"] == "provider_refused" for row in records),
        "api_attempts": sum(row["attempts"] for row in records),
        **{
            name: nullable_sum(row[name] for row in records)
            for name in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "estimated_cost_usd",
                "translation_seconds",
            )
        },
    }


def assemble_translations(config: Config, corpus: str, target: str, shard_size: int) -> Path:
    path, inputs, route, provenance, _ = _translation_job(config, corpus, target)
    schema = (
        pa.schema([*TRANSLATION_SCHEMA, *GENERATION_AUDIT_FIELDS])
        if provenance
        else TRANSLATION_SCHEMA
    )
    return assemble_generation_shards(
        path,
        inputs,
        provenance,
        shard_size,
        target,
        loader=_load_translations,
        configuration=asdict(route),
        schema=schema,
        filename="queries.parquet",
        fingerprint_key="query_fingerprint",
        summarize=translation_summary,
        seconds_key="translation_seconds",
    )


def prepare_translations(
    config: Config,
    corpus: str,
    target: str,
    *,
    translator_factory=ChatCompletionsTranslator,
    retry_failed: bool = False,
    retry_transient: bool = False,
    shard=None,
) -> Path:
    destination, inputs, route, provenance, _ = _translation_job(config, corpus, target, shard)
    cache = RequestCache(
        config.output_dir / "translation_requests",
        route,
        config.env_file,
        client_factory=translator_factory,
        label="Translation",
        command="translate",
        retry_transient=retry_transient,
        record_refusals=config.query_generation.record_refusals,
    )
    started = time.perf_counter()
    status = "interrupted"
    reused = destination.exists()
    try:
        if reused:
            _load_translations(destination, target, inputs, provenance)
        else:
            _prepare_translations(
                config,
                corpus,
                target,
                destination,
                inputs,
                provenance,
                cache,
                retry_failed=retry_failed,
            )
        status = "success"
        return destination
    except Exception:
        status = "failed"
        raise
    finally:
        if provenance:
            work = [
                {
                    **attempt_accounting(
                        {"attempts": [item["attempt"]]}, route, seconds_key="translation_seconds"
                    ),
                    **generation_audit({"attempts": [item["attempt"]]}, route),
                    "language": target,
                    "status": "provider_refused"
                    if provider_refused(item["attempt"].get("response"))
                    else item["attempt"]["status"],
                }
                for item in cache.new_attempts
            ]
            write_json(
                config.output_dir / "translation_invocations" / f"{uuid.uuid4().hex}.json",
                {
                    "schema": 1,
                    "corpus": corpus,
                    "translation_language": target,
                    "artifact": str(destination),
                    "artifact_reused": reused,
                    "configuration": asdict(route),
                    **provenance,
                    "status": status,
                    "new_api_attempts": len(work),
                    "retry_transient": retry_transient,
                    "new_work_by_language_and_model": generation_breakdown(
                        work, seconds_key="translation_seconds"
                    ),
                    "new_request_keys": [item["request_key"] for item in cache.new_attempts],
                    "cached_requests": cache.cache_hits,
                    "seconds": time.perf_counter() - started,
                },
            )


def _prepare_translations(
    config, corpus, target, destination, inputs, provenance, cache, *, retry_failed
):
    if not inputs:
        raise ValidationError("No eligible queries in ready settings to translate")
    route = cache.config
    records, new_records = [], []
    new_attempts = 0
    started = time.perf_counter()
    for row in inputs:
        record = {
            "id": row["id"],
            "language": target,
            "source_language": row["question_language"],
            "original_text": row["question"],
        }
        if row["question_language"].casefold() == target.casefold():
            records.append(
                {
                    **record,
                    "text": row["question"],
                    "request_key": None,
                    "status": "identity",
                    "attempts": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "translation_seconds": 0.0,
                    **(IDENTITY_AUDIT if provenance else {}),
                }
            )
            continue
        payload = request_payload(route, corpus, row, target)

        def key_for_payload(value, query_id=row["id"]):
            return digest(
                {
                    "schema": 1,
                    "query_id": query_id,
                    "endpoint": route.endpoint,
                    "payload": value,
                }
            )

        request_key = key_for_payload(payload)
        attempts_before = len(cache.new_attempts)
        state, is_new = cache.complete(
            request_key,
            payload,
            parse_response=response_translation,
            retry_failed=retry_failed,
            description=f"{row['id']} -> {target}",
            compatible_requests=completion_budget_candidates(
                payload,
                config.generation_cache_reuse.translation_max_completion_tokens,
                key_for_payload,
            ),
        )
        attempts_count = len(cache.new_attempts) - attempts_before
        new_attempts += attempts_count
        records.append(
            {
                **record,
                "text": state["text"],
                "request_key": state["request_key"],
                "status": "provider_refused"
                if state["status"] == "provider_refused"
                else "translated",
                **attempt_accounting(state, route, seconds_key="translation_seconds"),
                **(generation_audit(state, route) if provenance else {}),
            }
        )
        if is_new:
            new_records.append(
                {
                    **records[-1],
                    **attempt_accounting(
                        {"attempts": state["attempts"][-attempts_count:]},
                        route,
                        seconds_key="translation_seconds",
                    ),
                }
            )
    schema = (
        pa.schema([*TRANSLATION_SCHEMA, *GENERATION_AUDIT_FIELDS])
        if provenance
        else TRANSLATION_SCHEMA
    )
    with new_directory(destination) as temporary:
        with ParquetSink(temporary / "queries.parquet", schema) as sink:
            for record in records:
                sink.append(record)
        write_json(
            temporary / "manifest.json",
            {
                "schema": 1,
                "corpus": corpus,
                "translation_language": target,
                "configuration": asdict(route),
                **(
                    {
                        "cache_reuse_max_completion_tokens": config.generation_cache_reuse.translation_max_completion_tokens
                    }
                    if config.generation_cache_reuse.translation_max_completion_tokens
                    else {}
                ),
                "system_prompt": prompts(corpus)[0],
                "user_template": prompts(corpus)[1],
                "query_count": len(records),
                "query_fingerprint": digest(
                    [(row["id"], row["language"], row["text"]) for row in records]
                ),
                "summary": translation_summary(records),
                "new_api_attempts": new_attempts,
                "new_work": translation_summary(new_records),
                "cached_requests": cache.cache_hits,
                "seconds": time.perf_counter() - started,
                **provenance,
                **(
                    {
                        "by_language_and_model": generation_breakdown(
                            records, seconds_key="translation_seconds"
                        )
                    }
                    if provenance
                    else {}
                ),
            },
        )
    return destination
