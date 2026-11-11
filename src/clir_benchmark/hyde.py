from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import pyarrow as pa

from .artifacts import ParquetSink, rows
from .config import Config, HyDEConfig, language_tag
from .errors import ValidationError
from .generation import (
    ChatCompletionsClient,
    RequestCache,
    attempt_accounting,
    completion_budget_candidates,
    completion_payload,
    eligible_inputs,
    nullable_sum,
    parse_generated_text,
    response_text,
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
    languages = config.hyde.output_languages or config.target_languages
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
    return eligible_inputs(config, corpus, action="generating HyDE passages")


def hyde_path(config: Config, corpus: str, output_language: str) -> Path:
    language_tag(output_language, "HyDE output language")
    identity = {
        "schema": 1,
        "corpus": corpus,
        "output_language": output_language,
        "inputs": hyde_inputs(config, corpus),
        "prompts": prompts(corpus),
        "hyde": asdict(config.hyde),
        **(
            {
                "cache_reuse_max_completion_tokens": config.generation_cache_reuse.hyde_max_completion_tokens
            }
            if config.generation_cache_reuse.hyde_max_completion_tokens
            else {}
        ),
    }
    return config.output_dir / "hyde" / corpus / output_language / digest(identity)


def passage_fingerprint(records: list[dict]) -> str:
    return digest([(row["id"], row["language"], row["text"]) for row in records])


def load_hyde(config: Config, corpus: str, output_language: str) -> tuple[Path, list[dict], dict]:
    path = hyde_path(config, corpus, output_language)
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
) -> Path:
    destination = hyde_path(config, corpus, output_language)
    cache = RequestCache(
        config.output_dir / "hyde_requests",
        config.hyde,
        config.env_file,
        client_factory=generator_factory,
        label="HyDE",
        command="generate-hyde",
    )
    invocation = {
        "schema": 1,
        "corpus": corpus,
        "output_language": output_language,
        "artifact": str(destination),
        "artifact_reused": destination.exists(),
        "configuration": asdict(config.hyde),
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
        result = _prepare_hyde(
            config, corpus, output_language, destination, cache, retry_failed=retry_failed
        )
        invocation["status"] = "success"
        return result
    except Exception:
        invocation["status"] = "failed"
        raise
    finally:
        work = [
            attempt_accounting(
                {"attempts": [item["attempt"]]}, config.hyde, seconds_key="generation_seconds"
            )
            for item in cache.new_attempts
        ]
        invocation.update(
            new_work=hyde_summary(work),
            cached_requests=cache.cache_hits,
            new_request_keys=[item["request_key"] for item in cache.new_attempts],
            seconds=time.perf_counter() - started,
        )
        write_json(config.output_dir / "hyde_invocations" / f"{uuid.uuid4().hex}.json", invocation)


def _prepare_hyde(
    config: Config,
    corpus: str,
    output_language: str,
    destination: Path,
    cache: RequestCache,
    *,
    retry_failed: bool,
) -> Path:
    if destination.exists():
        load_hyde(config, corpus, output_language)
        return destination
    inputs = hyde_inputs(config, corpus)
    if not inputs:
        raise ValidationError("No eligible queries in ready settings to generate HyDE passages")
    started = time.perf_counter()
    records, new_records = [], []
    for row in inputs:
        payload = request_payload(config.hyde, corpus, row, output_language)

        def key_for_payload(value, query_id=row["id"]):
            return digest(
                {
                    "schema": 1,
                    "corpus": corpus,
                    "query_id": query_id,
                    "endpoint": config.hyde.endpoint,
                    "payload": value,
                }
            )

        request_key = key_for_payload(payload)
        state, is_new = cache.complete(
            request_key,
            payload,
            parse_response=response_hypothetical_document,
            retry_failed=retry_failed,
            description=f"{row['id']} -> {output_language}",
            compatible_requests=completion_budget_candidates(
                payload,
                config.generation_cache_reuse.hyde_max_completion_tokens,
                key_for_payload,
            ),
        )
        record = {
            "id": row["id"],
            "text": state["text"],
            "language": output_language,
            "source_language": row["question_language"],
            "original_text": row["question"],
            "request_key": state["request_key"],
            **attempt_accounting(state, config.hyde, seconds_key="generation_seconds"),
        }
        records.append(record)
        if is_new:
            # The last attempt was the only call made for this query in this invocation.
            new_records.append(
                {
                    **record,
                    **attempt_accounting(
                        {"attempts": state["attempts"][-1:]},
                        config.hyde,
                        seconds_key="generation_seconds",
                    ),
                }
            )
    with new_directory(destination) as temporary:
        with ParquetSink(temporary / "passages.parquet", HYDE_SCHEMA) as sink:
            for record in records:
                sink.append(record)
        write_json(
            temporary / "manifest.json",
            {
                "schema": 1,
                "corpus": corpus,
                "output_language": output_language,
                "configuration": asdict(config.hyde),
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
                "new_api_attempts": len(new_records),
                "new_work": hyde_summary(new_records),
                "cached_requests": len(records) - len(new_records),
                "seconds": time.perf_counter() - started,
            },
        )
    return destination
