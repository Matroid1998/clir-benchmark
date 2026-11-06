from __future__ import annotations

import time
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import pyarrow as pa

from .artifacts import ParquetSink, rows
from .config import Config, TranslationConfig, language_tag
from .errors import ValidationError
from .generation import (
    ChatCompletionsClient,
    RequestCache,
    attempt_accounting,
    completion_payload,
    eligible_inputs,
    parse_generated_text,
    response_text,
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
            set(config.target_languages)
            or set(read_json(config.corpus_dir(corpus) / "manifest.json")["languages"])
        )
    )


def translation_inputs(config: Config, corpus: str) -> list[dict]:
    return eligible_inputs(config, corpus, action="translating queries")


def request_payload(config: TranslationConfig, corpus: str, row: dict, target: str) -> dict:
    system, template = prompts(corpus)
    user = template.format(
        source_language=row["question_language"], target_language=target, query=row["question"]
    )
    return completion_payload(
        config, system, user, key="translation", schema_name="query_translation"
    )


def translation_path(config: Config, corpus: str, target: str) -> Path:
    language_tag(target, "translation language")
    if target not in translation_languages(config, corpus):
        raise ValidationError(
            f"Translation language {target} is outside the configured target language set"
        )
    identity = {
        "schema": 1,
        "corpus": corpus,
        "target": target,
        "inputs": translation_inputs(config, corpus),
        "prompts": prompts(corpus),
        "translation": asdict(config.translation),
    }
    return config.output_dir / "translations" / corpus / target / digest(identity)


def load_translations(config: Config, corpus: str, target: str) -> tuple[Path, list[dict], dict]:
    path = translation_path(config, corpus, target)
    if not (path / "manifest.json").is_file():
        raise ValidationError(f"Translations are not prepared for {target}. Run translate first.")
    manifest = read_json(path / "manifest.json")
    records = list(rows(path / "queries.parquet"))
    fingerprint = digest([(row["id"], row["language"], row["text"]) for row in records])
    if fingerprint != manifest["query_fingerprint"] or len(records) != manifest["query_count"]:
        raise ValidationError(f"Corrupt translation artifact: {path}")
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


def prepare_translations(
    config: Config,
    corpus: str,
    target: str,
    *,
    translator_factory=ChatCompletionsTranslator,
    retry_failed: bool = False,
) -> Path:
    destination = translation_path(config, corpus, target)
    if destination.exists():
        load_translations(config, corpus, target)
        return destination
    inputs = translation_inputs(config, corpus)
    if not inputs:
        raise ValidationError("No eligible queries in ready settings to translate")
    records = []
    new_attempts = 0
    started = time.perf_counter()
    cache_root = config.output_dir / "translation_requests"
    cache = RequestCache(
        cache_root,
        config.translation,
        config.env_file,
        client_factory=translator_factory,
        label="Translation",
        command="translate",
    )
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
                }
            )
            continue
        payload = request_payload(config.translation, corpus, row, target)
        request_key = digest(
            {
                "schema": 1,
                "query_id": row["id"],
                "endpoint": config.translation.endpoint,
                "payload": payload,
            }
        )
        state, is_new = cache.complete(
            request_key,
            payload,
            parse_response=response_translation,
            retry_failed=retry_failed,
            description=f"{row['id']} -> {target}",
        )
        new_attempts += int(is_new)
        records.append(
            {
                **record,
                "text": state["text"],
                "request_key": request_key,
                "status": "translated",
                **attempt_accounting(state, config.translation, seconds_key="translation_seconds"),
            }
        )
    with new_directory(destination) as temporary:
        with ParquetSink(temporary / "queries.parquet", TRANSLATION_SCHEMA) as sink:
            for record in records:
                sink.append(record)
        write_json(
            temporary / "manifest.json",
            {
                "schema": 1,
                "corpus": corpus,
                "translation_language": target,
                "configuration": asdict(config.translation),
                "system_prompt": prompts(corpus)[0],
                "user_template": prompts(corpus)[1],
                "query_count": len(records),
                "query_fingerprint": digest(
                    [(row["id"], row["language"], row["text"]) for row in records]
                ),
                "summary": translation_summary(records),
                "new_api_attempts": new_attempts,
                "seconds": time.perf_counter() - started,
            },
        )
    return destination
