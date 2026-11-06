from __future__ import annotations

import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import pyarrow as pa

from .artifacts import ParquetSink, rows
from .config import Config, TranslationConfig, language_tag
from .errors import ValidationError
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
    directory = config.relevance_dir(corpus)
    if not (directory / "manifest.json").is_file():
        raise ValidationError("Run prepare-relevance before translating queries")
    settings = read_json(directory / "settings.json")
    eligible = {
        row["query_id"]
        for row in rows(directory / "eligibility.parquet")
        if row["status"] == "eligible" and settings[row["setting_id"]]["status"] == "ready"
    }
    # Deliberately read only query text and language: never answers, evidence or documents.
    return sorted(
        (
            row
            for row in rows(directory / "queries.parquet", ["id", "question", "question_language"])
            if row["id"] in eligible
        ),
        key=lambda row: row["id"],
    )


def request_payload(config: TranslationConfig, corpus: str, row: dict, target: str) -> dict:
    system, template = prompts(corpus)
    result = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": template.format(
                    source_language=row["question_language"],
                    target_language=target,
                    query=row["question"],
                ),
            },
        ],
        **config.request_options,
    }
    if config.response_format == "json_schema":
        result["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "query_translation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"translation": {"type": "string"}},
                    "required": ["translation"],
                    "additionalProperties": False,
                },
            },
        }
    elif config.response_format == "json_object":
        result["response_format"] = {"type": "json_object"}
    return result


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
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError("Translation JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        result = json.loads(content, object_pairs_hook=unique)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Translation must be valid JSON without Markdown fences") from exc
    if (
        not isinstance(result, dict)
        or set(result) != {"translation"}
        or not isinstance(result["translation"], str)
        or not result["translation"].strip()
    ):
        raise ValidationError(
            'Translation must contain exactly one nonempty string key, "translation"'
        )
    return result["translation"]  # Never normalize or trim a model's query.


def response_translation(response: dict) -> str:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValidationError("Translation must return exactly one completion choice")
    choice = choices[0]
    message = choice.get("message")
    if (
        choice.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("refusal")
    ):
        raise ValidationError("Translation refused, truncated, or returned an invalid message")
    return parse_translation(message.get("content"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep credentials and query text on the explicitly configured endpoint.
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


class ChatCompletionsTranslator:
    """One non-streaming request, no implicit retries or provider/model selection."""

    def __init__(self, config: TranslationConfig, env_file: Path | None):
        from dotenv import dotenv_values

        if not config.endpoint or not config.model:
            raise ValidationError(
                "Choose translation.endpoint and translation.model before running translate"
            )
        self.config = config
        self.opener = urllib.request.build_opener(_NoRedirect())
        self.key = None
        if config.api_key_env:
            self.key = os.environ.get(config.api_key_env) or dotenv_values(
                env_file or Path.cwd() / ".env"
            ).get(config.api_key_env)
            if not self.key:
                raise ValidationError(
                    f"Set {config.api_key_env} before running translate (or api_key_env: null for an unauthenticated endpoint)"
                )

    def complete(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = urllib.request.Request(
            self.config.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with self.opener.open(request, timeout=self.config.timeout_seconds) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not log provider response bodies or headers, which may expose credentials.
            raise ValidationError(
                f"Translation HTTP error {exc.code}; no automatic retry"
            ) from None
        except (OSError, ValueError) as exc:
            raise ValidationError(
                f"Translation transport/response error ({type(exc).__name__}); no automatic retry"
            ) from None


def token_usage(response: dict | None) -> dict:
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details") or {}
    details = details if isinstance(details, dict) else {}

    def count(value):
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    return {
        "input_tokens": count(usage.get("prompt_tokens")),
        "output_tokens": count(usage.get("completion_tokens")),
        "cached_input_tokens": count(details.get("cached_tokens")),
    }


def estimated_cost(usage: dict, config: TranslationConfig) -> float | None:
    inp, out, cached = (
        usage[key] for key in ("input_tokens", "output_tokens", "cached_input_tokens")
    )
    if (
        inp is None
        or out is None
        or config.input_usd_per_million is None
        or config.output_usd_per_million is None
    ):
        return None
    rate = config.cached_input_usd_per_million
    if rate is None:
        rate = config.input_usd_per_million
        cached = 0  # No cache discount claimed unless explicitly configured and measured.
    if cached is None or cached > inp:
        return None
    return (
        (inp - cached) * config.input_usd_per_million
        + cached * rate
        + out * config.output_usd_per_million
    ) / 1_000_000


def nullable_sum(values) -> float | None:
    values = list(values)
    return None if any(value is None for value in values) else sum(values)


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
    translator = None
    new_attempts = 0
    started = time.perf_counter()
    cache_root = config.output_dir / "translation_requests"
    cache_root.mkdir(parents=True, exist_ok=True)
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
        path = cache_root / f"{request_key}.json"
        with (cache_root / f"{request_key}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValidationError(
                    f"Translation request {request_key} is running in another process"
                ) from None
            state = (
                read_json(path)
                if path.exists()
                else {"request_key": request_key, "payload": payload, "attempts": []}
            )

            def save(path=path, state=state):
                temporary = path.with_suffix(".tmp")
                write_json(temporary, state)
                temporary.replace(path)

            if state.get("status") != "success":
                if state["attempts"] and not retry_failed:
                    raise ValidationError(
                        f"Previous failed/interrupted translation: {path}. Inspect it, then use translate --retry-failed to retry explicitly."
                    )
                translator = translator or translator_factory(config.translation, config.env_file)
                attempt = {"status": "pending", "seconds": None, "response": None}
                state["attempts"].append(attempt)
                save()  # An interrupted or ambiguous request must not silently incur a second charge.
                call_started = time.perf_counter()
                try:
                    response = translator.complete(payload)
                    attempt["response"] = response
                    attempt["seconds"] = time.perf_counter() - call_started
                    state["text"] = response_translation(response)
                except (
                    ValidationError,
                    KeyError,
                    IndexError,
                    TypeError,
                    OSError,
                    ValueError,
                ) as exc:
                    attempt["status"] = state["status"] = "failed"
                    attempt["seconds"] = time.perf_counter() - call_started
                    attempt["error_type"] = type(exc).__name__
                    save()
                    raise ValidationError(
                        f"Translation failed for {row['id']} -> {target}; audit: {path}; no fallback to the original query"
                    ) from exc
                attempt["status"] = state["status"] = "success"
                save()
                new_attempts += 1
            usages = [token_usage(attempt["response"]) for attempt in state["attempts"]]
            records.append(
                {
                    **record,
                    "text": state["text"],
                    "request_key": request_key,
                    "status": "translated",
                    "attempts": len(usages),
                    **{key: nullable_sum(usage[key] for usage in usages) for key in usages[0]},
                    "estimated_cost_usd": nullable_sum(
                        estimated_cost(usage, config.translation) for usage in usages
                    ),
                    "translation_seconds": nullable_sum(
                        attempt["seconds"] for attempt in state["attempts"]
                    ),
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
