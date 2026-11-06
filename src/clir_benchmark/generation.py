from __future__ import annotations

import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .artifacts import rows
from .config import Config, GenerationConfig
from .errors import ValidationError
from .util import read_json, write_json


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep credentials and query text on the explicitly configured endpoint.
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


class ChatCompletionsClient:
    """One non-streaming request, no implicit retries or provider/model selection."""

    label = "Generation"
    command = "generate"

    def __init__(self, config: GenerationConfig, env_file: Path | None):
        from dotenv import dotenv_values

        if not config.endpoint or not config.model:
            raise ValidationError(
                f"Choose {config.section}.endpoint and {config.section}.model before running {self.command}"
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
                    f"Set {config.api_key_env} before running {self.command} (or api_key_env: null for an unauthenticated endpoint)"
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
                f"{self.label} HTTP error {exc.code}; no automatic retry"
            ) from None
        except (OSError, ValueError) as exc:
            raise ValidationError(
                f"{self.label} transport/response error ({type(exc).__name__}); no automatic retry"
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


def estimated_cost(usage: dict, config: GenerationConfig) -> float | None:
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


def eligible_inputs(config: Config, corpus: str, *, action: str) -> list[dict]:
    directory = config.relevance_dir(corpus)
    if not (directory / "manifest.json").is_file():
        raise ValidationError(f"Run prepare-relevance before {action}")
    settings = read_json(directory / "settings.json")
    eligible = {
        row["query_id"]
        for row in rows(directory / "eligibility.parquet")
        if row["status"] == "eligible" and settings[row["setting_id"]]["status"] == "ready"
    }
    # Never read answers, evidence or corpus text into a generation request.
    return sorted(
        (
            row
            for row in rows(directory / "queries.parquet", ["id", "question", "question_language"])
            if row["id"] in eligible
        ),
        key=lambda row: row["id"],
    )


def parse_generated_text(content: str, *, key: str, label: str) -> str:
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValidationError(f"{label} JSON contains duplicate keys")
            result[name] = value
        return result

    try:
        result = json.loads(content, object_pairs_hook=unique)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be valid JSON without Markdown fences") from exc
    if (
        not isinstance(result, dict)
        or set(result) != {key}
        or not isinstance(result[key], str)
        or not result[key].strip()
    ):
        raise ValidationError(f'{label} must contain exactly one nonempty string key, "{key}"')
    return result[key]  # Preserve generated text exactly, including whitespace and notation.


def response_text(response: dict, *, label: str) -> str:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValidationError(f"{label} must return exactly one completion choice")
    choice = choices[0]
    message = choice.get("message")
    if (
        choice.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("refusal")
    ):
        raise ValidationError(f"{label} refused, truncated, or returned an invalid message")
    return message.get("content")


class RequestCache:
    """Shared durable single-request execution; callers own namespaces and request keys."""

    def __init__(
        self,
        root: Path,
        config: GenerationConfig,
        env_file: Path | None,
        *,
        client_factory: Callable,
        label: str,
        command: str,
    ):
        self.root, self.config, self.env_file = root, config, env_file
        self.client_factory, self.label, self.command = client_factory, label, command
        self.client = None
        self.cache_hits = 0
        self.new_attempts: list[dict] = []
        root.mkdir(parents=True, exist_ok=True)

    def complete(
        self,
        request_key: str,
        payload: dict,
        *,
        parse_response: Callable,
        retry_failed: bool,
        description: str,
    ) -> tuple[dict, bool]:
        path = self.root / f"{request_key}.json"
        with (self.root / f"{request_key}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValidationError(
                    f"{self.label} request {request_key} is running in another process"
                ) from None
            state = (
                read_json(path)
                if path.exists()
                else {"request_key": request_key, "payload": payload, "attempts": []}
            )
            if state.get("status") == "success":
                self.cache_hits += 1
                return state, False
            if state["attempts"] and not retry_failed:
                raise ValidationError(
                    f"Previous failed/interrupted {self.label.lower()}: {path}. Inspect it, then use {self.command} --retry-failed to retry explicitly."
                )
            self.client = self.client or self.client_factory(self.config, self.env_file)

            def save():
                temporary = path.with_suffix(".tmp")
                write_json(temporary, state)
                temporary.replace(path)

            attempt = {"status": "pending", "seconds": None, "response": None}
            state["attempts"].append(attempt)
            self.new_attempts.append({"request_key": request_key, "attempt": attempt})
            save()  # Persist pending before the API call, including requests interrupted by a crash.
            started = time.perf_counter()
            try:
                response = self.client.complete(payload)
                attempt["response"] = response
                attempt["seconds"] = time.perf_counter() - started
                state["text"] = parse_response(response)
            except (ValidationError, KeyError, IndexError, TypeError, OSError, ValueError) as exc:
                attempt["status"] = state["status"] = "failed"
                attempt["seconds"] = time.perf_counter() - started
                attempt["error_type"] = type(exc).__name__
                save()
                raise ValidationError(
                    f"{self.label} failed for {description}; audit: {path}; no fallback to the original query"
                ) from exc
            attempt["status"] = state["status"] = "success"
            save()
            return state, True


def attempt_accounting(state: dict, config: GenerationConfig, *, seconds_key: str) -> dict:
    usages = [token_usage(attempt["response"]) for attempt in state["attempts"]]
    return {
        "attempts": len(usages),
        **{key: nullable_sum(usage[key] for usage in usages) for key in usages[0]},
        "estimated_cost_usd": nullable_sum(estimated_cost(usage, config) for usage in usages),
        seconds_key: nullable_sum(attempt["seconds"] for attempt in state["attempts"]),
    }


def completion_payload(
    config: GenerationConfig, system: str, user: str, *, key: str, schema_name: str
) -> dict:
    result = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **config.request_options,
    }
    if config.response_format == "json_schema":
        result["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {key: {"type": "string"}},
                    "required": [key],
                    "additionalProperties": False,
                },
            },
        }
    elif config.response_format == "json_object":
        result["response_format"] = {"type": "json_object"}
    return result
