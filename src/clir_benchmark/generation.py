from __future__ import annotations

import fcntl
import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow as pa

from .adapters import DatasetReader, required_string
from .artifacts import ParquetSink, rows
from .config import QUESTION_LANGUAGE, Config, GenerationConfig, language_tag
from .errors import ValidationError
from .util import digest, hf_token, new_directory, read_json, write_json

GENERATION_AUDIT_FIELDS = [
    ("requested_model", pa.string()),
    ("returned_model", pa.string()),
    ("provider_cost_usd", pa.float64()),
    ("api_cost_usd", pa.float64()),
]
IDENTITY_AUDIT = {
    "requested_model": None,
    "returned_model": None,
    "provider_cost_usd": 0.0,
    "api_cost_usd": 0.0,
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep credentials and query text on the explicitly configured endpoint.
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


class CompletionError(ValidationError):
    def __init__(self, message, *, status=None):
        super().__init__(message)
        self.status = status


def provider_refused(response):
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return False
    choice = choices[0]
    message = choice.get("message")
    return choice.get("finish_reason") == "content_filter" or (
        isinstance(message, dict) and bool(message.get("refusal"))
    )


def with_refusal_policy(config, enabled):
    if not enabled:
        return config
    return replace(config, query_generation=replace(config.query_generation, record_refusals=True))


def retryable_completion_failure(error, response):
    """Distinguish provider/network failures from validation of the request or file system."""
    code = error.status if isinstance(error, CompletionError) else None
    if isinstance(response, dict) and isinstance(response.get("error"), dict):
        code = response["error"].get("code")
        if not isinstance(code, int) and not (isinstance(code, str) and code.isdecimal()):
            return False, None
        code = int(code)
    if code is not None:
        return code in (408, 429) or 500 <= code <= 599, code
    if isinstance(error, (CompletionError, TimeoutError, ConnectionError)):
        return True, None
    if isinstance(response, dict):
        choices = response.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            if isinstance(choice, dict) and (
                choice.get("finish_reason") in ("content_filter", "length")
                or isinstance(choice.get("message"), dict)
                and choice["message"].get("refusal")
            ):
                return False, None
        return True, None  # Malformed completion, e.g. JSON schema violation.
    return False, None


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
            raise CompletionError(f"{self.label} HTTP error {exc.code}", status=exc.code) from None
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise CompletionError(
                f"{self.label} transport/response error ({type(exc).__name__})"
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


def generation_inputs(config: Config, corpus: str, *, action: str) -> list[dict]:
    if config.query_generation.input_mode != "verified-qas":
        return eligible_inputs(config, corpus, action=action)

    languages = config.query_generation.corpus_languages.get(corpus)
    if not languages:
        raise ValidationError(f"Set query_generation.corpus_languages for {corpus}")
    qa = config.datasets[corpus].qa
    reader = DatasetReader(qa, hf_token(config.env_file) if not qa.local_dir else None)
    inputs = {}
    for row in reader.read("qas", columns=["_id", "question", "question_language"]):
        identifier = required_string(row, "_id", f"{corpus} verified QA")
        label = f"{corpus} verified QA {identifier}"
        language = language_tag(required_string(row, "question_language", label), label)
        question = required_string(row, "question", label)
        if any(c.isspace() for c in identifier) or identifier in inputs:
            raise ValidationError(f"{label}: invalid or duplicate question ID")
        if not question.strip() or language not in languages:
            raise ValidationError(f"{label}: blank question or language outside {languages}")
        inputs[identifier] = {"id": identifier, "question": question, "question_language": language}
    if not inputs:
        raise ValidationError(f"No verified questions found for {corpus}")
    return [inputs[key] for key in sorted(inputs)]


def generation_provenance(config: Config, corpus: str, inputs: list[dict], prompts: tuple) -> dict:
    policy = {"record_refusals": True} if config.query_generation.record_refusals else {}
    if config.query_generation.input_mode != "verified-qas":
        return policy
    qa = config.datasets[corpus].qa
    return {
        **policy,
        "input_mode": "verified-qas",
        "qa_release": {"repo_id": qa.repo_id, "revision": qa.revision},
        "input_fingerprint": digest(inputs),
        "prompt_fingerprint": digest(prompts),
    }


def generation_partition(destination, inputs, provenance, shard):
    """Both verified and prepared inputs are already ordered by unique query ID."""
    if shard is None:
        return destination, inputs, provenance
    index, size = shard
    if size < 1 or index < 0 or index * size >= len(inputs):
        raise ValidationError(f"Invalid shard {index} of size {size} for {len(inputs)} questions")
    selected = inputs[index * size : (index + 1) * size]
    path = destination.parent / "shards" / destination.name / str(size) / str(index)
    return (
        path,
        selected,
        {
            **provenance,
            "shard": {
                "index": index,
                "size": size,
                "count": (len(inputs) + size - 1) // size,
                "input_fingerprint": digest(selected),
            },
        },
    )


def assemble_generation_shards(
    destination,
    inputs,
    provenance,
    size,
    language,
    *,
    loader,
    configuration,
    schema,
    filename,
    fingerprint_key,
    summarize,
    seconds_key,
):
    """Publish the canonical artifact from complete, validated local shards, without any client."""
    if size < 1 or not inputs:
        raise ValidationError("Shard size and question count must be positive")
    if destination.exists():
        loader(destination, language, inputs, provenance)
        return destination
    started = time.perf_counter()
    records, fingerprints = [], []
    for index in range((len(inputs) + size - 1) // size):
        path, subset, expected = generation_partition(
            destination, inputs, provenance, (index, size)
        )
        _, group, manifest = loader(path, language, subset, expected)
        if digest(manifest.get("configuration")) != digest(configuration):
            raise ValidationError(f"Generation model or options changed in shard {path}")
        records.extend(group)
        fingerprints.append(manifest[fingerprint_key])
    verify_generation_inputs(
        records,
        inputs,
        None if language == QUESTION_LANGUAGE else language,
        destination,
        record_refusals=provenance.get("record_refusals", False),
    )
    summary = summarize(records)
    assembled = {
        **manifest,
        **provenance,
        "query_count": len(records),
        fingerprint_key: digest([(row["id"], row["language"], row["text"]) for row in records]),
        "summary": summary,
        "new_work": summarize([]),
        "new_api_attempts": 0,
        "cached_requests": sum(row["request_key"] is not None for row in records),
        "seconds": time.perf_counter() - started,
        "assembled_from_shards": {"size": size, "fingerprints": fingerprints},
    }
    del assembled["shard"]
    if provenance or language == QUESTION_LANGUAGE:
        assembled["by_language_and_model"] = generation_breakdown(records, seconds_key=seconds_key)
    else:
        assembled.pop("by_language_and_model", None)
    with new_directory(destination) as temporary:
        with ParquetSink(temporary / filename, schema) as sink:
            for row in records:
                sink.append(row)
        write_json(temporary / "manifest.json", assembled)
        loader(temporary, language, inputs, provenance)
    return destination


def output_model(config: Config, language: str, default: GenerationConfig) -> GenerationConfig:
    assignments = config.query_generation.model_by_language
    if assignments:
        if language not in assignments:
            raise ValidationError(f"No generation model assigned to output language {language}")
        return replace(default, **asdict(config.query_generation.models[assignments[language]]))
    return default


def planned_generation(
    route: GenerationConfig, language: str, paid: int, identity: int = 0
) -> dict:
    if paid and (not route.model or not route.endpoint):
        raise ValidationError(f"Configure a generation model and endpoint for {language}")
    return {
        "output_language": language,
        "model": route.model if paid else None,
        "variants": paid + identity,
        "identity_variants": identity,
        "requests_before_cache_or_retries": paid,
    }


def verify_generation_inputs(
    records: list[dict],
    inputs: list[dict],
    target: str | None,
    artifact: Path,
    *,
    record_refusals=False,
) -> None:
    expected = {
        (
            query["id"],
            query["question"],
            query["question_language"],
            target or query["question_language"],
        )
        for query in inputs
    }
    observed = {
        (row["id"], row["original_text"], row["source_language"], row["language"])
        for row in records
    }
    if (
        observed != expected
        or len(observed) != len(records)
        or any(
            not (
                record_refusals
                and row["text"] is None
                and row.get("request_key")
                and row.get("attempts", 0) > 0
            )
            if row.get("status") == "provider_refused"
            else not isinstance(row["text"], str) or not row["text"].strip()
            for row in records
        )
    ):
        raise ValidationError(f"Query variants disagree with the pinned questions: {artifact}")


def provider_cost_usd(response: dict | None) -> float | None:
    usage = response.get("usage") if isinstance(response, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def generation_audit(state: dict, route: GenerationConfig) -> dict:
    attempts = state["attempts"]
    response = attempts[-1]["response"]
    reported = [provider_cost_usd(attempt["response"]) for attempt in attempts]
    return {
        "requested_model": route.model,
        "returned_model": response.get("model") if isinstance(response, dict) else None,
        "provider_cost_usd": nullable_sum(reported),
        "api_cost_usd": nullable_sum(
            cost if cost is not None else estimated_cost(token_usage(attempt["response"]), route)
            for cost, attempt in zip(reported, attempts, strict=True)
        ),
    }


def generation_breakdown(records: Iterable[dict], *, seconds_key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[(row["language"], row["requested_model"], row["returned_model"])].append(row)
    return [
        {
            "output_language": language,
            "requested_model": requested,
            "returned_model": returned,
            "query_count": len(group),
            "provider_refused_count": sum(row.get("status") == "provider_refused" for row in group),
            "api_attempts": sum(row["attempts"] for row in group),
            **{
                key: nullable_sum(row[key] for row in group)
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "estimated_cost_usd",
                    "provider_cost_usd",
                    "api_cost_usd",
                    seconds_key,
                )
            },
        }
        for (language, requested, returned), group in groups.items()
    ]


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


def _lower_completion_budget(payload: dict, candidate: dict) -> bool:
    current = payload.get("max_completion_tokens")
    previous = candidate.get("max_completion_tokens")
    return (
        isinstance(current, int)
        and not isinstance(current, bool)
        and isinstance(previous, int)
        and not isinstance(previous, bool)
        and 0 < previous < current
        and {key: value for key, value in payload.items() if key != "max_completion_tokens"}
        == {key: value for key, value in candidate.items() if key != "max_completion_tokens"}
    )


def completion_budget_candidates(
    payload: dict, budgets: tuple[int, ...], key_factory: Callable[[dict], str]
) -> dict[str, dict]:
    """Construct explicitly selected historical requests; never change a live payload."""
    candidates = {}
    for budget in budgets:
        candidate = {**payload, "max_completion_tokens": budget}
        if not _lower_completion_budget(payload, candidate):
            raise ValidationError(
                "Reusable completion budgets must be positive integers below the current limit"
            )
        key = key_factory(candidate)
        if key in candidates:
            raise ValidationError("Reusable completion budgets must identify distinct requests")
        candidates[key] = candidate
    return candidates


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
        retry_transient: bool = False,
        record_refusals: bool = False,
    ):
        self.root, self.config, self.env_file = root, config, env_file
        self.client_factory, self.label, self.command = client_factory, label, command
        self.retry_transient = retry_transient
        self.record_refusals = record_refusals
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
        compatible_requests: dict[str, dict] | None = None,
    ) -> tuple[dict, bool]:
        compatible_requests = compatible_requests or {}
        for key, candidate in compatible_requests.items():
            if (
                key == request_key
                or not isinstance(candidate, dict)
                or not _lower_completion_budget(payload, candidate)
            ):
                raise ValidationError(
                    "Compatible requests may differ only by a lower max_completion_tokens limit"
                )
        with ExitStack() as stack:
            locked = set()

            def lock_request(key):
                if not isinstance(key, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_-]{0,255}", key
                ):
                    raise ValidationError("Invalid generation request key")
                if key not in locked:
                    lock = stack.enter_context((self.root / f"{key}.lock").open("a"))
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise ValidationError(
                            f"{self.label} request {key} is running in another process"
                        ) from None
                    locked.add(key)

            def load_state(key, expected_payload=None):
                lock_request(key)
                try:
                    value = read_json(self.root / f"{key}.json")
                except (OSError, ValueError) as exc:
                    raise ValidationError(
                        f"Missing or corrupt generation request history: {key}"
                    ) from exc
                if (
                    not isinstance(value, dict)
                    or value.get("request_key") != key
                    or not isinstance(value.get("payload"), dict)
                    or (expected_payload is not None and value["payload"] != expected_payload)
                    or not isinstance(value.get("attempts"), list)
                    or not value["attempts"]
                    or value.get("status") not in {"pending", "failed", "success"}
                    or any(
                        not isinstance(attempt, dict)
                        or attempt.get("status") not in {"pending", "failed", "success"}
                        or "seconds" not in attempt
                        or "response" not in attempt
                        for attempt in value["attempts"]
                    )
                ):
                    raise ValidationError(
                        f"Corrupt or mismatched generation request history: {key}"
                    )
                if value["status"] == "success":
                    try:
                        text = parse_response(value["attempts"][-1]["response"])
                    except (ValidationError, KeyError, IndexError, TypeError, ValueError) as exc:
                        raise ValidationError(
                            f"Invalid successful generation request history: {key}"
                        ) from exc
                    if value["attempts"][-1]["status"] != "success" or text != value.get("text"):
                        raise ValidationError(f"Corrupt successful generation request text: {key}")
                return value

            def materialize(value):
                # Each physical API attempt lives in one file. Expand references
                # only in the returned view used for per-query accounting.
                seen, active, attempts = set(), set(), []

                def visit(current):
                    key = current["request_key"]
                    if key in active:
                        raise ValidationError("Generation request history contains a cycle")
                    if key in seen:
                        return
                    active.add(key)
                    parents = current.get("prior_request_keys", [])
                    if (
                        not isinstance(parents, list)
                        or any(not isinstance(parent, str) for parent in parents)
                        or len(set(parents)) != len(parents)
                    ):
                        raise ValidationError("Invalid generation request history references")
                    for parent in parents:
                        if parent in active:
                            raise ValidationError("Generation request history contains a cycle")
                        previous = load_state(parent)
                        if not _lower_completion_budget(current["payload"], previous["payload"]):
                            raise ValidationError(
                                "Generation request history has incompatible payloads"
                            )
                        if previous["status"] == "success":
                            raise ValidationError(
                                "A failed-request lineage cannot include an unused success"
                            )
                        visit(previous)
                    attempts.extend(current["attempts"])
                    active.remove(key)
                    seen.add(key)

                visit(value)
                return {**value, "attempts": attempts}

            def refusal_outcome(history):
                if history.get("status") != "failed" or not history["attempts"]:
                    return None
                last = history["attempts"][-1]
                if last["status"] != "failed" or not provider_refused(last.get("response")):
                    return None
                if not self.record_refusals:
                    raise ValidationError(
                        f"Provider refused {self.label.lower()} for {description}; audit: {self.root / (history['request_key'] + '.json')}; no fallback to the original query"
                    )
                return {**history, "status": "provider_refused", "text": None}

            for key in sorted({request_key, *compatible_requests}):
                lock_request(key)
            path = self.root / f"{request_key}.json"
            state = (
                read_json(path)
                if path.exists()
                else {"request_key": request_key, "payload": payload, "attempts": []}
            )
            if path.exists() and (
                self.record_refusals or compatible_requests or state.get("prior_request_keys")
            ):
                state = load_state(request_key, payload)
            if not path.exists():
                candidates = [
                    load_state(key, candidate)
                    for key, candidate in compatible_requests.items()
                    if (self.root / f"{key}.json").exists()
                ]
                if len(candidates) > 1:
                    raise ValidationError(
                        "Multiple compatible request histories require an explicit single source"
                    )
                if candidates:
                    previous = candidates[0]
                    history = materialize(previous)
                    if previous["status"] == "success":
                        self.cache_hits += 1
                        return history, False
                    refused = refusal_outcome(history)
                    if refused is not None:
                        self.cache_hits += 1
                        return refused, False
                    if not retry_failed:
                        raise ValidationError(
                            f"Previous failed/interrupted {self.label.lower()}: {previous['request_key']}. Inspect it, then use {self.command} --retry-failed to retry explicitly."
                        )
                    state["prior_request_keys"] = [previous["request_key"]]
            # Validate the accounting lineage before any paid request.
            history = materialize(state)
            if state.get("status") == "success":
                self.cache_hits += 1
                return history, False
            refused = refusal_outcome(history)
            if refused is not None:
                self.cache_hits += 1
                return refused, False

            def can_retry():
                return (
                    self.retry_transient
                    and len(state["attempts"]) < 3
                    and state["attempts"][-1].get("retryable") is True
                )

            if state["attempts"] and not retry_failed and not can_retry():
                raise ValidationError(
                    f"Previous failed/interrupted {self.label.lower()}: {path}. Inspect it, then use {self.command} --retry-failed to retry explicitly."
                )
            self.client = self.client or self.client_factory(self.config, self.env_file)

            def save():
                temporary = path.with_suffix(".tmp")
                write_json(temporary, state)
                temporary.replace(path)

            while True:
                attempt = {"status": "pending", "seconds": None, "response": None}
                state["attempts"].append(attempt)
                state["status"] = "pending"
                self.new_attempts.append({"request_key": request_key, "attempt": attempt})
                save()  # Persist pending before sending, including requests interrupted by a crash.
                started = time.perf_counter()
                try:
                    response = self.client.complete(payload)
                    attempt["response"] = response
                    attempt["seconds"] = time.perf_counter() - started
                    state["text"] = parse_response(response)
                except (
                    ValidationError,
                    KeyError,
                    IndexError,
                    TypeError,
                    OSError,
                    ValueError,
                ) as exc:
                    attempt["status"] = state["status"] = "failed"
                    attempt["seconds"] = time.perf_counter() - started
                    attempt["error_type"] = type(exc).__name__
                    attempt["retryable"], attempt["http_status"] = retryable_completion_failure(
                        exc, attempt["response"]
                    )
                    save()
                    refused = refusal_outcome(materialize(state))
                    if refused is not None:
                        return refused, True
                    if can_retry():
                        time.sleep(5 * 2 ** (len(state["attempts"]) - 1))
                        continue
                    raise ValidationError(
                        f"{self.label} failed for {description}; HTTP status {attempt['http_status']}; audit: {path}; no fallback to the original query"
                    ) from exc
                attempt["status"] = state["status"] = "success"
                save()
                return materialize(state), True


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
