import copy
import fcntl

import pytest

from clir_benchmark.config import GenerationConfig
from clir_benchmark.errors import ValidationError
from clir_benchmark.generation import (
    RequestCache,
    attempt_accounting,
    completion_budget_candidates,
    response_text,
)
from clir_benchmark.util import digest, read_json, write_json


def payload(budget=8192):
    return {
        "model": "fixture",
        "messages": [{"role": "user", "content": "query"}],
        "max_completion_tokens": budget,
        "reasoning_effort": "low",
    }


def response(*, failed=False):
    return {
        "choices": [
            {
                "finish_reason": "length" if failed else "stop",
                "message": {"content": "" if failed else "passage"},
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 5},
            "completion_tokens": 10,
        },
    }


def parse(value):
    return response_text(value, label="Fixture")


class Client:
    def __init__(self):
        self.calls = []
        self.responses = []

    def factory(self, config, env_file):
        return self

    def complete(self, value):
        self.calls.append(copy.deepcopy(value))
        return self.responses.pop(0) if self.responses else response()


def make_cache(tmp_path, client):
    return RequestCache(
        tmp_path,
        GenerationConfig(
            input_usd_per_million=1, cached_input_usd_per_million=0.1, output_usd_per_million=2
        ),
        None,
        client_factory=client.factory,
        label="Fixture",
        command="generate",
    )


def complete(cache, value, *, budgets=(), retry=False, candidates=None):
    return cache.complete(
        digest(value),
        value,
        parse_response=parse,
        retry_failed=retry,
        description="query -> fr",
        compatible_requests=candidates
        if candidates is not None
        else completion_budget_candidates(value, budgets, digest),
    )


def test_reuse_preserves_original_request_and_makes_no_alias_or_api_call(tmp_path):
    client = Client()
    cache = make_cache(tmp_path, client)
    original, _ = complete(cache, payload(2048))
    before = (tmp_path / f"{original['request_key']}.json").read_bytes()
    result, is_new = complete(cache, payload(), budgets=(2048,))
    assert not is_new
    assert result == original
    assert result["payload"]["max_completion_tokens"] == 2048
    assert len(client.calls) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert (tmp_path / f"{original['request_key']}.json").read_bytes() == before


def test_failed_history_is_linked_and_accounted_once_without_audit_duplication(tmp_path):
    client = Client()
    client.responses = [response(failed=True)]
    cache = make_cache(tmp_path, client)
    with pytest.raises(ValidationError, match="no fallback"):
        complete(cache, payload(2048))
    previous_key = digest(payload(2048))
    previous_path = tmp_path / f"{previous_key}.json"
    before = previous_path.read_bytes()
    with pytest.raises(ValidationError, match="retry-failed"):
        complete(cache, payload(), budgets=(2048,))
    assert len(client.calls) == 1
    result, is_new = complete(cache, payload(), budgets=(2048,), retry=True)
    assert is_new
    assert client.calls[-1]["max_completion_tokens"] == 8192
    assert len(result["attempts"]) == 2
    assert result["prior_request_keys"] == [previous_key]
    assert previous_path.read_bytes() == before
    saved = read_json(tmp_path / f"{result['request_key']}.json")
    assert len(saved["attempts"]) == 1
    assert sum(len(read_json(path)["attempts"]) for path in tmp_path.glob("*.json")) == 2
    summary = attempt_accounting(result, cache.config, seconds_key="seconds")
    assert summary["input_tokens"] == 40
    assert summary["estimated_cost_usd"] == pytest.approx(0.000071)
    again, is_new = complete(cache, payload(), budgets=(2048,))
    assert not is_new
    assert again == result
    assert len(client.calls) == 2
    # A new invocation records only its own call, never the referenced failure.
    assert len(cache.new_attempts) == 2
    fresh = make_cache(tmp_path, client)
    reused, _ = complete(fresh, payload())
    assert reused == result
    assert fresh.new_attempts == []


@pytest.mark.parametrize("budget", [0, -1, True, 8192, 16384, 2.5, "2048"])
def test_candidate_budgets_must_be_lower_positive_integers(budget):
    with pytest.raises(ValidationError, match="positive integers"):
        completion_budget_candidates(payload(), (budget,), digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "different"),
        ("reasoning_effort", "high"),
        ("messages", [{"role": "user", "content": "changed"}]),
    ],
)
def test_reuse_cannot_change_model_prompt_or_other_request_options(tmp_path, field, value):
    client = Client()
    cache = make_cache(tmp_path, client)
    candidate = {**payload(2048), field: value}
    with pytest.raises(ValidationError, match="differ only"):
        complete(cache, payload(), candidates={digest(candidate): candidate})
    assert client.calls == []


@pytest.mark.parametrize("mutation", ["request_key", "payload", "text", "response"])
def test_corrupt_candidates_are_rejected_before_calls(tmp_path, mutation):
    client = Client()
    cache = make_cache(tmp_path, client)
    state, _ = complete(cache, payload(2048))
    path = tmp_path / f"{state['request_key']}.json"
    if mutation == "request_key":
        state["request_key"] = "different"
    elif mutation == "payload":
        state["payload"]["model"] = "different"
    elif mutation == "text":
        state["text"] = "tampered"
    else:
        state["attempts"][-1]["response"] = response(failed=True)
    write_json(path, state)
    with pytest.raises(ValidationError, match="[Cc]orrupt|Invalid successful"):
        complete(cache, payload(), budgets=(2048,))
    assert len(client.calls) == 1


def test_candidate_lock_prevents_concurrent_duplicate_work(tmp_path):
    client = Client()
    cache = make_cache(tmp_path, client)
    key = digest(payload(2048))
    with (tmp_path / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValidationError, match="another process"):
            complete(cache, payload(), budgets=(2048,))
    assert client.calls == []


@pytest.mark.parametrize("prior", ["../outside", "missing", "self"])
def test_invalid_missing_or_cyclic_lineage_never_silently_loses_history(tmp_path, prior):
    client = Client()
    client.responses = [response(failed=True)]
    cache = make_cache(tmp_path, client)
    with pytest.raises(ValidationError):
        complete(cache, payload(2048))
    key = digest(payload(2048))
    path = tmp_path / f"{key}.json"
    state = read_json(path)
    state["prior_request_keys"] = [key if prior == "self" else prior]
    write_json(path, state)
    with pytest.raises(ValidationError, match="key|history|cycle"):
        complete(cache, payload(), budgets=(2048,), retry=True)
    assert len(client.calls) == 1


def test_multiple_source_histories_require_explicit_resolution(tmp_path):
    client = Client()
    cache = make_cache(tmp_path, client)
    complete(cache, payload(2048))
    complete(cache, payload(4096))
    with pytest.raises(ValidationError, match="single source"):
        complete(cache, payload(), budgets=(2048, 4096))
    assert len(client.calls) == 2


def test_shared_ancestor_attempts_are_counted_once(tmp_path):
    client = Client()
    cache = make_cache(tmp_path, client)
    budgets = [1024, 2048, 4096, 8192]
    keys = {budget: digest(payload(budget)) for budget in budgets}
    for budget in budgets:
        failed = budget != 8192
        state = {
            "request_key": keys[budget],
            "payload": payload(budget),
            "status": "failed" if failed else "success",
            "attempts": [
                {
                    "status": "failed" if failed else "success",
                    "response": response(failed=failed),
                    "seconds": 1,
                }
            ],
        }
        if budget in {2048, 4096}:
            state["prior_request_keys"] = [keys[1024]]
        elif budget == 8192:
            state["text"] = "passage"
            state["prior_request_keys"] = [keys[2048], keys[4096]]
        write_json(tmp_path / f"{keys[budget]}.json", state)
    result, is_new = complete(cache, payload())
    assert not is_new
    assert len(result["attempts"]) == 4
    assert attempt_accounting(result, cache.config, seconds_key="seconds")["seconds"] == 4
    assert client.calls == []
