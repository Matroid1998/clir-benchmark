import json
from dataclasses import replace

import pytest
import yaml
from test_pipeline import fixture_config
from test_translation import completion

from clir_benchmark.adore import (
    AdoreSession,
    ChatCompletionsADORE,
    expansion_payload,
    judgment_payload,
    parse_expansion,
    parse_judgment,
)
from clir_benchmark.config import ADOREConfig, load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.util import read_json, write_json

QUERY = {
    "id": "q1",
    "question": "Quelles dispositions s'appliquent ?",
    "question_language": "fr",
    "answer": "SECRET_GOLD_ANSWER",
    "evidence": "SECRET_GOLD_EVIDENCE",
}
DOCUMENT = {
    "id": "document1",
    "text": "These provisions apply to industrial chemicals.",
    "language": "en",
    "relevance": "SECRET_GOLD_RELEVANCE",
}


def adore_completion(content):
    result = completion()
    result["choices"][0]["message"]["content"] = json.dumps(content, ensure_ascii=False)
    return result


class FixtureADORE:
    def __init__(self):
        self.calls = []
        self.responses = []
        self.created = 0

    def factory(self, config, env_file):
        self.created += 1
        return self

    def complete(self, payload):
        self.calls.append(payload)
        if self.responses:
            value = self.responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        request = json.loads(payload["messages"][1]["content"])
        if request["task"] == "judge":
            return adore_completion({"score": 2})
        return adore_completion(
            {
                "language": request["query"]["question_language"],
                "passages": ["Les dispositions applicables sont précisées dans le règlement."]
                * request["passages_per_round"],
            }
        )


def priced_config(tmp_path):
    config, _ = fixture_config(tmp_path)
    return replace(
        config,
        adore=ADOREConfig(
            model="fixture-generator",
            endpoint="https://provider.invalid/chat",
            passages_per_round=2,
            input_usd_per_million=2,
            cached_input_usd_per_million=0.5,
            output_usd_per_million=4,
        ),
    )


def test_language_preserved_with_foreign_feedback_and_no_gold_information():
    config = ADOREConfig(passages_per_round=2)
    payload = expansion_payload(config, "chemistry", QUERY, 2, [{**DOCUMENT, "score": 1}])
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["language"]["enum"] == ["fr"]
    data = json.loads(payload["messages"][1]["content"])
    assert data["query"] == {key: QUERY[key] for key in ("id", "question", "question_language")}
    assert data["feedback"][0]["language"] == "en"
    assert "original query's question_language" in payload["messages"][0]["content"]
    assert "even when feedback documents use another language" in payload["messages"][0]["content"]
    assert "SECRET_GOLD" not in json.dumps(payload)
    judgment = judgment_payload(config, "chemistry", QUERY, DOCUMENT)
    judgment_data = json.loads(judgment["messages"][1]["content"])
    assert judgment_data["query"]["question"] == QUERY["question"]
    assert "SECRET_GOLD" not in json.dumps(judgment)


@pytest.mark.parametrize("language", ["en", "de", "es", "fr", "zh", "zh-Hans"])
def test_language_comes_from_original_query_in_every_round(language):
    config = ADOREConfig(passages_per_round=1)
    query = {**QUERY, "question_language": language}
    for round_number in (1, 2, 3):
        payload = expansion_payload(config, "eurlex", query, round_number, [])
        schema = payload["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["language"]["enum"] == [language]


@pytest.mark.parametrize(
    "content",
    [
        '{"language":"en","passages":["an English expansion"]}',
        '{"language":"fr","passages":[]}',
        '{"language":"fr","passages":[""]}',
        '{"language":"fr","passages":["a","b"]}',
        '{"language":"fr","passages":[null]}',
        '{"language":"fr","passages":"a"}',
        '{"language":"fr","passages":["a"],"extra":1}',
        '{"language":"fr","language":"fr","passages":["a"]}',
        '```json\n{"language":"fr","passages":["a"]}\n```',
        "[]",
        None,
    ],
)
def test_invalid_or_language_changed_expansions_fail(content):
    with pytest.raises(ValidationError):
        parse_expansion(content, "fr", 1)


@pytest.mark.parametrize(
    "content",
    [
        '{"score":true}',
        '{"score":2.0}',
        '{"score":"2"}',
        '{"score":4}',
        '{"score":-1}',
        '{"score":null}',
        '{"score":1,"extra":2}',
        '{"score":1,"score":2}',
        "[]",
        None,
    ],
)
def test_invalid_relevance_judgments_fail(content):
    with pytest.raises(ValidationError):
        parse_judgment(content)


def test_outputs_preserve_whitespace_notation_and_unicode():
    passage = "  À 20 °C, C₂H₅OH — disposition 3a.\n"
    assert parse_expansion(json.dumps({"language": "fr", "passages": [passage]}), "fr", 1) == {
        "language": "fr",
        "passages": [passage],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"max_rounds": 4},
        {"max_rounds": 0},
        {"max_rounds": True},
        {"max_rounds": 2.0},
        {"passages_per_round": 0},
        {"passages_per_round": False},
        {"feedback_k": 0},
        {"feedback_k": 101},
        {"feedback_k": "10"},
    ],
)
def test_adore_configuration_rejects_invalid_budgets(changes):
    with pytest.raises(ValidationError):
        ADOREConfig(**changes)


def test_adore_selection_and_configuration_preserve_shared_cache_identity(tmp_path):
    config, path = fixture_config(tmp_path)
    assert config.baselines == ("dense",)
    assert config.adore.model is config.adore.endpoint is None
    assert config.adore.api_key_env == "ADORE_API_KEY"
    assert config.adore.max_rounds == 3
    raw = yaml.safe_load(path.read_text())
    raw["baselines"] = ["adore-bm25", "adore-dense", "adore-hybrid"]
    raw["adore"] = {"max_rounds": 2, "passages_per_round": 3}
    path.write_text(yaml.safe_dump(raw))
    selected = load_config(path)
    assert selected.adore.max_rounds == 2
    assert selected.corpus_key("eurlex") == config.corpus_key("eurlex")
    assert selected.relevance_key("eurlex") == config.relevance_key("eurlex")
    raw["adore"]["output_language"] = "en"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="Invalid configuration"):
        load_config(path)


def test_round_limit_is_checked_before_any_generation(tmp_path):
    config = priced_config(tmp_path)
    client = FixtureADORE()
    session = AdoreSession(config, "eurlex", client_factory=client.factory)
    for round_number in (0, 4, True, "1"):
        with pytest.raises(ValidationError, match="round budget"):
            session.expand(QUERY, round_number, [])
    assert not client.calls
    assert client.created == 0
    session.expand(QUERY, 3, [])
    assert len(client.calls) == 1


def test_cached_work_is_reused_and_original_cost_is_deduplicated(tmp_path):
    config = priced_config(tmp_path)
    client = FixtureADORE()
    session = AdoreSession(config, "eurlex", client_factory=client.factory)
    expansion = session.expand(QUERY, 1, [])
    first = session.judge(QUERY, DOCUMENT)
    duplicate = session.judge(QUERY, DOCUMENT)
    assert len(client.calls) == 2
    assert first["request_key"] == duplicate["request_key"]
    assert not expansion["cached"] and not first["cached"] and duplicate["cached"]
    assert expansion["text"].startswith(QUERY["question"] + "\n\n")
    summary = session.summary([first["request_key"], first["request_key"]])
    assert summary["attempts"] == 1
    assert summary["estimated_cost_usd"] == pytest.approx(0.00022)
    assert session.summary()["attempts"] == 2
    assert session.invocation_summary()["new_work"]["attempts"] == 2
    assert session.invocation_summary()["cached_requests"] == 0

    def forbidden_factory(*_):
        pytest.fail("Cached ADORE work must not require credentials or instantiate a client")

    repeated = AdoreSession(config, "eurlex", client_factory=forbidden_factory)
    assert repeated.expand(QUERY, 1, [])["cached"]
    assert repeated.judge(QUERY, DOCUMENT)["cached"]
    assert repeated.summary() == session.summary()
    assert repeated.invocation_summary()["new_work"]["attempts"] == 0
    assert repeated.invocation_summary()["cached_requests"] == 2


def test_feedback_content_and_original_question_partition_request_cache(tmp_path):
    config = priced_config(tmp_path)
    client = FixtureADORE()
    session = AdoreSession(config, "eurlex", client_factory=client.factory)
    first = session.expand(QUERY, 2, [{**DOCUMENT, "score": 0}])
    next_score = session.expand(QUERY, 2, [{**DOCUMENT, "score": 3}])
    next_text = session.expand(QUERY, 2, [{**DOCUMENT, "text": "Other text", "score": 0}])
    original = session.judge(QUERY, DOCUMENT)
    next_query = session.judge({**QUERY, "question": "Autre question ?"}, DOCUMENT)
    assert len({row["request_key"] for row in (first, next_score, next_text)}) == 3
    assert original["request_key"] != next_query["request_key"]
    assert len(client.calls) == 5


@pytest.mark.parametrize("mutation", ["text", "response", "payload"])
def test_cached_success_is_validated_before_reuse(tmp_path, mutation):
    config = priced_config(tmp_path)
    client = FixtureADORE()
    session = AdoreSession(config, "eurlex", client_factory=client.factory)
    record = session.judge(QUERY, DOCUMENT)
    path = config.output_dir / "adore_requests" / f"{record['request_key']}.json"
    state = read_json(path)
    if mutation == "text":
        state["text"] = {"score": True}
    elif mutation == "response":
        state["attempts"][-1]["response"] = adore_completion({"score": True})
    else:
        state["payload"]["messages"][1]["content"] = "altered"
    write_json(path, state)
    with pytest.raises(ValidationError, match="Corrupt cached ADORE"):
        session.judge(QUERY, DOCUMENT)
    assert len(client.calls) == 1


def test_language_failure_requires_explicit_retry_and_keeps_paid_attempts(tmp_path):
    config = priced_config(tmp_path)
    client = FixtureADORE()
    client.responses = [adore_completion({"language": "en", "passages": ["wrong"] * 2})]
    session = AdoreSession(config, "eurlex", client_factory=client.factory)
    with pytest.raises(ValidationError, match="no fallback"):
        session.expand(QUERY, 1, [])
    assert session.invocation_summary()["new_work"]["estimated_cost_usd"] == pytest.approx(0.00022)
    with pytest.raises(ValidationError, match="retry-failed"):
        session.expand(QUERY, 1, [])
    assert len(client.calls) == 1
    retried = AdoreSession(config, "eurlex", client_factory=client.factory, retry_failed=True)
    result = retried.expand(QUERY, 1, [])
    assert result["attempts"] == 2
    assert result["estimated_cost_usd"] == pytest.approx(0.00044)
    assert retried.invocation_summary()["new_work"]["attempts"] == 1


def test_missing_usage_and_prices_remain_unknown(tmp_path):
    config = priced_config(tmp_path)
    client = FixtureADORE()
    response = adore_completion({"score": 0})
    del response["usage"]
    client.responses.append(response)
    session = AdoreSession(config, "eurlex", client_factory=client.factory)
    assert session.judge(QUERY, DOCUMENT)["estimated_cost_usd"] is None
    assert session.summary()["input_tokens"] is None
    assert session.invocation_summary()["new_work"]["estimated_cost_usd"] is None


@pytest.mark.parametrize("failure", ["refusal", "length", "multiple"])
def test_provider_refusals_and_truncation_do_not_enter_feedback(tmp_path, failure):
    config = priced_config(tmp_path)
    client = FixtureADORE()
    response = adore_completion({"score": 3})
    if failure == "refusal":
        response["choices"][0]["message"]["refusal"] = "refused"
    elif failure == "length":
        response["choices"][0]["finish_reason"] = "length"
    else:
        response["choices"] *= 2
    client.responses.append(response)
    session = AdoreSession(config, "eurlex", client_factory=client.factory)
    with pytest.raises(ValidationError, match="no fallback"):
        session.judge(QUERY, DOCUMENT)


def test_adore_credentials_are_independent(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="Choose adore.endpoint and adore.model"):
        ChatCompletionsADORE(ADOREConfig(), None)
    config = priced_config(tmp_path)
    monkeypatch.delenv("ADORE_API_KEY", raising=False)
    monkeypatch.setenv("HYDE_API_KEY", "not-used")
    with pytest.raises(ValidationError, match="ADORE_API_KEY"):
        ChatCompletionsADORE(config.adore, tmp_path / "nonexistent.env")
