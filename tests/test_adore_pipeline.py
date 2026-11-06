import json
from dataclasses import replace

import pytest
import yaml
from conftest import FixtureEncoder
from test_pipeline import fixture_config

from clir_benchmark.adore_pipeline import run_adore
from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.cli import main
from clir_benchmark.comparison import compare_adore
from clir_benchmark.config import ADOREConfig
from clir_benchmark.errors import ValidationError
from clir_benchmark.lexical import BM25Index, bm25_path, prepare_bm25
from clir_benchmark.pipeline import (
    embed,
    embedding_path,
    evaluate,
    model_spec,
    prepare_corpus,
    prepare_relevance,
    retrieve,
    run_path,
)
from clir_benchmark.util import file_digest, read_json, write_json


def completion(content):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }


class FixtureADORE:
    def __init__(self, *, score=1, passages=1):
        self.calls = []
        self.responses = []
        self.score = score
        self.passages = passages

    def factory(self, config, env_file):
        return self

    def complete(self, payload):
        self.calls.append(payload)
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        data = json.loads(payload["messages"][1]["content"])
        if data["task"] == "expand":
            number = data["round_number"]
            return completion(
                {
                    "language": data["query"]["question_language"],
                    "passages": [f"expansion round {number}"] * self.passages,
                }
            )
        return completion({"score": self.score})

    @property
    def expansions(self):
        return [
            json.loads(call["messages"][1]["content"])
            for call in self.calls
            if json.loads(call["messages"][1]["content"])["task"] == "expand"
        ]


def prepared_adore(tmp_path, tokenizer, *, annotations=True, monolingual=False, **options):
    config, path = fixture_config(tmp_path, annotations=annotations)
    config = replace(
        config,
        adore=ADOREConfig(
            model="fixture-adore",
            endpoint="https://example.invalid/chat",
            passages_per_round=1,
            input_usd_per_million=4,
            cached_input_usd_per_million=1,
            output_usd_per_million=8,
            **options,
        ),
    )
    if monolingual:
        config = replace(config, settings=("pairwise",), target_languages=("en",))
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    return config, path


def forbidden(*args, **kwargs):
    pytest.fail("This operation must not execute generation, embedding, or retrieval")


def test_three_rounds_return_last_ranking_not_union_and_keep_qrels(
    tmp_path, tokenizer, monkeypatch
):
    config, _ = prepared_adore(tmp_path, tokenizer, monolingual=True)
    index = prepare_bm25(config, "eurlex")
    index_hash = file_digest(index / "index.sqlite")
    qrels = config.relevance_dir("eurlex") / "qrels.parquet"
    qrels_hash = file_digest(qrels)
    chunks = {
        row["source_unit_id"]: row["id"]
        for row in rows(config.corpus_dir("eurlex") / "chunks.parquet")
    }
    relevant, distractor = chunks["a:en"], chunks["distractor:en"]
    expected = [[(relevant, 3.0)], [(relevant, 2.0), (distractor, 1.0)], [(distractor, 3.0)]]
    searches = []

    def search(self, queries, target_languages, *, k=100):
        assert target_languages == ("en",)
        for query in queries:
            searches.append(query)
            number = next(n for n in (1, 2, 3) if f"round {n}" in query)
            yield expected[number - 1]

    monkeypatch.setattr(BM25Index, "search", search)
    client = FixtureADORE()
    run = run_adore(config, "eurlex", baseline="adore-bm25", client_factory=client.factory)
    assert [item["round_number"] for item in client.expansions] == [1, 2, 3]
    assert len(searches) == 3
    assert [(row["chunk_id"], row["score"]) for row in rows(run / "run.parquet")] == expected[-1]
    trace = read_json(run / "traces.json")[0]
    assert trace["rounds_completed"] == 3
    assert len(trace["rounds"]) == 3
    assert trace["output_language"] == "en"
    assert "round 3" in trace["final_query"]
    assert "round 1" not in trace["final_query"]
    assert file_digest(index / "index.sqlite") == index_hash
    assert file_digest(qrels) == qrels_hash
    report_path = evaluate(config, "eurlex", baseline="adore-bm25")
    per_query = read_json(report_path / "per_query.json")
    assert per_query[0]["metrics"]["Hit@100"] == 0
    assert (report_path / "settings.csv").is_file()
    assert (report_path / "per_query.csv").is_file()


def test_cross_language_feedback_keeps_original_query_language_and_no_gold_leakage(
    tmp_path, tokenizer, monkeypatch
):
    config, _ = prepared_adore(tmp_path, tokenizer)
    prepare_bm25(config, "eurlex")
    chunks = list(rows(config.corpus_dir("eurlex") / "chunks.parquet"))

    def search(self, queries, target_languages, *, k=100):
        for _ in queries:
            yield [(row["id"], 1.0) for row in chunks if row["language"] in target_languages]

    monkeypatch.setattr(BM25Index, "search", search)
    client = FixtureADORE()
    run = run_adore(config, "eurlex", baseline="adore-bm25", client_factory=client.factory)
    traces = read_json(run / "traces.json")
    assert {trace["setting_id"] for trace in traces} == {
        "eurlex__en__en",
        "eurlex__en__fr",
        "eurlex__en__mixed",
    }
    assert {trace["output_language"] for trace in traces} == {"en"}
    assert all(call["query"]["question_language"] == "en" for call in client.expansions)
    assert any("preuve" in json.dumps(call.get("feedback")) for call in client.expansions)
    assert "unmatched" not in json.dumps(client.calls)
    assert "evidence_all_languages" not in json.dumps(client.calls)
    assert all("answer" not in call["query"] for call in client.expansions)
    # The identical first-round request is reusable across corpus language settings.
    assert sum(call["round_number"] == 1 for call in client.expansions) == 1
    summary = read_json(run / "manifest.json")["adore_summary"]
    assert summary["api_attempts"] == len(client.calls)
    assert summary["estimated_cost_usd"] == pytest.approx(len(client.calls) * 0.00044)


def test_sufficient_max_grade_feedback_stops_after_first_round(tmp_path, tokenizer):
    config, _ = prepared_adore(tmp_path, tokenizer, monolingual=True, feedback_k=1)
    prepare_bm25(config, "eurlex")
    client = FixtureADORE(score=3)
    run = run_adore(config, "eurlex", baseline="adore-bm25", client_factory=client.factory)
    assert len(client.expansions) == 1
    assert read_json(run / "traces.json")[0]["rounds_completed"] == 1


def test_short_high_grade_rankings_require_more_than_one_round(tmp_path, tokenizer):
    config, _ = prepared_adore(tmp_path, tokenizer, monolingual=True, feedback_k=10)
    prepare_bm25(config, "eurlex")
    client = FixtureADORE(score=3)
    run = run_adore(config, "eurlex", baseline="adore-bm25", client_factory=client.factory)
    trace = read_json(run / "traces.json")[0]
    assert trace["rounds_completed"] == 3
    assert len(client.expansions) == 3
    # Two available documents do not establish that all top-ten slots are relevant.
    assert len(trace["rounds"][0]["ranking"]) == 2


def test_complete_run_reuse_and_scoring_never_call_models(tmp_path, tokenizer, monkeypatch):
    config, _ = prepared_adore(tmp_path, tokenizer)
    prepare_bm25(config, "eurlex")
    client = FixtureADORE()
    run = run_adore(config, "eurlex", baseline="adore-bm25", client_factory=client.factory)
    summary = read_json(run / "manifest.json")["adore_summary"]
    monkeypatch.setattr(BM25Index, "search", forbidden)
    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.complete", forbidden)
    assert run_adore(config, "eurlex", baseline="adore-bm25", client_factory=forbidden) == run
    assert retrieve(config, "eurlex", baseline="adore-bm25") == run
    report = read_json(evaluate(config, "eurlex", baseline="adore-bm25") / "report.json")
    assert report["adore_summary"] == summary
    invocations = [
        read_json(path) for path in (config.output_dir / "adore_invocations").glob("*.json")
    ]
    reused = [entry for entry in invocations if entry["artifact_reused"]]
    assert len(reused) == 1
    assert reused[0]["new_work"]["api_attempts"] == 0
    assert reused[0]["new_work"]["estimated_cost_usd"] == 0


@pytest.mark.parametrize("baseline", ["adore-dense", "adore-hybrid"])
def test_dense_backends_reuse_document_embeddings_and_isolate_runs(tmp_path, tokenizer, baseline):
    config, _ = prepared_adore(tmp_path, tokenizer, monolingual=True)
    original = embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    document_hash = file_digest(original["document"] / "vectors.npy")
    if baseline == "adore-hybrid":
        prepare_bm25(config, "eurlex")
    client = FixtureADORE()
    run = run_adore(
        config,
        "eurlex",
        "bge-m3",
        baseline=baseline,
        client_factory=client.factory,
        encoder_factory=FixtureEncoder,
    )
    spec = model_spec(config, "bge-m3")
    assert run != run_path(config, "eurlex", spec, baseline=baseline.removeprefix("adore-"))
    assert original["document"] == embedding_path(config, "eurlex", spec, "document")
    assert file_digest(original["document"] / "vectors.npy") == document_hash
    report = read_json(evaluate(config, "eurlex", "bge-m3", baseline=baseline) / "report.json")
    assert report["baseline"] == baseline
    assert report["model"] == "bge-m3"
    assert report["metrics"]["setting"][0]["metrics"]["Hit@100"] == 1
    calls = FixtureEncoder.calls
    assert (
        run_adore(
            config,
            "eurlex",
            "bge-m3",
            baseline=baseline,
            client_factory=forbidden,
            encoder_factory=forbidden,
        )
        == run
    )
    assert FixtureEncoder.calls == calls


def test_missing_stages_do_not_execute_paid_work_and_blocked_settings_stay_blocked(
    tmp_path, tokenizer
):
    config, _ = prepared_adore(tmp_path, tokenizer, annotations=False)
    with pytest.raises(ValidationError, match="run-adore"):
        retrieve(config, "eurlex", baseline="adore-bm25")
    with pytest.raises(ValidationError, match="run-adore"):
        evaluate(config, "eurlex", baseline="adore-bm25")
    with pytest.raises(ValidationError, match="index-bm25"):
        run_adore(config, "eurlex", baseline="adore-bm25", client_factory=forbidden)
    prepare_bm25(config, "eurlex")
    client = FixtureADORE()
    run = run_adore(config, "eurlex", baseline="adore-bm25", client_factory=client.factory)
    assert {row["setting_id"] for row in rows(run / "run.parquet")} == {"eurlex__en__en"}
    assert not (run / "eurlex__en__fr.trec").exists()
    assert read_json(run / "manifest.json")["settings"]["eurlex__en__fr"]["status"] == "blocked"


def test_adore_cache_identity_preserves_shared_artifacts(tmp_path, tokenizer):
    config, _ = prepared_adore(tmp_path, tokenizer)
    prepare_bm25(config, "eurlex")
    base = run_path(config, "eurlex", baseline="adore-bm25")
    for updated in (
        replace(config, adore=replace(config.adore, max_rounds=2)),
        replace(config, adore=replace(config.adore, model="different-llm")),
        replace(config, bm25=replace(config.bm25, k1=2)),
    ):
        assert run_path(updated, "eurlex", baseline="adore-bm25") != base
        assert updated.corpus_key("eurlex") == config.corpus_key("eurlex")
        assert updated.relevance_key("eurlex") == config.relevance_key("eurlex")
        assert bm25_path(updated, "eurlex") == bm25_path(config, "eurlex")


def test_smaller_round_budget_reuses_generation_search_and_encoding(tmp_path, tokenizer):
    config, _ = prepared_adore(tmp_path, tokenizer, monolingual=True)
    embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    client = FixtureADORE()
    original = run_adore(
        config,
        "eurlex",
        "bge-m3",
        baseline="adore-dense",
        client_factory=client.factory,
        encoder_factory=FixtureEncoder,
    )
    original_calls = len(client.calls)
    short_config = replace(config, adore=replace(config.adore, max_rounds=1))
    short_run = run_adore(
        short_config,
        "eurlex",
        "bge-m3",
        baseline="adore-dense",
        client_factory=forbidden,
        encoder_factory=forbidden,
    )
    assert original != short_run
    assert len(client.calls) == original_calls
    trace = read_json(short_run / "traces.json")[0]
    assert trace["rounds_completed"] == 1
    assert trace["generation_accounting"]["estimated_cost_usd"] > 0
    assert trace["end_to_end_seconds"] is None
    assert trace["retrieval_seconds"] > 0
    assert trace["query_encoding_seconds"] > 0
    invocation = next(
        item
        for path in (config.output_dir / "adore_invocations").glob("*.json")
        if (item := read_json(path))["run"] == str(short_run)
    )
    assert not invocation["artifact_reused"]
    assert invocation["new_work"]["api_attempts"] == 0
    assert invocation["new_work"]["estimated_cost_usd"] == 0
    assert invocation["new_retrieval_seconds"] == 0
    assert invocation["new_query_encoding_seconds"] == 0


def test_cli_disabled_modes_and_validation_do_not_start_adore(tmp_path, tokenizer, monkeypatch):
    config, path = prepared_adore(tmp_path, tokenizer)
    prepare_bm25(config, "eurlex")
    monkeypatch.setattr("clir_benchmark.cli.run_adore", forbidden)
    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.complete", forbidden)
    prefix = ["--config", str(path)]
    assert main([*prefix, "validate"]) == 0
    assert main([*prefix, "retrieve", "--baseline", "bm25"]) == 0
    assert main([*prefix, "evaluate", "--baseline", "bm25"]) == 0
    raw = yaml.safe_load(path.read_text())
    raw["baselines"] = ["adore-bm25"]
    path.write_text(yaml.safe_dump(raw))
    assert main([*prefix, "validate"]) == 0
    assert main([*prefix, "retrieve"]) == 2
    assert main([*prefix, "evaluate"]) == 2
    assert main([*prefix, "retrieve", "--baseline", "bm25"]) == 0


def test_comparison_uses_matching_cohorts_and_preserves_unknown_costs(
    tmp_path, tokenizer, monkeypatch
):
    config, _ = prepared_adore(tmp_path, tokenizer)
    config = replace(config, adore=replace(config.adore, input_usd_per_million=None))
    prepare_bm25(config, "eurlex")
    retrieve(config, "eurlex", baseline="bm25")
    run = run_adore(config, "eurlex", baseline="adore-bm25", client_factory=FixtureADORE().factory)
    monkeypatch.setattr(BM25Index, "search", forbidden)
    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.complete", forbidden)
    comparison = compare_adore(config, "eurlex", baseline="adore-bm25")
    records = read_json(comparison / "per_query.json")
    assert len(records) == 3
    assert all(row["added_generation_cost_usd"] is None for row in records)
    for record in records:
        assert record["metric_deltas"]["MRR@10"] == (
            record["adore_metrics"]["MRR@10"] - record["control_metrics"]["MRR@10"]
        )
        assert record["added_component_seconds"] == pytest.approx(
            record["adore_component_seconds"] - record["control_component_seconds"]
        )
    report = read_json(comparison / "report.json")
    assert report["adore_summary"] == read_json(run / "manifest.json")["adore_summary"]
    assert report["adore_summary"]["estimated_cost_usd"] is None
    assert (comparison / "settings.csv").is_file()
    control = evaluate(config, "eurlex", baseline="bm25")
    control_report = read_json(control / "report.json")
    control_report["run"]["top_k"] = 50
    write_json(control / "report.json", control_report)
    with pytest.raises(ValidationError, match="top_k differ"):
        compare_adore(config, "eurlex", baseline="adore-bm25")
    control_report["run"]["top_k"] = 100
    write_json(control / "report.json", control_report)
    write_json(control / "per_query.json", read_json(control / "per_query.json")[:-1])
    with pytest.raises(ValidationError, match="cohorts differ"):
        compare_adore(config, "eurlex", baseline="adore-bm25")


def test_failed_request_requires_explicit_retry_and_retains_paid_cost(tmp_path, tokenizer):
    config, _ = prepared_adore(tmp_path, tokenizer, monolingual=True)
    prepare_bm25(config, "eurlex")
    client = FixtureADORE()
    invalid = completion({"language": "fr", "passages": ["mauvaise langue"]})
    client.responses = [invalid]
    with pytest.raises(ValidationError):
        run_adore(config, "eurlex", baseline="adore-bm25", client_factory=client.factory)
    assert not run_path(config, "eurlex", baseline="adore-bm25").exists()
    assert len(client.calls) == 1
    with pytest.raises(ValidationError, match="retry-failed"):
        run_adore(config, "eurlex", baseline="adore-bm25", client_factory=client.factory)
    assert len(client.calls) == 1
    run = run_adore(
        config, "eurlex", baseline="adore-bm25", client_factory=client.factory, retry_failed=True
    )
    summary = read_json(run / "manifest.json")["adore_summary"]
    assert summary["api_attempts"] == len(client.calls)
    assert summary["estimated_cost_usd"] == pytest.approx(len(client.calls) * 0.00044)
    audits = [read_json(path) for path in (config.output_dir / "adore_requests").rglob("*.json")]
    retried = [audit for audit in audits if len(audit["attempts"]) == 2]
    assert len(retried) == 1
    assert retried[0]["attempts"][0]["response"] == invalid
    invocations = [
        read_json(path) for path in (config.output_dir / "adore_invocations").glob("*.json")
    ]
    assert sum(item["new_work"]["api_attempts"] for item in invocations) == len(client.calls)


def test_cli_selects_only_enabled_adore_backends_and_caps_rounds(tmp_path, tokenizer, monkeypatch):
    _, path = prepared_adore(tmp_path, tokenizer)
    raw = yaml.safe_load(path.read_text())
    raw["baselines"] = ["bm25", "adore-bm25", "adore-dense"]
    path.write_text(yaml.safe_dump(raw))
    calls = []

    def run(config, corpus, alias=None, **kwargs):
        calls.append((kwargs["baseline"], alias, config.adore.max_rounds))
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.run_adore", run)
    prefix = ["--config", str(path)]
    assert main([*prefix, "run-adore", "--adore-rounds", "2"]) == 0
    assert calls == [("adore-bm25", None, 2), ("adore-dense", "bge-m3", 2)]
    calls.clear()
    assert main([*prefix, "run-adore", "--baseline", "adore-bm25"]) == 0
    assert calls == [("adore-bm25", None, 3)]
    with pytest.raises(SystemExit):
        main([*prefix, "run-adore", "--adore-rounds", "4"])


def test_bm25_adore_does_not_require_disabled_dense_model_registry(tmp_path, monkeypatch):
    _, path = fixture_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(
        baselines=["adore-bm25"],
        model_registry="missing-disabled-models.yaml",
        models=["unused-model"],
    )
    path.write_text(yaml.safe_dump(raw))
    calls = []

    def run(config, corpus, alias=None, **kwargs):
        calls.append((kwargs["baseline"], alias))
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.run_adore", run)
    monkeypatch.setattr("clir_benchmark.cli.load_registry", forbidden)
    assert main(["--config", str(path), "run-adore"]) == 0
    assert calls == [("adore-bm25", None)]
