import json
from dataclasses import asdict, replace

import pytest
import yaml
from conftest import FixtureEncoder
from test_adore_pipeline import FixtureADORE, prepared_adore
from test_hyde import FixtureGenerator
from test_hyde_pipeline import fixture_hyde
from test_pipeline import fixture_config
from test_translation import FixtureTranslator
from test_translation_pipeline import prepared

from clir_benchmark.adore_pipeline import run_adore
from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.cli import main
from clir_benchmark.errors import ValidationError
from clir_benchmark.hyde import prepare_hyde
from clir_benchmark.lexical import prepare_bm25
from clir_benchmark.pipeline import (
    embed,
    evaluate,
    model_spec,
    prepare_corpus,
    prepare_relevance,
    retrieve,
    run_path,
)
from clir_benchmark.rerank_pipeline import compare_rerank, rerank
from clir_benchmark.reranking import ScoredBatch
from clir_benchmark.translation import prepare_translations
from clir_benchmark.util import file_digest, read_json, write_json


def forbidden(*args, **kwargs):
    pytest.fail("This operation must not run retrieval, generation, or a reranking model")


class FixtureReranker:
    def __init__(self, score=None):
        self.calls = []
        self.loads = []
        self.fixed_score = score
        self.metadata = {"fixture": True, "device": "cpu", "dtype": "float32"}

    def factory(self, config, token=None):
        self.loads.append(asdict(config))
        return self

    def score(self, pairs):
        self.calls.extend(pairs)
        return ScoredBatch(
            [
                self.fixed_score
                if self.fixed_score is not None
                else 0.9
                if document == "preuve"
                else 0.2
                for _, document in pairs
            ],
            [
                {
                    "input_tokens": len(query) + len(document) + 1,
                    "encoded_tokens": len(query) + len(document) + 1,
                    "truncated": False,
                    "query_truncated": False,
                    "document_truncated": False,
                    "query_visible_ranges": [[0, len(query)]],
                    "document_visible_ranges": [[0, len(document)]],
                }
                for query, document in pairs
            ],
        )


def parent_fixture(tmp_path, tokenizer, parent="bm25"):
    if parent.startswith("translate-"):
        config, path = prepared(tmp_path, tokenizer)
    elif parent == "hyde-dense":
        config, path = fixture_hyde(tmp_path, tokenizer)
    elif parent.startswith("adore-"):
        config, path = prepared_adore(tmp_path, tokenizer, max_rounds=1)
    else:
        config, path = fixture_config(tmp_path)
        prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
        prepare_relevance(config, "eurlex")
    backend = parent.removeprefix("translate-").removeprefix("adore-")
    alias = None if backend == "bm25" else "bge-m3"
    if backend in {"bm25", "hybrid"}:
        prepare_bm25(config, "eurlex")
    languages = {}
    if parent.startswith("translate-"):
        prepare_translations(config, "eurlex", "fr", translator_factory=FixtureTranslator().factory)
        languages = {"translation_language": "fr"}
    elif parent == "hyde-dense":
        prepare_hyde(config, "eurlex", "fr", generator_factory=FixtureGenerator("preuve").factory)
        languages = {"hyde_language": "fr"}
    if alias:
        embed(config, "eurlex", alias, encoder_factory=FixtureEncoder, **languages)
    if parent.startswith("adore-"):
        parent_run = run_adore(
            config,
            "eurlex",
            alias,
            baseline=parent,
            client_factory=FixtureADORE().factory,
            encoder_factory=FixtureEncoder,
        )
    else:
        parent_run = retrieve(config, "eurlex", alias, baseline=parent, **languages)
    return config, path, alias, languages, parent_run


def grouped_rankings(path):
    result = {}
    for row in rows(path / "run.parquet"):
        result.setdefault((row["setting_id"], row["query_id"]), []).append(row)
    return {key: sorted(value, key=lambda row: row["rank"]) for key, value in result.items()}


def test_reranker_reorders_only_parent_candidates_and_preserves_sources(tmp_path, tokenizer):
    config, _, alias, _, parent = parent_fixture(tmp_path, tokenizer, "dense")
    parent_hash = file_digest(parent / "run.parquet")
    qrels = config.relevance_dir("eurlex") / "qrels.parquet"
    qrels_hash = file_digest(qrels)
    model = FixtureReranker()
    run = rerank(config, "eurlex", alias, baseline="rerank-dense", reranker_factory=model.factory)
    assert len(model.loads) == 1
    assert set(model.calls) == {("alpha", "alpha"), ("alpha", "preuve")}
    assert len(model.calls) == 2  # Repeated text and overlapping language pools share scores.
    corpus = {row["id"]: row for row in rows(config.corpus_dir("eurlex") / "chunks.parquet")}
    parents, variants = grouped_rankings(parent), grouped_rankings(run)
    assert parents.keys() == variants.keys()
    for key, parent_rows in parents.items():
        expected = sorted(
            parent_rows,
            key=lambda row: (
                -float(corpus[row["chunk_id"]]["text"] == "preuve"),
                row["rank"],
                row["chunk_id"],
            ),
        )
        assert [row["chunk_id"] for row in variants[key]] == [row["chunk_id"] for row in expected]
        assert [row["rank"] for row in variants[key]] == list(range(1, len(expected) + 1))
    assert file_digest(parent / "run.parquet") == parent_hash
    assert file_digest(qrels) == qrels_hash
    candidates = list(rows(run / "candidates.parquet"))
    assert len(candidates) == sum(map(len, parents.values()))
    assert all(row["parent_rank"] >= 1 for row in candidates)
    assert all(json.loads(row["audit_json"])["document_truncated"] is False for row in candidates)
    assert list(run.glob("*.trec"))
    # Distinct chunks with identical text share one inference and its elapsed time.
    trace = next(
        row for row in read_json(run / "traces.json") if row["setting_id"] == "eurlex__en__en"
    )
    assert len(trace["score_keys"]) == 2
    assert len(set(trace["score_keys"])) == 1
    cached_pair = read_json(
        config.output_dir / "reranking_pairs" / trace["score_keys"][0] / "result.json"
    )
    assert trace["reranking_seconds_amortized"] == cached_pair["seconds_amortized"]


@pytest.mark.parametrize(
    "parent",
    [
        "dense",
        "bm25",
        "hybrid",
        "translate-dense",
        "translate-bm25",
        "translate-hybrid",
        "hyde-dense",
        "adore-dense",
        "adore-bm25",
        "adore-hybrid",
    ],
)
def test_all_parent_modes_use_original_query_and_exact_document_text(tmp_path, tokenizer, parent):
    config, _, alias, languages, parent_run = parent_fixture(tmp_path, tokenizer, parent)
    model = FixtureReranker()
    baseline = f"rerank-{parent}"
    output = rerank(
        config, "eurlex", alias, baseline=baseline, reranker_factory=model.factory, **languages
    )
    assert output != parent_run
    assert model.calls and {query for query, _ in model.calls} == {"alpha"}
    assert {document for _, document in model.calls} <= {"alpha", "preuve"}
    assert "unmatched" not in json.dumps(model.calls)
    report_path = evaluate(config, "eurlex", alias, baseline=baseline, **languages)
    report = read_json(report_path / "report.json")
    assert report["baseline"] == baseline
    assert report["model"] == alias
    per_query = read_json(report_path / "per_query.json")
    assert len(per_query) == 3
    assert all(row["query_language"] == "en" for row in per_query)
    assert all(
        row["metrics"]["Recall@100"] == row["candidate_control_metrics"]["Recall@100"]
        for row in per_query
    )
    originals = {
        (row["setting_id"], row["query_id"]): row
        for row in read_json(
            evaluate(config, "eurlex", alias, baseline=parent, **languages) / "per_query.json"
        )
    }
    for row in per_query:
        original = originals[row["setting_id"], row["query_id"]]
        cost_field = (
            "generation_cost_usd"
            if parent == "hyde-dense" or parent.startswith("adore-")
            else "translation_cost_usd"
            if parent.startswith("translate-")
            else None
        )
        assert row["parent_api_cost_usd"] == (original[cost_field] if cost_field else 0)
        if cost_field:
            assert row[cost_field] == original[cost_field]
        assert row["reranking_cost_usd"] is None
        assert row["total_cost_usd"] is None
        assert row["end_to_end_seconds"] is None
        assert row["component_seconds"] == pytest.approx(
            row["parent_component_seconds"] + row["reranking_seconds_amortized"]
        )
    assert (report_path / "settings.csv").is_file()


def test_missing_parent_fails_before_loading_reranker(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    with pytest.raises(ValidationError, match="[Pp]arent|retrieve|prepared"):
        rerank(config, "eurlex", baseline="rerank-bm25", reranker_factory=forbidden)
    with pytest.raises(ValidationError):
        retrieve(config, "eurlex", baseline="rerank-bm25")
    with pytest.raises(ValidationError):
        evaluate(config, "eurlex", baseline="rerank-bm25")


def test_empty_bm25_candidate_lists_remain_in_cohort_and_scores(tmp_path, tokenizer):
    config, _, _, _, _ = parent_fixture(tmp_path, tokenizer)
    output = rerank(
        config, "eurlex", baseline="rerank-bm25", reranker_factory=FixtureReranker().factory
    )
    assert (output / "eurlex__en__fr.trec").read_text() == ""
    records = read_json(evaluate(config, "eurlex", baseline="rerank-bm25") / "per_query.json")
    empty = next(row for row in records if row["setting_id"] == "eurlex__en__fr")
    assert empty["metrics"]["Hit@100"] == 0
    assert empty["candidate_control_metrics"]["Hit@100"] == 0


def test_all_empty_candidates_do_not_initialize_reranker(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    config = replace(config, settings=("pairwise",), target_languages=("fr",))
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    prepare_bm25(config, "eurlex")
    retrieve(config, "eurlex", baseline="bm25")
    output = rerank(config, "eurlex", baseline="rerank-bm25", reranker_factory=forbidden)
    assert list(rows(output / "run.parquet")) == []
    records = read_json(evaluate(config, "eurlex", baseline="rerank-bm25") / "per_query.json")
    assert len(records) == 1
    assert records[0]["metrics"]["Hit@100"] == 0


def test_cached_runs_and_offline_comparisons_never_load_model(tmp_path, tokenizer, monkeypatch):
    config, _, _, _, _ = parent_fixture(tmp_path, tokenizer)
    model = FixtureReranker()
    output = rerank(config, "eurlex", baseline="rerank-bm25", reranker_factory=model.factory)
    original_manifest = read_json(output / "manifest.json")
    monkeypatch.setattr("clir_benchmark.reranking.Qwen3Reranker.score", forbidden)
    assert rerank(config, "eurlex", baseline="rerank-bm25", reranker_factory=forbidden) == output
    assert retrieve(config, "eurlex", baseline="rerank-bm25") == output
    evaluate(config, "eurlex", baseline="rerank-bm25")
    compared = compare_rerank(config, "eurlex", baseline="rerank-bm25")
    assert (compared / "report.json").is_file()
    assert read_json(output / "manifest.json") == original_manifest
    invocations = [
        read_json(path) for path in (config.output_dir / "reranking_invocations").glob("*.json")
    ]
    assert any(row["new_pairs"] == 0 for row in invocations)


def test_smaller_candidate_budget_uses_same_prefix_control_and_reuses_pair_scores(
    tmp_path, tokenizer
):
    config, _, alias, _, parent = parent_fixture(tmp_path, tokenizer, "dense")
    model = FixtureReranker()
    complete = rerank(
        config, "eurlex", alias, baseline="rerank-dense", reranker_factory=model.factory
    )
    restricted = replace(config, reranker=replace(config.reranker, candidate_k=1))
    output = rerank(
        restricted, "eurlex", alias, baseline="rerank-dense", reranker_factory=forbidden
    )
    assert output != complete
    parent_rows, variant_rows = grouped_rankings(parent), grouped_rankings(output)
    assert all(len(value) == 1 for value in variant_rows.values())
    for key, value in variant_rows.items():
        assert value[0]["chunk_id"] == parent_rows[key][0]["chunk_id"]
    comparison = compare_rerank(restricted, "eurlex", alias, baseline="rerank-dense")
    records = read_json(comparison / "per_query.json")
    assert records and all(
        all(value == 0 for value in row["metric_deltas"].values()) for row in records
    )
    assert all("full_parent_metric_deltas" in row for row in records)
    assert any(row["full_parent_metric_deltas"]["Recall@100"] < 0 for row in records)
    assert restricted.corpus_key("eurlex") == config.corpus_key("eurlex")
    assert restricted.relevance_key("eurlex") == config.relevance_key("eurlex")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), None, "0.5", True])
def test_invalid_reranking_scores_never_publish_run(tmp_path, tokenizer, invalid):
    config, _, _, _, _ = parent_fixture(tmp_path, tokenizer)

    class InvalidReranker(FixtureReranker):
        def score(self, pairs):
            batch = super().score(pairs)
            return ScoredBatch([invalid] * len(pairs), batch.audits)

    with pytest.raises(ValidationError):
        rerank(
            config,
            "eurlex",
            baseline="rerank-bm25",
            reranker_factory=InvalidReranker().factory,
        )
    assert not run_path(config, "eurlex", baseline="rerank-bm25").exists()


def test_configuration_changes_partition_reranked_runs_without_changing_parent(tmp_path, tokenizer):
    config, _, alias, _, parent = parent_fixture(tmp_path, tokenizer, "dense")
    spec = model_spec(config, alias)
    base = run_path(config, "eurlex", spec, baseline="rerank-dense")
    for reranker in (
        replace(config.reranker, candidate_k=10),
        replace(config.reranker, instruction="Find documents relevant to the query."),
        replace(config.reranker, max_length=4096),
        replace(config.reranker, revision="a" * 40),
    ):
        updated = replace(config, reranker=reranker)
        assert run_path(updated, "eurlex", spec, baseline="rerank-dense") != base
        assert run_path(updated, "eurlex", spec, baseline="dense") == parent


def test_pair_scores_reuse_across_distinct_retrievers_and_parent_fingerprints(tmp_path, tokenizer):
    config, _, alias, _, _ = parent_fixture(tmp_path, tokenizer, "dense")
    model = FixtureReranker()
    dense = rerank(config, "eurlex", alias, baseline="rerank-dense", reranker_factory=model.factory)
    prepare_bm25(config, "eurlex")
    parent = retrieve(config, "eurlex", baseline="bm25")
    lexical = rerank(config, "eurlex", baseline="rerank-bm25", reranker_factory=forbidden)
    assert dense != lexical
    for row in rows(lexical / "candidates.parquet"):
        assert row["score"] == 0.2
    manifest = read_json(parent / "manifest.json")
    manifest["run_name"] = "changed-parent-provenance"
    write_json(parent / "manifest.json", manifest)
    assert run_path(config, "eurlex", baseline="rerank-bm25") != lexical


def test_mismatched_parent_report_cohort_is_rejected(tmp_path, tokenizer):
    config, _, _, _, _ = parent_fixture(tmp_path, tokenizer)
    rerank(config, "eurlex", baseline="rerank-bm25", reranker_factory=FixtureReranker().factory)
    parent_report = evaluate(config, "eurlex", baseline="bm25")
    records = read_json(parent_report / "per_query.json")
    write_json(parent_report / "per_query.json", records[:-1])
    with pytest.raises(ValidationError, match="cohorts differ|stale"):
        compare_rerank(config, "eurlex", baseline="rerank-bm25")


@pytest.mark.parametrize("field", ["metrics", "query_language", "target_languages", "source_count"])
def test_stale_parent_metrics_or_cohort_metadata_cannot_enter_comparisons(
    tmp_path, tokenizer, field
):
    config, _, _, _, _ = parent_fixture(tmp_path, tokenizer)
    rerank(config, "eurlex", baseline="rerank-bm25", reranker_factory=FixtureReranker().factory)
    parent_report = evaluate(config, "eurlex", baseline="bm25")
    records = read_json(parent_report / "per_query.json")
    changed = next(row for row in records if row["setting_id"] == "eurlex__en__en")
    if field == "metrics":
        assert changed["metrics"]["Hit@100"] == 1
        changed["metrics"]["Hit@100"] = 0
    else:
        changed[field] = {"query_language": "fr", "target_languages": "fr", "source_count": 99}[
            field
        ]
    write_json(parent_report / "per_query.json", records)
    with pytest.raises(ValidationError):
        compare_rerank(config, "eurlex", baseline="rerank-bm25")


def test_duplicate_parent_report_rows_cannot_hide_cohort_mismatch(tmp_path, tokenizer):
    config, _, _, _, _ = parent_fixture(tmp_path, tokenizer)
    rerank(config, "eurlex", baseline="rerank-bm25", reranker_factory=FixtureReranker().factory)
    parent_report = evaluate(config, "eurlex", baseline="bm25")
    records = read_json(parent_report / "per_query.json")
    write_json(parent_report / "per_query.json", [*records, records[0]])
    with pytest.raises(ValidationError, match="[Dd]uplicate|cohort"):
        compare_rerank(config, "eurlex", baseline="rerank-bm25")


def test_cli_disabled_reranker_and_model_free_bm25_selection(tmp_path, tokenizer, monkeypatch):
    _, path, _, _, _ = parent_fixture(tmp_path, tokenizer)
    monkeypatch.setattr("clir_benchmark.cli.rerank", forbidden)
    prefix = ["--config", str(path)]
    assert main([*prefix, "validate"]) == 0
    assert main([*prefix, "retrieve", "--baseline", "bm25"]) == 0
    assert main([*prefix, "evaluate", "--baseline", "bm25"]) == 0
    raw = yaml.safe_load(path.read_text())
    raw.update(baselines=["rerank-bm25"], models=["disabled-encoder"], model_registry="absent.yaml")
    path.write_text(yaml.safe_dump(raw))
    calls = []

    def execute(config, corpus, alias=None, **kwargs):
        calls.append((kwargs["baseline"], alias, config.reranker.candidate_k))
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.rerank", execute)
    monkeypatch.setattr("clir_benchmark.cli.load_registry", forbidden)
    assert main([*prefix, "rerank", "--rerank-candidates", "20"]) == 0
    assert calls == [("rerank-bm25", None, 20)]


def test_cli_multiple_reranking_modes_and_language_variants(tmp_path, monkeypatch):
    _, path = fixture_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["baselines"] = ["dense", "rerank-dense", "rerank-bm25"]
    path.write_text(yaml.safe_dump(raw))
    calls = []

    def execute(config, corpus, alias=None, **kwargs):
        calls.append(
            (
                kwargs["baseline"],
                alias,
                kwargs.get("translation_language"),
                kwargs.get("hyde_language"),
                config.adore.max_rounds,
            )
        )
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.rerank", execute)
    prefix = ["--config", str(path)]
    assert main([*prefix, "rerank", "--model", "bge-m3", "--model", "LaBSE"]) == 0
    assert calls == [
        ("rerank-dense", "bge-m3", None, None, 3),
        ("rerank-dense", "LaBSE", None, None, 3),
        ("rerank-bm25", None, None, None, 3),
    ]
    calls.clear()
    assert (
        main(
            [
                *prefix,
                "rerank",
                "--baseline",
                "rerank-translate-bm25",
                "--translation-language",
                "fr",
                "--translation-language",
                "en",
            ]
        )
        == 0
    )
    assert calls == [
        ("rerank-translate-bm25", None, "fr", None, 3),
        ("rerank-translate-bm25", None, "en", None, 3),
    ]
    calls.clear()
    assert (
        main([*prefix, "rerank", "--baseline", "rerank-hyde-dense", "--hyde-language", "fr"]) == 0
    )
    assert calls == [("rerank-hyde-dense", "bge-m3", None, "fr", 3)]
    calls.clear()
    assert main([*prefix, "rerank", "--baseline", "rerank-adore-bm25", "--adore-rounds", "2"]) == 0
    assert calls == [("rerank-adore-bm25", None, None, None, 2)]
