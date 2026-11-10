from dataclasses import replace

import pyarrow.parquet as pq
import pytest
from conftest import FixtureEncoder
from test_hyde import FixtureGenerator
from test_pipeline import fixture_config, write_rows
from test_translation import FixtureTranslator

from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.comparison import compare_hyde, compare_translations
from clir_benchmark.config import HyDEConfig, TranslationConfig
from clir_benchmark.errors import ValidationError
from clir_benchmark.hyde import prepare_hyde
from clir_benchmark.pipeline import (
    embed,
    embedding_path,
    evaluate,
    model_spec,
    prepare_corpus,
    prepare_relevance,
    retrieve,
)
from clir_benchmark.translation import prepare_translations
from clir_benchmark.util import file_digest, read_json


def document_fixture(tmp_path, tokenizer):
    base, _ = fixture_config(tmp_path, annotations=False)
    dataset = tmp_path / "dataset"
    candidates = pq.read_table(dataset / "data/qas/train-00000.parquet").to_pylist()
    for row in candidates:
        if row["_id"] == "qdrop":
            row["total_score"] = 40.0
    write_rows(dataset, "qas", candidates)
    config = replace(
        base,
        candidate_mode="best-per-document",
        retrieval_unit="document",
        target_languages=("en", "fr"),
        baselines=("dense", "translate-dense", "hyde-dense"),
        translation=TranslationConfig(
            model="fixture",
            endpoint="https://fixture.invalid",
            api_key_env=None,
            input_usd_per_million=1,
            output_usd_per_million=2,
        ),
        hyde=HyDEConfig(
            model="fixture",
            endpoint="https://fixture.invalid",
            api_key_env=None,
            output_languages=("en", "fr"),
            input_usd_per_million=1,
            output_usd_per_million=2,
        ),
    )
    prepare_corpus(base, "eurlex", chunker=Chunker(tokenizer, base.chunking))
    return base, config


def forbidden(*args, **kwargs):
    pytest.fail("This operation must use stored artifacts only")


def test_document_pipeline_keeps_highest_score_without_spans_and_reuses_vectors(
    tmp_path,
    tokenizer,
    monkeypatch,
):
    base, config = document_fixture(tmp_path, tokenizer)
    before = embed(base, "eurlex", "bge-m3", kind="document", encoder_factory=FixtureEncoder)
    spec = model_spec(config, "bge-m3")
    assert before["document"] == embedding_path(config, "eurlex", spec, "document")
    vector_hash = file_digest(before["document"] / "vectors.npy")
    path = prepare_relevance(config, "eurlex")
    assert [r["id"] for r in rows(path / "queries.parquet")] == ["qdrop"]
    assert {r["id"] for r in rows(path / "input_queries.parquet")} == {"qbest", "qlow", "qdrop"}
    assert len((path / "input_qas.jsonl").read_text().splitlines()) == 3
    assert len((path / "input_source_qrels.jsonl").read_text().splitlines()) == 6
    assert read_json(path / "manifest.json")["selected_query_languages"] == {"en": 1}
    assert read_json(path / "selection.json")[0]["selected_ids"] == ["qdrop"]
    assert {r["document_id"] for r in rows(path / "qrels.parquet")} == {"a:en", "a:fr"}
    assert all(r["status"] == "eligible" for r in rows(path / "eligibility.parquet"))
    embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    run = retrieve(config, "eurlex", "bge-m3")
    rankings = list(rows(run / "run.parquet"))
    assert rankings and all("document_id" in r and "winning_chunk_id" in r for r in rankings)
    assert all("chunk_id" not in r for r in rankings)
    assert file_digest(before["document"] / "vectors.npy") == vector_hash
    for target in (
        "clir_benchmark.generation.ChatCompletionsClient.complete",
        "clir_benchmark.retrieval.DenseIndex.search_documents",
        "clir_benchmark.encoders.HuggingFaceEncoder.__init__",
    ):
        monkeypatch.setattr(target, forbidden)
    report_dir = evaluate(config, "eurlex", "bge-m3")
    report = read_json(report_dir / "report.json")
    assert report["retrieval_unit"] == "document"
    assert report["truncation"]["support_exposure"]["applicable"] is False
    assert read_json(report_dir / "mteb.json")["task_name"] == "CLIRBenchmark_eurlex_Document"
    detail = list(rows(report_dir / "analysis/rankings.parquet"))
    assert len(detail) == len(rankings)
    assert all(r["document_id"] == r["source_unit_id"] for r in detail)
    assert all(r["winning_chunk_id"] and r["text"] for r in detail)
    inventory = read_json(report_dir / "analysis_manifest.json")
    paths = [r["path"] for r in inventory["files"]]
    assert any(p.endswith("selection.json") for p in paths)
    assert any(p.endswith("vectors.npy") for p in paths)
    assert evaluate(config, "eurlex", "bge-m3") == report_dir


def test_document_variants_share_cohort_generation_and_accounting(tmp_path, tokenizer, monkeypatch):
    _, config = document_fixture(tmp_path, tokenizer)
    prepare_relevance(config, "eurlex")
    translator, generator = FixtureTranslator(), FixtureGenerator()
    for language in ("en", "fr"):
        prepare_translations(config, "eurlex", language, translator_factory=translator.factory)
        prepare_hyde(config, "eurlex", language, generator_factory=generator.factory)
    assert len(translator.calls) == 1
    assert len(generator.calls) == 2
    for alias in ("bge-m3", "LaBSE"):
        embed(config, "eurlex", alias, encoder_factory=FixtureEncoder)
        retrieve(config, "eurlex", alias)
        for language in ("en", "fr"):
            for baseline, keyword in (
                ("translate-dense", "translation_language"),
                ("hyde-dense", "hyde_language"),
            ):
                kwargs = {keyword: language}
                embed(
                    config, "eurlex", alias, kind="query", encoder_factory=FixtureEncoder, **kwargs
                )
                retrieve(config, "eurlex", alias, baseline=baseline, **kwargs)
    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.complete", forbidden)
    monkeypatch.setattr("clir_benchmark.retrieval.DenseIndex.search_documents", forbidden)
    for alias in ("bge-m3", "LaBSE"):
        original = evaluate(config, "eurlex", alias)
        cohort = {(r["query_id"], r["setting_id"]) for r in read_json(original / "per_query.json")}
        assert len(cohort) == 3
        for language in ("en", "fr"):
            for baseline, keyword in (
                ("translate-dense", "translation_language"),
                ("hyde-dense", "hyde_language"),
            ):
                report = evaluate(config, "eurlex", alias, baseline=baseline, **{keyword: language})
                assert {
                    (r["query_id"], r["setting_id"]) for r in read_json(report / "per_query.json")
                } == cohort
        translated = compare_translations(
            config, "eurlex", alias, baseline="dense", languages=("en", "fr")
        )
        hypothetical = compare_hyde(
            config, "eurlex", alias, languages=("en", "fr"), controls=("dense", "translate-dense")
        )
        assert read_json(translated / "report.json")["settings"]
        assert read_json(hypothetical / "report.json")["languages"]
        for path in (translated, hypothetical):
            assert read_json(path / "report.json")["retrieval_unit"] == "document"
            assert "retrieval_unit" in (path / "settings.csv").read_text().splitlines()[0]
    assert len(translator.calls) == 1 and len(generator.calls) == 2


def test_document_dense_does_not_require_generation_and_rejects_unsupported_modes(
    tmp_path, tokenizer, monkeypatch
):
    _, config = document_fixture(tmp_path, tokenizer)
    config = replace(
        config, baselines=("dense",), translation=TranslationConfig(), hyde=HyDEConfig()
    )
    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.__init__", forbidden)
    prepare_relevance(config, "eurlex")
    embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    retrieve(config, "eurlex", "bge-m3")
    evaluate(config, "eurlex", "bge-m3")
    for baseline in ("bm25", "hybrid", "adore-dense", "rerank-dense"):
        with pytest.raises(ValidationError, match="Document retrieval supports"):
            retrieve(config, "eurlex", "bge-m3", baseline=baseline)


def test_document_smoke_rejects_unsupported_mode_before_any_work(tmp_path, monkeypatch):
    from clir_benchmark import smoke

    config, _ = fixture_config(tmp_path)
    config = replace(config, retrieval_unit="document", baselines=("translate-hybrid",))
    monkeypatch.setattr(smoke, "prepare_translations", forbidden)
    monkeypatch.setattr(smoke, "prepare_corpus", forbidden)
    with pytest.raises(ValidationError, match="Document retrieval supports"):
        smoke.run_smoke(config, "eurlex", "bge-m3")
    assert not config.output_dir.exists()
