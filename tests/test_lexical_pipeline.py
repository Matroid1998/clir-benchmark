import json
from dataclasses import replace

import pytest
import yaml
from conftest import FixtureEncoder
from test_pipeline import fixture_config

from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.cli import main
from clir_benchmark.config import load_config
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
from clir_benchmark.retrieval import DenseIndex, reciprocal_rank_fusion
from clir_benchmark.util import file_digest, read_json


def prepare_fixture(tmp_path, tokenizer):
    config, path = fixture_config(tmp_path)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    return config, path


def test_bm25_cli_needs_no_embedding_models_and_scores_zero_hit_queries(
    tmp_path, tokenizer, monkeypatch
):
    config, path = prepare_fixture(tmp_path, tokenizer)

    def forbidden(*args, **kwargs):
        pytest.fail("BM25 must not load models, embeddings or execute dense retrieval")

    monkeypatch.setattr("clir_benchmark.pipeline.load_embeddings", forbidden)
    monkeypatch.setattr("clir_benchmark.pipeline.model_spec", forbidden)
    monkeypatch.setattr("clir_benchmark.cli.embed", forbidden)
    prefix = ["--config", str(path)]
    assert main([*prefix, "index-bm25"]) == 0
    assert main([*prefix, "retrieve", "--baseline", "bm25"]) == 0
    assert main([*prefix, "evaluate", "--baseline", "bm25"]) == 0
    run = run_path(config, "eurlex", baseline="bm25")
    rankings = list(rows(run / "run.parquet"))
    assert {r["query_id"] for r in rankings} == {"qbest"}
    mono = [r for r in rankings if r["setting_id"] == "eurlex__en__en"]
    assert len(mono) == 2  # The unrelated distractor is searched too.
    assert not [r for r in rankings if r["setting_id"] == "eurlex__en__fr"]
    report = read_json(config.output_dir / "reports" / "eurlex" / run.name / "report.json")
    assert report["baseline"] == "bm25"
    assert report["model"] is None
    assert report["truncation"]["applicable"] is False
    metrics = {r["setting_id"]: r["metrics"] for r in report["metrics"]["setting"]}
    assert metrics["eurlex__en__fr"]["Hit@100"] == 0
    assert metrics["eurlex__en__en"]["Hit@100"] == 1
    assert (run / "eurlex__en__fr.trec").read_text() == ""
    monkeypatch.setattr(BM25Index, "search", forbidden)
    assert retrieve(config, "eurlex", baseline="bm25") == run
    assert main([*prefix, "retrieve", "--baseline", "bm25", "--model", "bge-m3"]) == 2


def test_hybrid_reuses_dense_embeddings_and_fuses_each_language_pool(
    tmp_path, tokenizer, monkeypatch
):
    config, _ = prepare_fixture(tmp_path, tokenizer)
    prepare_bm25(config, "eurlex")
    embeddings = embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    encoder_calls = FixtureEncoder.calls
    dense_run = retrieve(config, "eurlex", "bge-m3")
    bm25_run = retrieve(config, "eurlex", baseline="bm25")
    relevance_file = config.relevance_dir("eurlex") / "qrels.parquet"
    qrels_hash = file_digest(relevance_file)
    depths = []
    original_dense = DenseIndex.search
    original_lexical = BM25Index.search

    def dense_search(self, *args, **kwargs):
        depths.append(("dense", kwargs["k"]))
        return original_dense(self, *args, **kwargs)

    def lexical_search(self, *args, **kwargs):
        depths.append(("bm25", kwargs["k"]))
        return original_lexical(self, *args, **kwargs)

    monkeypatch.setattr(DenseIndex, "search", dense_search)
    monkeypatch.setattr(BM25Index, "search", lexical_search)
    hybrid_run = retrieve(config, "eurlex", "bge-m3", baseline="hybrid")
    assert set(depths) == {("dense", 1000), ("bm25", 1000)}
    assert len({dense_run, bm25_run, hybrid_run}) == 3
    assert file_digest(relevance_file) == qrels_hash
    assert FixtureEncoder.calls == encoder_calls
    spec = model_spec(config, "bge-m3")
    assert embeddings["document"] == embedding_path(config, "eurlex", spec, "document")
    dense, lexical, hybrid = [
        list(rows(path / "run.parquet")) for path in (dense_run, bm25_run, hybrid_run)
    ]
    for setting in read_json(config.relevance_dir("eurlex") / "settings.json"):
        expected = reciprocal_rank_fusion(
            [(r["chunk_id"], r["score"]) for r in dense if r["setting_id"] == setting],
            [(r["chunk_id"], r["score"]) for r in lexical if r["setting_id"] == setting],
            config.hybrid,
        )
        actual = [(r["chunk_id"], r["score"]) for r in hybrid if r["setting_id"] == setting]
        assert actual == expected
    report_path = evaluate(config, "eurlex", "bge-m3", baseline="hybrid")
    report = read_json(report_path / "report.json")
    assert report["baseline"] == "hybrid"
    assert report["model"] == "bge-m3"
    assert report["truncation"]["scope"] == "dense_component"
    assert report["run"]["fusion"]["candidate_k"] == 1000
    assert all(r["metrics"]["Hit@100"] == 1 for r in report["metrics"]["setting"])
    assert "hybrid,bge-m3" in (report_path / "settings.csv").read_text()


def test_index_and_run_cache_dependencies_are_separate(tmp_path, tokenizer):
    config, _ = prepare_fixture(tmp_path, tokenizer)
    original_index = prepare_bm25(config, "eurlex")
    spec = model_spec(config, "bge-m3")
    index_hash = file_digest(original_index / "index.sqlite")
    changed_score = replace(config, bm25=replace(config.bm25, b=0.1))
    assert prepare_bm25(changed_score, "eurlex") == original_index
    assert file_digest(original_index / "index.sqlite") == index_hash
    assert run_path(changed_score, "eurlex", baseline="bm25") != run_path(
        config, "eurlex", baseline="bm25"
    )
    assert run_path(changed_score, "eurlex", spec, baseline="hybrid") != run_path(
        config, "eurlex", spec, baseline="hybrid"
    )
    assert run_path(changed_score, "eurlex", spec) == run_path(config, "eurlex", spec)
    changed_fusion = replace(config, hybrid=replace(config.hybrid, dense_weight=2))
    assert run_path(changed_fusion, "eurlex", spec, baseline="hybrid") != run_path(
        config, "eurlex", spec, baseline="hybrid"
    )
    assert run_path(changed_fusion, "eurlex", baseline="bm25") == run_path(
        config, "eurlex", baseline="bm25"
    )
    overlay = tmp_path / "evidence.jsonl"
    overlay.write_text(
        json.dumps(
            {
                "query_id": "qbest",
                "evidence_all_languages": {"fr": [{"source_unit_id": "a:fr", "evidence": []}]},
            }
        )
    )
    for updated in (replace(config, evidence_file=overlay), replace(config, candidate_mode="all")):
        assert prepare_bm25(updated, "eurlex") == original_index
        assert updated.corpus_key("eurlex") == config.corpus_key("eurlex")
        assert run_path(updated, "eurlex", baseline="bm25") != run_path(
            config, "eurlex", baseline="bm25"
        )


def test_bm25_index_preparation_is_independent_of_qas(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    assert not config.relevance_dir("eurlex").exists()
    index = prepare_bm25(config, "eurlex")
    assert read_json(index / "manifest.json")["chunk_count"] == 3


def test_missing_artifacts_and_blocked_settings_are_actionable(tmp_path, tokenizer):
    config, _ = prepare_fixture(tmp_path, tokenizer)
    with pytest.raises(ValidationError, match="index-bm25"):
        retrieve(config, "eurlex", baseline="bm25")
    with pytest.raises(ValidationError, match="retrieve explicitly"):
        evaluate(config, "eurlex", baseline="bm25")
    prepare_bm25(config, "eurlex")
    with pytest.raises(ValidationError, match="embed first"):
        retrieve(config, "eurlex", "bge-m3", baseline="hybrid")
    with pytest.raises(ValidationError, match="requires a dense model"):
        retrieve(config, "eurlex", baseline="hybrid")
    assert not (config.output_dir / "runs").exists()


def test_config_and_cli_dispatch_multiple_baselines_once(tmp_path, monkeypatch):
    config, path = fixture_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(baselines=["bm25", "hybrid"], bm25={"k1": 1.8}, hybrid={"candidate_k": 200})
    path.write_text(yaml.safe_dump(raw))
    updated = load_config(path)
    assert updated.baselines == ("bm25", "hybrid")
    assert updated.bm25.k1 == 1.8
    assert updated.hybrid.candidate_k == 200
    assert updated.corpus_key("eurlex") == config.corpus_key("eurlex")
    assert updated.relevance_key("eurlex") == config.relevance_key("eurlex")
    calls = []

    def fake_retrieve(config, corpus, alias=None, *, baseline="dense"):
        calls.append((baseline, alias))
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.retrieve", fake_retrieve)
    assert main(["--config", str(path), "retrieve", "--model", "bge-m3", "--model", "LaBSE"]) == 0
    assert calls == [("bm25", None), ("hybrid", "bge-m3"), ("hybrid", "LaBSE")]
    calls.clear()
    assert main(["--config", str(path), "retrieve", "--baseline", "dense"]) == 0
    assert calls == [("dense", "bge-m3")]


def test_blocked_language_settings_never_publish_lexical_runs(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path, annotations=False)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    prepare_bm25(config, "eurlex")
    run = retrieve(config, "eurlex", baseline="bm25")
    assert {r["setting_id"] for r in rows(run / "run.parquet")} == {"eurlex__en__en"}
    assert not (run / "eurlex__en__fr.trec").exists()
    assert bm25_path(config, "eurlex").exists()
