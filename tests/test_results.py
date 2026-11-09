from dataclasses import replace
from pathlib import Path

import pytest
from test_rerank_pipeline import FixtureReranker, parent_fixture

from clir_benchmark.artifacts import rows
from clir_benchmark.config import RETRIEVAL_BASELINES
from clir_benchmark.errors import ValidationError
from clir_benchmark.pipeline import evaluate
from clir_benchmark.rerank_pipeline import rerank
from clir_benchmark.results import language_script, metric_name
from clir_benchmark.util import file_digest, read_json


def forbidden(*args, **kwargs):
    pytest.fail("Result export must only read stored artifacts")


@pytest.mark.parametrize("baseline", RETRIEVAL_BASELINES)
def test_all_twenty_baselines_export_metrics_and_complete_analysis(
    tmp_path, tokenizer, monkeypatch, baseline
):
    config, _, alias, languages, parent = parent_fixture(tmp_path, tokenizer, baseline)
    config = replace(config, baselines=(baseline, f"rerank-{baseline}"))
    variant = rerank(
        config,
        "eurlex",
        alias,
        baseline=f"rerank-{baseline}",
        **languages,
        reranker_factory=FixtureReranker().factory,
    )
    # Even credentials stored in the artifact directory must never be copied.
    secret = config.output_dir / ".env"
    secret.write_text("HF_TOKEN=DO_NOT_COPY_THIS_SECRET\n")
    config = replace(config, env_file=secret)
    for target in (
        "clir_benchmark.encoders.HuggingFaceEncoder.__init__",
        "clir_benchmark.reranking.Qwen3Reranker.__init__",
        "clir_benchmark.generation.ChatCompletionsClient.complete",
        "clir_benchmark.lexical.BM25Index.search",
        "clir_benchmark.retrieval.DenseIndex.search",
    ):
        monkeypatch.setattr(target, forbidden)
    qrels = config.relevance_dir("eurlex") / "qrels.parquet"
    qrels_before = file_digest(qrels)
    for mode, run in ((baseline, parent), (f"rerank-{baseline}", variant)):
        report_dir = evaluate(config, "eurlex", alias, baseline=mode, **languages)
        report = read_json(report_dir / "report.json")
        export = read_json(report_dir / "mteb.json")
        assert export["task_name"] == "CLIRBenchmark_eurlex"
        assert export["dataset_revision"] == config.relevance_key("eurlex")
        assert export["mteb_version"] is None
        assert export["evaluation_time"] is None
        assert export["kg_co2_emissions"] is None
        assert len(export["scores"]["test"]) == len(report["metrics"]["setting"])
        for score, setting in zip(
            export["scores"]["test"], report["metrics"]["setting"], strict=True
        ):
            assert score["hf_subset"] == setting["setting_id"]
            assert score["main_score"] == score["ndcg_at_10"] == setting["metrics"]["nDCG@10"]
            assert score["recall_at_100"] == setting["metrics"]["Recall@100"]
            assert score["all_source_success_at_100"] == setting["metrics"]["AllSourceSuccess@100"]
            assert score["query_count"] == setting["query_count"]
            assert "eng-Latn" in score["languages"]
        details = list(rows(report_dir / "analysis/rankings.parquet"))
        original = list(rows(run / "run.parquet"))
        assert len(details) == len(original)
        for row, ranked in zip(details, original, strict=True):
            assert row["rank"] == ranked["rank"]
            assert row["score"] == ranked["score"]
            assert row["text"] in {"alpha", "preuve"}
            assert row["query"] == "alpha"
            assert row["relevance"] == int(row["source_unit_id"] in {"a:en", "a:fr"})
        manifest = read_json(report_dir / "analysis_manifest.json")
        assert (report_dir / manifest["artifact_root"]).resolve() == config.output_dir.resolve()
        assert read_json(config.output_dir / manifest["mteb_result"]) == export
        assert "/experiments/" in manifest["mteb_result"]
        assert f"{mode}__{run.name}" in manifest["mteb_result"]
        paths = {file["path"] for file in manifest["files"]}
        assert all(not path.startswith("/") for path in paths)
        namespaces = {Path(path).parts[0] for path in paths}
        assert {"corpus", "relevance", "runs", "reports", "input_snapshots", "mteb"} <= namespaces
        assert any(path.endswith("data/qas/train-00000.parquet") for path in paths)
        assert any(path.endswith("data/source_qrels/train-00000.parquet") for path in paths)
        if baseline.startswith("translate-"):
            assert {"translation_requests", "translations"} <= namespaces
        elif baseline == "hyde-dense":
            assert {"hyde_requests", "hyde", "hyde_invocations"} <= namespaces
        elif baseline.startswith("adore-"):
            assert {"adore_requests", "adore_searches", "adore_invocations"} <= namespaces
        if alias:
            assert "embeddings" in namespaces
            assert any(path.endswith("vectors.npy") for path in paths)
        if mode.startswith("rerank-"):
            assert {"reranking_pairs", "reranking_invocations"} <= namespaces
            assert str(parent.relative_to(config.output_dir) / "run.parquet") in paths
        for file in manifest["files"]:
            path = config.output_dir / file["path"]
            assert path.is_file()
            assert file_digest(path) == file["sha256"]
            assert path.stat().st_size == file["bytes"]
            assert path.name != ".env"
            if path.suffix == ".json":
                assert "DO_NOT_COPY_THIS_SECRET" not in path.read_text()
    assert file_digest(qrels) == qrels_before


def test_cached_evaluation_backfills_exports_and_preserves_config_history(
    tmp_path, tokenizer, monkeypatch
):
    config, _, alias, languages, _ = parent_fixture(tmp_path, tokenizer, "bm25")
    report = evaluate(config, "eurlex", baseline="bm25")
    original_report = file_digest(report / "report.json")
    first_config = next((report / "evaluation_configs").glob("*.json"))
    saved = read_json(first_config)
    (report / "mteb.json").unlink()
    index = read_json(report / "analysis_manifest.json")
    (config.output_dir / index["mteb_result"]).unlink()
    monkeypatch.setattr("clir_benchmark.lexical.BM25Index.search", forbidden)
    changed = replace(config, baselines=("bm25", "rerank-bm25"))
    assert evaluate(changed, "eurlex", baseline="bm25") == report
    assert file_digest(report / "report.json") == original_report
    assert read_json(first_config) == saved
    assert len(list((report / "evaluation_configs").glob("*.json"))) == 2
    assert (report / "mteb.json").is_file()
    assert (config.output_dir / index["mteb_result"]).is_file()


def test_corrupt_analysis_ranking_is_not_silently_relabelled(tmp_path, tokenizer):
    config, _, _, _, _ = parent_fixture(tmp_path, tokenizer, "bm25")
    report = evaluate(config, "eurlex", baseline="bm25")
    with (report / "analysis/rankings.parquet").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValidationError, match="Corrupt analysis"):
        evaluate(config, "eurlex", baseline="bm25")


def test_metric_and_language_names_preserve_semantics():
    assert metric_name("SourceRecall@10") == "source_recall_at_10"
    assert language_script("zh-Hant") == "zho-Hant"
    assert language_script("fr") == "fra-Latn"
    assert language_script("unknown") == "unknown"


def test_subset_raw_inputs_and_selection_provenance_are_retained(tmp_path, tokenizer):
    from test_pipeline import fixture_config

    from clir_benchmark.chunking import Chunker
    from clir_benchmark.config import load_config
    from clir_benchmark.lexical import prepare_bm25
    from clir_benchmark.pipeline import prepare_corpus, prepare_relevance, retrieve
    from clir_benchmark.subset import prepare_subset

    config, _ = fixture_config(tmp_path)
    subset = prepare_subset(
        config,
        "eurlex",
        max_questions=1,
        max_sources=3,
        chunker=Chunker(tokenizer, config.chunking),
    )
    config = load_config(subset / "config.yaml")
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    prepare_bm25(config, "eurlex")
    retrieve(config, "eurlex", baseline="bm25")
    report = evaluate(config, "eurlex", baseline="bm25")
    inventory = read_json(report / "analysis_manifest.json")["files"]
    snapshots = [
        config.output_dir / entry["path"]
        for entry in inventory
        if entry["path"].startswith("input_snapshots/")
    ]
    for name in ("config.yaml", "manifest.json", "selection.json"):
        copies = [path for path in snapshots if path.parent.name == "subset" and path.name == name]
        assert len(copies) == 2
        assert all(file_digest(path) == file_digest(subset / name) for path in copies)
    original_qas = subset / "qa/data/qas/train-00000.parquet"
    copied_qas = next(path for path in snapshots if path.parent.name == "qas")
    assert file_digest(original_qas) == file_digest(copied_qas)
    assert {row["_id"] for row in rows(copied_qas)} == {"qbest", "qlow"}
    previous = set(config.output_dir.joinpath("input_snapshots").iterdir())
    evaluate(config, "eurlex", baseline="bm25")
    assert set(config.output_dir.joinpath("input_snapshots").iterdir()) == previous


def test_child_inventory_survives_parent_invocation_history_refresh(tmp_path, tokenizer):
    from clir_benchmark.adore_pipeline import run_adore

    config, _, alias, languages, _ = parent_fixture(tmp_path, tokenizer, "adore-bm25")
    rerank(
        config, "eurlex", baseline="rerank-adore-bm25", reranker_factory=FixtureReranker().factory
    )
    report = evaluate(config, "eurlex", baseline="rerank-adore-bm25")
    inventory = read_json(report / "analysis_manifest.json")["files"]
    assert not any(entry["path"].endswith("analysis_manifest.json") for entry in inventory)
    # A cache-only run adds a new invocation; refreshing the parent must not
    # invalidate the child's existing immutable artifact references.
    run_adore(config, "eurlex", baseline="adore-bm25", client_factory=forbidden)
    evaluate(config, "eurlex", baseline="adore-bm25")
    assert all(
        file_digest(config.output_dir / entry["path"]) == entry["sha256"] for entry in inventory
    )
