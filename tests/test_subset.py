from dataclasses import replace

import pyarrow.parquet as pq
import pytest
from test_pipeline import fixture_config, write_rows

from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.config import load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.pipeline import prepare_corpus, prepare_relevance
from clir_benchmark.subset import prepare_subset
from clir_benchmark.util import read_json


def test_subset_preserves_candidates_sources_and_eligible_settings(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    config = replace(config, target_languages=("en", "fr"))
    subset = prepare_subset(
        config,
        "eurlex",
        max_questions=1,
        max_sources=3,
        chunker=Chunker(tokenizer, config.chunking),
    )
    manifest = read_json(subset / "manifest.json")
    selected = read_json(subset / "selection.json")
    assert manifest["question_count"] == 1
    assert manifest["source_count"] == 3
    assert manifest["retained_candidate_count"] == 2
    assert selected["selected_query_ids"] == ["qbest"]
    assert selected["retained_candidate_ids"] == ["qbest", "qlow"]
    assert selected["distractor_source_ids"] == ["distractor:en"]
    assert len(selected["eligible_settings"]["qbest"]) == 3
    subset_config = load_config(subset / "config.yaml")
    prepare_corpus(subset_config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    relevance = prepare_relevance(subset_config, "eurlex")
    assert {row["query_id"] for row in rows(relevance / "qrels.parquet")} == {"qbest"}
    assert all(
        setting["status"] == "ready" for setting in read_json(relevance / "settings.json").values()
    )
    original = pq.read_table(tmp_path / "dataset/data/corpus/train-00000.parquet").to_pylist()
    exported = pq.read_table(subset / "corpus/data/corpus/train-00000.parquet").to_pylist()
    assert sorted(original, key=lambda row: row["_id"]) == exported


def test_subset_reuses_validated_cache_without_reads(tmp_path, tokenizer, monkeypatch):
    config, _ = fixture_config(tmp_path)
    kwargs = dict(max_questions=1, max_sources=3, chunker=Chunker(tokenizer, config.chunking))
    subset = prepare_subset(config, "eurlex", **kwargs)
    monkeypatch.setattr(
        "clir_benchmark.subset.DatasetReader.read",
        lambda *args: pytest.fail("cached subset read upstream input"),
    )
    assert prepare_subset(config, "eurlex", **kwargs) == subset
    inactive_policy = replace(
        config, candidate_selection=replace(config.candidate_selection, seed=999)
    )
    assert prepare_subset(inactive_policy, "eurlex", **kwargs) == subset
    (subset / "selection.json").write_text("{}")
    with pytest.raises(ValidationError, match="artifact changed"):
        prepare_subset(config, "eurlex", **kwargs)


def test_subset_rejects_insufficient_budget_and_never_replaces_winner(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    with pytest.raises(ValidationError, match="Found only 0"):
        prepare_subset(
            config,
            "eurlex",
            max_questions=1,
            max_sources=1,
            chunker=Chunker(tokenizer, config.chunking),
        )
    with pytest.raises(ValidationError, match="Found only 1"):
        prepare_subset(
            config,
            "eurlex",
            max_questions=2,
            max_sources=3,
            chunker=Chunker(tokenizer, config.chunking),
        )


def test_subset_audits_invalid_unselected_groups(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    path = tmp_path / "dataset/data/qas/train-00000.parquet"
    qas = pq.read_table(path).to_pylist()
    for row in qas:
        if row["_id"] == "qdrop":
            row["total_score"] = None
    write_rows(tmp_path / "dataset", "qas", qas)
    subset = prepare_subset(
        config,
        "eurlex",
        max_questions=1,
        max_sources=3,
        chunker=Chunker(tokenizer, config.chunking),
    )
    audit = read_json(subset / "selection.json")
    assert len(audit["rejected_upstream_groups"]) == 1
    assert "missing grader" in audit["rejected_upstream_groups"][0]["reason"]


def test_subset_all_mode_does_not_restore_unselected_siblings(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    config = replace(config, candidate_mode="all")
    subset = prepare_subset(
        config,
        "eurlex",
        max_questions=1,
        max_sources=3,
        chunker=Chunker(tokenizer, config.chunking),
    )
    manifest = read_json(subset / "manifest.json")
    assert manifest["retained_candidate_count"] == manifest["question_count"] == 1


@pytest.mark.parametrize("kwargs", [{"max_questions": 0}, {"max_sources": True}, {"seed": False}])
def test_subset_limits_are_validated(tmp_path, kwargs):
    config, _ = fixture_config(tmp_path)
    with pytest.raises(ValidationError):
        prepare_subset(config, "eurlex", **kwargs)


def test_chemistry_subset_requires_its_own_sampling_policy(tmp_path):
    config, _ = fixture_config(tmp_path)
    with pytest.raises(ValidationError, match="eurlex and un"):
        prepare_subset(config, "chemistry")


def test_subset_rejects_retained_unsupported_associations_before_publishing(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    qrels_path = tmp_path / "dataset/data/source_qrels/train-00000.parquet"
    qrels = pq.read_table(qrels_path).to_pylist()
    qrels.append({"query_id": "qbest", "corpus_id": "distractor:en", "score": 1})
    write_rows(tmp_path / "dataset", "source_qrels", qrels)
    destination = tmp_path / "subset"
    with pytest.raises(ValidationError, match="unsupported associations"):
        prepare_subset(
            config,
            "eurlex",
            max_questions=1,
            max_sources=3,
            destination=destination,
            chunker=Chunker(tokenizer, config.chunking),
        )
    assert not destination.exists()


def test_subset_deduplicates_distractors_with_different_provenance(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    source_path = tmp_path / "dataset/data/corpus/train-00000.parquet"
    sources = pq.read_table(source_path).to_pylist()
    sources.append({**sources[-1], "parent_id": "duplicate-provenance"})
    write_rows(tmp_path / "dataset", "corpus", sources)
    subset = prepare_subset(
        config,
        "eurlex",
        max_questions=1,
        max_sources=3,
        chunker=Chunker(tokenizer, config.chunking),
    )
    manifest = read_json(subset / "manifest.json")
    assert manifest["upstream_source_count"] == 4
    assert manifest["source_count"] == 3
    assert manifest["distractor_count"] == 1


def test_document_subset_selects_globally_and_retains_candidates_and_unannotated_versions(
    tmp_path, tokenizer
):
    config, _ = fixture_config(tmp_path, annotations=False)
    qas = pq.read_table(tmp_path / "dataset/data/qas/train-00000.parquet").to_pylist()
    for row in qas:
        if row["_id"] == "qdrop":
            row.update(
                total_score=32.0,
                source_unit_id="a:fr",
                involved_source_unit_ids=["a:fr"],
                question_language="fr",
                context_language="fr",
                generator_model_id="other-generator",
                evidence_all_languages="{}",
            )
    write_rows(tmp_path / "dataset", "qas", qas)
    config = replace(
        config,
        candidate_mode="best-per-document",
        retrieval_unit="document",
        target_languages=("en", "fr", "zh"),
    )
    subset = prepare_subset(
        config,
        "eurlex",
        max_questions=1,
        max_sources=3,
        chunker=Chunker(tokenizer, config.chunking),
    )
    audit = read_json(subset / "selection.json")
    assert audit["selected_query_ids"] == ["qdrop"]
    assert audit["retained_candidate_ids"] == ["qbest", "qdrop", "qlow"]
    assert {"a:en", "a:fr"} <= set(audit["selected_source_ids"])
    assert audit["candidate_groups"][0]["candidate_count"] == 3
    assert audit["eligible_settings"]["qdrop"] == [
        "eurlex__fr__en",
        "eurlex__fr__fr",
        "eurlex__fr__mixed",
    ]
    subset_config = load_config(subset / "config.yaml")
    prepare_corpus(subset_config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    relevance = prepare_relevance(subset_config, "eurlex")
    assert read_json(relevance / "manifest.json")["selected_query_count"] == 1
    assert {row["query_id"] for row in rows(relevance / "qrels.parquet")} == {"qdrop"}
    assert {row["document_id"] for row in rows(relevance / "qrels.parquet")} == {"a:en", "a:fr"}


def test_document_subset_preserves_random_winner_when_prepared_again(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path, annotations=False)
    config = replace(config, candidate_mode="best-per-document", retrieval_unit="document")
    subset = prepare_subset(
        config,
        "eurlex",
        max_questions=1,
        max_sources=3,
        chunker=Chunker(tokenizer, config.chunking),
    )
    audit = read_json(subset / "selection.json")
    subset_config = load_config(subset / "config.yaml")
    prepare_corpus(subset_config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    relevance = prepare_relevance(subset_config, "eurlex")
    assert read_json(relevance / "selection.json")[0]["selected_ids"] == audit["selected_query_ids"]
    assert audit["candidate_groups"][0] == read_json(relevance / "selection.json")[0]


def test_document_retrieval_subset_without_new_selection_keeps_legacy_best_groups(
    tmp_path, tokenizer
):
    config, _ = fixture_config(tmp_path, annotations=False)
    config = replace(config, retrieval_unit="document")
    subset = prepare_subset(
        config,
        "eurlex",
        max_questions=1,
        max_sources=3,
        chunker=Chunker(tokenizer, config.chunking),
    )
    audit = read_json(subset / "selection.json")
    assert len(audit["selected_query_ids"]) == 1
    assert {"a:en", "a:fr"} <= set(audit["selected_source_ids"])
    assert all("task_id" in row for row in audit["candidate_groups"])
