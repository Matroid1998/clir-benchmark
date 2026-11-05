from __future__ import annotations

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from conftest import FixtureEncoder

from clir_benchmark.artifacts import read_queries, rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.cli import main
from clir_benchmark.config import load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.evidence import ALIGNMENT_CONTRACT
from clir_benchmark.pipeline import (
    embed,
    embedding_path,
    evaluate,
    model_spec,
    prepare_corpus,
    prepare_relevance,
    retrieve,
)
from clir_benchmark.util import read_json


def write_rows(root, configuration, data):
    path = root / "data" / configuration / "train-00000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(data), path)


def fixture_config(tmp_path, *, annotations=True):
    dataset = tmp_path / "dataset"
    units = [
        {
            "_id": "a:en",
            "source_id": "a",
            "parent_id": "parent",
            "language": "en",
            "text": "alpha",
            "unit_type": "article",
        },
        {
            "_id": "a:fr",
            "source_id": "a",
            "parent_id": "parent",
            "language": "fr",
            "text": "preuve",
            "unit_type": "article",
        },
        {
            "_id": "distractor:en",
            "source_id": "distractor",
            "parent_id": "other",
            "language": "en",
            "text": "alpha",
            "unit_type": "article",
        },
    ]
    qas = [
        {
            "_id": "qbest",
            "question": "alpha",
            "question_language": "en",
            "context_language": "en",
            "source_unit_id": "a:en",
            "involved_source_unit_ids": ["a:en"],
            "answer": "alpha",
            "evidence": "unmatched",
            "generator_model_id": "generator",
            "persona": "lookup",
            "provenance_json": json.dumps({"file": "run.sqlite", "task": "task1"}),
            "metadata_json": "{}",
            "candidate_rank": 2,
            "total_score": 20.0,
            "is_best": None,
            "evidence_all_languages": json.dumps(
                {"fr": [{"source_unit_id": "a:fr", "evidence": ["preuve"]}]}
            )
            if annotations
            else "{}",
        }
    ]
    qas.append(
        {
            **qas[0],
            "_id": "qlow",
            "question": "another question",
            "candidate_rank": 1,
            "total_score": 10.0,
        }
    )
    qas.append(
        {
            **qas[0],
            "_id": "qdrop",
            "question": "no support",
            "answer": "absent",
            "evidence": "",
            "provenance_json": json.dumps({"file": "run.sqlite", "task": "task2"}),
            "evidence_all_languages": json.dumps(
                {"fr": [{"source_unit_id": "a:fr", "evidence": []}]}
            )
            if annotations
            else "{}",
        }
    )
    write_rows(dataset, "corpus", units)
    write_rows(dataset, "qas", qas)
    write_rows(
        dataset,
        "source_qrels",
        [
            {"query_id": query["_id"], "corpus_id": source_id, "score": 1}
            for query in qas
            for source_id in ("a:en", "a:fr")
        ],
    )
    source = {"repo_id": "fixture/eurlex", "revision": "a" * 40, "local_dir": "dataset"}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "output_dir": "artifacts",
                "datasets": {"eurlex": {"qa": source, "corpus": source}},
                "models": ["bge-m3"],
                "runtime": {"batch_size": 2, "query_batch_size": 1, "corpus_block_size": 1},
            }
        )
    )
    return load_config(config_path), config_path


def test_complete_fixture_pipeline_and_full_corpus(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    corpus_dir = prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    relevance_dir = prepare_relevance(config, "eurlex")
    corpus_manifest = read_json(corpus_dir / "manifest.json")
    assert corpus_manifest["source_count"] == corpus_manifest["chunk_count"] == 3
    assert read_json(relevance_dir / "manifest.json")["selected_query_count"] == 2
    assert {row["query_id"] for row in rows(relevance_dir / "qrels.parquet")} == {"qbest"}
    selection = read_json(relevance_dir / "selection.json")
    assert selection[0]["method"] == "grader_total_score"
    embeddings = embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    assert read_json(embeddings["document"] / "manifest.json")["rows"] == 3
    run_dir = retrieve(config, "eurlex", "bge-m3")
    mono = [row for row in rows(run_dir / "run.parquet") if row["setting_id"] == "eurlex__en__en"]
    assert len(mono) == 2  # Includes the unrelated distractor, not an oracle pool.
    report_dir = evaluate(config, "eurlex", "bge-m3")
    report = read_json(report_dir / "report.json")
    assert len(report["metrics"]["macro_by_regime"]) == 3
    assert all(row["metrics"]["AllSourceSuccess@100"] == 1 for row in report["metrics"]["setting"])
    assert report["settings"]["eurlex__en__en"]["counts"]["dropped"] == 1
    assert list((relevance_dir / "qrels").glob("*.tsv"))
    assert list(run_dir.glob("*.trec"))
    assert (report_dir / "settings.csv").is_file()
    before = FixtureEncoder.calls
    assert embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder) == embeddings
    assert FixtureEncoder.calls == before


def test_evidence_and_candidate_changes_reuse_corpus_and_document_vectors(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    chunker = Chunker(tokenizer, config.chunking)
    corpus_dir = prepare_corpus(config, "eurlex", chunker=chunker)
    prepare_relevance(config, "eurlex")
    embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    spec = model_spec(config, "bge-m3")
    old_docs = embedding_path(config, "eurlex", spec, "document")
    old_queries = embedding_path(config, "eurlex", spec, "query")
    overlay = tmp_path / "evidence.jsonl"
    overlay.write_text(
        json.dumps(
            {
                "query_id": "qbest",
                "evidence_all_languages": {
                    "fr": [{"source_unit_id": "a:fr", "evidence": [], "answer": "preuve"}]
                },
            }
        )
        + "\n"
    )
    updated = replace(config, evidence_file=overlay)
    assert prepare_corpus(updated, "eurlex", chunker=chunker) == corpus_dir
    prepare_relevance(updated, "eurlex")
    assert updated.relevance_dir("eurlex") != config.relevance_dir("eurlex")
    assert embedding_path(updated, "eurlex", spec, "document") == old_docs
    assert embedding_path(updated, "eurlex", spec, "query") == old_queries
    all_candidates = replace(updated, candidate_mode="all")
    assert prepare_corpus(all_candidates, "eurlex", chunker=chunker) == corpus_dir
    assert (
        read_json(prepare_relevance(all_candidates, "eurlex") / "manifest.json")[
            "selected_query_count"
        ]
        == 3
    )
    assert embedding_path(all_candidates, "eurlex", spec, "document") == old_docs


def test_missing_annotations_block_only_affected_settings(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path, annotations=False)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    path = prepare_relevance(config, "eurlex")
    settings = read_json(path / "settings.json")
    assert settings["eurlex__en__en"]["status"] == "ready"
    assert settings["eurlex__en__fr"]["status"] == "blocked"
    assert settings["eurlex__en__mixed"]["status"] == "blocked"
    assert {row["setting_id"] for row in rows(path / "qrels.parquet")} == {"eurlex__en__en"}


def test_best_candidate_is_not_replaced_after_filtering(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    dataset = tmp_path / "dataset"
    qas = list(rows(dataset / "data/qas/train-00000.parquet"))
    qas[0]["answer"] = "not in the source"
    qas[0]["evidence"] = "not in the source"
    write_rows(dataset, "qas", qas)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    path = prepare_relevance(config, "eurlex")
    assert not [
        row for row in rows(path / "qrels.parquet") if row["setting_id"] == "eurlex__en__en"
    ]
    assert {row["id"] for row in rows(path / "queries.parquet")} == {"qbest", "qdrop"}


def test_cli_validation_and_preparation_never_embed(tmp_path, tokenizer, monkeypatch, capsys):
    _, path = fixture_config(tmp_path)
    monkeypatch.setattr("clir_benchmark.pipeline.load_tokenizer", lambda *args: tokenizer)

    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not load encoders or run retrieval/evaluation")

    monkeypatch.setattr("clir_benchmark.cli.embed", forbidden)
    monkeypatch.setattr("clir_benchmark.cli.retrieve", forbidden)
    monkeypatch.setattr("clir_benchmark.cli.evaluate", forbidden)
    assert main(["--config", str(path), "validate", "--inputs"]) == 0
    assert main(["--config", str(path), "prepare-corpus"]) == 0
    assert main(["--config", str(path), "prepare-relevance"]) == 0
    assert "evaluation_executed" in capsys.readouterr().out


def test_conflicting_duplicate_source_rejected(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    dataset = tmp_path / "dataset"
    units = list(rows(dataset / "data/corpus/train-00000.parquet"))
    write_rows(dataset, "corpus", units + [{**units[0], "text": "changed"}])
    with pytest.raises(ValidationError, match="Conflicting"):
        prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    assert not config.corpus_dir("eurlex").exists()


def test_published_annotation_structs_prepare_and_preserve_provenance(tmp_path, tokenizer):
    config, config_path = fixture_config(tmp_path)
    dataset = tmp_path / "dataset"
    qas = list(rows(dataset / "data/qas/train-00000.parquet"))
    for row in qas:
        dropped = row["_id"] == "qdrop"
        row["evidence_all_languages"] = {
            "en": [],
            "fr": []
            if dropped
            else [
                {
                    "source_unit_id": "a:fr",
                    "language": None,
                    "text": "preuve",
                    "anchor_id": "original-anchor",
                    "origin": "cross_language_alignment",
                    "validation_status": "pending_review",
                    "match_status": "exact",
                }
            ],
        }
        row["evidence_alignment_status"] = {
            "en": [],
            "fr": [
                {
                    "source_unit_id": "a:fr",
                    "status": "not_found" if dropped else "matched",
                    "reason": "no supporting passage" if dropped else None,
                }
            ],
        }
        row["evidence_alignment_contract"] = ALIGNMENT_CONTRACT
        row["evidence_alignment_eligibility"] = "eligible"
    write_rows(dataset, "qas", qas)
    assert main(["--config", str(config_path), "validate", "--inputs"]) == 0
    corpus_path = prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    relevance_path = prepare_relevance(config, "eurlex")
    manifest = read_json(relevance_path / "manifest.json")
    assert manifest["schema"] == 2
    assert not manifest["blocked_settings"]
    assert manifest["alignment_contracts"] == {ALIGNMENT_CONTRACT: 2}
    retained = {q.id: q for q in read_queries(relevance_path / "queries.parquet")}
    assert retained["qbest"].evidence_alignment_status == qas[0]["evidence_alignment_status"]
    positives = [
        m for m in rows(relevance_path / "matches.parquet") if m["setting_id"] == "eurlex__en__fr"
    ]
    assert len(positives) == 1
    assert positives[0]["query_id"] == "qbest"
    provenance = json.loads(positives[0]["annotation_json"])
    assert provenance["anchor_id"] == "original-anchor"
    assert provenance["validation_status"] == "pending_review"
    assert {r["query_id"] for r in rows(relevance_path / "qrels.parquet")} == {"qbest"}
    # Replacement native annotations affect relevance only, including statuses.
    overlay = tmp_path / "native.jsonl"
    overlay.write_text(
        json.dumps(
            {
                "query_id": "qbest",
                "evidence_all_languages": {"fr": []},
                "evidence_alignment_status": {
                    "fr": [{"source_unit_id": "a:fr", "status": "uncertain"}]
                },
            }
        )
    )
    updated = replace(config, evidence_file=overlay)
    assert prepare_corpus(updated, "eurlex") == corpus_path
    updated_path = prepare_relevance(updated, "eurlex")
    assert updated_path != relevance_path
    assert not [
        r for r in rows(updated_path / "qrels.parquet") if r["setting_id"] == "eurlex__en__fr"
    ]
    assert read_json(corpus_path / "manifest.json")["source_count"] == 3
