import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import yaml
from conftest import FixtureEncoder
from test_hyde import FixtureGenerator, hyde_completion
from test_pipeline import fixture_config, write_rows
from test_translation import FixtureTranslator
from test_translation_pipeline import prepared

from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.cli import main
from clir_benchmark.comparison import compare_hyde
from clir_benchmark.config import HyDEConfig
from clir_benchmark.embedding import load_embeddings
from clir_benchmark.errors import ValidationError
from clir_benchmark.hyde import hyde_languages, hyde_path, load_hyde, prepare_hyde
from clir_benchmark.pipeline import (
    embed,
    evaluate,
    model_spec,
    prepare_corpus,
    prepare_relevance,
    retrieve,
)
from clir_benchmark.retrieval import DenseIndex
from clir_benchmark.translation import (
    prepare_translations,
    translation_path,
)
from clir_benchmark.translation import (
    request_payload as translation_payload,
)
from clir_benchmark.util import digest, file_digest, read_json, write_json


def fixture_hyde(tmp_path, tokenizer, *, annotations=True):
    config, path = prepared(tmp_path, tokenizer, annotations=annotations)
    return replace(
        config,
        hyde=HyDEConfig(
            model="hyde-model",
            endpoint="https://example.invalid/chat",
            input_usd_per_million=4,
            cached_input_usd_per_million=1,
            output_usd_per_million=8,
        ),
    ), path


def test_same_language_generation_independent_variants_and_current_invocation_cost(
    tmp_path, tokenizer
):
    config, _ = fixture_hyde(tmp_path, tokenizer)
    generator = FixtureGenerator()
    en = prepare_hyde(config, "eurlex", "en", generator_factory=generator.factory)
    fr = prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory)
    # An explicit output language outside the corpus remains a valid experiment.
    ja = prepare_hyde(config, "eurlex", "ja", generator_factory=generator.factory)
    assert len({en, fr, ja}) == len(generator.calls) == 3
    assert hyde_languages(config, "eurlex") == ("en", "fr")
    for path in (en, fr, ja):
        manifest = read_json(path / "manifest.json")
        assert manifest["query_count"] == manifest["summary"]["api_attempts"] == 1
        assert manifest["summary"]["estimated_cost_usd"] == pytest.approx(0.00044)
        assert [row["id"] for row in rows(path / "passages.parquet")] == ["qbest"]
    assert (
        generator.calls[0]["messages"][1]["content"]
        == "collection: eurlex\nsource_language: en\noutput_language: en\n\n<query>\nalpha\n</query>"
    )
    assert "preuve" not in json.dumps(
        generator.calls
    )  # The answer/evidence never entered generation.
    assert prepare_hyde(config, "eurlex", "en", generator_factory=generator.factory) == en
    assert len(generator.calls) == 3
    invocations = [read_json(p) for p in (config.output_dir / "hyde_invocations").glob("*.json")]
    reused = [entry for entry in invocations if entry["artifact_reused"]]
    assert len(reused) == 1
    assert reused[0]["new_work"]["api_attempts"] == 0
    assert reused[0]["new_work"]["estimated_cost_usd"] == 0
    assert read_json(en / "manifest.json")["summary"]["api_attempts"] == 1


class PrefixEncoder(FixtureEncoder):
    inputs = []

    def encode(self, texts, kind):
        prefix = self.spec.document_prefix if kind == "document" else self.spec.query_prefix
        self.inputs.append((kind, list(texts), prefix))
        return super().encode([prefix + text for text in texts], kind)


def test_document_adapter_vectors_corpus_reuse_and_exact_retrieval(tmp_path, tokenizer):
    config, _ = fixture_hyde(tmp_path, tokenizer)
    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"bge-m3": {"query_prefix": "Q:", "document_prefix": "D:"}}))
    config = replace(config, model_registry=registry)
    generator = FixtureGenerator("preuve")
    passage_path = prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory)
    qrels = config.relevance_dir("eurlex") / "qrels.parquet"
    qrels_hash = file_digest(qrels)
    original = embed(config, "eurlex", "bge-m3", encoder_factory=PrefixEncoder)
    original_calls = len(PrefixEncoder.inputs)
    variant = embed(config, "eurlex", "bge-m3", hyde_language="fr", encoder_factory=PrefixEncoder)
    assert PrefixEncoder.inputs[original_calls:] == [("document", ["preuve"], "D:")]
    assert variant["document"] == original["document"]
    assert variant["query"] != original["query"]
    assert variant["query"].parent.name == "hyde"
    spec = model_spec(config, "bge-m3")
    vectors, metadata, manifest = load_embeddings(variant["query"])
    expected = PrefixEncoder(spec, config.runtime).encode(["preuve"], "document").vectors
    np.testing.assert_allclose(vectors, expected)
    assert manifest["kind"] == "document"
    assert [row["id"] for row in metadata] == ["qbest"]
    assert metadata[0]["language"] == "fr"
    run = retrieve(config, "eurlex", "bge-m3", baseline="hyde-dense", hyde_language="fr")
    docs, doc_rows, _ = load_embeddings(original["document"])
    index = DenseIndex(docs, [row["id"] for row in doc_rows], [row["language"] for row in doc_rows])
    for setting, info in read_json(config.relevance_dir("eurlex") / "settings.json").items():
        expected_ranking = list(index.search(vectors, tuple(info["target_languages"]), k=100))[0]
        actual = [
            (row["chunk_id"], row["score"])
            for row in rows(run / "run.parquet")
            if row["setting_id"] == setting
        ]
        assert actual == expected_ranking
    report_dir = evaluate(config, "eurlex", "bge-m3", baseline="hyde-dense", hyde_language="fr")
    report = read_json(report_dir / "report.json")
    assert report["output_language"] == "fr" and report["generation_model"] == "hyde-model"
    assert report["truncation"]["hypothetical_documents"]["fr"]["inputs"] == 1
    assert report["truncation"]["search_encoder_kind"] == "document"
    assert (
        report["run"]["hyde"]["passage_fingerprint"]
        == read_json(passage_path / "manifest.json")["passage_fingerprint"]
    )
    assert report["hyde_summary"]["api_attempts"] == 1
    assert "generation_input_tokens" in (report_dir / "per_query.csv").read_text()
    assert "output_language" in (report_dir / "settings.csv").read_text()
    assert file_digest(qrels) == qrels_hash
    assert len(generator.calls) == 1
    calls = len(PrefixEncoder.inputs)
    assert (
        embed(config, "eurlex", "bge-m3", hyde_language="fr", encoder_factory=PrefixEncoder)
        == variant
    )
    assert len(PrefixEncoder.inputs) == calls
    assert retrieve(config, "eurlex", "bge-m3", baseline="hyde-dense", hyde_language="fr") == run


def prepare_comparison(config):
    prepare_translations(config, "eurlex", "fr", translator_factory=FixtureTranslator().factory)
    prepare_hyde(config, "eurlex", "fr", generator_factory=FixtureGenerator().factory)
    for baseline, kwargs in (
        ("dense", {}),
        ("translate-dense", {"translation_language": "fr"}),
        ("hyde-dense", {"hyde_language": "fr"}),
    ):
        embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder, **kwargs)
        retrieve(config, "eurlex", "bge-m3", baseline=baseline, **kwargs)
        evaluate(config, "eurlex", "bge-m3", baseline=baseline, **kwargs)


def test_comparison_subtracts_both_controls_without_double_counting(
    tmp_path, tokenizer, monkeypatch
):
    config, _ = fixture_hyde(tmp_path, tokenizer)
    prepare_comparison(config)

    def forbidden(*args, **kwargs):
        pytest.fail("Comparison must never generate, embed, or retrieve")

    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.complete", forbidden)
    monkeypatch.setattr("clir_benchmark.pipeline.HuggingFaceEncoder", forbidden)
    monkeypatch.setattr("clir_benchmark.pipeline.DenseIndex.search", forbidden)
    path = compare_hyde(
        config, "eurlex", "bge-m3", languages=("fr",), controls=("dense", "translate-dense")
    )
    records = read_json(path / "per_query.json")
    assert {row["control"] for row in records} == {"dense", "translate-dense"}
    for row in records:
        assert row["added_generation_cost_usd"] == pytest.approx(
            0.00044 if row["control"] == "dense" else 0.00022
        )
        assert row["added_generation_seconds"] == pytest.approx(
            row["hyde_generation_seconds"] - row["control_generation_seconds"]
        )
        expected = (
            row["hyde_generation_seconds"]
            + row["hyde_encoding_seconds_amortized"]
            + row["hyde_retrieval_seconds_amortized"]
            - row["control_generation_seconds"]
            - row["control_encoding_seconds_amortized"]
            - row["control_retrieval_seconds_amortized"]
        )
        assert row["added_component_seconds"] == pytest.approx(expected)
        assert (
            row["metric_deltas"]["MRR@10"]
            == row["hyde_metrics"]["MRR@10"] - row["control_metrics"]["MRR@10"]
        )
    report = read_json(path / "report.json")
    assert len(report["languages"]) == 1
    assert report["languages"][0]["hyde_summary"]["api_attempts"] == 1
    assert len(report["settings"]) == 6
    assert (path / "settings.csv").is_file()


def test_missing_prices_propagate_through_comparison(tmp_path, tokenizer):
    config, _ = fixture_hyde(tmp_path, tokenizer)
    config = replace(config, hyde=replace(config.hyde, input_usd_per_million=None))
    prepare_comparison(config)
    path = compare_hyde(
        config, "eurlex", "bge-m3", languages=("fr",), controls=("translate-dense",)
    )
    assert all(
        row["added_generation_cost_usd"] is None for row in read_json(path / "per_query.json")
    )


def test_comparison_rejects_mismatched_cohort_and_handles_old_missing_timings(tmp_path, tokenizer):
    config, _ = fixture_hyde(tmp_path, tokenizer)
    prepare_comparison(config)
    original_dir = evaluate(config, "eurlex", "bge-m3")
    originals = read_json(original_dir / "per_query.json")
    for row in originals:
        row.pop("retrieval_seconds_amortized")
    write_json(original_dir / "per_query.json", originals)
    comparison = compare_hyde(config, "eurlex", "bge-m3", languages=("fr",))
    assert all(
        row["added_component_seconds"] is None for row in read_json(comparison / "per_query.json")
    )
    write_json(original_dir / "per_query.json", originals[:-1])
    with pytest.raises(ValidationError, match="cohorts differ"):
        compare_hyde(config, "eurlex", "bge-m3", languages=("fr",))


def test_failed_and_interrupted_generation_preserves_audit_and_explicit_retry_costs(
    tmp_path, tokenizer
):
    config, _ = fixture_hyde(tmp_path, tokenizer)
    generator = FixtureGenerator()
    invalid = hyde_completion()
    invalid["choices"][0]["message"]["content"] = '{"hypothetical_document":"a","extra":1}'
    generator.responses = [invalid, KeyboardInterrupt(), hyde_completion()]
    with pytest.raises(ValidationError, match="no fallback"):
        prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory)
    assert not hyde_path(config, "eurlex", "fr").exists()
    with pytest.raises(ValidationError, match="generate-hyde --retry-failed"):
        prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory)
    assert len(generator.calls) == 1
    with pytest.raises(KeyboardInterrupt):
        prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory, retry_failed=True)
    path = prepare_hyde(
        config, "eurlex", "fr", generator_factory=generator.factory, retry_failed=True
    )
    assert len(generator.calls) == 3
    record = list(rows(path / "passages.parquet"))[0]
    assert record["attempts"] == 3
    assert record["estimated_cost_usd"] is record["generation_seconds"] is None
    audit = read_json(next((config.output_dir / "hyde_requests").glob("*.json")))
    assert audit["attempts"][0]["response"] == invalid
    assert audit["attempts"][1]["status"] == "pending"
    invocations = [read_json(p) for p in (config.output_dir / "hyde_invocations").glob("*.json")]
    assert sorted(row["status"] for row in invocations) == [
        "failed",
        "failed",
        "interrupted",
        "success",
    ]
    assert sum(row["new_work"]["api_attempts"] for row in invocations) == 3


def test_paid_invalid_response_cost_is_retained_after_retry(tmp_path, tokenizer):
    config, _ = fixture_hyde(tmp_path, tokenizer)
    generator = FixtureGenerator()
    invalid = hyde_completion()
    invalid["choices"][0]["finish_reason"] = "length"
    generator.responses = [invalid, hyde_completion()]
    with pytest.raises(ValidationError):
        prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory)
    path = prepare_hyde(
        config, "eurlex", "fr", generator_factory=generator.factory, retry_failed=True
    )
    summary = read_json(path / "manifest.json")["summary"]
    assert summary["api_attempts"] == 2
    assert summary["input_tokens"] == 200
    assert summary["estimated_cost_usd"] == pytest.approx(0.00088)


def test_resume_reuses_earlier_successful_queries(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    qas = list(rows(tmp_path / "dataset/data/qas/train-00000.parquet"))
    qas[-1]["answer"] = "alpha"
    write_rows(tmp_path / "dataset", "qas", qas)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    generator = FixtureGenerator()
    generator.responses = [hyde_completion(), {}, hyde_completion()]
    with pytest.raises(ValidationError):
        prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory)
    prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory, retry_failed=True)
    assert len(generator.calls) == 3
    _, records, manifest = load_hyde(config, "eurlex", "fr")
    assert [row["attempts"] for row in records] == [1, 2]
    assert manifest["cached_requests"] == manifest["new_api_attempts"] == 1


def test_cache_identity_repricing_and_model_prompt_language_changes(
    tmp_path, tokenizer, monkeypatch
):
    from clir_benchmark import hyde

    config, _ = fixture_hyde(tmp_path, tokenizer)
    generator = FixtureGenerator()
    original = prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory)
    repriced = replace(config, hyde=replace(config.hyde, output_usd_per_million=10))
    repriced_path = prepare_hyde(repriced, "eurlex", "fr", generator_factory=generator.factory)
    assert repriced_path != original and len(generator.calls) == 1
    assert read_json(repriced_path / "manifest.json")["summary"][
        "estimated_cost_usd"
    ] == pytest.approx(0.00048)
    for changes in ({"model": "new-model"}, {"request_options": {"temperature": 0}}):
        changed = replace(config, hyde=replace(config.hyde, **changes))
        assert changed.corpus_key("eurlex") == config.corpus_key("eurlex")
        assert changed.relevance_key("eurlex") == config.relevance_key("eurlex")
        assert translation_path(changed, "eurlex", "fr") == translation_path(config, "eurlex", "fr")
        prepare_hyde(changed, "eurlex", "fr", generator_factory=generator.factory)
    original_prompts = hyde.prompts
    monkeypatch.setattr(
        hyde,
        "prompts",
        lambda corpus: (original_prompts(corpus)[0] + "\nRevision.", original_prompts(corpus)[1]),
    )
    prepare_hyde(config, "eurlex", "fr", generator_factory=generator.factory)
    assert len(generator.calls) == 4


def test_blocked_settings_and_missing_stages_remain_explicit(tmp_path, tokenizer):
    config, _ = fixture_hyde(tmp_path, tokenizer, annotations=False)
    with pytest.raises(ValidationError, match="generate-hyde first"):
        retrieve(config, "eurlex", "bge-m3", baseline="hyde-dense", hyde_language="fr")
    prepare_hyde(config, "eurlex", "fr", generator_factory=FixtureGenerator().factory)
    with pytest.raises(ValidationError, match="embed first"):
        retrieve(config, "eurlex", "bge-m3", baseline="hyde-dense", hyde_language="fr")
    embed(config, "eurlex", "bge-m3", hyde_language="fr", encoder_factory=FixtureEncoder)
    with pytest.raises(ValidationError, match="retrieve explicitly"):
        evaluate(config, "eurlex", "bge-m3", baseline="hyde-dense", hyde_language="fr")
    path = retrieve(config, "eurlex", "bge-m3", baseline="hyde-dense", hyde_language="fr")
    assert read_json(path / "manifest.json")["settings"]["eurlex__en__fr"]["status"] == "blocked"
    assert not (path / "eurlex__en__fr.trec").exists()
    with pytest.raises(ValidationError, match="retrieve explicitly"):
        compare_hyde(config, "eurlex", "bge-m3", languages=("fr",))


def test_legacy_translation_request_and_artifact_keys_are_reused(tmp_path, tokenizer):
    config, _ = fixture_hyde(tmp_path, tokenizer)
    row = {"id": "qbest", "question": "alpha", "question_language": "en"}
    payload = translation_payload(config.translation, "eurlex", row, "fr")
    key = digest(
        {
            "schema": 1,
            "query_id": row["id"],
            "endpoint": config.translation.endpoint,
            "payload": payload,
        }
    )
    from test_translation import completion

    write_json(
        config.output_dir / "translation_requests" / f"{key}.json",
        {
            "request_key": key,
            "payload": payload,
            "status": "success",
            "text": "preuve",
            "attempts": [{"status": "success", "seconds": 2.0, "response": completion()}],
        },
    )

    def forbidden(*args):
        pytest.fail("Existing translation must not be regenerated")

    result = prepare_translations(config, "eurlex", "fr", translator_factory=forbidden)
    from clir_benchmark.translation import prompts

    expected_key = digest(
        {
            "schema": 1,
            "corpus": "eurlex",
            "target": "fr",
            "inputs": [row],
            "prompts": prompts("eurlex"),
            "translation": asdict(config.translation),
        }
    )
    assert result.name == expected_key
    assert read_json(result / "manifest.json")["new_api_attempts"] == 0
    assert list(rows(result / "queries.parquet"))[0]["translation_seconds"] == 2


def test_cli_language_precedence_controls_and_mode_isolation(tmp_path, tokenizer, monkeypatch):
    config, path = fixture_hyde(tmp_path, tokenizer)
    raw = yaml.safe_load(path.read_text())
    raw["hyde"] = {"output_languages": ["ja", "fr"]}
    raw["baselines"] = ["hyde-dense", "dense", "translate-dense"]
    path.write_text(yaml.safe_dump(raw))
    calls = []

    def generation(config, corpus, language, **kwargs):
        calls.append(("generate", language))
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.prepare_hyde", generation)
    prefix = ["--config", str(path)]
    assert main([*prefix, "generate-hyde"]) == 0
    assert calls == [("generate", "fr"), ("generate", "ja")]
    calls.clear()
    assert main([*prefix, "generate-hyde", "--hyde-language", "en"]) == 0
    assert calls == [("generate", "en")]
    calls.clear()

    def retrieval(config, corpus, alias=None, *, baseline, **kwargs):
        calls.append((baseline, kwargs))
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.retrieve", retrieval)
    assert main([*prefix, "retrieve", "--hyde-language", "en", "--translation-language", "fr"]) == 0
    assert calls == [
        ("hyde-dense", {"hyde_language": "en"}),
        ("dense", {}),
        ("translate-dense", {"translation_language": "fr"}),
    ]
    calls.clear()
    assert main([*prefix, "retrieve", "--baseline", "dense"]) == 0
    assert calls == [("dense", {})]
    assert main([*prefix, "retrieve", "--baseline", "dense", "--hyde-language", "fr"]) == 2
    assert main([*prefix, "embed", "--hyde-language", "fr", "--translation-language", "fr"]) == 2
    assert main([*prefix, "embed", "--kind", "document", "--hyde-language", "fr"]) == 2
    assert main([*prefix, "validate"]) == 0
    calls.clear()

    def comparison(config, corpus, alias, **kwargs):
        calls.append(kwargs)
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.compare_hyde", comparison)
    assert (
        main(
            [
                *prefix,
                "compare-hyde",
                "--hyde-language",
                "fr",
                "--control",
                "dense",
                "--control",
                "translate-dense",
            ]
        )
        == 0
    )
    assert calls == [{"languages": ("fr",), "controls": ("dense", "translate-dense")}]
    calls.clear()
    assert main([*prefix, "compare-hyde"]) == 0
    assert calls == [{"languages": ("fr", "ja"), "controls": ("dense",)}]


def test_cli_hyde_workflow_and_generation_reuse_across_retrieval_models(
    tmp_path, tokenizer, monkeypatch
):
    config, path = fixture_hyde(tmp_path, tokenizer)
    raw = yaml.safe_load(path.read_text())
    raw["hyde"] = {
        "model": "fixture",
        "endpoint": "https://example.invalid/chat",
        "api_key_env": None,
        "output_languages": ["fr"],
    }
    path.write_text(yaml.safe_dump(raw))
    generator = FixtureGenerator()
    monkeypatch.setattr(
        "clir_benchmark.hyde.ChatCompletionsHyDE.complete",
        lambda self, payload: generator.complete(payload),
    )
    monkeypatch.setattr(
        "clir_benchmark.cli.embed",
        lambda config, corpus, alias, **kwargs: embed(
            config, corpus, alias, encoder_factory=FixtureEncoder, **kwargs
        ),
    )
    prefix = ["--config", str(path)]
    assert main([*prefix, "generate-hyde"]) == 0
    for alias in ("bge-m3", "LaBSE"):
        assert main([*prefix, "embed", "--model", alias, "--hyde-language", "fr"]) == 0
        assert main([*prefix, "retrieve", "--model", alias, "--baseline", "hyde-dense"]) == 0
        assert main([*prefix, "evaluate", "--model", alias, "--baseline", "hyde-dense"]) == 0
    assert len(generator.calls) == 1
    assert main([*prefix, "embed", "--model", "bge-m3"]) == 0
    assert main([*prefix, "retrieve", "--model", "bge-m3"]) == 0
    assert main([*prefix, "compare-hyde", "--model", "bge-m3"]) == 0
    assert len(generator.calls) == 1
