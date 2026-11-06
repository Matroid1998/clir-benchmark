import json
from dataclasses import replace

import pytest
import yaml
from conftest import FixtureEncoder
from test_pipeline import fixture_config, write_rows
from test_translation import FixtureTranslator, completion

from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.cli import main
from clir_benchmark.comparison import compare_translations
from clir_benchmark.errors import ValidationError
from clir_benchmark.lexical import prepare_bm25
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
from clir_benchmark.translation import load_translations, prepare_translations, translation_path
from clir_benchmark.util import file_digest, read_json


def prepared(tmp_path, tokenizer, *, annotations=True):
    config, path = fixture_config(tmp_path, annotations=annotations)
    config = replace(
        config,
        translation=replace(
            config.translation,
            model="fixture-model",
            endpoint="https://provider.invalid/v1/chat/completions",
            input_usd_per_million=2,
            cached_input_usd_per_million=0.5,
            output_usd_per_million=4,
        ),
    )
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    return config, path


def test_translate_bm25_improvement_and_cost_comparison(tmp_path, tokenizer):
    config, _ = prepared(tmp_path, tokenizer)
    translator = FixtureTranslator()
    qrels_path = config.relevance_dir("eurlex") / "qrels.parquet"
    qrels_hash = file_digest(qrels_path)
    prepare_bm25(config, "eurlex")
    original = retrieve(config, "eurlex", baseline="bm25")
    fr = prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory)
    en = prepare_translations(config, "eurlex", "en", translator_factory=translator.factory)
    assert len(translator.calls) == 1  # Only eligible qbest; source language uses identity.
    assert [row["id"] for row in rows(fr / "queries.parquet")] == ["qbest"]
    assert read_json(en / "manifest.json")["summary"]["api_attempts"] == 0
    assert (
        translator.calls[0]["messages"][1]["content"]
        == "Source language: en\nTarget language: fr\n\n<query>\nalpha\n</query>"
    )
    assert "preuve" not in json.dumps(translator.calls)
    variants = {
        language: retrieve(
            config, "eurlex", baseline="translate-bm25", translation_language=language
        )
        for language in ("en", "fr")
    }
    assert len({original, *variants.values()}) == 3
    assert list(rows(original / "run.parquet")) == list(rows(variants["en"] / "run.parquet"))
    comparison = compare_translations(
        config, "eurlex", None, baseline="bm25", languages=("fr", "en")
    )
    report = read_json(comparison / "report.json")
    french = next(
        row
        for row in report["settings"]
        if row["translation_language"] == "fr" and row["setting_id"] == "eurlex__en__fr"
    )
    assert french["original_metrics.Hit@100"] == 0
    assert french["translated_metrics.Hit@100"] == french["metric_deltas.Hit@100"] == 1
    assert french["mean_translation_cost_usd"] == pytest.approx(0.00022)
    assert french["mean_added_component_seconds"] is not None
    assert (
        report["translations"][0]["translation_summary"]["api_attempts"] == 1
    )  # Not once per setting.
    assert file_digest(qrels_path) == qrels_hash
    assert prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory) == fr
    assert len(translator.calls) == 1


@pytest.mark.parametrize("baseline", ["dense", "hybrid"])
def test_translated_dense_and_hybrid_reuse_document_vectors(tmp_path, tokenizer, baseline):
    config, _ = prepared(tmp_path, tokenizer)
    translator = FixtureTranslator()
    prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory)
    originals = embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)
    translated = embed(
        config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder, translation_language="fr"
    )
    assert originals["document"] == translated["document"]
    assert originals["query"] != translated["query"]
    spec = model_spec(config, "bge-m3")
    assert translated["query"] == embedding_path(
        config, "eurlex", spec, "query", translation_language="fr"
    )
    if baseline == "hybrid":
        prepare_bm25(config, "eurlex")
    retrieve(config, "eurlex", "bge-m3", baseline=baseline)
    variant = retrieve(
        config, "eurlex", "bge-m3", baseline=f"translate-{baseline}", translation_language="fr"
    )
    report_dir = evaluate(
        config, "eurlex", "bge-m3", baseline=f"translate-{baseline}", translation_language="fr"
    )
    report = read_json(report_dir / "report.json")
    assert report["translation_language"] == "fr"
    assert report["truncation"]["queries"]["fr"]["inputs"] == 1
    assert report["run"]["translation"]["summary"]["api_attempts"] == 1
    assert {row["query_id"] for row in rows(variant / "run.parquet")} == {"qbest"}
    compare_translations(config, "eurlex", "bge-m3", baseline=baseline, languages=("fr",))
    assert len(translator.calls) == 1


def test_failed_responses_are_audited_and_retry_costs_are_included(tmp_path, tokenizer):
    config, _ = prepared(tmp_path, tokenizer)
    translator = FixtureTranslator()
    invalid = completion()
    invalid["choices"][0]["message"]["content"] = '{"translation":"preuve", "extra":"not allowed"}'
    translator.responses = [invalid, completion()]
    with pytest.raises(ValidationError, match="no fallback"):
        prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory)
    assert not translation_path(config, "eurlex", "fr").exists()
    audit = next((config.output_dir / "translation_requests").glob("*.json"))
    assert read_json(audit)["attempts"][0]["response"] == invalid
    with pytest.raises(ValidationError, match="retry-failed"):
        prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory)
    assert len(translator.calls) == 1
    prepare_translations(
        config, "eurlex", "fr", translator_factory=translator.factory, retry_failed=True
    )
    _, records, _ = load_translations(config, "eurlex", "fr")
    assert records[0]["attempts"] == 2
    assert records[0]["input_tokens"] == 200
    assert records[0]["estimated_cost_usd"] == pytest.approx(0.00044)


def test_resume_preserves_successes_after_later_failure(tmp_path, tokenizer):
    config, _ = fixture_config(tmp_path)
    qas = list(rows(tmp_path / "dataset/data/qas/train-00000.parquet"))
    qas[-1]["answer"] = "alpha"  # qdrop now eligible in the English setting.
    write_rows(tmp_path / "dataset", "qas", qas)
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    translator = FixtureTranslator()
    translator.responses = [completion(), {"choices": []}, completion()]
    with pytest.raises(ValidationError):
        prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory)
    assert len(translator.calls) == 2
    path = prepare_translations(
        config, "eurlex", "fr", translator_factory=translator.factory, retry_failed=True
    )
    assert len(translator.calls) == 3
    records = list(rows(path / "queries.parquet"))
    assert [row["attempts"] for row in records] == [1, 2]
    assert records[1]["estimated_cost_usd"] is None


def test_prices_recompute_without_paid_calls_and_models_invalidate_translation_cache(
    tmp_path, tokenizer
):
    config, _ = prepared(tmp_path, tokenizer)
    translator = FixtureTranslator()
    original = prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory)
    repriced = replace(config, translation=replace(config.translation, output_usd_per_million=8))
    new = prepare_translations(repriced, "eurlex", "fr", translator_factory=translator.factory)
    assert new != original and len(translator.calls) == 1
    assert read_json(new / "manifest.json")["summary"]["estimated_cost_usd"] == pytest.approx(
        0.0003
    )
    changed = replace(config, translation=replace(config.translation, model="other-model"))
    prepare_translations(changed, "eurlex", "fr", translator_factory=translator.factory)
    assert len(translator.calls) == 2
    assert changed.corpus_key("eurlex") == config.corpus_key("eurlex")
    assert changed.relevance_key("eurlex") == config.relevance_key("eurlex")
    assert run_path(changed, "eurlex", baseline="bm25") == run_path(
        config, "eurlex", baseline="bm25"
    )


def test_blocked_settings_stay_blocked_and_missing_artifacts_are_actionable(tmp_path, tokenizer):
    config, _ = prepared(tmp_path, tokenizer, annotations=False)
    with pytest.raises(ValidationError, match="Run translate first"):
        retrieve(config, "eurlex", baseline="translate-bm25", translation_language="fr")
    with pytest.raises(ValidationError, match="outside"):
        prepare_translations(config, "eurlex", "de")
    prepare_translations(config, "eurlex", "fr", translator_factory=FixtureTranslator().factory)
    prepare_bm25(config, "eurlex")
    path = retrieve(config, "eurlex", baseline="translate-bm25", translation_language="fr")
    assert not (path / "eurlex__en__fr.trec").exists()
    assert read_json(path / "manifest.json")["settings"]["eurlex__en__fr"]["status"] == "blocked"


def test_cli_dispatch_and_no_implicit_paid_translation(tmp_path, tokenizer, monkeypatch):
    config, path = prepared(tmp_path, tokenizer)
    # CLI config intentionally has no translation provider or model selected.
    assert main(["--config", str(path), "validate"]) == 0
    calls = []

    def fake_translate(config, corpus, language, **kwargs):
        calls.append(("translate", language))
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.prepare_translations", fake_translate)
    assert main(["--config", str(path), "translate"]) == 0
    assert calls == [("translate", "en"), ("translate", "fr")]
    calls.clear()

    def fake_retrieve(config, corpus, alias=None, *, baseline, translation_language=None):
        calls.append((baseline, alias, translation_language))
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.retrieve", fake_retrieve)
    assert main(["--config", str(path), "retrieve", "--baseline", "translate-bm25"]) == 0
    assert calls == [("translate-bm25", None, "en"), ("translate-bm25", None, "fr")]
    assert (
        main(
            [
                "--config",
                str(path),
                "retrieve",
                "--baseline",
                "bm25",
                "--translation-language",
                "fr",
            ]
        )
        == 2
    )
    assert (
        main(
            ["--config", str(path), "retrieve", "--baseline", "translate-bm25", "--model", "bge-m3"]
        )
        == 2
    )
    raw = yaml.safe_load(path.read_text())
    raw["baselines"] = ["translate-bm25", "translate-dense", "translate-hybrid"]
    path.write_text(yaml.safe_dump(raw))
    assert main(["--config", str(path), "validate"]) == 0


def test_prompt_and_generation_option_changes_invalidate_paid_request_cache(
    tmp_path, tokenizer, monkeypatch
):
    from clir_benchmark import translation

    config, _ = prepared(tmp_path, tokenizer)
    translator = FixtureTranslator()
    original = prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory)
    updated = replace(
        config, translation=replace(config.translation, request_options={"temperature": 0})
    )
    different_options = prepare_translations(
        updated, "eurlex", "fr", translator_factory=translator.factory
    )
    original_prompts = translation.prompts
    monkeypatch.setattr(
        translation,
        "prompts",
        lambda corpus: (
            original_prompts(corpus)[0] + "\nPrompt revision.",
            original_prompts(corpus)[1],
        ),
    )
    different_prompt = prepare_translations(
        config, "eurlex", "fr", translator_factory=translator.factory
    )
    assert len({original, different_options, different_prompt}) == len(translator.calls) == 3


def test_interrupted_request_is_not_implicitly_retried(tmp_path, tokenizer):
    config, _ = prepared(tmp_path, tokenizer)

    class InterruptingTranslator:
        def complete(self, payload):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        prepare_translations(
            config, "eurlex", "fr", translator_factory=lambda *args: InterruptingTranslator()
        )
    audit = next((config.output_dir / "translation_requests").glob("*.json"))
    assert read_json(audit)["attempts"][0]["status"] == "pending"
    translator = FixtureTranslator()
    with pytest.raises(ValidationError, match="retry-failed"):
        prepare_translations(config, "eurlex", "fr", translator_factory=translator.factory)
    assert not translator.calls
    path = prepare_translations(
        config, "eurlex", "fr", translator_factory=translator.factory, retry_failed=True
    )
    summary = read_json(path / "manifest.json")["summary"]
    assert summary["api_attempts"] == 2
    assert summary["estimated_cost_usd"] is None
    assert summary["translation_seconds"] is None
