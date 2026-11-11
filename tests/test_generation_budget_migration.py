from dataclasses import replace

import pytest
import yaml
from conftest import FixtureEncoder
from test_hyde import FixtureGenerator, hyde_completion
from test_pipeline import fixture_config, write_rows
from test_translation import FixtureTranslator, completion

from clir_benchmark.artifacts import rows
from clir_benchmark.chunking import Chunker
from clir_benchmark.cli import main
from clir_benchmark.config import GenerationCacheReuseConfig, HyDEConfig, load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.hyde import hyde_path, load_hyde, prepare_hyde
from clir_benchmark.pipeline import embed, prepare_corpus, prepare_relevance
from clir_benchmark.translation import (
    load_translations,
    prepare_translations,
    translation_path,
)
from clir_benchmark.util import read_json


def migration_fixture(tmp_path, tokenizer, *, two_queries=False):
    config, path = fixture_config(tmp_path)
    if two_queries:
        queries = list(rows(tmp_path / "dataset/data/qas/train-00000.parquet"))
        queries[-1]["answer"] = "alpha"
        write_rows(tmp_path / "dataset", "qas", queries)
    options = {"max_completion_tokens": 2048, "reasoning_effort": "low"}
    generation = {
        "model": "fixture-model",
        "endpoint": "https://provider.invalid/v1/chat/completions",
        "request_options": options,
        "input_usd_per_million": 2,
        "cached_input_usd_per_million": 0.5,
        "output_usd_per_million": 4,
    }
    config = replace(
        config,
        translation=replace(config.translation, **generation),
        hyde=HyDEConfig(**generation),
    )
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    return config, path


def migrate(config, method):
    section = getattr(config, method)
    return replace(
        config,
        **{
            method: replace(
                section,
                request_options={**section.request_options, "max_completion_tokens": 8192},
            ),
            "generation_cache_reuse": GenerationCacheReuseConfig(
                **{f"{method}_max_completion_tokens": (2048,)}
            ),
        },
    )


def method_functions(method):
    if method == "translation":
        return (
            FixtureTranslator,
            completion,
            prepare_translations,
            load_translations,
            translation_path,
            "translator_factory",
        )
    return (
        FixtureGenerator,
        hyde_completion,
        prepare_hyde,
        load_hyde,
        hyde_path,
        "generator_factory",
    )


def forbidden_client(*args, **kwargs):
    pytest.fail("Reusing successful generation must not initialize an API client")


@pytest.mark.parametrize("method", ["translation", "hyde"])
def test_larger_budget_reuses_success_with_original_provenance_and_corpus_vectors(
    tmp_path, tokenizer, method
):
    config, _ = migration_fixture(tmp_path, tokenizer)
    client_type, _, prepare, load, path_for, factory_name = method_functions(method)
    client = client_type()
    original = prepare(config, "eurlex", "fr", **{factory_name: client.factory})
    _, old_records, old_manifest = load(config, "eurlex", "fr")
    old_request_key = old_records[0]["request_key"]
    request_file = config.output_dir / f"{method}_requests" / f"{old_request_key}.json"
    old_request_bytes = request_file.read_bytes()
    old_manifest_bytes = (original / "manifest.json").read_bytes()
    original_vectors = embed(config, "eurlex", "bge-m3", encoder_factory=FixtureEncoder)

    changed = migrate(config, method)
    assert changed.corpus_key("eurlex") == config.corpus_key("eurlex")
    assert changed.relevance_key("eurlex") == config.relevance_key("eurlex")
    assert path_for(changed, "eurlex", "fr") != original
    # Reuse is opt-in and contributes to artifact identity independently of the new cap.
    strict = replace(changed, generation_cache_reuse=GenerationCacheReuseConfig())
    assert path_for(changed, "eurlex", "fr") != path_for(strict, "eurlex", "fr")
    unused_method = "hyde" if method == "translation" else "translation"
    unrelated_policy = replace(
        config,
        generation_cache_reuse=GenerationCacheReuseConfig(
            **{f"{unused_method}_max_completion_tokens": (1024,)}
        ),
    )
    assert path_for(unrelated_policy, "eurlex", "fr") == original

    migrated = prepare(changed, "eurlex", "fr", **{factory_name: forbidden_client})
    _, new_records, new_manifest = load(changed, "eurlex", "fr")
    assert migrated != original
    assert new_records == old_records
    assert new_records[0]["request_key"] == old_request_key
    assert new_manifest["configuration"]["request_options"]["max_completion_tokens"] == 8192
    assert new_manifest["summary"] == old_manifest["summary"]
    assert new_manifest["new_api_attempts"] == 0
    assert new_manifest["new_work"]["api_attempts"] == 0
    assert new_manifest["new_work"]["estimated_cost_usd"] == 0
    assert request_file.read_bytes() == old_request_bytes
    assert (original / "manifest.json").read_bytes() == old_manifest_bytes
    assert len(list(request_file.parent.glob("*.json"))) == 1
    assert len(client.calls) == 1
    assert embed(changed, "eurlex", "bge-m3", encoder_factory=FixtureEncoder) == original_vectors


@pytest.mark.parametrize("method", ["translation", "hyde"])
def test_partial_batch_reuses_success_and_accounts_for_failed_old_budget_exactly_once(
    tmp_path, tokenizer, method
):
    config, _ = migration_fixture(tmp_path, tokenizer, two_queries=True)
    client_type, response, prepare, load, path_for, factory_name = method_functions(method)
    client = client_type()
    truncated = response()
    truncated["choices"][0]["finish_reason"] = "length"
    truncated["choices"][0]["message"]["content"] = ""
    client.responses = [response(), truncated, response()]
    with pytest.raises(ValidationError, match="no fallback"):
        prepare(config, "eurlex", "fr", **{factory_name: client.factory})
    assert len(client.calls) == 2
    assert not path_for(config, "eurlex", "fr").exists()
    cache_root = config.output_dir / f"{method}_requests"
    old_audits = {p: p.read_bytes() for p in cache_root.glob("*.json")}
    old_states = [read_json(p) for p in old_audits]
    failed = next(state for state in old_states if state["status"] == "failed")
    succeeded = next(state for state in old_states if state["status"] == "success")

    changed = migrate(config, method)
    with pytest.raises(ValidationError, match="retry-failed"):
        prepare(changed, "eurlex", "fr", **{factory_name: forbidden_client})
    assert len(client.calls) == 2
    assert not path_for(changed, "eurlex", "fr").exists()
    prepare(changed, "eurlex", "fr", retry_failed=True, **{factory_name: client.factory})
    assert len(client.calls) == 3
    assert client.calls[-1]["max_completion_tokens"] == 8192
    assert all(path.read_bytes() == contents for path, contents in old_audits.items())
    _, records, manifest = load(changed, "eurlex", "fr")
    by_id = {record["id"]: record for record in records}
    assert by_id["qbest"]["request_key"] == succeeded["request_key"]
    assert by_id["qbest"]["attempts"] == 1
    retried = by_id["qdrop"]
    assert retried["request_key"] != failed["request_key"]
    new_state = read_json(cache_root / f"{retried['request_key']}.json")
    assert new_state["prior_request_keys"] == [failed["request_key"]]
    assert len(new_state["attempts"]) == 1
    assert new_state["payload"]["max_completion_tokens"] == 8192
    assert retried["attempts"] == 2
    assert retried["input_tokens"] == 200
    assert retried["cached_input_tokens"] == 80
    assert retried["output_tokens"] == 40
    assert retried["estimated_cost_usd"] == pytest.approx(0.00044)
    assert manifest["summary"]["api_attempts"] == 3
    assert manifest["summary"]["estimated_cost_usd"] == pytest.approx(0.00066)
    assert manifest["new_api_attempts"] == 1
    assert manifest["new_work"]["api_attempts"] == 1
    assert manifest["new_work"]["estimated_cost_usd"] == pytest.approx(0.00022)

    # Reading/reusing the migrated artifact must not add the old failed attempt again.
    prepare(changed, "eurlex", "fr", **{factory_name: forbidden_client})
    assert load(changed, "eurlex", "fr")[1] == records
    assert len(client.calls) == 3


@pytest.mark.parametrize("method", ["translation", "hyde"])
@pytest.mark.parametrize("value", ["2048", None, (0,), (-1,), (True,), (2048.0,), ("2048",), (2048, 2048)])
def test_invalid_generation_cache_reuse_budgets(method, value):
    with pytest.raises(ValidationError):
        GenerationCacheReuseConfig(**{f"{method}_max_completion_tokens": value})


def test_generation_reuse_configuration_loads_without_enabling_generation(tmp_path):
    original, path = fixture_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["generation_cache_reuse"] = {
        "translation_max_completion_tokens": [2048],
        "hyde_max_completion_tokens": [1024, 2048],
    }
    path.write_text(yaml.safe_dump(raw))
    loaded = load_config(path)
    assert loaded.generation_cache_reuse.translation_max_completion_tokens == (2048,)
    assert loaded.generation_cache_reuse.hyde_max_completion_tokens == (1024, 2048)
    assert loaded.baselines == original.baselines == ("dense",)
    assert loaded.translation.model is loaded.hyde.model is None
    assert loaded.corpus_key("eurlex") == original.corpus_key("eurlex")
    assert loaded.relevance_key("eurlex") == original.relevance_key("eurlex")


@pytest.mark.parametrize("before_subcommand", [True, False])
@pytest.mark.parametrize("method", ["translation", "hyde"])
def test_cli_reuse_policy_reaches_only_the_explicit_generation_stage(
    tmp_path, monkeypatch, method, before_subcommand
):
    _, path = fixture_config(tmp_path)
    command = "translate" if method == "translation" else "generate-hyde"
    language_flag = "--translation-language" if method == "translation" else "--hyde-language"
    function = "prepare_translations" if method == "translation" else "prepare_hyde"
    calls = []

    def capture(config, corpus, language, **kwargs):
        calls.append((config, corpus, language))
        return tmp_path

    monkeypatch.setattr(f"clir_benchmark.cli.{function}", capture)
    other_function = "prepare_hyde" if method == "translation" else "prepare_translations"
    monkeypatch.setattr(f"clir_benchmark.cli.{other_function}", forbidden_client)
    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.complete", forbidden_client)
    flags = [f"--reuse-{method}-token-limit", "1024", f"--reuse-{method}-token-limit", "2048"]
    arguments = (
        [*flags, command, language_flag, "fr"]
        if before_subcommand
        else [command, language_flag, "fr", *flags]
    )
    assert main(["--config", str(path), *arguments]) == 0
    assert len(calls) == 1
    received, corpus, language = calls[0]
    assert (corpus, language) == ("eurlex", "fr")
    assert getattr(received.generation_cache_reuse, f"{method}_max_completion_tokens") == (
        1024,
        2048,
    )


@pytest.mark.parametrize("before_subcommand", [True, False])
def test_cli_disable_clears_configured_reuse_for_both_methods(
    tmp_path, monkeypatch, before_subcommand
):
    _, path = fixture_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["generation_cache_reuse"] = {
        "translation_max_completion_tokens": [2048],
        "hyde_max_completion_tokens": [2048],
    }
    path.write_text(yaml.safe_dump(raw))
    observed = []

    def capture(config, corpus, language, **kwargs):
        observed.append(config.generation_cache_reuse)
        return tmp_path

    monkeypatch.setattr("clir_benchmark.cli.prepare_hyde", capture)
    monkeypatch.setattr("clir_benchmark.cli.prepare_translations", forbidden_client)
    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.complete", forbidden_client)
    stage = ["generate-hyde", "--hyde-language", "fr"]
    flag = ["--no-generation-cache-reuse"]
    arguments = [*flag, *stage] if before_subcommand else [*stage, *flag]
    assert main(["--config", str(path), *arguments]) == 0
    assert observed == [GenerationCacheReuseConfig()]


@pytest.mark.parametrize("method", ["translation", "hyde"])
def test_cli_validation_is_offline_and_rejects_conflicting_reuse_flags(
    tmp_path, monkeypatch, capsys, method
):
    _, path = fixture_config(tmp_path)
    monkeypatch.setattr("clir_benchmark.cli.prepare_hyde", forbidden_client)
    monkeypatch.setattr("clir_benchmark.cli.prepare_translations", forbidden_client)
    monkeypatch.setattr("clir_benchmark.generation.ChatCompletionsClient.complete", forbidden_client)
    prefix = ["--config", str(path)]
    reuse = [f"--reuse-{method}-token-limit", "2048"]
    # Dense-only validation needs neither credentials nor a generation token limit.
    assert main([*prefix, "validate", *reuse]) == 0
    assert main([*prefix, *reuse, "validate", "--no-generation-cache-reuse"]) == 2
    assert "Cannot enable and disable generation cache reuse together" in capsys.readouterr().err
