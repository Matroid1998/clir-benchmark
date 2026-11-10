from dataclasses import replace

import pytest
import yaml

from clir_benchmark.cli import main
from clir_benchmark.config import (
    CandidateSelectionConfig,
    ChunkingConfig,
    RerankerConfig,
    SourceSpec,
    load_config,
    validate_retrieval_baselines,
)
from clir_benchmark.errors import ValidationError


@pytest.mark.parametrize("revision", ["main", "latest", "abc", None])
def test_moving_revisions_rejected(revision):
    with pytest.raises(ValidationError, match="commit SHA"):
        SourceSpec("fixture/data", revision)


@pytest.mark.parametrize("max_tokens,overlap", [(513, 64), (512, 510), (512, -1), (2, 0)])
def test_invalid_chunk_policies_rejected(max_tokens, overlap):
    with pytest.raises(ValidationError):
        ChunkingConfig(max_tokens=max_tokens, overlap_tokens=overlap)


def test_dataset_structure_error_is_actionable(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "datasets": {"eurlex": {"qa": {}}}}))
    with pytest.raises(ValidationError, match="qa and corpus"):
        load_config(path)
    assert main(["--config", str(path), "validate"]) == 2


def test_default_config_never_loads_encoder(capsys):
    assert main(["--config", "configs/benchmark.yaml", "list-models"]) == 0
    assert "SapBERT" in capsys.readouterr().out


@pytest.mark.parametrize(
    "options",
    [
        {"revision": "main"},
        {"repo_id": "Qwen/Qwen3-Embedding-0.6B"},
        {"candidate_k": 0},
        {"candidate_k": 101},
        {"candidate_k": True},
        {"max_length": 32769},
        {"max_length": 64},
        {"batch_size": 0},
        {"dtype": "int8"},
        {"device": ""},
        {"instruction": " "},
    ],
)
def test_reranker_invalid_model_or_budget_rejected(options):
    with pytest.raises(ValidationError):
        RerankerConfig(**options)


def test_reranker_choices_preserve_shared_preparation_identity():
    config = load_config("configs/benchmark.yaml")
    assert config.baselines == ("dense",)
    assert config.reranker.repo_id == "Qwen/Qwen3-Reranker-0.6B"
    changed = replace(
        config,
        baselines=("bm25", "rerank-bm25", "rerank-adore-dense"),
        reranker=replace(config.reranker, candidate_k=20, batch_size=4),
    )
    for corpus in config.datasets:
        assert changed.corpus_key(corpus) == config.corpus_key(corpus)
        assert changed.relevance_key(corpus) == config.relevance_key(corpus)


def test_alignment_release_keeps_original_corpus_cache_keys():
    config = load_config("configs/benchmark.yaml")
    old_pins = {
        "eurlex": (
            "8b5d1bd5604e7ebb7c27bc6d877b6b3ef712a006",
            "8b5d1bd5604e7ebb7c27bc6d877b6b3ef712a006",
        ),
        "un": (
            "79b190ab68940cfed6236c372cbfe53404339c05",
            "79b190ab68940cfed6236c372cbfe53404339c05",
        ),
        "chemistry": (
            "d390d599ef42870bc5fe2349403ba36924546022",
            "24c1f894591a5aa6f745ac233d5091fdbc30ad3c",
        ),
    }
    assert set(config.target_languages) == {"en", "de", "es", "fr", "zh"}
    for name, (old_qa_pin, corpus_pin) in old_pins.items():
        assert config.datasets[name].corpus.revision == corpus_pin
        old_dataset = replace(
            config.datasets[name], qa=replace(config.datasets[name].qa, revision=old_qa_pin)
        )
        previous = replace(
            config, datasets={**config.datasets, name: old_dataset}, target_languages=()
        )
        assert previous.corpus_key(name) == config.corpus_key(name)
        assert previous.relevance_key(name) != config.relevance_key(name)


@pytest.mark.parametrize(
    "options",
    [
        {"seed": True},
        {"seed": "42"},
        {"seed": 2.5},
        {"preferred_generators": "model"},
        {"preferred_generators": [""]},
        {"preferred_generators": ["same", "same"]},
    ],
)
def test_invalid_document_selection_policy(options):
    with pytest.raises(ValidationError, match="candidate_selection"):
        CandidateSelectionConfig(**options)


def test_selection_tier_order_is_not_significant():
    assert CandidateSelectionConfig(preferred_generators=["a", "b"]) == CandidateSelectionConfig(
        preferred_generators=["b", "a"]
    )


def test_new_selection_and_document_mode_preserve_legacy_cache_identities():
    from clir_benchmark.util import digest

    original = load_config("configs/benchmark.yaml")
    for mode in ("best", "all"):
        config = replace(original, candidate_mode=mode)
        for corpus in config.datasets:
            legacy_identity = {
                "schema": 2,
                "corpus_key": config.corpus_key(corpus),
                "qa": config.datasets[corpus].qa.identity(),
                "candidate_mode": mode,
                "settings": config.settings,
                "query_languages": config.query_languages,
                "target_languages": config.target_languages,
                "evidence": None,
            }
            assert config.relevance_key(corpus) == digest(legacy_identity)
            inactive_policy = replace(config, candidate_selection=CandidateSelectionConfig(seed=7))
            assert inactive_policy.relevance_key(corpus) == config.relevance_key(corpus)
    selected = replace(original, candidate_mode="best-per-document")
    changed_seed = replace(selected, candidate_selection=CandidateSelectionConfig(seed=7))
    documents = replace(selected, retrieval_unit="document")
    for corpus in original.datasets:
        assert (
            len({c.relevance_key(corpus) for c in (original, selected, changed_seed, documents)})
            == 4
        )
        assert (
            len({c.corpus_key(corpus) for c in (original, selected, changed_seed, documents)}) == 1
        )


def test_document_baseline_validation_happens_after_explicit_override(tmp_path, monkeypatch):
    raw = yaml.safe_load(open("configs/benchmark.yaml"))
    raw.update(retrieval_unit="document", baselines=["bm25"])
    path = tmp_path / "document.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    with pytest.raises(ValidationError, match="supports only"):
        validate_retrieval_baselines(config)
    validate_retrieval_baselines(config, ("dense", "translate-dense", "hyde-dense"))
    monkeypatch.setattr("clir_benchmark.cli.retrieve", lambda *args, **kwargs: tmp_path)
    assert main(["--config", str(path), "retrieve", "--corpus", "un", "--baseline", "dense"]) == 0
    assert main(["--config", str(path), "retrieve", "--corpus", "un"]) == 2


@pytest.mark.parametrize("before", [False, True])
def test_document_selection_cli_validation_reads_sources_without_preparing_artifacts(
    tmp_path, capsys, before
):
    import json

    from test_pipeline import fixture_config

    config, path = fixture_config(tmp_path)
    flags = [
        "--candidate-mode",
        "best-per-document",
        "--selection-seed",
        "42",
        "--retrieval-unit",
        "document",
    ]
    command = [*flags, "validate", "--inputs"] if before else ["validate", "--inputs", *flags]
    assert main(["--config", str(path), *command]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidate_mode"] == "best-per-document"
    assert result["retrieval_unit"] == "document"
    assert result["inputs"]["eurlex"]["selected_queries"] == 1
    assert result["inputs"]["eurlex"]["canonical_documents"] == 1
    assert not config.output_dir.exists()
