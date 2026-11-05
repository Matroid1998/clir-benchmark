from dataclasses import replace

import pytest
import yaml

from clir_benchmark.cli import main
from clir_benchmark.config import ChunkingConfig, SourceSpec, load_config
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
