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
