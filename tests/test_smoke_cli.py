from types import SimpleNamespace

import pytest

from clir_benchmark.cli import main
from clir_benchmark.util import write_json


@pytest.mark.parametrize("status,expected", [("success", 0), ("failed", 2)])
def test_smoke_cli_preserves_selection_and_returns_outcome(tmp_path, monkeypatch, status, expected):
    import sys

    captured = {}

    def run(config, corpus, alias, **kwargs):
        captured.update(config=config, corpus=corpus, alias=alias, **kwargs)
        write_json(tmp_path / "status.json", {"status": status})
        return tmp_path

    monkeypatch.setitem(sys.modules, "clir_benchmark.smoke", SimpleNamespace(run_smoke=run))
    assert (
        main(
            [
                "--config",
                "configs/benchmark.yaml",
                "smoke",
                "--corpus",
                "eurlex",
                "--model",
                "minilm-l6",
                "--baseline",
                "dense",
                "--baseline",
                "rerank-adore-dense",
                "--adore-rounds",
                "3",
                "--rerank-candidates",
                "3",
            ]
        )
        == expected
    )
    assert captured["config"].baselines == ("dense", "rerank-adore-dense")
    assert captured["config"].adore.max_rounds == 3
    assert captured["config"].reranker.candidate_k == 3
    assert captured["alias"] == "minilm-l6"


def test_bm25_smoke_requires_no_embedding_model(tmp_path, monkeypatch):
    import sys

    def run(config, corpus, alias, **kwargs):
        assert alias is None
        assert config.baselines == ("bm25",)
        write_json(tmp_path / "status.json", {"status": "success"})
        return tmp_path

    monkeypatch.setitem(sys.modules, "clir_benchmark.smoke", SimpleNamespace(run_smoke=run))
    assert main(["smoke", "--corpus", "eurlex", "--baseline", "bm25"]) == 0


def test_prepare_subset_cli_never_runs_baselines(tmp_path, monkeypatch, capsys):
    import sys

    def prepare(config, corpus, **kwargs):
        assert corpus == "eurlex"
        assert kwargs["max_questions"] == 2
        assert kwargs["max_sources"] == 12
        return tmp_path

    monkeypatch.setitem(
        sys.modules, "clir_benchmark.subset", SimpleNamespace(prepare_subset=prepare)
    )
    assert main(["prepare-subset", "--corpus", "eurlex"]) == 0
    assert str(tmp_path / "config.yaml") in capsys.readouterr().out
