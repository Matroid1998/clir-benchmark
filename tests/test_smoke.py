from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FixtureEncoder
from test_pipeline import fixture_config

from clir_benchmark import smoke
from clir_benchmark.chunking import Chunker
from clir_benchmark.config import BASELINES, DatasetSpec, SourceSpec
from clir_benchmark.errors import ValidationError
from clir_benchmark.pipeline import embed, prepare_corpus, prepare_relevance
from clir_benchmark.util import read_json, write_json


def fixture(tmp_path, tokenizer, baselines):
    config, _ = fixture_config(tmp_path)
    config = replace(config, baselines=baselines, target_languages=("en", "fr"))
    prepare_corpus(config, "eurlex", chunker=Chunker(tokenizer, config.chunking))
    prepare_relevance(config, "eurlex")
    return config


def fake_stages(monkeypatch, tmp_path):
    calls = []

    def path(name, *args, **kwargs):
        suffix = "-".join(
            str(value)
            for value in (
                kwargs.get("baseline"),
                kwargs.get("translation_language"),
                kwargs.get("hyde_language"),
                kwargs.get("kind"),
            )
            if value
        )
        return tmp_path / "fake-stages" / name / (suffix or "default")

    def operation(name):
        def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            destination = path(name, *args, **kwargs)
            write_json(destination / "fixture.json", {"fixture": True})
            if name == "embed":
                return {kwargs["kind"]: destination}
            return destination

        return run

    for name in (
        "embed",
        "prepare_bm25",
        "prepare_translations",
        "prepare_hyde",
        "retrieve",
        "run_adore",
        "rerank",
        "evaluate",
        "compare_translations",
        "compare_hyde",
        "compare_adore",
        "compare_rerank",
    ):
        monkeypatch.setattr(smoke, name, operation(name))
    for name in (
        "embedding_path",
        "bm25_path",
        "translation_path",
        "hyde_path",
        "run_path",
        "adore_run_path",
        "rerank_run_path",
    ):
        monkeypatch.setattr(
            smoke, name, lambda *args, _name=name, **kwargs: path(_name, *args, **kwargs)
        )
    return calls


def test_smoke_schedules_every_baseline_language_and_shared_stage_once(
    tmp_path, tokenizer, monkeypatch
):
    config = fixture(tmp_path, tokenizer, BASELINES)
    calls = fake_stages(monkeypatch, tmp_path)
    destination = smoke.run_smoke(config, "eurlex", "bge-m3", retry_failed=True)
    status = read_json(destination / "status.json")
    assert status["status"] == "success"
    assert set(status["selected_baselines"]) == set(BASELINES)
    assert len(status["baselines"]) == 28  # Four language-sensitive modes, plus their rerankers.
    assert all(row["status"] == "success" for row in status["baselines"])
    assert {row["baseline"] for row in status["baselines"]} == set(BASELINES)
    embeddings = [kwargs for name, _, kwargs in calls if name == "embed"]
    assert len([row for row in embeddings if row["kind"] == "document"]) == 1
    assert len([row for row in embeddings if row["kind"] == "query"]) == 5
    assert len([name for name, _, _ in calls if name == "prepare_bm25"]) == 1
    for name in ("prepare_translations", "prepare_hyde"):
        generated = [(args, kwargs) for operation, args, kwargs in calls if operation == name]
        assert len(generated) == 2
        assert {args[2] for args, _ in generated} == {"en", "fr"}
        assert all(kwargs["retry_failed"] for _, kwargs in generated)
    assert len([name for name, _, _ in calls if name == "run_adore"]) == 3
    assert len([name for name, _, _ in calls if name == "rerank"]) == 14
    assert len([name for name, _, _ in calls if name == "compare_rerank"]) == 14
    assert len({row["id"] for row in status["stages"]}) == len(status["stages"])
    assert status["configuration"]["baselines"] == list(BASELINES)
    assert status["cohort"]["queries"] == 2
    assert status["seconds"] > 0


def test_smoke_omitted_modes_make_no_calls_and_bm25_needs_no_model(
    tmp_path, tokenizer, monkeypatch
):
    config = fixture(tmp_path, tokenizer, ("bm25",))
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("SMOKE_UNRELATED_SECRET", "do-not-record")

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled stages must not execute")

    for name in (
        "embed",
        "prepare_translations",
        "prepare_hyde",
        "run_adore",
        "rerank",
        "model_spec",
    ):
        monkeypatch.setattr(smoke, name, forbidden)
    destination = smoke.run_smoke(config, "eurlex")
    status = read_json(destination / "status.json")
    assert status["status"] == "success"
    assert [row["baseline"] for row in status["baselines"]] == ["bm25"]
    assert status["model"] is None
    hardware = status["hardware_and_runtime"]
    assert hardware["thread_environment"]["OMP_NUM_THREADS"] == "2"
    assert hardware["configured_reranker_device"] is None
    assert hardware["python_version"]
    assert "do-not-record" not in (destination / "status.json").read_text()
    assert read_json(Path(status["baselines"][0]["report"]) / "report.json")["baseline"] == "bm25"
    first_run = status["baselines"][0]["run"]
    again = read_json(smoke.run_smoke(config, "eurlex") / "status.json")
    assert again["baselines"][0]["run"] == first_run
    retrieval = next(row for row in again["stages"] if row["id"] == "run:bm25:original")
    assert all(retrieval["artifacts_existed_before"].values())


def test_smoke_real_offline_dense_hybrid_and_bm25_share_embeddings(
    tmp_path, tokenizer, monkeypatch
):
    config = fixture(tmp_path, tokenizer, ("dense", "bm25", "hybrid"))
    monkeypatch.setattr(
        smoke,
        "embed",
        lambda *args, **kwargs: embed(*args, **kwargs, encoder_factory=FixtureEncoder),
    )
    destination = smoke.run_smoke(config, "eurlex", "bge-m3")
    status = read_json(destination / "status.json")
    assert status["status"] == "success"
    assert all(row["status"] == "success" for row in status["baselines"])
    assert len([row for row in status["stages"] if row["id"].startswith("embed:")]) == 2
    assert (
        len(
            {
                read_json(Path(row["report"]) / "report.json")["run"]["corpus_fingerprint"]
                for row in status["baselines"]
            }
        )
        == 1
    )


def test_failed_generation_is_retained_without_blocking_controls(tmp_path, tokenizer, monkeypatch):
    config = fixture(tmp_path, tokenizer, BASELINES)
    calls = fake_stages(monkeypatch, tmp_path)

    def failed(*args, **kwargs):
        raise ValidationError("Provider rejected Authorization=sk-secret123 bearer hf_secret456")

    monkeypatch.setattr(smoke, "prepare_hyde", failed)
    destination = smoke.run_smoke(config, "eurlex", "bge-m3")
    status = read_json(destination / "status.json")
    assert status["status"] == "partial"
    assert all(
        row["status"] == "success" for row in status["baselines"] if "hyde" not in row["baseline"]
    )
    assert all(
        row["status"] == "blocked" for row in status["baselines"] if "hyde" in row["baseline"]
    )
    failures = [row for row in status["stages"] if row["status"] == "failed"]
    assert len(failures) == 2
    assert all(row["error_type"] == "ValidationError" and row["seconds"] >= 0 for row in failures)
    assert "secret123" not in (destination / "status.json").read_text()
    assert "secret456" not in (destination / "status.json").read_text()
    assert not [
        kwargs
        for name, _, kwargs in calls
        if name in {"rerank", "retrieve"} and "hyde" in kwargs.get("baseline", "")
    ]


def test_rerank_only_executes_required_parent_and_no_unselected_controls(
    tmp_path, tokenizer, monkeypatch
):
    config = fixture(tmp_path, tokenizer, ("rerank-translate-bm25",))
    calls = fake_stages(monkeypatch, tmp_path)
    status = read_json(smoke.run_smoke(config, "eurlex") / "status.json")
    assert status["status"] == "success"
    assert status["required_parent_baselines"] == ["translate-bm25"]
    assert {row["baseline"] for row in status["baselines"]} == {
        "rerank-translate-bm25",
        "translate-bm25",
    }
    assert all(
        row["role"] == "required_rerank_parent" and row["comparison_status"] == "not_requested"
        for row in status["baselines"]
        if row["baseline"] == "translate-bm25"
    )
    assert not [
        name
        for name, _, _ in calls
        if name in {"embed", "run_adore", "prepare_hyde", "compare_translations"}
    ]
    assert len([name for name, _, _ in calls if name == "retrieve"]) == 2


@pytest.mark.parametrize("limits", [{"max_queries": 1}, {"max_sources": 2}, {"max_chunks": 2}])
def test_size_bounds_prevent_inference_and_journal_all_modes(
    tmp_path, tokenizer, monkeypatch, limits
):
    config = fixture(tmp_path, tokenizer, BASELINES)
    calls = fake_stages(monkeypatch, tmp_path)
    status = read_json(smoke.run_smoke(config, "eurlex", "bge-m3", **limits) / "status.json")
    assert status["status"] == "failed"
    assert {row["baseline"] for row in status["baselines"]} == set(BASELINES)
    assert all(row["status"] == "blocked" for row in status["baselines"])
    assert not calls


def test_remote_full_corpus_refused_before_preparation(tmp_path, monkeypatch):
    config, _ = fixture_config(tmp_path)
    source = SourceSpec("remote/dataset", "a" * 40)
    config = replace(config, datasets={"eurlex": DatasetSpec(source, source)}, baselines=("bm25",))

    def forbidden(*args, **kwargs):
        pytest.fail("A full remote dataset must not be fetched by smoke execution")

    monkeypatch.setattr(smoke, "prepare_corpus", forbidden)
    status = read_json(smoke.run_smoke(config, "eurlex") / "status.json")
    assert status["status"] == "failed"
    assert status["stages"][0]["status"] == "failed"
    assert "local QA and corpus subsets" in status["stages"][0]["error"]


def test_interrupt_still_persists_invocation_status(tmp_path, tokenizer, monkeypatch):
    config = fixture(tmp_path, tokenizer, ("bm25",))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(smoke, "prepare_bm25", interrupted)
    with pytest.raises(KeyboardInterrupt):
        smoke.run_smoke(config, "eurlex")
    paths = list((config.output_dir / "smoke_invocations").glob("*/status.json"))
    assert len(paths) == 1
    status = read_json(paths[0])
    assert status["status"] == "interrupted"
    assert status["finished_at"]
