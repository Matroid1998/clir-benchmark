from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from clir_benchmark.cluster import (
    JobResources,
    collect_status,
    main,
    retry_jobs,
    run_job,
    submit_plan,
    write_plan,
)
from clir_benchmark.config import load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.util import read_json, write_json


def plan(tmp_path, *, baselines=("dense", "bm25", "hybrid"), split=False, **resources):
    config = replace(
        load_config("configs/benchmark.yaml"), baselines=baselines, candidate_mode="all"
    )
    remote = tmp_path / "remote plan"
    path = write_plan(
        config,
        tmp_path / "plan",
        remote_project="/remote/project with spaces",
        remote_directory=str(remote),
        remote_cpu_config="/remote/cpu.yaml",
        remote_model_config="/remote/models.yaml",
        python="/remote/venv/bin/python",
        environment="/remote/env.sh",
        account="def-example_cpu",
        gpu_account="def-example_gpu",
        split_embedding=split,
        **resources,
    )
    return path, config


def test_plan_uses_independent_model_corpus_jobs_and_shared_preparation(tmp_path):
    path, config = plan(tmp_path)
    data = read_json(path)
    assert len(data["jobs"]) == 30
    assert data["candidate_mode"] == "all"
    assert data["selected_baselines"] == ["dense", "bm25", "hybrid"]
    assert len([job for job in data["jobs"] if job["kind"] == "model"]) == 24
    for job in data["jobs"]:
        if job["kind"] != "prepare":
            assert job["dependencies"] == [f"clir-prepare-{job['corpus']}"]
        script = (path.parent / "scripts" / Path(job["script"]).name).read_text()
        assert "#SBATCH --array" not in script
        assert "cd '/remote/project with spaces'" in script
        assert f"export HF_HUB_OFFLINE={0 if job['kind'] == 'prepare' else 1}" in script
        assert "gpt" not in script
        assert ("--account=def-example_gpu" in script) == (job["kind"] == "model")
        assert ("--gpus=h100:1" in script) == (job["kind"] == "model")
        assert job["resources"]["cpus"] == 1
        assert "#SBATCH --cpus-per-task=1" in script
    assert set(data["models"]) == set(config.models)


def test_split_releases_gpus_before_scoring_and_preserves_model_config(tmp_path):
    path, _ = plan(tmp_path, split=True)
    jobs = read_json(path)["jobs"]
    assert len(jobs) == 54
    embeds = {job["label"]: job for job in jobs if job["kind"] == "embed"}
    assert len(embeds) == 24
    for job in (job for job in jobs if job["kind"] == "score"):
        parent = embeds[job["dependencies"][0]]
        assert job["configuration"] == parent["configuration"] == "/remote/models.yaml"
        assert job["resources"]["gpus"] is None
    assert all(job["resources"]["cpus"] == 1 for job in jobs)


def test_stage_resources_are_independent_and_do_not_change_experiment_identity(tmp_path):
    default_path, _ = plan(tmp_path / "default", split=True)
    default = read_json(default_path)
    overrides = {
        "prepare_resources": JobResources(cpus=2),
        "bm25_resources": JobResources(cpus=3, memory="16G"),
        "model_resources": JobResources(cpus=4, gpus="h100:1"),
        "score_resources": JobResources(cpus=5, memory="32G"),
    }
    path, _ = plan(tmp_path / "override", split=True, **overrides)
    data = read_json(path)
    assert data["configuration"] == default["configuration"]
    assert data["selected_baselines"] == default["selected_baselines"]
    expected_cpus = {"prepare": 2, "bm25": 3, "embed": 4, "score": 5}
    for actual, original in zip(data["jobs"], default["jobs"], strict=True):
        for field in ("label", "kind", "corpus", "model", "dependencies", "configuration"):
            assert actual[field] == original[field]
        assert actual["resources"]["cpus"] == expected_cpus[actual["kind"]]
        script = (path.parent / "scripts" / Path(actual["script"]).name).read_text()
        assert f"#SBATCH --cpus-per-task={expected_cpus[actual['kind']]}" in script
        assert 'export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"' in script
    bm25_jobs = [job for job in data["jobs"] if job["kind"] == "bm25"]
    assert all(job["resources"]["memory"] == "16G" for job in bm25_jobs)


def test_score_resource_override_does_not_reserve_cpus_for_bm25(tmp_path):
    path, _ = plan(tmp_path, split=True, score_resources=JobResources(cpus=4))
    jobs = read_json(path)["jobs"]
    assert all(job["resources"]["cpus"] == 1 for job in jobs if job["kind"] == "bm25")
    assert all(job["resources"]["cpus"] == 4 for job in jobs if job["kind"] == "score")


@pytest.mark.parametrize("split", [False, True])
def test_model_resources_override_only_selected_alias_without_changing_modes(tmp_path, split):
    default_path, _ = plan(tmp_path / "default", split=split)
    default = read_json(default_path)
    alias = default["models"][0]
    gpu = "nvidia_h100_80gb_hbm3_1g.10gb:1"
    path, _ = plan(
        tmp_path / "custom",
        split=split,
        model_resources_by_alias={alias: JobResources(gpus=gpu, memory="16G")},
    )
    data = read_json(path)
    assert data["configuration"] == default["configuration"]
    assert data["selected_baselines"] == default["selected_baselines"]
    for actual, original in zip(data["jobs"], default["jobs"], strict=True):
        assert actual["label"] == original["label"]
        assert actual["dependencies"] == original["dependencies"]
        assert actual["configuration"] == original["configuration"]
        if actual["kind"] in {"model", "embed"} and actual["model"] == alias:
            assert actual["resources"]["gpus"] == gpu
            assert actual["resources"]["memory"] == "16G"
            script = (path.parent / "scripts" / Path(actual["script"]).name).read_text()
            assert "#SBATCH --gpus=" + gpu in script
            assert "#SBATCH --account=def-example_gpu" in script
        else:
            assert actual["resources"] == original["resources"]


def test_model_resources_do_not_enable_gpu_jobs_when_dense_modes_disabled(tmp_path):
    path, config = plan(
        tmp_path,
        baselines=("bm25",),
        model_resources_by_alias={"LaBSE": JobResources(gpus="h100:1")},
    )
    assert "LaBSE" in config.models
    jobs = read_json(path)["jobs"]
    assert len(jobs) == 6
    assert all(job["kind"] in {"prepare", "bm25"} for job in jobs)
    assert all(job["resources"]["gpus"] is None for job in jobs)


def test_unknown_model_resource_alias_fails_before_plan_writes(tmp_path):
    with pytest.raises(ValidationError, match="Unknown resource model aliases: typo"):
        plan(tmp_path, model_resources_by_alias={"typo": JobResources(gpus="h100:1")})
    assert not (tmp_path / "plan").exists()


def document_plan(
    tmp_path, *, baselines=("dense", "translate-dense", "hyde-dense"), dependencies=None
):
    config = replace(
        load_config("configs/selected-document.yaml"),
        output_dir=tmp_path / "artifacts",
        models=("bge-m3", "LaBSE"),
        target_languages=("en", "fr"),
        baselines=baselines,
    )
    config = replace(
        config,
        datasets={"eurlex": config.datasets["eurlex"]},
        hyde=replace(config.hyde, output_languages=("en", "fr")),
    )
    write_json(config.corpus_dir("eurlex") / "manifest.json", {"languages": {"en": 2, "fr": 2}})
    path = write_plan(
        config,
        tmp_path / "plan",
        remote_project="/remote/project",
        remote_directory=str(tmp_path / "remote"),
        remote_cpu_config="/remote/cpu.yaml",
        remote_model_config="/remote/models.yaml",
        python="/remote/python",
        environment="/remote/env.sh",
        account="def-example_cpu",
        gpu_account="def-example_gpu",
        document_embedding_dependencies=dependencies
        if dependencies is not None
        else {("eurlex", "bge-m3"): "499", ("eurlex", "LaBSE"): "500"},
    )
    return path, config


def test_document_plan_shares_generation_and_reuses_existing_document_jobs(tmp_path):
    path, _ = document_plan(tmp_path)
    data = read_json(path)
    assert data["retrieval_unit"] == "document"
    assert len(data["jobs"]) == 9
    by_kind = {}
    for job in data["jobs"]:
        by_kind.setdefault(job["kind"], []).append(job)
    assert set(by_kind) == {"prepare-selected", "translate", "hyde", "query-embed", "variant-score"}
    assert len(by_kind["translate"]) == len(by_kind["hyde"]) == 2
    generation = {job["label"] for kind in ("translate", "hyde") for job in by_kind[kind]}
    for job in by_kind["query-embed"]:
        assert set(job["dependencies"]) == generation | {"clir-prepare-selected-eurlex"}
        assert job["external_dependencies"] == []
    for job in by_kind["variant-score"]:
        assert job["dependencies"] == [f"clir-query-embed-eurlex-{job['model']}"]
        assert len(job["external_dependencies"]) == 1
        assert job["resources"]["gpus"] is None
        assert job["configuration"] == "/remote/models.yaml"
    assert all(job["model"] is None for kind in ("translate", "hyde") for job in by_kind[kind])
    for job in data["jobs"]:
        script = (path.parent / "scripts" / Path(job["script"]).name).read_text()
        offline = 0 if job["kind"] == "prepare-selected" else 1
        assert f"export HF_HUB_OFFLINE={offline}" in script


@pytest.mark.parametrize(
    "baselines,disabled",
    [
        (("dense",), {"translate", "hyde"}),
        (("hyde-dense",), {"translate"}),
        (("translate-dense",), {"hyde"}),
    ],
)
def test_document_planning_does_not_require_disabled_generation_configuration(
    tmp_path, monkeypatch, baselines, disabled
):
    monkeypatch.setattr(
        "clir_benchmark.generation.ChatCompletionsClient.__init__",
        lambda *a, **k: pytest.fail("Planning loaded an API client"),
    )
    path, _ = document_plan(tmp_path, baselines=baselines)
    jobs = read_json(path)["jobs"]
    assert not ({job["kind"] for job in jobs} & disabled)
    assert not any(job["kind"] in {"prepare", "embed", "bm25", "model", "score"} for job in jobs)


def test_document_plan_requires_explicit_cache_or_dependency_before_writing(tmp_path):
    with pytest.raises(ValidationError, match="require document_embedding_dependencies"):
        document_plan(tmp_path, dependencies={})
    assert not (tmp_path / "plan").exists()


def test_document_plan_verifies_completed_cache_without_loading_vectors(tmp_path, monkeypatch):
    from clir_benchmark import cluster

    calls = []
    monkeypatch.setattr(
        cluster,
        "_document_embedding_readiness",
        lambda config, corpus, alias: (
            calls.append((corpus, alias)) or {"path": "cached", "manifest_sha256": "hash"}
        ),
    )
    path, _ = document_plan(
        tmp_path, dependencies={("eurlex", "bge-m3"): None, ("eurlex", "LaBSE"): "500"}
    )
    assert calls == [("eurlex", "bge-m3")]
    jobs = read_json(path)["jobs"]
    assert (
        next(j for j in jobs if j["kind"] == "variant-score" and j["model"] == "bge-m3")[
            "external_dependencies"
        ]
        == []
    )


def test_document_cache_readiness_checks_files_without_reading_vector_contents(
    tmp_path, monkeypatch
):
    from clir_benchmark.cluster import _document_embedding_readiness

    path, config = document_plan(tmp_path)
    cached = path.parent / "cached"
    write_json(cached / "manifest.json", {"rows": 4, "dimension": 3})
    (cached / "vectors.npy").write_text("not opened by plan validation")
    (cached / "rows.parquet").write_text("not opened by plan validation")
    monkeypatch.setattr("clir_benchmark.pipeline.embedding_path", lambda *a, **k: cached)
    monkeypatch.setattr(
        "clir_benchmark.embedding.load_embeddings",
        lambda *a, **k: pytest.fail("Read vector matrix"),
    )
    assert _document_embedding_readiness(config, "eurlex", "bge-m3")["rows"] == 4
    (cached / "vectors.npy").unlink()
    with pytest.raises(ValidationError, match="Document embeddings are not ready"):
        _document_embedding_readiness(config, "eurlex", "bge-m3")


@pytest.mark.parametrize(
    "kind,baseline,method",
    [
        ("translate", "translate-dense", "clir_benchmark.translation.prepare_translations"),
        ("hyde", "hyde-dense", "clir_benchmark.hyde.prepare_hyde"),
    ],
)
def test_explicit_generation_stage_runs_only_requested_method_and_language(
    tmp_path, monkeypatch, kind, baseline, method
):
    _, config = document_plan(tmp_path, baselines=(baseline,))
    calls = []
    monkeypatch.setattr(
        "clir_benchmark.pipeline.embed", lambda *a, **k: pytest.fail("Generation encoded queries")
    )
    monkeypatch.setattr(
        "clir_benchmark.pipeline.retrieve",
        lambda *a, **k: pytest.fail("Generation retrieved documents"),
    )
    monkeypatch.setattr(
        method, lambda cfg, corpus, lang, **options: calls.append((corpus, lang, options))
    )
    journal = run_job(
        config,
        "eurlex",
        kind,
        language="fr",
        retry_failed=True,
        journal_directory=tmp_path / "journals",
    )
    assert calls == [("eurlex", "fr", {"retry_failed": True})]
    assert read_json(journal)["retry_failed"] is True


def test_document_query_embeddings_never_build_documents_or_disabled_modes(tmp_path, monkeypatch):
    _, config = document_plan(tmp_path, baselines=("hyde-dense",))
    calls = []
    monkeypatch.setattr("clir_benchmark.pipeline.embed", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(
        "clir_benchmark.hyde.prepare_hyde",
        lambda *a, **k: pytest.fail("Embedding generated passages"),
    )
    monkeypatch.setattr(
        "clir_benchmark.translation.prepare_translations",
        lambda *a, **k: pytest.fail("Embedding translated queries"),
    )
    run_job(
        config, "eurlex", "query-embed", alias="bge-m3", journal_directory=tmp_path / "journals"
    )
    assert calls == [
        {"kind": "query", "hyde_language": "en"},
        {"kind": "query", "hyde_language": "fr"},
    ]
    with pytest.raises(ValidationError, match="disabled"):
        run_job(
            config, "eurlex", "translate", language="en", journal_directory=tmp_path / "disabled"
        )
    assert not (tmp_path / "disabled").exists()


def test_document_preparation_never_rebuilds_corpus_or_index(tmp_path, monkeypatch):
    _, config = document_plan(tmp_path)
    relevance = tmp_path / "relevance"
    write_json(relevance / "manifest.json", {"blocked_settings": []})
    monkeypatch.setattr(
        "clir_benchmark.pipeline.prepare_corpus", lambda *a, **k: pytest.fail("Rebuilt corpus")
    )
    monkeypatch.setattr(
        "clir_benchmark.lexical.prepare_bm25", lambda *a, **k: pytest.fail("Built BM25")
    )
    monkeypatch.setattr("clir_benchmark.pipeline.prepare_relevance", lambda *a, **k: relevance)
    journal = run_job(config, "eurlex", "prepare-selected", journal_directory=tmp_path / "journals")
    assert [row["stage"] for row in read_json(journal)["stages"]] == ["prepare-relevance"]


def test_document_scoring_compares_matching_enabled_controls_without_generation(
    tmp_path, monkeypatch
):
    _, config = document_plan(tmp_path)
    calls = []
    monkeypatch.setattr(
        "clir_benchmark.cluster._document_embedding_readiness", lambda *a: {"path": "cache"}
    )
    monkeypatch.setattr(
        "clir_benchmark.pipeline.embed", lambda *a, **k: pytest.fail("Scoring encoded queries")
    )
    monkeypatch.setattr(
        "clir_benchmark.hyde.prepare_hyde",
        lambda *a, **k: pytest.fail("Scoring generated passages"),
    )
    monkeypatch.setattr(
        "clir_benchmark.translation.prepare_translations",
        lambda *a, **k: pytest.fail("Scoring translated queries"),
    )
    for name in ("retrieve", "evaluate"):
        monkeypatch.setattr(
            f"clir_benchmark.pipeline.{name}", lambda *a, op=name, **k: calls.append((op, k))
        )
    for name in ("compare_translations", "compare_hyde"):
        monkeypatch.setattr(
            f"clir_benchmark.comparison.{name}", lambda *a, op=name, **k: calls.append((op, k))
        )
    run_job(
        config, "eurlex", "variant-score", alias="bge-m3", journal_directory=tmp_path / "journals"
    )
    retrieval = [value for name, value in calls if name == "retrieve"]
    assert retrieval == [
        {"baseline": "dense"},
        {"baseline": "translate-dense", "translation_language": "en"},
        {"baseline": "translate-dense", "translation_language": "fr"},
        {"baseline": "hyde-dense", "hyde_language": "en"},
        {"baseline": "hyde-dense", "hyde_language": "fr"},
    ]
    assert [value for name, value in calls if name == "compare_hyde"] == [
        {"languages": ("en",), "controls": ("dense", "translate-dense")},
        {"languages": ("fr",), "controls": ("dense", "translate-dense")},
    ]


def test_document_submit_verifies_external_jobs_and_never_resubmits_them(tmp_path):
    path, _ = document_plan(tmp_path)
    counter = iter(range(1000, 1100))
    submitted_commands = []

    def runner(command, **kwargs):
        if command[0] == "sacct":
            return SimpleNamespace(
                returncode=0, stdout="499|RUNNING|0:0\n500|COMPLETED|0:0\n", stderr=""
            )
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="499|RUNNING\n", stderr="")
        submitted_commands.append(command)
        return SimpleNamespace(returncode=0, stdout=str(next(counter)), stderr="")

    result = read_json(submit_plan(path, runner=runner))
    assert len(submitted_commands) == 9
    score_jobs = [j for j in result["jobs"] if j["kind"] == "variant-score"]
    for job in score_jobs:
        dependency = next(arg for arg in job["command"] if arg.startswith("--dependency="))
        assert dependency.endswith(":499") == (job["model"] == "bge-m3")
        assert "500" not in dependency
    assert result["external_dependency_verification"]["500"]["action"] == "already_completed"


def test_failed_external_document_job_blocks_submission_before_any_sbatch(tmp_path):
    path, _ = document_plan(tmp_path)

    def runner(command, **kwargs):
        assert command[0] in {"sacct", "squeue"}
        return SimpleNamespace(
            returncode=0,
            stdout="499|FAILED|1:0\n500|COMPLETED|0:0\n" if command[0] == "sacct" else "",
            stderr="",
        )

    with pytest.raises(ValidationError, match="External document job 499 is FAILED"):
        submit_plan(path, runner=runner)
    assert not (path.parent / "submission.json").exists()


@pytest.mark.parametrize("baselines,count", [(("bm25",), 6), (("dense",), 27)])
def test_disabled_modes_have_no_jobs(tmp_path, baselines, count):
    path, _ = plan(tmp_path, baselines=baselines)
    jobs = read_json(path)["jobs"]
    assert len(jobs) == count
    if baselines == ("bm25",):
        assert all(job["resources"]["gpus"] is None for job in jobs)
    else:
        assert all(job["kind"] != "bm25" for job in jobs)


def test_job_stage_dispatch_never_runs_disabled_or_unrequested_stages(tmp_path, monkeypatch):
    config = replace(load_config("configs/benchmark.yaml"), baselines=("dense",))
    calls = []
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setattr("clir_benchmark.pipeline.embed", lambda *a, **k: calls.append("embed"))
    monkeypatch.setattr(
        "clir_benchmark.pipeline.retrieve",
        lambda *a, **k: calls.append("retrieve-" + k["baseline"]),
    )
    monkeypatch.setattr(
        "clir_benchmark.pipeline.evaluate",
        lambda *a, **k: calls.append("evaluate-" + k["baseline"]),
    )
    journal = run_job(config, "eurlex", "embed", alias=config.models[0], journal_directory=tmp_path)
    assert calls == ["embed"]
    assert read_json(journal)["status"] == "success"
    assert read_json(journal)["slurm"]["SLURM_CPUS_PER_TASK"] == "1"
    assert read_json(journal)["thread_environment"]["OPENBLAS_NUM_THREADS"] == "1"
    assert read_json(journal)["thread_environment"]["TOKENIZERS_PARALLELISM"] == "false"
    calls.clear()
    run_job(config, "eurlex", "score", alias=config.models[0], journal_directory=tmp_path)
    assert calls == ["retrieve-dense", "evaluate-dense"]
    with pytest.raises(ValidationError, match="disabled"):
        run_job(config, "eurlex", "bm25", journal_directory=tmp_path)


def test_preparation_blocked_settings_need_explicit_choice(tmp_path, monkeypatch):
    config = replace(load_config("configs/benchmark.yaml"), baselines=("bm25",))
    relevance = tmp_path / "relevance"
    write_json(relevance / "manifest.json", {"blocked_settings": ["missing-labels"]})
    calls = []
    monkeypatch.setattr("clir_benchmark.pipeline.prepare_corpus", lambda *a: calls.append("corpus"))
    monkeypatch.setattr("clir_benchmark.pipeline.prepare_relevance", lambda *a: relevance)
    monkeypatch.setattr("clir_benchmark.lexical.prepare_bm25", lambda *a: calls.append("index"))
    with pytest.raises(ValidationError, match="settings are blocked"):
        run_job(config, "eurlex", "prepare", journal_directory=tmp_path / "strict")
    assert calls == ["corpus"]
    journal = read_json(next((tmp_path / "strict").glob("*.json")))
    assert journal["status"] == "failed"
    assert journal["relevance"]["blocked_settings"] == ["missing-labels"]
    run_job(
        config,
        "eurlex",
        "prepare",
        journal_directory=tmp_path / "allow",
        allow_blocked_settings=True,
    )
    assert calls == ["corpus", "corpus", "index"]


def test_job_failure_retains_status_and_redacts_token(tmp_path, monkeypatch):
    config = replace(load_config("configs/benchmark.yaml"), baselines=("bm25",))

    def fail(*args, **kwargs):
        raise RuntimeError("Transport error with hf_secret_value")

    monkeypatch.setattr("clir_benchmark.pipeline.retrieve", fail)
    with pytest.raises(RuntimeError):
        run_job(config, "eurlex", "bm25", journal_directory=tmp_path)
    path = next(tmp_path.glob("*.json"))
    data = read_json(path)
    assert data["status"] == "failed"
    assert len(data["stages"]) == 1
    assert data["stages"][0]["seconds"] >= 0
    assert "hf_secret_value" not in path.read_text()


def test_submit_all_jobs_records_dependencies_and_never_resubmits(tmp_path):
    path, _ = plan(tmp_path, split=True)
    commands = []

    def submit(command, **kwargs):
        commands.append(command)
        number = str(1000 + len(commands))
        return SimpleNamespace(returncode=0, stdout=number + ";fir\n", stderr="")

    submitted = submit_plan(path, workers=8, runner=submit)
    data = read_json(submitted)
    assert data["status"] == "submitted"
    assert len(commands) == 54
    by_label = {job["label"]: job for job in data["jobs"]}
    for job in data["jobs"]:
        if job["dependencies"]:
            parent_id = by_label[job["dependencies"][0]]["job_id"]
            assert "--dependency=afterok:" + parent_id in job["command"]
    assert (
        submit_plan(path, runner=lambda *a, **k: pytest.fail("duplicate submission")) == submitted
    )


def test_submission_failure_blocks_only_dependent_jobs(tmp_path):
    path, _ = plan(tmp_path, baselines=("bm25",))

    def submit(command, **kwargs):
        failed = command[-1].endswith("clir-prepare-un.sh")
        return SimpleNamespace(
            returncode=1 if failed else 0,
            stdout="" if failed else "100\n",
            stderr="allocation problem" if failed else "",
        )

    journal = submit_plan(path, runner=submit)
    jobs = {job["label"]: job for job in read_json(journal)["jobs"]}
    assert jobs["clir-bm25-un"]["status"] == "blocked"
    assert jobs["clir-bm25-eurlex"]["status"] == "submitted"
    with pytest.raises(ValidationError, match="previous submission is incomplete"):
        submit_plan(path, runner=submit)


def test_status_retains_queue_accounting_and_job_journals_without_mutations(tmp_path):
    path, _ = plan(tmp_path, baselines=("bm25",))
    counter = iter(range(100, 200))
    submitted = submit_plan(
        path,
        runner=lambda *a, **k: SimpleNamespace(returncode=0, stdout=str(next(counter)), stderr=""),
    )
    labels = read_json(submitted)["jobs"]
    first = labels[0]
    job_status = (
        Path(read_json(path)["remote_directory"]) / "job_status" / first["label"] / "attempt.json"
    )
    write_json(job_status, {"job_id": first["job_id"], "status": "success", "seconds": 10.0})

    def query(command, **kwargs):
        if command[0] == "sacct":
            fields = [
                first["job_id"],
                "COMPLETED",
                "0:0",
                "11",
                "2026-11-09",
                "2026-11-09",
                "12",
                "96G",
                "node1",
                "None",
                "cpu=12",
                "132",
                "00:10",
                "",
            ]
            return SimpleNamespace(returncode=0, stdout="|".join(fields) + "\n", stderr="")
        assert command[0] == "squeue"
        return SimpleNamespace(
            returncode=0,
            stdout=labels[1]["job_id"] + "|PENDING|DependencyNeverSatisfied|0:00\n",
            stderr="",
        )

    result = read_json(collect_status(submitted, runner=query))
    assert result["all_terminal"] is False
    assert result["all_succeeded"] is False
    assert result["jobs"][0]["state"] == "COMPLETED"
    assert result["jobs"][0]["stage_journals"][0]["seconds"] == 10.0
    assert result["jobs"][1]["blocked_dependency"] is True
    assert result["jobs"][2]["state"] == "UNKNOWN"


def retry_fixture(tmp_path):
    path, _ = plan(
        tmp_path,
        split=True,
        model_resources=JobResources(cpus=12, gpus="h100:1"),
        score_resources=JobResources(cpus=12),
    )
    counter = iter(range(1000, 2000))
    submitted = submit_plan(
        path,
        runner=lambda *a, **k: SimpleNamespace(returncode=0, stdout=str(next(counter)), stderr=""),
    )
    original = read_json(submitted)
    labels = ["clir-score-chemistry-SapBERT", "clir-embed-chemistry-SapBERT"]
    selected = {job["job_id"]: job for job in original["jobs"] if job["label"] in labels}
    return submitted, original, labels, selected


def retry_runner(
    selected, commands, *, active=False, state="FAILED", fail_submit=False, unknown=False
):
    counter = iter(range(9000, 9100))

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "sacct":
            records = []
            for job_id in command[command.index("--jobs") + 1].split(","):
                job = selected.get(job_id)
                records.append(
                    "|".join(
                        [
                            job_id,
                            "COMPLETED"
                            if job is None
                            else state
                            if job["kind"] == "embed"
                            else "CANCELLED",
                            "1:0",
                            "36",
                            "start",
                            "end",
                            "12",
                            "96G",
                            "node",
                            "None",
                            "cpu=12,gres/gpu=1",
                            "432",
                            "00:20",
                            "1G",
                        ]
                    )
                )
            return SimpleNamespace(returncode=0, stdout="\n".join(records), stderr="")
        if command[0] == "squeue":
            return SimpleNamespace(
                returncode=0,
                stdout=next(iter(selected)) + "|RUNNING\n" if active else "",
                stderr="",
            )
        assert command[0] == "sbatch"
        if unknown:
            raise OSError("connection interrupted after request")
        return SimpleNamespace(
            returncode=1 if fail_submit else 0,
            stdout="" if fail_submit else str(next(counter)),
            stderr="scheduler rejection" if fail_submit else "",
        )

    return run


def test_retry_preserves_attempt_accounting_and_updates_only_selected_ids(tmp_path):
    submitted, original, labels, selected = retry_fixture(tmp_path)
    write_json(
        submitted.parent / "status.json",
        {"jobs": [{"job_id": key, "state": "FAILED"} for key in selected]},
    )
    commands = []
    history = retry_jobs(
        submitted,
        labels,
        walltime="03:00:00",
        cpus=1,
        reason="tokenizer dependencies repaired",
        runner=retry_runner(selected, commands),
    )
    retry = read_json(history)
    updated = read_json(submitted)
    assert retry["status"] == updated["status"] == "submitted"
    assert retry["submission_order"] == list(reversed(labels))
    assert retry["old_status_snapshot"]["jobs"]
    assert retry["cpus_override"] == 1
    current = {job["label"]: job for job in updated["jobs"]}
    for old in original["jobs"]:
        new = current[old["label"]]
        if old["label"] not in labels:
            assert new == old
        else:
            assert new["job_id"] != old["job_id"]
            assert new["previous_attempts"][0]["job_id"] == old["job_id"]
            assert new["previous_attempts"][0]["accounting_at_retry"]["ElapsedRaw"] == "36"
            assert (
                new["previous_attempts"][0]["accounting_at_retry"]["AllocTRES"]
                == "cpu=12,gres/gpu=1"
            )
            assert "--time=03:00:00" in new["command"]
            assert "--cpus-per-task=1" in new["command"]
            assert new["cpus_override"] == 1
            assert new["resources"] == old["resources"]
            assert new["resources"]["cpus"] == 12
            script = submitted.parent / "scripts" / Path(new["script"]).name
            assert "#SBATCH --cpus-per-task=12" in script.read_text()
    assert "--dependency=afterok:" + current[labels[1]]["job_id"] in current[labels[0]]["command"]
    assert not any(value.startswith("--dependency") for value in current[labels[1]]["command"])
    assert (
        retry["dependency_verification"]["clir-prepare-chemistry"]["action"] == "already_completed"
    )
    assert updated["plan_sha256"] == original["plan_sha256"]


def test_retry_retains_active_external_dependencies_without_submitting_their_jobs(tmp_path):
    submitted, _, labels, selected = retry_fixture(tmp_path)
    data = read_json(submitted)
    next(job for job in data["jobs"] if job["label"] == labels[0])["external_dependencies"] = [
        "777"
    ]
    write_json(submitted, data)
    commands = []
    base_runner = retry_runner(selected, commands)

    def runner(command, **kwargs):
        if command[0] == "sacct" and command[-1] == "JobIDRaw,State,ExitCode":
            return SimpleNamespace(returncode=0, stdout="777|RUNNING|0:0\n", stderr="")
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="777|RUNNING\n", stderr="")
        return base_runner(command, **kwargs)

    retry = read_json(retry_jobs(submitted, labels, runner=runner))
    assert retry["external_dependency_verification"]["777"]["action"] == "afterok"
    assert len(retry["jobs"]) == 2
    scored = next(job for job in retry["jobs"] if job["label"] == labels[0])
    assert next(arg for arg in scored["command"] if arg.startswith("--dependency=")).endswith(
        ":777"
    )


def test_retry_uses_original_script_cpu_request_unless_explicitly_overridden(tmp_path):
    submitted, _, labels, selected = retry_fixture(tmp_path)
    history = retry_jobs(submitted, labels, runner=retry_runner(selected, []))
    retry = read_json(history)
    assert retry["cpus_override"] is None
    assert retry["gpus_override"] is None
    assert retry["memory_override"] is None
    assert all(
        not any(
            value.startswith(("--cpus-per-task", "--gpus", "--mem")) for value in job["command"]
        )
        for job in retry["jobs"]
    )


def test_retry_gpu_override_preserves_original_requests_and_cpu_children(tmp_path):
    submitted, original, labels, selected = retry_fixture(tmp_path)
    gpu = "nvidia_h100_80gb_hbm3_1g.10gb:1"
    history = retry_jobs(
        submitted, labels, gpus=gpu, memory="16G", runner=retry_runner(selected, [])
    )
    retry = read_json(history)
    assert retry["gpus_override"] == gpu
    assert retry["memory_override"] == "16G"
    updated = {job["label"]: job for job in read_json(submitted)["jobs"]}
    for old in original["jobs"]:
        new = updated[old["label"]]
        if old["label"] not in labels:
            assert new == old
            continue
        assert new["resources"] == old["resources"]
        assert new["previous_attempts"][0]["resources"] == old["resources"]
        assert new["resources"]["memory"] == "96G"
        assert new["memory_override"] == "16G"
        assert "--mem=16G" in new["command"]
        script = submitted.parent / "scripts" / Path(new["script"]).name
        assert "#SBATCH --mem=96G" in script.read_text()
        if old["kind"] == "embed":
            assert new["gpus_override"] == gpu
            assert "--gpus=" + gpu in new["command"]
            script = submitted.parent / "scripts" / Path(new["script"]).name
            assert "#SBATCH --gpus=h100:1" in script.read_text()
        else:
            assert new["gpus_override"] is None
            assert not any(value.startswith("--gpus") for value in new["command"])


def test_retry_gpu_override_refuses_cpu_only_selection_before_scheduler_access(tmp_path):
    submitted, original, labels, _ = retry_fixture(tmp_path)
    with pytest.raises(ValidationError, match="requires at least one selected GPU job"):
        retry_jobs(
            submitted,
            [labels[0]],
            gpus="h100:1",
            runner=lambda *a, **k: pytest.fail("scheduler must not be queried"),
        )
    assert read_json(submitted) == original


@pytest.mark.parametrize("gpus", ["", "h100:1\n#SBATCH --x", 1])
def test_retry_invalid_gpu_request_fails_before_scheduler_access(tmp_path, gpus):
    with pytest.raises(ValidationError, match="Invalid Slurm gpus"):
        retry_jobs(
            tmp_path / "missing.json",
            ["selected-job"],
            gpus=gpus,
            runner=lambda *a, **k: pytest.fail("scheduler must not be queried"),
        )


@pytest.mark.parametrize("memory", ["", "16G\n#SBATCH --x", 16])
def test_retry_invalid_memory_request_fails_before_scheduler_access(tmp_path, memory):
    with pytest.raises(ValidationError, match="Invalid Slurm memory"):
        retry_jobs(
            tmp_path / "missing.json",
            ["selected-job"],
            memory=memory,
            runner=lambda *a, **k: pytest.fail("scheduler must not be queried"),
        )


@pytest.mark.parametrize("cpus", [0, -1, True, 1.5, "1"])
def test_retry_invalid_cpu_request_fails_before_scheduler_access(tmp_path, cpus):
    with pytest.raises(ValidationError, match="CPU count"):
        retry_jobs(
            tmp_path / "missing.json",
            ["selected-job"],
            cpus=cpus,
            runner=lambda *a, **k: pytest.fail("scheduler must not be queried"),
        )


def test_retry_cli_passes_explicit_cpu_override(tmp_path, monkeypatch):
    path = tmp_path / "retry.json"
    write_json(path, {"status": "submitted"})
    calls = []

    def retry(*args, **kwargs):
        calls.append((args, kwargs))
        return path

    monkeypatch.setattr("clir_benchmark.cluster.retry_jobs", retry)
    assert (
        main(
            [
                "retry",
                "--submission",
                str(path),
                "--label",
                "selected",
                "--cpus",
                "1",
                "--gpus",
                "h100:1",
                "--memory",
                "16G",
            ]
        )
        == 0
    )
    assert calls == [
        (
            (path, ["selected"]),
            {"walltime": None, "cpus": 1, "gpus": "h100:1", "memory": "16G", "reason": None},
        )
    ]


@pytest.mark.parametrize(
    "active,state", [(True, "FAILED"), (False, "RUNNING"), (False, "PENDING"), (False, "UNKNOWN")]
)
def test_retry_refuses_active_or_unknown_jobs_before_mutation(tmp_path, active, state):
    submitted, original, labels, selected = retry_fixture(tmp_path)
    commands = []
    with pytest.raises(ValidationError, match="Refusing duplicate retry"):
        retry_jobs(
            submitted, labels, runner=retry_runner(selected, commands, active=active, state=state)
        )
    assert read_json(submitted) == original
    assert not any(command[0] == "sbatch" for command in commands)


def test_retry_rejection_blocks_child_and_prevents_blind_resubmission(tmp_path):
    submitted, original, labels, selected = retry_fixture(tmp_path)
    history = retry_jobs(submitted, labels, runner=retry_runner(selected, [], fail_submit=True))
    assert read_json(history)["status"] == "incomplete"
    updated = {job["label"]: job for job in read_json(submitted)["jobs"]}
    assert updated[labels[1]]["status"] == "submission_failed"
    assert updated[labels[0]]["status"] == "blocked"
    with pytest.raises(ValidationError, match="incomplete or retrying"):
        retry_jobs(submitted, labels, runner=lambda *a, **k: pytest.fail("duplicate retry"))


def test_retry_transport_interruption_is_explicitly_ambiguous(tmp_path):
    submitted, original, labels, selected = retry_fixture(tmp_path)
    with pytest.raises(OSError):
        retry_jobs(submitted, labels, runner=retry_runner(selected, [], unknown=True))
    updated = {job["label"]: job for job in read_json(submitted)["jobs"]}
    assert updated[labels[1]]["status"] == "retry_unknown"
    history = read_json(next((submitted.parent / "retry_history").glob("*.json")))
    assert history["jobs"][0]["status"] == "unknown"
    with pytest.raises(ValidationError, match="incomplete or retrying"):
        retry_jobs(submitted, labels)


@pytest.mark.parametrize("args", [{"cpus": 0}, {"cpus": True}, {"gpus": "h100:1\n#SBATCH --x"}])
def test_invalid_resource_values_rejected(args):
    with pytest.raises(ValidationError):
        JobResources(**args)
