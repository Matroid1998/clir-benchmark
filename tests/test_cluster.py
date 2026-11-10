from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from clir_benchmark.cluster import (
    JobResources,
    collect_status,
    retry_jobs,
    run_job,
    submit_plan,
    write_plan,
)
from clir_benchmark.config import load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.util import read_json, write_json


def plan(tmp_path, *, baselines=("dense", "bm25", "hybrid"), split=False):
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
    path, _ = plan(tmp_path, split=True)
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
        reason="tokenizer dependencies repaired",
        runner=retry_runner(selected, commands),
    )
    retry = read_json(history)
    updated = read_json(submitted)
    assert retry["status"] == updated["status"] == "submitted"
    assert retry["submission_order"] == list(reversed(labels))
    assert retry["old_status_snapshot"]["jobs"]
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
    assert "--dependency=afterok:" + current[labels[1]]["job_id"] in current[labels[0]]["command"]
    assert not any(value.startswith("--dependency") for value in current[labels[1]]["command"])
    assert (
        retry["dependency_verification"]["clir-prepare-chemistry"]["action"] == "already_completed"
    )
    assert updated["plan_sha256"] == original["plan_sha256"]


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
