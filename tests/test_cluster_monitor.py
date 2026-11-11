import fcntl
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from clir_benchmark.cluster_monitor import (
    fetch_status,
    main,
    monitor,
    sync_once,
    verify_local_results,
)
from clir_benchmark.errors import ValidationError
from clir_benchmark.util import file_digest, read_json, write_json


def test_remote_status_only_runs_read_only_command_and_quotes_paths():
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"jobs": [], "all_terminal": False}), stderr=""
        )

    result = fetch_status("fir", "/remote/my run", "/remote/env.sh", "/remote/python", runner=run)
    assert result["jobs"] == []
    command, kwargs = calls[0]
    assert command[0] == "ssh" and "BatchMode=yes" in command
    assert "cluster status" in command[-1]
    assert "sbatch" not in command[-1] and "scancel" not in command[-1]
    assert kwargs["timeout"] <= 60


def test_nested_campaign_status_can_use_the_shared_project():
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"jobs": []}), stderr="")

    fetch_status(
        "fir",
        "/remote/run/selected",
        "/env",
        "/python",
        project_directory="/remote/run/project",
        runner=run,
    )
    assert "cd /remote/run/project" in calls[0][-1]
    assert "/remote/run/selected/submission.json" in calls[0][-1]


def test_sync_is_non_destructive_and_excludes_partial_artifacts_and_secrets(tmp_path):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 3600
        return SimpleNamespace(returncode=0, stdout="copied", stderr="")

    result = sync_once("fir", "/remote/run", tmp_path, runner=run)
    assert result["status"] == "success"
    command = commands[0]
    assert "--delete" not in command
    assert "--exclude=.*" in command and "--exclude=*.tmp" in command
    assert "--exclude=.env" in command
    assert "--partial-dir=.rsync-partial" in command
    assert command[-2] == "fir:/remote/run/"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="Local rsync is unavailable")
def test_sync_skips_hidden_upload_files_and_preserves_visible_artifacts(tmp_path):
    source = tmp_path / "server"
    destination = tmp_path / "collected"
    visible = [
        "corpus/lookup.sqlite",
        "embeddings/vectors.npy",
        "reports/report.json",
        "reports/mteb.json",
    ]
    hidden = [
        "corpus/.lookup.sqlite.8QOkrv",
        "embeddings/.vectors.npy.Abc123",
        ".uploading/unfinished.json",
    ]
    for relative in [*visible, *hidden]:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)

    def local_transfer(command, **kwargs):
        # Exercise the real rsync filters without SSH or a remote server.
        return subprocess.run([*command[:-2], str(source) + "/", command[-1]], **kwargs)

    result = sync_once("fir", "/remote/run", destination, runner=local_transfer)
    assert result["status"] == "success"
    assert sorted(
        str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()
    ) == sorted(visible)
    assert all((destination / relative).read_text() == relative for relative in visible)


def local_result(tmp_path):
    remote = "/remote/run"
    report = tmp_path / "artifacts/reports/corpus/run1"
    report.mkdir(parents=True)
    for name in ("report.json", "per_query.json", "mteb.json"):
        write_json(report / name, {})
    artifact = tmp_path / "artifacts/corpus/source.json"
    write_json(artifact, {"text": "original source"})
    write_json(
        report / "analysis_manifest.json",
        {
            "artifact_root": "../../..",
            "files": [{"path": "corpus/source.json", "sha256": file_digest(artifact)}],
        },
    )
    journal = tmp_path / "job_status/bm25/100.json"
    write_json(
        journal,
        {
            "status": "success",
            "stages": [
                {
                    "stage": "evaluate-bm25",
                    "status": "success",
                    "result": remote + "/artifacts/reports/corpus/run1",
                }
            ],
        },
    )
    status = {
        "all_terminal": True,
        "all_succeeded": True,
        "jobs": [
            {
                "label": "bm25",
                "kind": "bm25",
                "state": "COMPLETED",
                "stage_journals": [{"path": remote + "/job_status/bm25/100.json"}],
            }
        ],
    }
    return remote, status, artifact


def test_local_verification_relocates_remote_paths_and_checks_every_artifact(tmp_path):
    remote, status, artifact = local_result(tmp_path)
    result = verify_local_results(tmp_path, remote, status)
    assert result["status"] == "success"
    assert result["reports_verified"] == 1
    assert result["unique_artifact_checksums_verified"] == 1
    artifact.write_text("changed")
    result = verify_local_results(tmp_path, remote, status)
    assert result["status"] == "failed"
    assert result["problems"][0]["reason"] == "checksum_mismatch"


def test_variant_reports_use_shared_root_and_resolved_plan_languages(tmp_path, monkeypatch):
    remote, original, artifact = local_result(tmp_path)
    nested = tmp_path / "selected"
    languages = ["en", "de", "es", "fr", "zh"]
    write_json(
        nested / "plan.json",
        {"output_languages": {"corpus": {"translation": languages, "hyde": languages}}},
    )
    stages = []
    for number in range(11):
        report = tmp_path / f"artifacts/reports/corpus/variant{number}"
        for name in ("report.json", "per_query.json", "mteb.json"):
            write_json(report / name, {})
        write_json(
            report / "analysis_manifest.json",
            {
                "artifact_root": "../../..",
                "files": [{"path": "corpus/source.json", "sha256": file_digest(artifact)}],
            },
        )
        stages.append(
            {
                "stage": f"evaluate-variant-{number}",
                "status": "success",
                "result": remote + f"/artifacts/reports/corpus/variant{number}",
            }
        )
    journal = nested / "job_status/variant/200.json"
    write_json(
        journal,
        {
            "status": "success",
            "corpus": "corpus",
            "configuration": {
                "baselines": ["dense", "translate-dense", "hyde-dense"],
                "target_languages": [],
            },
            "stages": stages,
        },
    )
    status = {
        "all_terminal": True,
        "all_succeeded": True,
        "jobs": [
            {
                "label": "variant",
                "kind": "variant-score",
                "corpus": "corpus",
                "state": "COMPLETED",
                "stage_journals": [{"path": remote + "/selected/job_status/variant/200.json"}],
            }
        ],
    }
    digests = []

    def counted_digest(path):
        digests.append(path)
        return file_digest(path)

    monkeypatch.setattr("clir_benchmark.cluster_monitor.file_digest", counted_digest)
    cache = {}
    assert (
        verify_local_results(tmp_path, remote, original, checksum_cache=cache)["status"]
        == "success"
    )
    result = verify_local_results(tmp_path, remote, status, checksum_cache=cache)
    assert result["status"] == "success"
    assert result["reports_verified"] == 11
    assert digests == [artifact]
    with pytest.raises(ValidationError, match="outside the campaign"):
        verify_local_results(nested, remote + "/selected", status)
    stages.pop()
    data = read_json(journal)
    data["stages"] = stages
    write_json(journal, data)
    result = verify_local_results(tmp_path, remote, status)
    assert {"reason": "report_count", "expected": 11, "actual": 10} in result["problems"]


def test_missing_journal_or_failed_scheduler_job_cannot_pass(tmp_path):
    remote, status, _ = local_result(tmp_path)
    status["jobs"][0]["state"] = "FAILED"
    status["all_succeeded"] = False
    result = verify_local_results(tmp_path, remote, status)
    assert result["status"] == "failed"
    assert result["problems"][0]["reason"] == "scheduler_state"


def test_monitor_retries_network_and_sync_failures_then_verifies(tmp_path):
    polls = []
    syncs = []
    checks = []

    def poll(*args):
        polls.append(True)
        if len(polls) == 1:
            raise OSError("temporary disconnected SSH")
        return {"all_terminal": True, "all_succeeded": True, "jobs": [{"state": "COMPLETED"}]}

    def sync(*args):
        syncs.append(True)
        return {
            "status": "failed" if len(syncs) == 1 else "success",
            "returncode": 1 if len(syncs) == 1 else 0,
        }

    def verify(*args):
        checks.append(True)
        return {"status": "success", "reports_verified": 51}

    path = monitor(
        host="fir",
        remote_directory="/remote/run",
        local_directory=tmp_path,
        remote_environment="/env.sh",
        remote_python="/python",
        interval=0.01,
        status_reader=poll,
        synchronizer=sync,
        verifier=verify,
    )
    data = read_json(path)
    assert data["status"] == "complete"
    assert len(data["status_errors"]) == 1
    assert len(syncs) == 2
    assert checks == [True]
    assert data["sync_attempts"][-1]["final_sync"] is True
    assert (tmp_path / ".local-monitor/latest_status.json").is_file()


def test_terminal_failures_are_synced_but_never_reported_as_success(tmp_path):
    status = {"all_terminal": True, "all_succeeded": False, "jobs": [{"state": "FAILED"}]}
    path = monitor(
        host="fir",
        remote_directory="/remote/run",
        local_directory=tmp_path,
        remote_environment="/env.sh",
        remote_python="/python",
        interval=0.01,
        status_reader=lambda *a: status,
        synchronizer=lambda *a: {"status": "success"},
        verifier=lambda *a: {"status": "failed", "problems": ["job failed"]},
    )
    assert read_json(path)["status"] == "complete_with_failures"


def test_combined_monitor_waits_for_both_campaigns_and_syncs_parent_once(tmp_path):
    polls, syncs, verifications = [], [], []
    child_polls = 0

    def poll(host, remote, env, python, **kwargs):
        nonlocal child_polls
        polls.append((remote, kwargs))
        if remote == "/remote/run":
            return {"all_terminal": True, "all_succeeded": False, "jobs": [{"state": "CANCELLED"}]}
        child_polls += 1
        done = child_polls > 1
        return {
            "all_terminal": done,
            "all_succeeded": done,
            "jobs": [{"state": "COMPLETED" if done else "RUNNING"}],
        }

    def sync(host, remote, local):
        syncs.append((remote, local))
        return {"status": "success"}

    def verify(local, remote, status):
        verifications.append((local, remote, status))
        if status["all_succeeded"]:
            return {"status": "success", "reports_verified": 264, "problems": []}
        return {
            "status": "failed",
            "reports_verified": 42,
            "problems": [{"reason": "scheduler_state", "state": "CANCELLED"}],
        }

    path = monitor(
        host="fir",
        remote_directory="/remote/run",
        local_directory=tmp_path,
        remote_environment="/env",
        remote_python="/python",
        interval=0.01,
        additional_campaigns=("selected",),
        status_reader=poll,
        synchronizer=sync,
        verifier=verify,
    )
    state = read_json(path)
    assert state["status"] == "complete"
    assert state["campaigns"]["selected"]["status"] == "complete"
    assert state["campaigns"]["."]["status"] == "complete_with_failures"
    assert state["verification"]["primary_cancellations_retained"] is True
    assert state["verification"]["reports_verified"] == 306
    assert child_polls >= 2
    assert all(remote == "/remote/run" and local == tmp_path for remote, local in syncs)
    assert len(verifications) == 2
    assert all(remote == "/remote/run" for _, remote, _ in verifications)
    assert any(options == {"project_directory": "/remote/run/project"} for _, options in polls)
    assert (tmp_path / ".local-monitor/campaigns/selected/verification.json").is_file()


def test_combined_monitor_does_not_hide_additional_campaign_failures(tmp_path):
    status = {"all_terminal": True, "all_succeeded": False, "jobs": [{"state": "CANCELLED"}]}
    path = monitor(
        host="fir",
        remote_directory="/remote/run",
        local_directory=tmp_path,
        remote_environment="/env",
        remote_python="/python",
        interval=0.01,
        additional_campaigns=("selected",),
        status_reader=lambda *a, **k: status,
        synchronizer=lambda *a: {"status": "success"},
        verifier=lambda *a: {
            "status": "failed",
            "problems": [{"reason": "scheduler_state", "state": "CANCELLED"}],
        },
    )
    state = read_json(path)
    assert state["status"] == "complete_with_failures"
    assert state["campaigns"]["selected"]["status"] == "complete_with_failures"


@pytest.mark.parametrize("name", ["/outside", "../outside", ".", "nested/../outside", ".hidden"])
def test_additional_campaign_paths_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValidationError, match="relative subdirectories"):
        monitor(
            host="fir",
            remote_directory="/remote/run",
            local_directory=tmp_path,
            remote_environment="/env",
            remote_python="/python",
            additional_campaigns=(name,),
        )


@pytest.mark.parametrize("host", ["fir;rm", "-oProxyCommand=bad", "fir:/tmp", "user@fir\nexit"])
def test_host_rejects_shell_syntax(host, tmp_path):
    with pytest.raises(ValidationError):
        sync_once(host, "/remote/run", tmp_path)


@pytest.mark.parametrize("interval", [0, -1, 61])
def test_monitor_interval_is_bounded(tmp_path, interval):
    with pytest.raises(ValidationError):
        monitor(
            host="fir",
            remote_directory="/remote",
            local_directory=tmp_path,
            remote_environment="/env.sh",
            remote_python="/python",
            interval=interval,
        )


def external_artifacts_result(tmp_path):
    shared = tmp_path / "fir"
    remote, status, artifact = local_result(shared)
    control = tmp_path / "narval"
    remote_control = "/remote/narval"
    journal = read_json(shared / "job_status/bm25/100.json")
    write_json(control / "job_status/bm25/100.json", journal)
    status["jobs"][0]["stage_journals"] = [{"path": remote_control + "/job_status/bm25/100.json"}]
    options = {
        "remote_artifact_directory": remote + "/artifacts",
        "local_artifact_directory": shared / "artifacts",
    }
    return control, remote_control, status, artifact, options


def test_external_artifact_mapping_verifies_reports_and_full_inventory(tmp_path):
    control, remote, status, artifact, options = external_artifacts_result(tmp_path)
    shared = options["local_artifact_directory"]
    report = shared / "reports/corpus/run1"
    manifest = read_json(report / "analysis_manifest.json")
    for relative in ("qrels/source.json", "hyde_requests/failed_prior_attempt.json"):
        path = shared / relative
        write_json(path, {"audit": relative})
        manifest["files"].append({"path": relative, "sha256": file_digest(path)})
    write_json(report / "analysis_manifest.json", manifest)
    result = verify_local_results(control, remote, status, **options)
    assert result["status"] == "success"
    assert result["reports_verified"] == 1
    assert result["unique_artifact_checksums_verified"] == 3
    assert result["path_mapping"] == {"remote": remote, "local": str(control)}
    assert result["artifact_path_mapping"] == {
        "remote": options["remote_artifact_directory"],
        "local": str(shared),
    }
    assert not (control / "artifacts").exists()
    (shared / "hyde_requests/failed_prior_attempt.json").write_text("changed")
    result = verify_local_results(control, remote, status, **options)
    assert result["status"] == "failed"
    assert [item["reason"] for item in result["problems"]] == ["checksum_mismatch"]
    assert artifact.is_file()


@pytest.mark.parametrize("escape", ["prefix", "inventory_root", "inventory_path", "symlink"])
def test_external_mapping_and_inventory_cannot_escape_selected_roots(tmp_path, escape):
    control, remote, status, _, options = external_artifacts_result(tmp_path)
    shared = options["local_artifact_directory"]
    report = shared / "reports/corpus/run1"
    journal = control / "job_status/bm25/100.json"
    if escape == "prefix":
        data = read_json(journal)
        data["stages"][0]["result"] = options["remote_artifact_directory"] + "-other/reports/run"
        write_json(journal, data)
    elif escape == "symlink":
        (shared / "escape").symlink_to(control, target_is_directory=True)
        data = read_json(journal)
        data["stages"][0]["result"] = options["remote_artifact_directory"] + "/escape"
        write_json(journal, data)
    else:
        manifest = read_json(report / "analysis_manifest.json")
        if escape == "inventory_root":
            manifest["artifact_root"] = "../../../.."
        else:
            manifest["files"][0]["path"] = "../../outside"
        write_json(report / "analysis_manifest.json", manifest)
    with pytest.raises(ValidationError, match="outside|escapes"):
        verify_local_results(control, remote, status, **options)


def test_separate_root_monitor_coordinates_both_syncs_and_retries_partial_failure(tmp_path):
    control = tmp_path / "narval"
    shared = tmp_path / "fir/artifacts"
    calls, verifications = [], []

    def synchronize(host, remote, local):
        calls.append((host, remote, local))
        if len(calls) == 2:
            raise OSError("shared artifact transfer interrupted")
        return {
            "status": "success",
            "returncode": 0,
            "command": ["rsync", remote, str(local)],
            "stdout": "copied " + remote,
            "stderr": "",
        }

    def verify(local, remote, status, **options):
        verifications.append(options)
        assert len(calls) == 4
        return {"status": "success", "reports_verified": 1}

    path = monitor(
        host="narval",
        remote_directory="/remote/narval",
        local_directory=control,
        remote_artifact_directory="/remote/fir/artifacts",
        local_artifact_directory=shared,
        remote_environment="/env",
        remote_python="/python",
        interval=0.01,
        status_reader=lambda *a: {"all_terminal": True, "all_succeeded": True, "jobs": []},
        synchronizer=synchronize,
        verifier=verify,
    )
    state = read_json(path)
    assert state["status"] == "complete"
    assert (
        calls
        == [
            ("narval", "/remote/narval", control),
            ("narval", "/remote/fir/artifacts", shared),
        ]
        * 2
    )
    assert len(state["sync_attempts"]) == 2
    first, second = state["sync_attempts"]
    assert first["status"] == "failed"
    assert first["transfers"][1]["error_type"] == "OSError"
    assert second["status"] == "success"
    assert [item["role"] for item in second["transfers"]] == ["campaign", "artifacts"]
    assert all(item["stdout"].startswith("copied") for item in second["transfers"])
    assert second["final_sync"] is True
    assert state["artifact_path_mapping"] == {
        "remote": "/remote/fir/artifacts",
        "local": str(shared),
    }
    assert verifications == [
        {
            "remote_artifact_directory": "/remote/fir/artifacts",
            "local_artifact_directory": shared,
        }
    ]


@pytest.mark.parametrize("only", ["remote", "local"])
def test_artifact_directory_options_are_a_required_pair(tmp_path, only):
    options = {f"{only}_artifact_directory": "/shared" if only == "remote" else tmp_path}
    with pytest.raises(ValidationError, match="provided together"):
        monitor(
            host="narval",
            remote_directory="/remote/narval",
            local_directory=tmp_path / "control",
            remote_environment="/env",
            remote_python="/python",
            **options,
        )
    with pytest.raises(ValidationError, match="provided together"):
        verify_local_results(tmp_path, "/remote", {"jobs": []}, **options)


@pytest.mark.parametrize(
    "remote_artifacts,local_relative",
    [
        ("/remote/narval/artifacts", "shared"),
        ("/remote", "shared"),
        ("/shared", "control/artifacts"),
    ],
)
def test_separate_sync_roots_cannot_overlap_and_duplicate_artifacts(
    tmp_path, remote_artifacts, local_relative
):
    with pytest.raises(ValidationError, match="must not overlap"):
        monitor(
            host="narval",
            remote_directory="/remote/narval",
            local_directory=tmp_path / "control",
            remote_artifact_directory=remote_artifacts,
            local_artifact_directory=tmp_path / local_relative,
            remote_environment="/env",
            remote_python="/python",
        )


def test_separate_root_collector_still_refuses_duplicate_campaign_monitor(tmp_path):
    control = tmp_path / "narval"
    lock_path = control / ".local-monitor/monitor.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValidationError, match="already holds"):
            monitor(
                host="narval",
                remote_directory="/remote/narval",
                local_directory=control,
                remote_artifact_directory="/remote/fir/artifacts",
                local_artifact_directory=tmp_path / "fir/artifacts",
                remote_environment="/env",
                remote_python="/python",
            )


def test_cli_passes_the_artifact_mapping_pair(tmp_path, monkeypatch):
    seen = []
    result = tmp_path / "monitor.json"
    write_json(result, {"status": "complete"})

    def fake_monitor(**options):
        seen.append(options)
        return result

    monkeypatch.setattr("clir_benchmark.cluster_monitor.monitor", fake_monitor)
    assert (
        main(
            [
                "--host",
                "narval",
                "--remote-directory",
                "/remote/control",
                "--local-directory",
                str(tmp_path / "control"),
                "--remote-artifact-directory",
                "/remote/shared/artifacts",
                "--local-artifact-directory",
                str(tmp_path / "shared/artifacts"),
                "--remote-environment",
                "/env",
                "--remote-python",
                "/python",
            ]
        )
        == 0
    )
    assert seen[0]["remote_artifact_directory"] == "/remote/shared/artifacts"
    assert seen[0]["local_artifact_directory"] == Path(tmp_path / "shared/artifacts")
