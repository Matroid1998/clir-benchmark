import json
from types import SimpleNamespace

import pytest

from clir_benchmark.cluster_monitor import fetch_status, monitor, sync_once, verify_local_results
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
    assert "--exclude=.*/" in command and "--exclude=*.tmp" in command
    assert "--exclude=.env" in command
    assert "--partial-dir=.rsync-partial" in command
    assert command[-2] == "fir:/remote/run/"


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
