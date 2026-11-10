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
