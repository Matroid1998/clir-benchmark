"""Durable local monitoring and non-destructive collection of a Slurm campaign."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .errors import ValidationError
from .util import file_digest, read_json, write_json


def _atomic(path: Path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _validate_host(host):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
        raise ValidationError("Use an SSH host alias or user@hostname without shell syntax")


def fetch_status(
    host, remote_directory, remote_environment, remote_python, *, runner=subprocess.run
):
    """Ask the remote package for scheduler status; never changes scheduled jobs."""
    _validate_host(host)
    root = Path(remote_directory)
    command = "\n".join(
        [
            "set -e",
            "source " + shlex.quote(remote_environment),
            "cd " + shlex.quote(str(root / "project")),
            shlex.join(
                [
                    remote_python,
                    "-m",
                    "clir_benchmark.cluster",
                    "status",
                    "--submission",
                    str(root / "submission.json"),
                ]
            )
            + " >/dev/null",
            "cat " + shlex.quote(str(root / "status.json")),
        ]
    )
    response = runner(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            host,
            "bash -lc " + shlex.quote(command),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=55,
    )
    if response.returncode:
        raise ValidationError(
            f"Remote status returned {response.returncode}: {response.stderr[-2000:]}"
        )
    result = json.loads(response.stdout)
    if not isinstance(result, dict) or not isinstance(result.get("jobs"), list):
        raise ValidationError("Remote scheduler status has an invalid structure")
    return result


def sync_once(host, remote_directory, local_directory, *, runner=subprocess.run, timeout=3600):
    """Copy completed files and logs while preserving local data and partial transfers."""
    _validate_host(host)
    local = Path(local_directory)
    local.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "--archive",
        "--protect-args",
        "--human-readable",
        "--stats",
        "--partial-dir=.rsync-partial",
        "--delay-updates",
        "--timeout=120",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=15",
        "--exclude=.*/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=*.tmp",
        "--exclude=.env",
        "--exclude=.env.*",
        f"{host}:{str(remote_directory).rstrip('/')}/",
        str(local) + "/",
    ]
    started = time.perf_counter()
    response = runner(command, capture_output=True, text=True, check=False, timeout=timeout)
    return {
        "status": "success" if response.returncode == 0 else "failed",
        "returncode": response.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": response.stdout,
        "stderr": response.stderr,
        "command": command,
    }


def verify_local_results(local_directory: Path, remote_directory: str, status: dict) -> dict:
    """Verify every completed stage journal, report, MTEB file and indexed artifact."""
    local = Path(local_directory).resolve()
    remote = Path(remote_directory)
    problems = []
    reports = set()
    verified = {}
    expected_reports = 0

    def relocated(value):
        try:
            relative = Path(value).relative_to(remote)
        except ValueError as exc:
            raise ValidationError(f"Artifact is outside the campaign: {value}") from exc
        path = (local / relative).resolve()
        if not path.is_relative_to(local):
            raise ValidationError(f"Artifact resolves outside the local campaign: {value}")
        return path

    for job in status["jobs"]:
        if job["state"] != "COMPLETED":
            problems.append(
                {"job": job["label"], "reason": "scheduler_state", "state": job["state"]}
            )
            continue
        journals = job.get("stage_journals", [])
        if not journals:
            problems.append({"job": job["label"], "reason": "missing_stage_journal"})
        # A requeued Slurm job can have multiple attempts. Retain every journal
        # on disk, but final completion is assessed using its newest attempt.
        for entry in journals[-1:]:
            path = relocated(entry["path"])
            if not path.is_file():
                problems.append({"job": job["label"], "reason": "missing_file", "path": str(path)})
                continue
            journal = read_json(path)
            if job["kind"] in {"model", "score"}:
                expected_reports += len(
                    {"dense", "hybrid"}
                    & set(journal.get("configuration", {}).get("baselines", ["dense", "hybrid"]))
                )
            elif job["kind"] == "bm25":
                expected_reports += 1
            if journal["status"] != "success":
                problems.append({"job": job["label"], "reason": "stage_failure", "path": str(path)})
            for stage in journal.get("stages", []):
                if stage["stage"].startswith("evaluate-") and stage.get("status") == "success":
                    reports.add(relocated(stage["result"]))
    for report in sorted(reports):
        for filename in ("report.json", "per_query.json", "mteb.json", "analysis_manifest.json"):
            if not (report / filename).is_file():
                problems.append({"reason": "missing_report_file", "path": str(report / filename)})
        manifest_path = report / "analysis_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        root = (report / manifest["artifact_root"]).resolve()
        if not root.is_relative_to(local):
            raise ValidationError("Analysis inventory root escapes the local campaign")
        for item in manifest["files"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValidationError("Analysis inventory path escapes its artifact root")
            if not path.is_file():
                problems.append({"reason": "missing_inventory_file", "path": str(path)})
                continue
            actual = verified.setdefault(path, None)
            if actual is None:
                actual = verified[path] = file_digest(path)
            if actual != item["sha256"]:
                problems.append({"reason": "checksum_mismatch", "path": str(path)})
    if len(reports) != expected_reports:
        problems.append(
            {"reason": "report_count", "expected": expected_reports, "actual": len(reports)}
        )
    return {
        "status": "success" if not problems and status.get("all_succeeded") else "failed",
        "reports_verified": len(reports),
        "unique_artifact_checksums_verified": len(verified),
        "problems": problems,
        "path_mapping": {"remote": str(remote), "local": str(local)},
        "notes": "Remote paths remain in original provenance. Analysis inventories use relative paths and were verified against local files. No remote jobs were changed.",
    }


def monitor(
    *,
    host: str,
    remote_directory: str,
    local_directory: Path,
    remote_environment: str,
    remote_python: str,
    interval: float = 30,
    status_reader=fetch_status,
    synchronizer=sync_once,
    verifier=verify_local_results,
) -> Path:
    """Poll at most every 60 seconds while a single independent rsync runs."""
    if not 0 < interval <= 60:
        raise ValidationError("Monitor interval must be greater than zero and at most 60 seconds")
    _validate_host(host)
    local = Path(local_directory).resolve()
    state_directory = local / ".local-monitor"
    state_directory.mkdir(parents=True, exist_ok=True)
    lock = (state_directory / "monitor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise ValidationError(
            "A local collector already holds this campaign's monitor lock"
        ) from exc
    state_path = state_directory / "monitor.json"
    state = {
        "schema": 1,
        "status": "monitoring",
        "pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat(),
        "host": host,
        "remote_directory": remote_directory,
        "local_directory": str(local),
        "poll_interval_seconds": interval,
        "sync_attempts": [],
        "status_errors": [],
        "path_mapping": {"remote": remote_directory, "local": str(local)},
    }
    if state_path.exists():
        previous = read_json(state_path)
        _atomic(state_directory / ("previous-" + str(previous["pid"]) + ".json"), previous)
    future = None
    final_sync = False
    latest = None
    terminal_observed = False
    executor = ThreadPoolExecutor(max_workers=1)

    def event(kind, **details):
        record = {"time": datetime.now(UTC).isoformat(), "kind": kind, **details}
        with (state_directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save():
        state["updated_at"] = datetime.now(UTC).isoformat()
        _atomic(state_path, state)

    event("monitor_started", pid=os.getpid())
    save()
    try:
        while True:
            began = time.monotonic()
            try:
                latest = status_reader(host, remote_directory, remote_environment, remote_python)
                _atomic(state_directory / "latest_status.json", latest)
                counts = dict(Counter(job["state"] for job in latest["jobs"]))
                state["scheduler_counts"] = counts
                state["last_successful_poll"] = datetime.now(UTC).isoformat()
                event("scheduler_status", counts=counts, all_terminal=latest["all_terminal"])
                terminal_observed = bool(latest["all_terminal"])
            except Exception as exc:
                error = {
                    "time": datetime.now(UTC).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                state["status_errors"].append(error)
                event("status_error", **error)
            if future is not None and future.done():
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                result["final_sync"] = final_sync
                state["sync_attempts"].append(result)
                event("sync_finished", status=result["status"], final_sync=final_sync)
                future = None
                if final_sync and result["status"] == "success" and terminal_observed:
                    try:
                        verification = verifier(local, remote_directory, latest)
                    except Exception as exc:
                        verification = {
                            "status": "failed",
                            "problems": [
                                {
                                    "reason": "verification_error",
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            ],
                        }
                    _atomic(state_directory / "verification.json", verification)
                    state["verification"] = verification
                    state["status"] = (
                        "complete"
                        if verification["status"] == "success"
                        else "complete_with_failures"
                    )
                    state["finished_at"] = datetime.now(UTC).isoformat()
                    event("monitor_finished", status=state["status"])
                    save()
                    return state_path
            if future is None:
                final_sync = terminal_observed
                future = executor.submit(synchronizer, host, remote_directory, local)
                event("sync_started", final_sync=final_sync)
            save()
            time.sleep(max(0.01, interval - (time.monotonic() - began)))
    except BaseException as exc:
        state.update(status="interrupted", error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="fir")
    parser.add_argument("--remote-directory", required=True)
    parser.add_argument("--local-directory", type=Path, required=True)
    parser.add_argument("--remote-environment", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args(argv)
    path = monitor(**vars(args))
    print(path)
    return 0 if read_json(path)["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
