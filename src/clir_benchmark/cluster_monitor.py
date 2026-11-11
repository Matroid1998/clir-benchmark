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
    host,
    remote_directory,
    remote_environment,
    remote_python,
    *,
    runner=subprocess.run,
    project_directory=None,
):
    """Ask the remote package for scheduler status; never changes scheduled jobs."""
    _validate_host(host)
    root = Path(remote_directory)
    command = "\n".join(
        [
            "set -e",
            "source " + shlex.quote(remote_environment),
            "cd " + shlex.quote(str(project_directory or root / "project")),
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


def _expected_reports(job: dict, journal: dict, journal_path: Path) -> int:
    configuration = journal.get("configuration", {})
    baselines = set(configuration.get("baselines", ["dense", "hybrid"]))
    if job["kind"] in {"model", "score"}:
        return len({"dense", "hybrid"} & baselines)
    if job["kind"] == "bm25":
        return 1
    if job["kind"] != "variant-score":
        return 0
    # The plan records resolved output languages, including defaults derived from
    # corpus metadata. Reading it never loads models or performs generation.
    plan_path = journal_path.parents[2] / "plan.json"
    corpus = job.get("corpus") or journal.get("corpus")
    languages = (
        read_json(plan_path).get("output_languages", {}).get(corpus, {})
        if plan_path.is_file()
        else {}
    )
    total = int("dense" in baselines)
    for baseline, name in (("translate-dense", "translation"), ("hyde-dense", "hyde")):
        if baseline not in baselines:
            continue
        selected = languages.get(name)
        if selected is None:
            selected = (
                configuration.get("hyde", {}).get("output_languages") if name == "hyde" else None
            ) or configuration.get("target_languages")
        if not selected:
            raise ValidationError(f"Missing resolved {name} languages for {job['label']}")
        total += len(set(selected))
    return total


def verify_local_results(
    local_directory: Path, remote_directory: str, status: dict, *, checksum_cache=None
) -> dict:
    """Verify every completed stage journal, report, MTEB file and indexed artifact."""
    local = Path(local_directory).resolve()
    remote = Path(remote_directory)
    problems = []
    reports = set()
    verified = set()
    checksums = {} if checksum_cache is None else checksum_cache
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
            expected_reports += _expected_reports(job, journal, path)
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
            actual = checksums.get(path)
            if actual is None:
                actual = checksums[path] = file_digest(path)
            verified.add(path)
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
    additional_campaigns: tuple[str, ...] = (),
    status_reader=fetch_status,
    synchronizer=sync_once,
    verifier=verify_local_results,
) -> Path:
    """Poll at most every 60 seconds while a single independent rsync runs."""
    if not 0 < interval <= 60:
        raise ValidationError("Monitor interval must be greater than zero and at most 60 seconds")
    _validate_host(host)
    campaigns = ["."]
    for value in additional_campaigns:
        relative = Path(value)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part.startswith(".") for part in relative.parts)
        ):
            raise ValidationError(
                "Additional campaigns must be relative subdirectories without dot components"
            )
        if str(relative) not in campaigns:
            campaigns.append(str(relative))
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
    if additional_campaigns:
        state["additional_campaigns"] = campaigns[1:]
        state["campaigns"] = {}
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

    def poll_campaigns():
        if len(campaigns) == 1:
            return status_reader(host, remote_directory, remote_environment, remote_python)

        def poll(name):
            root = str(Path(remote_directory) / name)
            options = (
                {"project_directory": str(Path(remote_directory) / "project")}
                if name != "."
                else {}
            )
            return status_reader(host, root, remote_environment, remote_python, **options)

        # SSH timeouts run concurrently, so a disconnected host does not multiply
        # the polling delay by the number of campaigns.
        with ThreadPoolExecutor(max_workers=len(campaigns)) as polls:
            futures = {name: polls.submit(poll, name) for name in campaigns}
            statuses = {name: future.result() for name, future in futures.items()}
        for name, status in statuses.items():
            _atomic(state_directory / "campaigns" / name / "latest_status.json", status)
            state["campaigns"][name] = {
                "all_terminal": status["all_terminal"],
                "all_succeeded": status.get("all_succeeded", False),
                "scheduler_counts": dict(Counter(job["state"] for job in status["jobs"])),
            }
        return {
            "schema": 1,
            "campaigns": statuses,
            "jobs": [
                {**job, "campaign": name}
                for name, status in statuses.items()
                for job in status["jobs"]
            ],
            "all_terminal": all(status["all_terminal"] for status in statuses.values()),
            "all_succeeded": all(status.get("all_succeeded") for status in statuses.values()),
        }

    def verify_campaigns(status):
        if len(campaigns) == 1:
            return verifier(local, remote_directory, status)
        checksums = {}
        results = {}
        for name, campaign_status in status["campaigns"].items():
            try:
                options = {"checksum_cache": checksums} if verifier is verify_local_results else {}
                result = verifier(local, remote_directory, campaign_status, **options)
            except Exception as exc:
                result = {
                    "status": "failed",
                    "problems": [
                        {
                            "reason": "verification_error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ],
                }
            results[name] = result
            _atomic(state_directory / "campaigns" / name / "verification.json", result)
            state["campaigns"][name].update(
                status="complete" if result["status"] == "success" else "complete_with_failures",
                verification=result,
            )
        primary = results["."]
        primary_cancellations_only = bool(primary.get("problems")) and all(
            item.get("reason") == "scheduler_state" and item.get("state") == "CANCELLED"
            for item in primary["problems"]
        )
        successful = all(
            result["status"] == "success" or (name == "." and primary_cancellations_only)
            for name, result in results.items()
        )
        return {
            "status": "success" if successful else "failed",
            "campaigns": results,
            "reports_verified": sum(
                result.get("reports_verified", 0) for result in results.values()
            ),
            "unique_artifact_checksums_verified": len(checksums),
            "primary_cancellations_retained": primary_cancellations_only,
            "notes": "Each campaign retains its own scheduler and verification outcome. Terminal cancellations in the primary campaign do not fail additional campaigns; other verification errors do. Shared files are synchronized once and checksummed once.",
        }

    event("monitor_started", pid=os.getpid())
    save()
    try:
        while True:
            began = time.monotonic()
            try:
                latest = poll_campaigns()
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
                        verification = verify_campaigns(latest)
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
    parser.add_argument(
        "--additional-campaign",
        action="append",
        dest="additional_campaigns",
        default=[],
        help="Also monitor a campaign subdirectory; synchronize and verify the shared parent tree once",
    )
    args = parser.parse_args(argv)
    path = monitor(**vars(args))
    print(path)
    return 0 if read_json(path)["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
