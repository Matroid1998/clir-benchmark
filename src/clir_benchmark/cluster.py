"""Explicit Slurm job planning and execution for the original retrieval baselines.

Plan creation is offline. Submission is intentionally separate: the resulting
dependency graph can be submitted without an artificial concurrency throttle.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import getpass
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import Config, load_config
from .errors import ValidationError
from .util import file_digest, read_json, versions, write_json

ORIGINAL_BASELINES = {"dense", "bm25", "hybrid"}


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


@dataclass(frozen=True)
class JobResources:
    cpus: int = 1
    memory: str = "96G"
    walltime: str = "12:00:00"
    gpus: str | None = None
    partition: str | None = None

    def __post_init__(self):
        if isinstance(self.cpus, bool) or not isinstance(self.cpus, int) or self.cpus < 1:
            raise ValidationError("Slurm CPU count must be a positive integer")
        for name in ("memory", "walltime", "gpus", "partition"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or re.search(r"\s", value)
            ):
                raise ValidationError(f"Invalid Slurm {name}")


def write_plan(
    config: Config,
    destination: Path,
    *,
    remote_project: str,
    remote_directory: str,
    remote_cpu_config: str,
    remote_model_config: str,
    python: str,
    environment: str,
    account: str,
    gpu_account: str | None = None,
    prepare_resources: JobResources | None = None,
    bm25_resources: JobResources | None = None,
    model_resources: JobResources | None = None,
    model_resources_by_alias: Mapping[str, JobResources] | None = None,
    score_resources: JobResources | None = None,
    split_embedding: bool = False,
    allow_blocked_settings: bool = False,
) -> Path:
    """Write one independent model/corpus script and its shared dependencies.

    With eight models and three corpora, the default graph has 30 jobs: three
    preparation/index jobs, three BM25 jobs and 24 dense/hybrid model jobs.
    ``split_embedding`` creates separate GPU embedding and CPU scoring jobs,
    releasing accelerators before NumPy/SQLite retrieval and result export.
    Each stage defaults to one CPU. Resource overrides are independent of
    baseline selection and artifact identities; BM25 has its own resource request.
    """
    if not config.baselines or set(config.baselines) - ORIGINAL_BASELINES:
        raise ValidationError("Cluster plans support only dense, bm25 and hybrid baselines")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", account) or (
        gpu_account is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", gpu_account)
    ):
        raise ValidationError("Invalid Slurm account")
    if len(set(config.models)) != len(config.models):
        raise ValidationError("Cluster model aliases must be unique")
    model_resources_by_alias = dict(model_resources_by_alias or {})
    if any(
        not isinstance(alias, str) or not isinstance(resources, JobResources)
        for alias, resources in model_resources_by_alias.items()
    ):
        raise ValidationError(
            "Per-model resources require model alias keys and JobResources values"
        )
    unknown_aliases = set(model_resources_by_alias) - set(config.models)
    if unknown_aliases:
        raise ValidationError(
            "Unknown resource model aliases: " + ", ".join(sorted(unknown_aliases))
        )
    prepare_resources = prepare_resources or JobResources()
    bm25_resources = bm25_resources or JobResources()
    model_resources = model_resources or JobResources(gpus="h100:1")
    score_resources = score_resources or JobResources()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scripts = destination / "scripts"
    scripts.mkdir(exist_ok=True)
    remote = Path(remote_directory)
    jobs = []
    labels = set()

    def add(kind, corpus, alias, dependencies, resources):
        label = "clir-" + "-".join(_label(value) for value in (kind, corpus, alias) if value)
        if label in labels:
            raise ValidationError("Model aliases produce colliding Slurm job names")
        labels.add(label)
        configuration = remote_cpu_config if kind in {"prepare", "bm25"} else remote_model_config
        command = [
            python,
            "-m",
            "clir_benchmark.cluster",
            "run",
            "--config",
            configuration,
            "--kind",
            kind,
            "--corpus",
            corpus,
            "--journal-directory",
            str(remote / "job_status" / label),
        ]
        if alias:
            command += ["--model", alias]
        if allow_blocked_settings:
            command += ["--allow-blocked-settings"]
        directives = {
            "account": gpu_account if resources.gpus and gpu_account else account,
            "job-name": label,
            "cpus-per-task": resources.cpus,
            "mem": resources.memory,
            "time": resources.walltime,
            "output": str(remote / "logs" / (label + "-%j.out")),
            "error": str(remote / "logs" / (label + "-%j.err")),
        }
        if resources.gpus:
            directives["gpus"] = resources.gpus
        if resources.partition:
            directives["partition"] = resources.partition
        lines = ["#!/bin/bash"]
        lines += [
            f"#SBATCH --{name}={shlex.quote(str(value))}" for name, value in directives.items()
        ]
        lines += [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_project)}",
            f"source {shlex.quote(environment)}",
            'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
            'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
            'export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
            "export TOKENIZERS_PARALLELISM=false",
            f"export HF_HUB_OFFLINE={0 if kind == 'prepare' else 1}",
            f"export TRANSFORMERS_OFFLINE={0 if kind == 'prepare' else 1}",
            shlex.join(command),
            "",
        ]
        filename = label + ".sh"
        (scripts / filename).write_text("\n".join(lines), encoding="utf-8")
        (scripts / filename).chmod(0o755)
        jobs.append(
            {
                "label": label,
                "kind": kind,
                "corpus": corpus,
                "model": alias,
                "dependencies": list(dependencies),
                "script": str(remote / "scripts" / filename),
                "resources": asdict(resources),
                "configuration": configuration,
            }
        )
        return label

    # Every preparation job is submitted before any dependent job; all corpus
    # graphs are independent and no array concurrency limit is imposed.
    preparation = {
        corpus: add("prepare", corpus, None, (), prepare_resources) for corpus in config.datasets
    }
    for corpus in config.datasets:
        if "bm25" in config.baselines:
            add("bm25", corpus, None, (preparation[corpus],), bm25_resources)
        if not ({"dense", "hybrid"} & set(config.baselines)):
            continue
        for alias in config.models:
            embedding = add(
                "embed" if split_embedding else "model",
                corpus,
                alias,
                (preparation[corpus],),
                model_resources_by_alias.get(alias, model_resources),
            )
            if split_embedding:
                add("score", corpus, alias, (embedding,), score_resources)
    plan = {
        "schema": 1,
        "execution": "not_submitted",
        "account": account,
        "gpu_account": gpu_account,
        "remote_project": remote_project,
        "remote_directory": remote_directory,
        "selected_baselines": list(config.baselines),
        "models": list(config.models),
        "corpora": list(config.datasets),
        "candidate_mode": config.candidate_mode,
        "configuration": _plain(asdict(config)),
        "split_embedding": split_embedding,
        "allow_blocked_settings": allow_blocked_settings,
        "concurrency": "All jobs may be submitted immediately with afterok dependencies; no artificial running-job throttle.",
        "submission_notes": "Create remote_directory/logs before sbatch. Submit topologically with --dependency=afterok:<ids>, recording every returned job ID. Job submission is separate from plan creation.",
        "jobs": jobs,
    }
    write_json(destination / "plan.json", plan)
    return destination / "plan.json"


def run_job(
    config: Config,
    corpus: str,
    kind: str,
    *,
    alias: str | None = None,
    journal_directory: Path,
    allow_blocked_settings: bool = False,
) -> Path:
    """Explicitly run one scheduled stage group, preserving a durable journal."""
    from .lexical import prepare_bm25
    from .pipeline import embed, evaluate, prepare_corpus, prepare_relevance, retrieve

    if kind not in {"prepare", "bm25", "model", "embed", "score"}:
        raise ValidationError(f"Unknown cluster job kind: {kind}")
    if corpus not in config.datasets:
        raise ValidationError(f"Corpus is not configured: {corpus}")
    if set(config.baselines) - ORIGINAL_BASELINES:
        raise ValidationError("Cluster execution supports only dense, bm25 and hybrid")
    if kind in {"model", "embed", "score"} and alias not in config.models:
        raise ValidationError("Model job requires a configured model alias")
    if kind == "bm25" and "bm25" not in config.baselines:
        raise ValidationError("BM25 baseline is disabled")
    journal_directory = Path(journal_directory)
    journal_directory.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    journal_path = journal_directory / f"{_label(job_id)}-{timestamp}.json"
    started = time.perf_counter()
    journal = {
        "schema": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "job_id": job_id,
        "slurm": {
            name: os.environ[name]
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_JOB_GPUS",
                "CUDA_VISIBLE_DEVICES",
            )
            if name in os.environ
        },
        "thread_environment": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
            if name in os.environ
        },
        "kind": kind,
        "corpus": corpus,
        "model": alias,
        "configuration": _plain(asdict(config)),
        "versions": versions(),
        "stages": [],
        "measurement_methodology": "Stage seconds measure current invocation wall time including loading/cache validation/export. Historical model/retrieval timings remain in dependency manifests. Local compute monetary price is unknown.",
    }

    def save():
        journal["seconds"] = time.perf_counter() - started
        temporary = journal_path.with_suffix(".tmp")
        write_json(temporary, journal)
        temporary.replace(journal_path)

    def stage(name, operation):
        record = {"stage": name, "status": "running"}
        journal["stages"].append(record)
        print(f"cluster: {kind}/{corpus}/{alias or '-'}: {name}", flush=True)
        save()
        began = time.perf_counter()
        try:
            result = operation()
            record.update(status="success", result=_plain(result))
            return result
        except BaseException as exc:

            def redact(value):
                value = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", value)
                return re.sub(r"\b(?:sk-|hf_)[A-Za-z0-9_-]+", "[REDACTED]", value)

            record.update(
                status="failed",
                error_type=type(exc).__name__,
                error=redact(str(exc)),
                traceback=redact(traceback.format_exc()),
            )
            raise
        finally:
            record["seconds"] = time.perf_counter() - began
            save()

    def relevance():
        path = prepare_relevance(config, corpus)
        manifest = read_json(path / "manifest.json")
        journal["relevance"] = manifest
        if manifest["blocked_settings"] and not allow_blocked_settings:
            raise ValidationError(
                f"{len(manifest['blocked_settings'])} relevance settings are blocked; inspect the eligibility artifacts. "
                "Use --allow-blocked-settings only to deliberately run the remaining settings."
            )
        return path

    save()
    try:
        if kind == "prepare":
            stage("prepare-corpus", lambda: prepare_corpus(config, corpus))
            stage("prepare-relevance", relevance)
            if {"bm25", "hybrid"} & set(config.baselines):
                stage("index-bm25", lambda: prepare_bm25(config, corpus))
        elif kind == "bm25":
            stage("retrieve-bm25", lambda: retrieve(config, corpus, baseline="bm25"))
            stage("evaluate-bm25", lambda: evaluate(config, corpus, baseline="bm25"))
        else:
            if kind in {"embed", "model"}:
                stage("embed", lambda: embed(config, corpus, alias))
            if kind in {"score", "model"}:
                for baseline in ("dense", "hybrid"):
                    if baseline in config.baselines:
                        stage(
                            f"retrieve-{baseline}",
                            lambda b=baseline: retrieve(config, corpus, alias, baseline=b),
                        )
                        stage(
                            f"evaluate-{baseline}",
                            lambda b=baseline: evaluate(config, corpus, alias, baseline=b),
                        )
        journal["status"] = "success"
    except BaseException:
        journal["status"] = "failed"
        raise
    finally:
        journal["finished_at"] = datetime.now(UTC).isoformat()
        save()
    return journal_path


def submit_plan(plan_path: Path, *, workers: int = 8, runner=subprocess.run) -> Path:
    """Submit independent jobs concurrently, recording IDs before the next wave.

    Rerunning a completed submission reuses IDs. An interrupted submission with
    unknown job IDs is never repeated automatically, avoiding duplicate jobs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValidationError("Submission worker count must be positive")
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    journal_path = plan_path.parent / "submission.json"
    fingerprint = file_digest(plan_path)
    if journal_path.exists():
        existing = read_json(journal_path)
        if existing["plan_sha256"] != fingerprint:
            raise ValidationError("Submission plan differs from its saved journal")
        if existing["status"] == "submitted":
            return journal_path
        raise ValidationError(
            "A previous submission is incomplete; inspect submission.json and the scheduler before retrying to avoid duplicate jobs"
        )
    labels = [job["label"] for job in plan["jobs"]]
    if len(set(labels)) != len(labels):
        raise ValidationError("Duplicate labels in cluster plan")
    if any(set(job["dependencies"]) - set(labels) for job in plan["jobs"]):
        raise ValidationError("Unknown dependency in cluster plan")
    Path(plan["remote_directory"], "logs").mkdir(parents=True, exist_ok=True)
    journal = {
        "schema": 1,
        "status": "submitting",
        "plan_sha256": fingerprint,
        "started_at": datetime.now(UTC).isoformat(),
        "submission_workers": workers,
        "jobs": [{**job, "status": "not_submitted", "job_id": None} for job in plan["jobs"]],
    }
    by_label = {job["label"]: job for job in journal["jobs"]}

    def save():
        temporary = journal_path.with_suffix(".tmp")
        write_json(temporary, journal)
        temporary.replace(journal_path)

    def submit(job):
        command = ["sbatch", "--parsable"]
        parents = [by_label[label]["job_id"] for label in job["dependencies"]]
        if parents:
            command.append("--dependency=afterok:" + ":".join(parents))
        command.append(job["script"])
        result = runner(command, capture_output=True, text=True, check=False)
        job_id = result.stdout.strip().split(";", 1)[0]
        return result, job_id, command

    save()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while pending := [j for j in journal["jobs"] if j["status"] == "not_submitted"]:
                ready = [
                    job
                    for job in pending
                    if all(
                        by_label[label]["status"] == "submitted" for label in job["dependencies"]
                    )
                ]
                if not ready:
                    for job in pending:
                        job["status"] = "blocked"
                    save()
                    break
                for job in ready:
                    job["status"] = "submitting"
                save()  # Persist before invoking sbatch: an interruption is ambiguous.
                futures = {executor.submit(submit, job): job for job in ready}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        result, job_id, command = future.result()
                        job.update(
                            command=command,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            returncode=result.returncode,
                        )
                        if result.returncode == 0 and re.fullmatch(r"\d+", job_id):
                            job.update(status="submitted", job_id=job_id)
                        else:
                            job["status"] = "submission_failed"
                    except Exception as exc:
                        job.update(
                            status="submission_failed",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    save()
        journal["status"] = (
            "submitted"
            if all(job["status"] == "submitted" for job in journal["jobs"])
            else "incomplete"
        )
    finally:
        journal["finished_at"] = datetime.now(UTC).isoformat()
        save()
    return journal_path


def retry_jobs(
    submission_path: Path,
    labels: list[str] | tuple[str, ...],
    *,
    walltime: str | None = None,
    cpus: int | None = None,
    gpus: str | None = None,
    memory: str | None = None,
    reason: str | None = None,
    runner=subprocess.run,
) -> Path:
    """Explicitly replace selected terminal jobs while preserving every attempt.

    Selected parents are submitted first; selected children depend on their new
    IDs. Unselected jobs are untouched. Interrupted or ambiguous sbatch attempts
    require scheduler inspection before another retry, never automatic replay.
    """
    submission_path = Path(submission_path)
    labels = list(dict.fromkeys(labels))
    if not labels:
        raise ValidationError("Select at least one job label to retry")
    if walltime is not None and not re.fullmatch(r"(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d", walltime):
        raise ValidationError("Retry walltime must use [days-]hours:minutes:seconds")
    if cpus is not None:
        JobResources(cpus=cpus)
    if gpus is not None:
        JobResources(gpus=gpus)
    if memory is not None:
        JobResources(memory=memory)
    lock = submission_path.with_suffix(".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise ValidationError("Another submission update is running") from exc
    try:
        submitted = read_json(submission_path)
        if submitted["status"] != "submitted":
            raise ValidationError(
                "Submission journal is incomplete or retrying; inspect the scheduler and retry journal before resubmitting"
            )
        by_label = {job["label"]: job for job in submitted["jobs"]}
        if set(labels) - by_label.keys():
            raise ValidationError(
                "Unknown retry job labels: " + ", ".join(sorted(set(labels) - by_label.keys()))
            )
        selected = [by_label[label] for label in labels]
        if gpus is not None and not any(job["resources"].get("gpus") for job in selected):
            raise ValidationError("A GPU override requires at least one selected GPU job")
        if any(
            job["status"] != "submitted" or not re.fullmatch(r"\d+", job.get("job_id") or "")
            for job in selected
        ):
            raise ValidationError(
                "Selected jobs have ambiguous submission state; inspect the scheduler before retrying"
            )
        order = []
        pending = list(labels)
        while pending:
            ready = [
                label
                for label in pending
                if not (set(by_label[label]["dependencies"]) & set(pending))
            ]
            if not ready:
                raise ValidationError("Selected retry dependencies contain a cycle")
            order.extend(ready)
            pending = [label for label in pending if label not in ready]
        old_ids = [job["job_id"] for job in selected]
        existing_parents = sorted(
            {parent for job in selected for parent in job["dependencies"] if parent not in labels}
        )
        if any(
            by_label[parent]["status"] != "submitted"
            or not re.fullmatch(r"\d+", by_label[parent].get("job_id") or "")
            for parent in existing_parents
        ):
            raise ValidationError(
                "A retry dependency has no confirmed job ID; repair its submission first"
            )
        query_ids = list(
            dict.fromkeys([*old_ids, *(by_label[parent]["job_id"] for parent in existing_parents)])
        )
        fields = [
            "JobIDRaw",
            "State",
            "ExitCode",
            "ElapsedRaw",
            "Start",
            "End",
            "AllocCPUS",
            "ReqMem",
            "NodeList",
            "Reason",
            "AllocTRES",
            "CPUTimeRAW",
            "TotalCPU",
            "MaxRSS",
        ]
        command = [
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(query_ids),
            "--format",
            ",".join(fields),
        ]
        response = runner(command, capture_output=True, text=True, check=False)
        if response.returncode:
            raise ValidationError("Cannot verify terminal job accounting: " + response.stderr)
        accounting = {}
        for line in response.stdout.splitlines():
            values = line.split("|")
            if len(values) == len(fields) and values[0] in query_ids:
                accounting[values[0]] = dict(zip(fields, values, strict=True))
        # Query the user's queue, rather than -j old IDs (which can error after
        # terminal jobs leave the controller). A live entry always prevents retry.
        queue_command = ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%T"]
        queue = runner(queue_command, capture_output=True, text=True, check=False)
        if queue.returncode:
            raise ValidationError(
                "Cannot verify that previous jobs left the queue: " + queue.stderr
            )
        active = {line.split("|", 1)[0].strip() for line in queue.stdout.splitlines()}
        terminal = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
            "REVOKED",
            "SPECIAL_EXIT",
        }
        for job in selected:
            old_id = job["job_id"]
            state = accounting.get(old_id, {}).get("State", "UNKNOWN").split(" ")[0].rstrip("+")
            if old_id in active or state not in terminal:
                raise ValidationError(
                    f"Refusing duplicate retry for {job['label']}: job {old_id} is {state} or still in the queue"
                )
        dependency_verification = {}
        for parent in existing_parents:
            parent_id = by_label[parent]["job_id"]
            parent_accounting = accounting.get(parent_id, {})
            state = parent_accounting.get("State", "UNKNOWN").split(" ")[0].rstrip("+")
            if state == "COMPLETED" and parent_id not in active:
                action = "already_completed"
            elif parent_id in active or state not in terminal and state != "UNKNOWN":
                action = "afterok"
            else:
                raise ValidationError(
                    f"Retry dependency {parent} is {state}; include failed parents in the retry or verify unknown jobs before submission"
                )
            dependency_verification[parent] = {
                "job_id": parent_id,
                "action": action,
                "accounting": parent_accounting,
            }
        prior_status_path = submission_path.parent / "status.json"
        prior_status = read_json(prior_status_path) if prior_status_path.exists() else None
        retry_id = uuid.uuid4().hex
        retry_path = submission_path.parent / "retry_history" / f"{retry_id}.json"
        timestamp = datetime.now(UTC).isoformat()
        retry = {
            "schema": 1,
            "retry_id": retry_id,
            "status": "retrying",
            "started_at": timestamp,
            "reason": reason,
            "labels": labels,
            "submission_order": order,
            "walltime_override": walltime,
            "cpus_override": cpus,
            "gpus_override": gpus,
            "memory_override": memory,
            "old_accounting": accounting,
            "dependency_verification": dependency_verification,
            "old_status_snapshot": prior_status,
            "verification_commands": [command, queue_command],
            "jobs": [],
        }
        submitted.setdefault("retry_history", []).append(
            {"retry_id": retry_id, "path": str(retry_path)}
        )
        submitted["status"] = "retrying"

        def save():
            for path, data in ((retry_path, retry), (submission_path, submitted)):
                temporary = path.with_suffix(".tmp")
                write_json(temporary, data)
                temporary.replace(path)

        # Snapshot all selected old attempts before changing any current ID.
        for job in selected:
            previous = copy.deepcopy(
                {key: value for key, value in job.items() if key != "previous_attempts"}
            )
            previous.update(
                retried_at=timestamp,
                retry_id=retry_id,
                accounting_at_retry=accounting[job["job_id"]],
            )
            job.setdefault("previous_attempts", []).append(previous)
            job.update(status="retry_pending", job_id=None, retry_id=retry_id)
        save()
        try:
            for label in order:
                job = by_label[label]
                if any(by_label[parent]["status"] != "submitted" for parent in job["dependencies"]):
                    job["status"] = "blocked"
                    retry["jobs"].append({"label": label, "status": "blocked"})
                    save()
                    continue
                command = ["sbatch", "--parsable"]
                if walltime:
                    command.append("--time=" + walltime)
                if cpus is not None:
                    command.append("--cpus-per-task=" + str(cpus))
                if memory is not None:
                    command.append("--mem=" + memory)
                gpu_override = gpus if job["resources"].get("gpus") else None
                if gpu_override is not None:
                    command.append("--gpus=" + gpu_override)
                parents = [
                    by_label[parent]["job_id"]
                    for parent in job["dependencies"]
                    if parent in labels or dependency_verification[parent]["action"] == "afterok"
                ]
                if parents:
                    command.append("--dependency=afterok:" + ":".join(parents))
                command.append(job["script"])
                attempt = {"label": label, "status": "submitting", "command": command}
                retry["jobs"].append(attempt)
                job.update(
                    status="retry_submitting",
                    command=command,
                    walltime_override=walltime,
                    cpus_override=cpus,
                    gpus_override=gpu_override,
                    memory_override=memory,
                )
                job["dependency_verification"] = {
                    parent: dependency_verification[parent]
                    for parent in job["dependencies"]
                    if parent in dependency_verification
                }
                save()  # Crash or transport interruption after this point is ambiguous.
                try:
                    result = runner(command, capture_output=True, text=True, check=False)
                except BaseException as exc:
                    attempt.update(status="unknown", error_type=type(exc).__name__, error=str(exc))
                    job["status"] = "retry_unknown"
                    raise
                job_id = result.stdout.strip().split(";", 1)[0]
                attempt.update(
                    stdout=result.stdout, stderr=result.stderr, returncode=result.returncode
                )
                job.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
                if result.returncode == 0 and re.fullmatch(r"\d+", job_id):
                    attempt.update(status="submitted", job_id=job_id)
                    job.update(status="submitted", job_id=job_id)
                else:
                    # A zero exit with unparseable output may have created a job.
                    state = "retry_unknown" if result.returncode == 0 else "submission_failed"
                    attempt["status"] = job["status"] = state
                save()
            retry["status"] = (
                "submitted"
                if all(by_label[label]["status"] == "submitted" for label in labels)
                else "incomplete"
            )
            submitted["status"] = (
                "submitted"
                if all(job["status"] == "submitted" for job in submitted["jobs"])
                else "incomplete"
            )
        finally:
            retry["finished_at"] = datetime.now(UTC).isoformat()
            save()
        return retry_path
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def collect_status(
    submission_path: Path, *, destination: Path | None = None, runner=subprocess.run
) -> Path:
    """Read Slurm accounting and stage journals without changing any job."""
    submission_path = Path(submission_path)
    submitted = read_json(submission_path)
    ids = [job["job_id"] for job in submitted["jobs"] if job.get("job_id")]
    accounting = {}
    queued = {}
    commands = []
    fields = [
        "JobIDRaw",
        "State",
        "ExitCode",
        "ElapsedRaw",
        "Start",
        "End",
        "AllocCPUS",
        "ReqMem",
        "NodeList",
        "Reason",
        "AllocTRES",
        "CPUTimeRAW",
        "TotalCPU",
        "MaxRSS",
    ]
    if ids:
        command = [
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(ids),
            "--format",
            ",".join(fields),
        ]
        response = runner(command, capture_output=True, text=True, check=False)
        commands.append(
            {"command": command, "returncode": response.returncode, "stderr": response.stderr}
        )
        if response.returncode == 0:
            for line in response.stdout.splitlines():
                values = line.split("|")
                if len(values) == len(fields) and values[0] in ids:
                    accounting[values[0]] = dict(zip(fields, values, strict=True))
        command = ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i|%T|%R|%M"]
        response = runner(command, capture_output=True, text=True, check=False)
        commands.append(
            {"command": command, "returncode": response.returncode, "stderr": response.stderr}
        )
        if response.returncode == 0:
            for line in response.stdout.splitlines():
                values = line.split("|", 3)
                if len(values) == 4 and values[0] in ids:
                    queued[values[0]] = dict(
                        zip(("job_id", "state", "reason", "elapsed"), values, strict=True)
                    )
    terminal = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
        "SPECIAL_EXIT",
    }
    plan = read_json(submission_path.parent / "plan.json")
    records = []
    for job in submitted["jobs"]:
        job_id = job.get("job_id")
        queue = queued.get(job_id, {})
        account = accounting.get(job_id, {})
        state = queue.get("state") or account.get("State") or "UNKNOWN"
        normalized = state.split(" ")[0].rstrip("+")
        journals = []
        for path in sorted(
            (Path(plan["remote_directory"]) / "job_status" / job["label"]).glob("*.json")
        ):
            record = read_json(path)
            if record.get("job_id") == job_id:
                journals.append(
                    {
                        "path": str(path),
                        "status": record["status"],
                        "seconds": record.get("seconds"),
                    }
                )
        records.append(
            {
                "label": job["label"],
                "kind": job["kind"],
                "corpus": job["corpus"],
                "model": job["model"],
                "job_id": job_id,
                "submission_status": job["status"],
                "state": normalized,
                "terminal": normalized in terminal
                or job["status"] in {"blocked", "submission_failed"},
                "queue": queue or None,
                "accounting": account or None,
                "stage_journals": journals,
                "blocked_dependency": queue.get("reason") == "DependencyNeverSatisfied",
                "previous_attempts": job.get("previous_attempts", []),
            }
        )
    data = {
        "schema": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "submission": str(submission_path),
        "all_terminal": bool(records) and all(record["terminal"] for record in records),
        "all_succeeded": bool(records)
        and all(record["state"] == "COMPLETED" for record in records),
        "jobs": records,
        "scheduler_queries": commands,
        "notes": "Read-only Slurm status. UNKNOWN may mean accounting has not appeared yet. GPU/CPU allocation and scheduler elapsed time are not measured model latency; original stage measurements remain in the benchmark artifacts. Monetary compute costs remain unknown.",
    }
    destination = destination or submission_path.parent / "status.json"
    temporary = destination.with_suffix(".tmp")
    write_json(temporary, data)
    temporary.replace(destination)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Explicit execution inside a scheduled allocation")
    run.add_argument("--config", required=True)
    run.add_argument(
        "--kind", choices=["prepare", "bm25", "model", "embed", "score"], required=True
    )
    run.add_argument("--corpus", required=True)
    run.add_argument("--model")
    run.add_argument("--journal-directory", type=Path, required=True)
    run.add_argument("--allow-blocked-settings", action="store_true")
    submit = sub.add_parser("submit", help="Explicitly submit an existing plan to Slurm")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--workers", type=int, default=8)
    retry = sub.add_parser(
        "retry", help="Explicitly replace selected terminal jobs, preserving history"
    )
    retry.add_argument("--submission", type=Path, required=True)
    retry.add_argument("--label", action="append", required=True)
    retry.add_argument("--time", dest="walltime")
    retry.add_argument("--cpus", type=int, help="Override CPUs per task for the selected retries")
    retry.add_argument("--gpus", help="Override GPU requests for selected GPU jobs only")
    retry.add_argument("--memory", help="Override host memory for the selected retries, e.g. 16G")
    retry.add_argument("--reason")
    status = sub.add_parser(
        "status", help="Read scheduler status and accounting; never changes jobs"
    )
    status.add_argument("--submission", type=Path, required=True)
    status.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "retry":
            path = retry_jobs(
                args.submission,
                args.label,
                walltime=args.walltime,
                cpus=args.cpus,
                gpus=args.gpus,
                memory=args.memory,
                reason=args.reason,
            )
            print(path)
            return 0 if read_json(path)["status"] == "submitted" else 2
        if args.command == "status":
            print(collect_status(args.submission, destination=args.destination))
            return 0
        if args.command == "submit":
            path = submit_plan(args.plan, workers=args.workers)
            print(path)
            return 0 if read_json(path)["status"] == "submitted" else 2
        path = run_job(
            load_config(args.config),
            args.corpus,
            args.kind,
            alias=args.model,
            journal_directory=args.journal_directory,
            allow_blocked_settings=args.allow_blocked_settings,
        )
        print(path)
        return 0
    except (ValidationError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
