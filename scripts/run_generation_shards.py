"""Run local query-generation workers and publish their offline, verified aggregates."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from clir_benchmark.config import load_config
from clir_benchmark.generation import with_refusal_policy
from clir_benchmark.hyde import assemble_hyde, hyde_languages, hyde_plan
from clir_benchmark.translation import (
    assemble_translations,
    translation_languages,
    translation_plan,
)
from clir_benchmark.util import digest, file_digest, write_json


def campaign(config, stages, size):
    lanes = []
    for stage in stages:
        translate = stage == "translate"
        languages = translation_languages if translate else hyde_languages
        plan = translation_plan if translate else hyde_plan
        for corpus in config.datasets:
            for language in languages(config, corpus):
                lane = plan(config, corpus, language)
                outputs = [lane] if translate else lane["outputs"]
                variants = sum(item["variants"] for item in outputs)
                lanes.append(
                    {
                        "stage": stage,
                        "corpus": corpus,
                        "language": language,
                        "artifact": lane["artifact"],
                        "question_count": variants,
                        "num_shards": (variants + size - 1) // size,
                        "planned_calls": sum(
                            item["requests_before_cache_or_retries"] for item in outputs
                        ),
                        "input_fingerprint": lane.get("input_fingerprint"),
                        "prompt_fingerprint": lane.get("prompt_fingerprint"),
                    }
                )
    return {
        "shard_size": size,
        "lanes": lanes,
        "shards": sum(lane["num_shards"] for lane in lanes),
        "outputs": sum(lane["question_count"] for lane in lanes),
        "calls_before_cache_and_retries": sum(lane["planned_calls"] for lane in lanes),
    }


def run_shard(options, config_hash, directory, lane, index):
    if file_digest(options.config) != config_hash:
        raise RuntimeError("Generation configuration changed after planning")
    canonical = Path(lane["artifact"])
    shard_dir = canonical.parent / "shards" / canonical.name / str(options.shard_size) / str(index)
    if (canonical / "manifest.json").is_file() or (shard_dir / "manifest.json").is_file():
        return  # The offline assembly step verifies each existing publication.
    stage, corpus, language = lane["stage"], lane["corpus"], lane["language"]
    selector = "--translation-language" if stage == "translate" else "--hyde-language"
    command = [
        sys.executable,
        "-m",
        "clir_benchmark",
        "--config",
        str(options.config),
        stage,
        "--corpus",
        corpus,
        selector,
        language,
        "--shard-index",
        str(index),
        "--shard-size",
        str(options.shard_size),
    ]
    if options.retry_failed:
        command.append("--retry-failed")
    if options.retry_transient:
        command.append("--retry-transient")
    if options.record_refusals:
        command.append("--record-refusals")
    log = directory / f"{stage}-{corpus}-{language}-{index:04d}.log"
    with log.open("a") as handle:
        print(datetime.now(UTC).isoformat(), "starting", flush=True, file=handle)
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        print(
            datetime.now(UTC).isoformat(),
            "exit",
            result.returncode,
            flush=True,
            file=handle,
        )
    if result.returncode:
        raise RuntimeError(f"Generation worker exited {result.returncode}. See {log}")


def run_workers(options, config_hash, directory, plan):
    work = iter((lane, i) for lane in plan["lanes"] for i in range(lane["num_shards"]))
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        active = set()
        while True:
            while len(active) < options.workers and (item := next(work, None)) is not None:
                active.add(pool.submit(run_shard, options, config_hash, directory, *item))
            if not active:
                break
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="Campaign plans, verification receipts, and per-shard logs",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument(
        "--record-refusals",
        action="store_true",
        help="Record provider refusals as terminal missing outputs",
    )
    parser.add_argument(
        "--retry-transient",
        action="store_true",
        help="Permit up to three attempts per request for retryable errors",
    )
    parser.add_argument(
        "--stage",
        choices=("translate", "generate-hyde"),
        action="append",
        help="Repeat to select stages; defaults to both",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Permit a new attempt on a previously failed or interrupted request",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--plan", action="store_true", help="Count the work without sending requests"
    )
    action.add_argument(
        "--run", action="store_true", help="Execute generation workers, then assemble"
    )
    action.add_argument(
        "--assemble", action="store_true", help="Verify and assemble saved shards offline"
    )
    options = parser.parse_args()
    if (
        options.shard_size < 1
        or options.workers < 1
        or (options.retry_failed or options.retry_transient)
        and not options.run
    ):
        parser.error("Use positive shard size and worker count, and --retry-failed only with --run")
    options.config = options.config.resolve()
    config_hash = file_digest(options.config)
    config = with_refusal_policy(load_config(options.config), options.record_refusals)
    plan = campaign(
        config, dict.fromkeys(options.stage or ["translate", "generate-hyde"]), options.shard_size
    )
    directory = options.state_dir.resolve() / digest(plan)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(directory / "plan.json", plan)
        print(
            json.dumps(
                {"directory": str(directory), **{k: v for k, v in plan.items() if k != "lanes"}},
                indent=2,
            ),
            flush=True,
        )
        if options.plan:
            return
        if options.run:
            run_workers(options, config_hash, directory, plan)
        verified = []
        for lane in plan["lanes"]:
            assemble = assemble_translations if lane["stage"] == "translate" else assemble_hyde
            path = assemble(config, lane["corpus"], lane["language"], options.shard_size)
            if str(path) != lane["artifact"]:
                raise RuntimeError("Cohort or model configuration changed since planning")
            verified.append(
                {
                    "stage": lane["stage"],
                    "path": str(path),
                    "manifest_sha256": file_digest(path / "manifest.json"),
                }
            )
        receipt = directory / f"verified-{os.getpid()}.tmp"
        write_json(receipt, verified)
        receipt.replace(directory / "verified.json")
        print(f"Verified all {len(verified)} complete artifacts.", flush=True)


if __name__ == "__main__":
    main()
