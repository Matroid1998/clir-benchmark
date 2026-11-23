"""Submit and execute retrieval from verified generation artifacts on Narval."""

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from clir_benchmark.config import load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.fingerprints import cached_file_digest
from clir_benchmark.hyde import hyde_languages, hyde_path, load_hyde
from clir_benchmark.lexical import bm25_path
from clir_benchmark.pipeline import embed, embedding_path, evaluate, model_spec, retrieve, run_path
from clir_benchmark.translation import load_translations, translation_languages, translation_path
from clir_benchmark.util import read_json, write_json

PROJECT = Path(__file__).resolve().parents[1]
CAMPAIGN = PROJECT.parent
CONFIG = PROJECT / "configs/verified-20261120-transformed-retrieval.yaml"
STATE = CAMPAIGN / "job_status"
LOGS = CAMPAIGN / "logs"
BATCH = PROJECT / "scripts/submit_transformed_retrieval.sh"
METHODS = ("hyde", "translate")


def languages(config, method, corpus):
    select = hyde_languages if method == "hyde" else translation_languages
    return select(config, corpus)


def link_inputs(config):
    shared, generation = config.read_only_artifact_roots
    for source, names in (
        (shared, ("corpus", "relevance", "lexical", "embeddings/document")),
        (
            generation,
            (
                "translations",
                "hyde",
                "translation_requests",
                "hyde_requests",
                "translation_invocations",
                "hyde_invocations",
            ),
        ),
    ):
        for name in names:
            link = config.output_dir / name
            target = source / name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.is_symlink() and not link.exists():
                link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
            if not link.is_symlink() or link.resolve() != target:
                raise ValueError(f"Unexpected artifact link: {link}")


def validate_indexes(config):
    counts = {}
    for corpus in config.datasets:
        shared = read_json(config.corpus_dir(corpus) / "manifest.json")
        relevance = read_json(config.relevance_dir(corpus) / "manifest.json")
        if shared["corpus_fingerprint"] != relevance["corpus_fingerprint"]:
            raise ValueError(f"Corpus and relevance differ for {corpus}")
        lexical = bm25_path(config, corpus)
        if not (lexical / "index.sqlite").is_file() or not (lexical / "manifest.json").is_file():
            raise ValueError(f"BM25 index missing for {corpus}")
        for alias in config.models:
            document = embedding_path(config, corpus, model_spec(config, alias), "document")
            manifest = read_json(document / "manifest.json")
            if (
                manifest["rows"] != shared["chunk_count"]
                or manifest["input_fingerprint"] != shared["corpus_fingerprint"]
                or manifest["truncation"]["all"]["truncated"]
                or not (document / "vectors.npy").is_file()
                or not (document / "rows.parquet").is_file()
            ):
                raise ValueError(f"Document index differs for {corpus}/{alias}")
        counts[corpus] = {
            "questions": relevance["selected_query_count"],
            "chunks": shared["chunk_count"],
        }
    return counts


def verify_generation(config, method):
    generation = config.read_only_artifact_roots[1]
    receipt_dir = generation.parent / "transfer_receipts"
    receipt_prefix = "generate-hyde" if method == "hyde" else "translate"
    path_for, load, table = (
        (hyde_path, load_hyde, "passages.parquet")
        if method == "hyde"
        else (translation_path, load_translations, "queries.parquet")
    )
    if not list(receipt_dir.glob(f"{receipt_prefix}-*.sha256")):
        return None
    lanes = [
        (c, lang, path_for(config, c, lang))
        for c in config.datasets
        for lang in languages(config, method, c)
    ]
    keys = {
        (path / name).relative_to(config.output_dir).as_posix()
        for _, _, path in lanes
        for name in ("manifest.json", table)
    }
    for receipt in sorted(receipt_dir.glob(f"{receipt_prefix}-*.sha256")):
        expected = dict(
            (name, digest)
            for digest, name in (
                line.split(maxsplit=1) for line in receipt.read_text().splitlines()
            )
        )
        if keys <= expected.keys():
            break
    else:
        return None
    for name in sorted(keys):
        if cached_file_digest(config.output_dir / name) != expected[name]:
            raise ValueError(f"Published file differs from the Narval transfer receipt: {name}")
    counts = {}
    for corpus, lang, _ in lanes:
        _, records, _ = load(config, corpus, lang)
        counts[f"{corpus}/{lang}"] = {
            "rows": len(records),
            "refusals": sum(row.get("status") == "provider_refused" for row in records),
        }
    return {"receipt": receipt.name, "lanes": counts}


def run_job(config, stage, method, corpus, ledger):
    if stage == "check":
        for _ in range(432):
            verified = verify_generation(config, method)
            if verified is not None:
                write_json(STATE / f"verified-{method}.json", verified)
                print(json.dumps(verified, indent=2))
                return
            time.sleep(60)
        raise ValueError(f"No verified Narval transfer arrived for {method}")

    verified = read_json(STATE / f"verified-{method}.json")
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    slot = read_json(ledger)[index]
    if stage.startswith("stacked-"):
        stage = stage.removeprefix("stacked-")
        failures = []
        status_file = STATE / f"{ledger.stem}-{os.environ['SLURM_ARRAY_JOB_ID']}_{index}.json"
        status = {}
        for language, alias in slot:
            key = f"{language}/{alias}"
            try:
                print(f"{method}-{stage} {corpus}/{key}", flush=True)
                run_lane(config, stage, method, corpus, language, alias, verified)
                status[key] = "completed"
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"{method}-{stage} {corpus}/{key} failed: {exc}", flush=True)
                status[key] = str(exc)
                failures.append(key)
            write_json(status_file, status)
        if failures:
            raise ValueError(f"Failed lanes: {', '.join(failures)}. See {status_file}")
        return
    language, alias = slot
    run_lane(config, stage, method, corpus, language, alias, verified)


def run_lane(config, stage, method, corpus, language, alias, verified):
    if f"{corpus}/{language}" not in verified["lanes"]:
        raise ValueError(f"Unverified generation lane {corpus}/{language}")
    options = {"hyde_language" if method == "hyde" else "translation_language": language}
    if stage == "embed":
        print(embed(config, corpus, alias, kind="query", **options))
        return
    baseline = f"{method}-{stage}"
    if stage != "bm25":
        ranking = run_path(config, corpus, model_spec(config, alias), baseline=baseline, **options)
        report = config.output_dir / "reports" / corpus / ranking.name
        if (report / "analysis_manifest.json").is_file():
            print(report)
            return
    print(retrieve(config, corpus, alias, baseline=baseline, **options))
    if stage != "bm25":
        for attempt in range(3):
            try:
                print(evaluate(config, corpus, alias, baseline=baseline, **options))
                return
            except ValidationError as exc:
                if "File changed while hashing:" not in str(exc) or attempt == 2:
                    raise
                print(f"Retrying report verification: {exc}", flush=True)
                time.sleep(10)


def retire_pending_arrays(arrays):
    """Remove queued tasks while retaining running tasks and their artifacts."""
    if not arrays:
        return
    job_ids = ",".join(sorted(arrays))
    check_pending = [
        "squeue",
        "--noheader",
        "--array",
        f"--jobs={job_ids}",
        "--states=PENDING",
        "--format=%i",
    ]
    for _ in range(3):
        queue = subprocess.check_output(check_pending, text=True).split()
        if not queue:
            return
        if any(
            not task.partition("_")[2].isdigit() or task.partition("_")[0] not in arrays
            for task in queue
        ):
            raise ValueError(f"Cannot identify individual pending tasks: {queue}")
        print(f"Retiring {len(queue)} pending fine-grained translation tasks", flush=True)
        subprocess.run(["scancel", "--quiet", "--state=PENDING", *queue], check=False)
        time.sleep(2)
    if subprocess.check_output(check_pending, text=True).strip():
        raise ValueError("Some fine-grained tasks are still pending after cancellation")


def submit(config, methods, *, retry_cancelled=False, stack_translation=False):
    from collections import defaultdict

    receipts = {}
    submitted = STATE / "submissions.tsv"
    if submitted.exists():
        for row in submitted.read_text().splitlines():
            key, job, _, _ = row.split("\t")
            receipts[key] = job

    def job(kind, method, corpus, tasks=(), dependency=""):
        key = "-".join(filter(None, (method, corpus, kind)))
        previous = receipts.get(key)
        if previous is not None:
            if not retry_cancelled:
                return previous
            accounting = subprocess.check_output(
                ["sacct", "-XnP", "-j", previous, "-o", "State%40"], text=True
            )
            states = {line.strip().split()[0] for line in accounting.splitlines() if line.strip()}
            if not states:
                raise ValueError(f"No Slurm accounting for previous job {previous} ({key})")
            if states != {"CANCELLED"}:
                return previous
        task_path = STATE / f"{key}.json"
        if tasks:
            write_json(task_path, tasks)
        cpu = "--account=def-mahyarh_cpu"
        resources = {
            "check": [cpu, "--cpus-per-task=1", "--mem=3G", "--time=08:00:00"],
            "embed": [
                "--account=def-mahyarh_gpu",
                "--gpus=a100:1",
                "--cpus-per-task=4",
                "--mem=32G",
                "--time=03:00:00",
            ],
            "dense": [cpu, "--cpus-per-task=8", "--mem=48G", "--time=08:00:00"],
            "bm25": [cpu, "--cpus-per-task=1", "--mem=32G", "--tmp=160G", "--time=7-00:00:00"],
            "hybrid": [cpu, "--cpus-per-task=2", "--mem=24G", "--time=12:00:00"],
        }
        stage = kind.removeprefix("stacked-")
        spec = resources[stage]
        if kind != stage:
            spec = [item for item in spec if not item.startswith("--time=")]
            spec.append(
                {
                    "embed": "--time=1-00:00:00",
                    "dense": "--time=3-00:00:00",
                    "hybrid": "--time=2-00:00:00",
                }[stage]
            )
        pattern = "%j"
        if tasks:
            limit = {"embed": 8, "dense": 12, "bm25": 8, "hybrid": 12}[stage]
            spec.append(f"--array=0-{len(tasks) - 1}%{limit}")
            pattern = "%A_%a"
        if dependency:
            spec.extend([f"--dependency={dependency}", "--kill-on-invalid-dep=yes"])
        result = (
            subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    f"--job-name=clir-{key}",
                    f"--output={LOGS}/{key}-{pattern}.out",
                    f"--error={LOGS}/{key}-{pattern}.err",
                    *spec,
                    "--wrap="
                    + shlex.join(
                        ["bash", str(BATCH), "worker", kind, method, corpus, str(task_path)]
                    ),
                ],
                text=True,
            )
            .strip()
            .split(";")[0]
        )
        if not result.isdecimal():
            raise ValueError(f"Unexpected sbatch output: {result}")
        with submitted.open("a") as out:
            print(key, result, len(tasks), dependency, sep="\t", file=out)
        print(key, result, len(tasks), dependency, sep="\t", flush=True)
        receipts[key] = result
        return result

    total = defaultdict(int)
    for method in methods:
        verified = verify_generation(config, method)
        if verified is None:
            if retry_cancelled or stack_translation:
                raise ValueError(f"Verified transfer receipt required before retrying {method}")
            gate = f"afterok:{job('check', method, '')}"
        else:
            temporary = STATE / f".verified-{method}.tmp"
            write_json(temporary, verified)
            temporary.replace(STATE / f"verified-{method}.json")
            gate = ""
        if stack_translation:
            originals = {
                (corpus, stage): receipts[f"{method}-{corpus}-{stage}"]
                for corpus in config.datasets
                for stage in ("embed", "dense", "hybrid")
                if f"{method}-{corpus}-{stage}" in receipts
            }
            retire_pending_arrays(set(originals.values()))
            for corpus in config.datasets:
                lane_languages = languages(config, method, corpus)
                lexical = job("bm25", method, corpus, [(lang, None) for lang in lane_languages])

                def tasks(size, target_languages=lane_languages):
                    return [
                        [
                            (lang, model)
                            for model in config.models[i : i + size]
                            for lang in target_languages
                        ]
                        for i in range(0, len(config.models), size)
                    ]

                def after(*parents):
                    return "afterany:" + ":".join(filter(None, parents)) if any(parents) else ""

                encoding = job(
                    "stacked-embed",
                    method,
                    corpus,
                    tasks(4),
                    after(originals.get((corpus, "embed"))),
                )
                dense = job(
                    "stacked-dense",
                    method,
                    corpus,
                    tasks(2),
                    after(encoding, originals.get((corpus, "dense"))),
                )
                job(
                    "stacked-hybrid",
                    method,
                    corpus,
                    tasks(2),
                    after(dense, lexical, originals.get((corpus, "hybrid"))),
                )
                total[method] += 2 * len(lane_languages) * len(config.models)
            continue
        for corpus in config.datasets:
            lane_languages = languages(config, method, corpus)
            tasks = [(lang, model) for lang in lane_languages for model in config.models]
            lexical = job("bm25", method, corpus, [(lang, None) for lang in lane_languages], gate)
            encoding = job("embed", method, corpus, tasks, gate)
            dense = job("dense", method, corpus, tasks, f"aftercorr:{encoding}")
            job("hybrid", method, corpus, tasks, f"aftercorr:{dense},afterok:{lexical}")
            total[method] += 2 * len(tasks)
    print("Planned evaluated retrieval runs:", dict(total), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "submit", "worker"))
    parser.add_argument("worker_args", nargs="*")
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--stack-translation", action="store_true")
    parser.add_argument(
        "--retry-cancelled",
        action="store_true",
        help="Resubmit arrays whose prior attempt was wholly cancelled",
    )
    args = parser.parse_args()
    if args.retry_cancelled and args.action != "submit":
        parser.error("--retry-cancelled requires submit")
    config = load_config(CONFIG)
    methods = args.method or METHODS
    if args.stack_translation and (args.action != "submit" or methods != ["translate"]):
        parser.error("--stack-translation requires submit --method translate")
    if args.stack_translation and args.retry_cancelled:
        parser.error("Use one of --stack-translation and --retry-cancelled")
    if args.action == "worker":
        kind, method, corpus, ledger = args.worker_args
        run_job(config, kind, method, corpus, Path(ledger))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (STATE / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        link_inputs(config)
        indexes = validate_indexes(config)
        if args.action == "preflight":
            print(
                json.dumps(
                    {
                        "indexes": indexes,
                        **{method: verify_generation(config, method) for method in methods},
                    },
                    indent=2,
                )
            )
            return
        submit(
            config,
            methods,
            retry_cancelled=args.retry_cancelled,
            stack_translation=args.stack_translation,
        )


if __name__ == "__main__":
    main()
