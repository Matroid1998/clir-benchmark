"""Run opt-in reranking of saved original-query candidates on the verified cohort."""

import argparse
import fcntl
import logging
import os
import shlex
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from clir_benchmark.artifacts import rows
from clir_benchmark.config import load_config
from clir_benchmark.errors import ValidationError
from clir_benchmark.pipeline import evaluate, retrieval_spec, retrieve, run_path
from clir_benchmark.rerank_pipeline import compare_rerank, rerank
from clir_benchmark.util import file_digest, read_json, write_json

PROJECT = Path(__file__).resolve().parents[1]
CAMPAIGN = PROJECT.parent
STATE = CAMPAIGN / "job_status"
LOGS = CAMPAIGN / "logs"
BATCH = PROJECT / "scripts/submit_verified_rerank.sh"
CONFIG = PROJECT / "configs/verified-20261124-rerank.yaml"
PLAN = STATE / "plan.json"
MODES = ("rerank-bm25", "rerank-dense", "rerank-hybrid")


def write_receipt(path, value):
    tmp = path.with_suffix(".tmp")
    write_json(tmp, value)
    tmp.replace(path)


def lanes(config):
    result = []
    if "rerank-bm25" in config.baselines:
        result.append([(None, "rerank-bm25")])
    modes = [mode for mode in config.baselines if mode in ("rerank-dense", "rerank-hybrid")]
    if modes:
        result.extend([[(alias, mode) for mode in modes] for alias in config.models])
    return result


def parent_path(config, corpus, alias, parent):
    return run_path(config, corpus, retrieval_spec(config, alias, parent), baseline=parent)


def preflight(config):
    if not config.cache_hybrid_candidates or config.retrieval_unit != "chunk":
        raise ValidationError(
            "This campaign requires saved component candidates and chunk judgments"
        )
    if set(config.baselines) - set(MODES):
        raise ValidationError("This campaign supports reranking original-query candidates only")
    snapshot = Path(
        snapshot_download(
            config.reranker.repo_id, revision=config.reranker.revision, local_files_only=True
        )
    )
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (snapshot / name).is_file():
            raise ValidationError(f"Missing pinned reranker file: {snapshot / name}")
    result = {
        "baselines": list(config.baselines),
        "configuration_sha256": file_digest(CONFIG),
        "reranker": asdict(config.reranker),
        "snapshot": str(snapshot),
        "corpora": {},
    }
    for corpus in config.datasets:
        corpus_dir, relevance = config.corpus_dir(corpus), config.relevance_dir(corpus)
        metadata = read_json(relevance / "manifest.json")
        corpus_hash = read_json(corpus_dir / "manifest.json")["corpus_fingerprint"]
        ids = [row["id"] for row in rows(relevance / "queries.parquet", ["id"])]
        if len(set(ids)) != len(ids) or not ids:
            raise ValidationError(f"Invalid saved cohort for {corpus}")
        for path in (
            corpus_dir / "chunks.parquet",
            relevance / "qrels.parquet",
            relevance / "eligibility.parquet",
        ):
            if pq.ParquetFile(path).metadata.num_rows == 0:
                raise ValidationError(f"Empty benchmark input: {path}")
        groups = lanes(config)
        missing, parents = [], {}
        for group in groups:
            for alias, baseline in group:
                base = baseline.removeprefix("rerank-")
                path = parent_path(config, corpus, alias, base)
                key = f"{base}-{alias or 'bm25'}"
                if (path / "run.parquet").is_file():
                    manifest = read_json(path / "manifest.json")
                    spec = retrieval_spec(config, alias, base)
                    expected = {
                        "baseline": base,
                        "corpus_fingerprint": corpus_hash,
                        "query_fingerprint": metadata["query_fingerprint"],
                        "relevance_hash": config.relevance_key(corpus),
                        "model": asdict(spec) if spec else None,
                    }
                    if any(manifest.get(k) != v for k, v in expected.items()):
                        raise ValidationError(f"Parent configuration mismatch: {path}")
                    parents[key] = str(path)
                    continue
                if path.exists() or base != "hybrid":
                    raise ValidationError(f"Missing or incomplete saved {base} ranking: {path}")
                for component, model in (("bm25", None), ("dense", alias)):
                    cache = parent_path(config, corpus, model, component) / "candidates.parquet"
                    if not cache.is_file():
                        raise ValidationError(f"Missing candidates needed for fusion: {cache}")
                missing.append(alias)
        result["corpora"][corpus] = {
            "queries": len(ids),
            "relevance": str(relevance),
            "qrels_sha256": file_digest(relevance / "qrels.parquet"),
            "lanes": groups,
            "saved_parents": parents,
            "fuse_from_components": missing,
        }
        print(
            f"{corpus}: {len(ids)} queries, {len(parents)} saved parents, fuse {missing}",
            flush=True,
        )
    return result


def submit(config):
    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (STATE / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if PLAN.exists():
            plan = read_json(PLAN)
            if plan["configuration_sha256"] != file_digest(CONFIG) or plan["baselines"] != list(
                config.baselines
            ):
                raise ValidationError("Submission parameters differ from the saved campaign plan")
        else:
            plan = preflight(config)
            write_receipt(PLAN, plan)
        ledger = STATE / "submissions.tsv"
        jobs = (
            {r[0]: r[1] for r in (line.split("\t") for line in ledger.read_text().splitlines())}
            if ledger.exists()
            else {}
        )

        def job(key, stage, corpus="", indices=None, deps=()):
            if key in jobs:
                return jobs[key]
            args = ["sbatch", "--parsable", f"--job-name=rr-{key}", "--nodes=1", "--ntasks=1"]
            if stage == "score":
                args += [
                    "--account=def-mahyarh_gpu",
                    "--gpus=a100:1",
                    "--cpus-per-task=4",
                    "--mem=64G",
                    "--time=2-00:00:00",
                ]
            else:
                args += [
                    "--account=def-mahyarh_cpu",
                    "--cpus-per-task=4",
                    "--mem=48G" if stage != "figure" else "--mem=4G",
                    "--time=1-12:00:00"
                    if stage == "report"
                    else "--time=12:00:00"
                    if stage == "parents"
                    else "--time=00:30:00",
                ]
            pattern = "%j"
            if indices is not None:
                args.append(
                    "--array=" + ",".join(map(str, indices)) + ("%4" if stage == "score" else "")
                )
                pattern = "%A_%a"
            if deps:
                args += ["--dependency=afterok:" + ":".join(deps), "--kill-on-invalid-dep=yes"]
            command = ["bash", str(BATCH), stage] + ([corpus] if corpus else [])
            args += [
                f"--output={LOGS}/{key}-{pattern}.out",
                f"--error={LOGS}/{key}-{pattern}.err",
                "--wrap=" + shlex.join(command),
            ]
            identifier = subprocess.check_output(args, text=True).strip().split(";")[0]
            if not identifier.isdecimal():
                raise ValueError(f"Unexpected sbatch response: {identifier}")
            with ledger.open("a") as handle:
                print(
                    key,
                    identifier,
                    len(indices) if indices is not None else 1,
                    ",".join(deps),
                    sep="\t",
                    file=handle,
                )
            print(f"Submitted {key} {identifier}", flush=True)
            jobs[key] = identifier
            return identifier

        reports = []
        for corpus, spec in plan["corpora"].items():
            delayed = [
                i
                for i, group in enumerate(spec["lanes"])
                if any(
                    mode == "rerank-hybrid" and alias in spec["fuse_from_components"]
                    for alias, mode in group
                )
            ]
            ready = [i for i in range(len(spec["lanes"])) if i not in delayed]
            scores = []
            if ready:
                scores.append(job(corpus + "-score", "score", corpus, ready))
            if delayed:
                parent = job(corpus + "-fusion", "parents", corpus)
                scores.append(job(corpus + "-score-fused", "score", corpus, delayed, [parent]))
            shards = list(range((len(spec["lanes"]) + 2) // 3))
            reports.append(job(corpus + "-report", "report", corpus, shards, scores))
        job("table", "figure", deps=reports)


def checked_report(function, *args, **kwargs):
    for attempt in range(3):
        try:
            return function(*args, **kwargs)
        except ValidationError as exc:
            if attempt == 2 or "File changed while hashing:" not in str(exc):
                raise
            logging.warning("Retrying checksum verification: %s", exc)
            time.sleep(10)


def worker(stage, corpus):
    plan = read_json(PLAN)
    if file_digest(CONFIG) != plan["configuration_sha256"]:
        raise ValidationError("Campaign configuration changed after submission")
    config = replace(load_config(CONFIG), baselines=tuple(plan["baselines"]))
    spec = plan["corpora"][corpus]
    if stage == "parents":
        for alias in spec["fuse_from_components"]:
            print("Fuse saved candidates", corpus, alias, flush=True)
            path = retrieve(config, corpus, alias, baseline="hybrid")
            write_receipt(STATE / f"parent-{corpus}-{alias}.json", {"run": str(path)})
        return
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    groups = (
        spec["lanes"][index : index + 1]
        if stage == "score"
        else spec["lanes"][3 * index : 3 * index + 3]
    )
    for group in groups:
        for alias, baseline in group:
            key = f"{corpus}-{baseline}-{alias or 'lexical'}"
            receipt = STATE / f"{stage}-{key}.json"
            print(stage, corpus, baseline, alias, flush=True)
            if stage == "score":
                directory = rerank(config, corpus, alias, baseline=baseline)
                write_receipt(receipt, {"run": str(directory)})
            else:
                base = baseline.removeprefix("rerank-")
                parent = parent_path(config, corpus, alias, base)
                parent_report = config.output_dir / "reports" / corpus / parent.name
                if not (parent_report / "analysis_manifest.json").is_file():
                    checked_report(evaluate, config, corpus, alias, baseline=base)
                comparison = checked_report(
                    compare_rerank, config, corpus, alias, baseline=baseline
                )
                write_receipt(
                    receipt,
                    {
                        "corpus": corpus,
                        "model": alias,
                        "baseline": baseline,
                        "comparison": str(comparison),
                        "parent_run": str(parent),
                    },
                )
            print(receipt, flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "submit", "parents", "score", "report"))
    parser.add_argument("corpus", nargs="?")
    parser.add_argument("--baseline", action="append", choices=MODES)
    args = parser.parse_args()
    if args.stage in ("check", "submit"):
        cfg = load_config(CONFIG)
        if args.baseline:
            cfg = replace(cfg, baselines=tuple(dict.fromkeys(args.baseline)))
        if args.stage == "check":
            preflight(cfg)
        else:
            submit(cfg)
    else:
        worker(args.stage, args.corpus)
