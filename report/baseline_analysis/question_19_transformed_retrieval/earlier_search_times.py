"""Export the November 20 original-query and HyDE search timings on Narval.

Run with the benchmark Python environment and the campaign runs directory as
the only argument. The CSV on stdout uses Toronto time. Timed search seconds
sum the batch-amortized values over the queries actually searched.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def export(runs):
    jobs = {}
    campaigns = {
        "original": runs / "narval-20261120-verified-retrieval",
        "hyde": runs / "narval-20261120-transformed-retrieval",
    }
    for variant, campaign in campaigns.items():
        for retriever in ("bm25", "dense"):
            pattern = f"{retriever}-*.out" if variant == "original" else f"hyde-*-{retriever}-*.out"
            for log in sorted((campaign / "logs").glob(pattern)):
                job_id = log.stem.rsplit("-", 1)[1]
                jobs[job_id] = (variant, retriever, log)

    fields = "JobID,State,AllocCPUS,Submit,Start,End,ElapsedRaw,TotalCPU"
    array_ids = sorted({job.split("_", 1)[0] for job in jobs})
    accounting = subprocess.check_output(
        ["sacct", "-nP", "-j", ",".join(array_ids), "-o", fields],
        env={**os.environ, "TZ": "America/Toronto"},
        text=True,
    )
    records = {row[0]: row[1:] for row in csv.reader(accounting.splitlines(), delimiter="|") if row}
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow([
        "job_id", "query_variant", "retriever", "corpus", "model", "state", "cpus",
        "submitted_toronto", "started_toronto", "finished_toronto", "elapsed_seconds",
        "total_cpu_time", "eligible_searches", "executed_searches", "timed_search_seconds", "run_id",
    ])
    for job, (variant, retriever, log) in jobs.items():
        run = next(
            Path(line) for line in log.read_text().splitlines()
            if line.startswith("/") and "/artifacts/runs/" in line
        )
        manifest = json.loads((run / "manifest.json").read_text())
        column = "retrieval_seconds_amortized"
        times = pq.read_table(run / "timings.parquet", columns=[column]).column(column)
        writer.writerow([
            job, variant, retriever, manifest["corpus"], (manifest.get("model") or {}).get("alias"),
            *records[job], len(times), len(times) - times.null_count,
            round(pc.sum(times).as_py(), 3), run.name,
        ])


if __name__ == "__main__":
    export(Path(sys.argv[1]))
