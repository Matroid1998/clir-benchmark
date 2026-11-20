"""Read the verified BM25 workers' Linux wait channels and CPU time on their compute node."""

import json
import os
import socket
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def active_workers():
    workers = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            b"prepare-corpus" in args
            and b"--corpus" in args
            and any(a.endswith(b"/configs/verified-20261118-bm25.yaml") for a in args)
        ):
            corpus = args[args.index(b"--corpus") + 1].decode()
            workers[corpus] = proc
    return workers


def cpu_seconds(proc):
    fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def main():
    workers = active_workers()
    if not workers:
        raise SystemExit("No verified-release corpus preparation workers on this node")
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    before = {corpus: cpu_seconds(proc) for corpus, proc in workers.items()}
    channels = {corpus: Counter() for corpus in workers}
    clock = time.monotonic()
    for index in range(21):
        if index:
            time.sleep(1)
        for corpus, proc in workers.items():
            channels[corpus][(proc / "wchan").read_text().strip()] += 1
    elapsed = time.monotonic() - clock
    summary = {}
    for corpus, proc in workers.items():
        now = cpu_seconds(proc)
        paths = []
        for fd in (proc / "fd").iterdir():
            try:
                target = os.readlink(fd)
            except FileNotFoundError:
                continue
            if Path(target).name in {"lookup.sqlite", "lookup.sqlite-journal"}:
                paths.append(target)
        summary[corpus] = {
            "pid": int(proc.name),
            "wait_channel_samples": channels[corpus],
            "process_cpu_seconds": now,
            "cpu_seconds_during_sample": round(now - before[corpus], 2),
            "open_preparation_database_files": sorted(set(paths)),
        }
    print(
        json.dumps(
            {
                "sample_started": started,
                "node": socket.gethostname(),
                "sample_elapsed_seconds": round(elapsed, 2),
                "workers": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
