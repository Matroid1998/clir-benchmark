"""Parallel source chunking with small, restartable on-disk boundary records."""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
import sys
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile

from .chunking import make_chunker, source_chunks
from .config import Config
from .types import Chunk, SourceUnit
from .util import digest, hf_token


def _source_batches(units: Iterable[SourceUnit]) -> Iterator[list[SourceUnit]]:
    # The caps bound both checkpoint frequency and interprocess transfers, including
    # corpora with very short patent titles and very long UN editions.
    batch = []
    characters = 0
    for unit in units:
        batch.append(unit)
        characters += len(unit.text)
        if len(batch) >= 256 or characters >= 256_000:
            yield batch
            batch, characters = [], 0
    if batch:
        yield batch


_worker_chunker = None


def _start_worker(config, chunker):
    global _worker_chunker
    # Give each process a CPU without also starting a Rayon thread pool per tokenizer.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _worker_chunker = chunker or make_chunker(config, hf_token(config.env_file))


def _write_boundaries(path: Path, units: list[SourceUnit]) -> None:
    boundaries = [
        [[c.start, c.end, c.token_count, c.overlap_tokens] for c in _worker_chunker.split(unit)]
        for unit in units
    ]
    # Atomic replacement allows colliding identical jobs and prevents a preempted
    # job from leaving a partial checkpoint that a retry might mistake for a hit.
    with NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            json.dump(boundaries, handle, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _batch_identity(units: list[SourceUnit]) -> str:
    return digest(
        [
            (
                u.id,
                u.corpus,
                u.source_id,
                u.parent_id,
                u.language,
                hashlib.sha256(u.text.encode()).hexdigest(),
            )
            for u in units
        ]
    )


def checkpointed_chunks(
    units: Iterable[SourceUnit],
    config: Config,
    chunker,
    fingerprint: str,
    directory: Path,
    workers: int,
) -> Iterator[tuple[SourceUnit, list[Chunk]]]:
    directory.mkdir(parents=True, exist_ok=True)
    batches = enumerate(_source_batches(units), 1)
    # Spawn isolated processes before loading their tokenizers. Forking after
    # Arrow, ICU or the tokenizer runtime starts threads can deadlock on HPC nodes.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(config, chunker),
    ) as pool:

        def schedule(item):
            number, batch = item
            path = directory / f"{number:06d}-{_batch_identity(batch)}.json.gz"
            future = None if path.is_file() else pool.submit(_write_boundaries, path, batch)
            return number, batch, path, future

        pending = deque(schedule(item) for item in islice(batches, workers * 2))
        reused = editions = 0
        while pending:
            number, batch, path, future = pending.popleft()
            if future is not None:
                future.result()
            else:
                reused += 1
            item = next(batches, None)
            if item is not None:
                pending.append(schedule(item))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                bounds = json.load(handle)
            for unit, values in zip(batch, bounds, strict=True):
                # Replay the same ID/offset constructor as the serial chunker.
                windows = [(start, end, count) for start, end, count, _ in values]
                overlaps = iter(value[3] for value in values)
                yield (
                    unit,
                    source_chunks(
                        unit, windows, fingerprint, lambda _, counts=overlaps: next(counts)
                    ),
                )
            editions += len(batch)
            if number == 1 or number % 25 == 0 or not pending:
                print(
                    f"{editions:,} editions assembled, {number:,} boundary batches, "
                    f"{reused:,} reused",
                    file=sys.stderr,
                    flush=True,
                )
