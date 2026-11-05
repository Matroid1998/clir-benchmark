from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import EMBED_ROW_SCHEMA, ParquetSink, rows
from .config import RuntimeConfig
from .encoders import Encoder, ModelSpec, normalize
from .errors import ValidationError
from .util import digest, json_dumps, new_directory, read_json, versions, write_json


def embedding_key(
    input_fingerprint: str, kind: str, spec: ModelSpec, runtime: RuntimeConfig
) -> str:
    # No qrels, answer/evidence strings, or annotation hashes enter this key.
    return digest(
        {
            "schema": 1,
            "input": input_fingerprint,
            "kind": kind,
            "model": asdict(spec),
            "device": runtime.device,
            "dtype": runtime.dtype,
            "batch_size": runtime.batch_size,
            "versions": versions(),
        }
    )


def encode_records(
    destination: Path,
    records: Callable[[], Iterable[dict]],
    count: int,
    encoder: Encoder,
    kind: str,
    runtime: RuntimeConfig,
    input_fingerprint: str,
) -> Path:
    if destination.exists():
        manifest = read_json(destination / "manifest.json")
        if (
            manifest["input_fingerprint"] != input_fingerprint
            or manifest["kind"] != kind
            or manifest["model"] != asdict(encoder.spec)
        ):
            raise ValidationError(f"Embedding cache identity mismatch: {destination}")
        return destination
    started = time.perf_counter()
    with new_directory(destination) as temporary:
        vectors = np.lib.format.open_memmap(
            temporary / "vectors.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, encoder.spec.dimension),
        )
        counts = defaultdict(Counter)
        position = 0
        with ParquetSink(temporary / "rows.parquet", EMBED_ROW_SCHEMA) as sink:
            batch = []

            def flush() -> None:
                nonlocal position
                if not batch:
                    return
                encoded = encoder.encode([row["text"] for row in batch], kind)
                if encoded.vectors.shape != (len(batch), encoder.spec.dimension) or len(
                    encoded.audits
                ) != len(batch):
                    raise ValidationError("Encoder returned an inconsistent batch")
                if position + len(batch) > count:
                    raise ValidationError("Input has more rows than declared")
                vectors[position : position + len(batch)] = normalize(encoded.vectors)
                for row, audit in zip(batch, encoded.audits, strict=True):
                    sink.append(
                        {
                            "id": row["id"],
                            "language": row["language"],
                            "source_unit_id": row.get("source_unit_id", ""),
                            "source_start": row.get("start", 0),
                            "input_tokens": audit.input_tokens,
                            "encoded_tokens": audit.encoded_tokens,
                            "truncated": audit.truncated,
                            "visible_ranges_json": json_dumps(audit.visible_ranges),
                        }
                    )
                    for language in ("all", row["language"]):
                        counts[language]["inputs"] += 1
                        counts[language]["truncated"] += int(audit.truncated)
                        counts[language]["input_tokens"] += audit.input_tokens
                        counts[language]["encoded_tokens"] += audit.encoded_tokens
                position += len(batch)
                batch.clear()

            for row in records():
                batch.append(row)
                if len(batch) >= runtime.batch_size:
                    flush()
            flush()
        if position != count:
            raise ValidationError(f"Expected {count} embedding inputs but received {position}")
        vectors.flush()
        vectors._mmap.close()
        summary = {
            language: {**values, "truncation_rate": values["truncated"] / values["inputs"]}
            for language, values in counts.items()
        }
        write_json(
            temporary / "manifest.json",
            {
                "schema": 1,
                "input_fingerprint": input_fingerprint,
                "kind": kind,
                "model": asdict(encoder.spec),
                "encoder": encoder.metadata,
                "rows": count,
                "dimension": encoder.spec.dimension,
                "runtime": asdict(runtime),
                "versions": versions(),
                "seconds": time.perf_counter() - started,
                "truncation": summary,
            },
        )
    return destination


def load_embeddings(
    path: Path, columns: list[str] | None = None
) -> tuple[np.ndarray, list[dict], dict]:
    if not (path / "manifest.json").is_file():
        raise ValidationError(f"Embeddings are not prepared: {path}. Run embed first.")
    manifest = read_json(path / "manifest.json")
    matrix = np.load(path / "vectors.npy", mmap_mode="r", allow_pickle=False)
    metadata = list(rows(path / "rows.parquet", columns))
    if (
        matrix.shape != (manifest["rows"], manifest["dimension"])
        or len(metadata) != manifest["rows"]
    ):
        raise ValidationError(f"Corrupt embedding cache shape: {path}")
    if len({row["id"] for row in metadata}) != len(metadata):
        raise ValidationError(f"Duplicate embedding IDs: {path}")
    return matrix, metadata, manifest


def span_visible(match: dict, embedding_row: dict) -> bool:
    import json

    start = match["start"] - embedding_row["source_start"]
    end = match["end"] - embedding_row["source_start"]
    return any(
        left <= start and end <= right
        for left, right in json.loads(embedding_row["visible_ranges_json"])
    )


def support_exposure(matches: Iterable[dict], embedding_rows: list[dict]) -> dict:
    indexed = {row["id"]: row for row in embedding_rows}
    visible = {}
    for match in matches:
        key = (
            match["setting_id"],
            match["query_id"],
            match["chunk_id"],
            match["required_source_id"],
        )
        if match["chunk_id"] not in indexed:
            raise ValidationError(f"Missing embedding for positive chunk {match['chunk_id']}")
        visible[key] = visible.get(key, False) or span_visible(match, indexed[match["chunk_id"]])
    by_setting = defaultdict(Counter)
    query_groups = defaultdict(dict)
    chunk_visibility = {}
    for (setting, query_id, chunk_id, group), survives in visible.items():
        chunk_key = (setting, query_id, chunk_id)
        chunk_visibility[chunk_key] = chunk_visibility.get(chunk_key, False) or survives
        query_groups[(setting, query_id)][group] = (
            query_groups[(setting, query_id)].get(group, False) or survives
        )
    for (setting, _, _), survives in chunk_visibility.items():
        by_setting[setting]["positive_query_chunk_pairs"] += 1
        by_setting[setting]["positive_query_chunk_pairs_without_visible_span"] += int(not survives)
    for (setting, _), groups in query_groups.items():
        by_setting[setting]["queries"] += 1
        by_setting[setting]["queries_with_all_sources_visible"] += int(all(groups.values()))
    return dict(by_setting)
