"""Extract mixed-pool ranking columns from the completed, selected-query reports.

This is an offline projection and validation step. It never executes retrieval,
loads embedding models, or calls a generation API. Original artifacts are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_COLUMNS = [
    "setting_id",
    "query_id",
    "rank",
    "score",
    "relevance",
    "document_id",
    "source_id",
    "document_language",
]
ANNOTATION_COLUMNS = ["corpus", "model", "variant", "method", "output_language"]
OUTPUT_SCHEMA = pa.schema(
    [(name, pa.string()) for name in ANNOTATION_COLUMNS]
    + [("setting_id", pa.string()), ("query_id", pa.string())]
    + [("rank", pa.int32()), ("score", pa.float64()), ("relevance", pa.int8())]
    + [
        ("document_id", pa.string()),
        ("source_id", pa.string()),
        ("document_language", pa.string()),
    ]
)


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    analysis = args.analysis_root.resolve()
    repo = args.repo_root.resolve()
    data = analysis / "data"
    catalog_path = data / "report_catalog.parquet"
    per_query_path = data / "per_query.parquet"
    qrels_path = data / "qrels.parquet"
    destination = data / "mixed_rankings.parquet"
    temporary = data / "mixed_rankings.building.parquet"
    metadata_path = data / "mixed_rankings_metadata.json"

    catalog = pq.read_table(catalog_path).to_pylist()
    catalog.sort(key=lambda row: (row["corpus"], row["model"], row["variant"]))
    configuration_keys = [tuple(row[name] for name in ("corpus", "model", "variant")) for row in catalog]
    if len(catalog) != 264 or len(set(configuration_keys)) != 264:
        raise ValueError("Expected 264 distinct selected-document report configurations")
    if any(row["retrieval_unit"] != "document" for row in catalog):
        raise ValueError("Chunk-level reports are outside this extraction scope")

    expected_columns = [
        "corpus", "model", "variant", "query_id", "setting_id",
        "mrr10", "ndcg10", "hit10", "recall10", "relevant_document_count",
    ]
    expected_table = pq.read_table(
        per_query_path, columns=expected_columns, filters=[("regime", "=", "mixed")]
    )
    expected = defaultdict(dict)
    for row in expected_table.to_pylist():
        key = tuple(row[name] for name in ("corpus", "model", "variant"))
        if row["query_id"] in expected[key]:
            raise ValueError(f"Duplicate mixed query configuration: {key}, {row['query_id']}")
        expected[key][row["query_id"]] = row
    if sum(map(len, expected.values())) != 638 * 8 * 11:
        raise ValueError("Expected 56,144 mixed query configurations")

    gold = defaultdict(set)
    mixed_settings = sorted({row["setting_id"] for row in expected_table.to_pylist()})
    for row in pq.read_table(
        qrels_path,
        columns=["corpus", "setting_id", "query_id", "document_id"],
        filters=[("setting_id", "in", mixed_settings)],
    ).to_pylist():
        gold[(row["corpus"], row["setting_id"], row["query_id"])].add(row["document_id"])

    began = time.perf_counter()
    sources = []
    row_counts = Counter()
    query_counts = Counter()
    rank_lengths = Counter()
    metric_max_error = {name: 0.0 for name in ("mrr10", "ndcg10", "hit10", "recall10")}
    discount = 1.0 / np.log2(np.arange(1, 11) + 1)
    total_rows = 0
    with pq.ParquetWriter(
        temporary, OUTPUT_SCHEMA, compression="zstd", compression_level=5, use_dictionary=True
    ) as writer:
        for position, record in enumerate(catalog, 1):
            key = tuple(record[name] for name in ("corpus", "model", "variant"))
            metrics = expected[key]
            report = Path(record["report_path"])
            report = report if report.is_absolute() else repo / report
            source = report / "analysis" / "rankings.parquet"
            source_stat = source.stat()
            settings = sorted({row["setting_id"] for row in metrics.values()})
            projected = pq.read_table(
                source, columns=SOURCE_COLUMNS, filters=[("setting_id", "in", settings)]
            ).sort_by([("query_id", "ascending"), ("rank", "ascending")])
            if projected.num_rows == 0:
                raise ValueError(f"No mixed rankings in {source}")

            query_ids = projected["query_id"].to_numpy()
            rank = projected["rank"].to_numpy()
            score = projected["score"].to_numpy()
            relevance = projected["relevance"].to_numpy()
            document_ids = projected["document_id"].to_numpy()
            setting_ids = projected["setting_id"].to_numpy()
            if not np.isfinite(score).all() or not np.isin(relevance, [0, 1]).all():
                raise ValueError(f"Nonfinite scores or nonbinary relevance in {source}")
            unique_queries, starts, lengths = np.unique(query_ids, return_index=True, return_counts=True)
            if set(unique_queries) != set(metrics):
                raise ValueError(f"Ranking and per-query cohorts differ in {source}")
            for query_id, start, length in zip(unique_queries, starts, lengths, strict=True):
                end = start + length
                measured = metrics[query_id]
                if length > 100 or not np.array_equal(rank[start:end], np.arange(1, length + 1)):
                    raise ValueError(f"Noncontiguous or duplicate rank for {key}, {query_id}")
                if len(set(document_ids[start:end])) != length:
                    raise ValueError(f"Duplicate retrieved document for {key}, {query_id}")
                if set(setting_ids[start:end]) != {measured["setting_id"]}:
                    raise ValueError(f"Wrong mixed setting for {key}, {query_id}")
                expected_gold = gold[(record["corpus"], measured["setting_id"], query_id)]
                gold_count = len(expected_gold)
                if gold_count != measured["relevant_document_count"] or gold_count == 0:
                    raise ValueError(f"Wrong relevance denominator for {key}, {query_id}")
                expected_binary = np.fromiter(
                    (doc_id in expected_gold for doc_id in document_ids[start:end]), dtype=np.int8
                )
                if not np.array_equal(expected_binary, relevance[start:end]):
                    raise ValueError(f"Ranking relevance differs from qrels for {key}, {query_id}")
                first_ten = relevance[start:min(end, start + 10)]
                hits = np.flatnonzero(first_ten)
                dcg = float(discount[:len(first_ten)] @ first_ten)
                ideal = math.fsum(discount[:min(gold_count, 10)])
                reconstructed = {
                    "mrr10": float(1 / (hits[0] + 1)) if len(hits) else 0.0,
                    "ndcg10": dcg / ideal,
                    "hit10": float(bool(len(hits))),
                    "recall10": float(first_ten.sum() / gold_count),
                }
                for name, value in reconstructed.items():
                    error = abs(value - measured[name])
                    metric_max_error[name] = max(metric_max_error[name], error)
                    if error > 1e-12:
                        raise ValueError(f"Metric {name} disagrees for {key}, {query_id}: {error}")
                query_counts[record["corpus"]] += 1
                rank_lengths[int(length)] += 1

            columns = [pa.array([record[name]] * projected.num_rows) for name in ANNOTATION_COLUMNS]
            columns += [projected[name] for name in SOURCE_COLUMNS]
            output = pa.Table.from_arrays(columns, names=OUTPUT_SCHEMA.names).cast(OUTPUT_SCHEMA)
            writer.write_table(output, row_group_size=32768)
            row_counts[record["corpus"]] += output.num_rows
            total_rows += output.num_rows

            inventory_path = report / "analysis_manifest.json"
            inventory = json.loads(inventory_path.read_text())
            relative_source = source.relative_to((report / inventory["artifact_root"]).resolve()).as_posix()
            entry = next(item for item in inventory["files"] if item["path"] == relative_source)
            after_stat = source.stat()
            if (source_stat.st_size, source_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
                raise ValueError(f"Ranking source changed during extraction: {source}")
            sources.append({
                **dict(zip(("corpus", "model", "variant"), key, strict=True)),
                "path": str(source.relative_to(repo)),
                "source_size_bytes": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_sha256_from_verified_inventory": entry["sha256"],
                "analysis_manifest_sha256": file_hash(inventory_path),
                "mixed_rows": output.num_rows,
                "mixed_queries": len(unique_queries),
            })
            if position % 24 == 0:
                print(f"Extracted {position}/264 reports; {total_rows:,} rank rows", flush=True)

    persisted = pq.ParquetFile(temporary)
    if persisted.metadata.num_rows != total_rows or persisted.schema_arrow.names != OUTPUT_SCHEMA.names:
        raise ValueError("Persisted Parquet metadata differs from the extracted data")
    output_hash = file_hash(temporary)
    temporary.replace(destination)
    metadata = {
        "status": "success",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Mixed-language pools from all264 selected-document reports; no legacy chunk runs",
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "script_sha256": file_hash(Path(__file__)),
        "report_catalog_sha256": file_hash(catalog_path),
        "per_query_dataset_sha256": file_hash(per_query_path),
        "qrels_dataset_sha256": file_hash(qrels_path),
        "reports": len(sources),
        "unique_query_configurations": sum(query_counts.values()),
        "ranking_rows": total_rows,
        "rows_by_corpus": dict(row_counts),
        "query_configurations_by_corpus": dict(query_counts),
        "ranking_length_distribution": dict(rank_lengths),
        "columns": OUTPUT_SCHEMA.names,
        "source_projection": SOURCE_COLUMNS,
        "text_columns_read": False,
        "unique_configuration_query_rank": True,
        "unique_documents_per_query_configuration": True,
        "relevance_flags_match_qrels": True,
        "metric_max_absolute_errors": metric_max_error,
        "validation_queries": sum(query_counts.values()),
        "output_sha256": output_hash,
        "output_bytes": destination.stat().st_size,
        "seconds": time.perf_counter() - began,
        "notes": [
            "Reads one projected/filtered report file at a time; no model, API, or retrieval calls.",
            "Only saved top100 document-language ranks are available; collapsed source rankings are censored beyond that boundary.",
            "Source ranking hashes are recorded from the previously verified result inventories, not recomputed by reading unprojected source text columns.",
            "Original qrel flags are preserved; source_id is not permission to expand chemistry relevance beyond accepted direct pairs.",
        ],
        "sources": sources,
    }
    write_json(metadata_path, metadata)
    print(json.dumps({name: metadata[name] for name in (
        "status", "reports", "unique_query_configurations", "ranking_rows",
        "metric_max_absolute_errors", "output_bytes", "seconds",
    )}), flush=True)


if __name__ == "__main__":
    main()
