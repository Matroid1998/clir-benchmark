#!/usr/bin/env python3
"""Bounded, explicit GPU sizing probes; never publish benchmark embeddings.

Run ``prepare`` once on a CPU allocation, then ``probe`` on GPU allocations.
GPU probes retain the configured dtype and batch size. Sample timing includes
the production encoder's tokenization, truncation audit, transfer and forward.
Synthetic shape tests measure forward memory only, not benchmark throughput.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import math
import os
import subprocess
import time
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from clir_benchmark.config import load_config
from clir_benchmark.encoders import HuggingFaceEncoder, load_registry


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parquet_rows(path, columns):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns):
        yield from batch.to_pylist()


def retain_largest(heap, key, row, count):
    item = (key, row["id"], row)
    if len(heap) < count:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def select_documents(records, longest=64, sample=32):
    large, random = defaultdict(list), defaultdict(list)
    counts = Counter()
    for row in records:
        language = row["language"]
        counts[language] += 1
        retain_largest(large[language], len(row["text"]), row, longest)
        key = int(hashlib.sha256(row["id"].encode()).hexdigest(), 16)
        retain_largest(random[language], -key, row, sample)
    selected = {}
    for selection, groups in (("longest_characters", large), ("deterministic_sample", random)):
        for heap in groups.values():
            for _, identifier, row in heap:
                value = selected.setdefault(identifier, {**row, "selection": []})
                value["selection"].append(selection)
    return sorted(selected.values(), key=lambda row: (row["language"], row["id"])), dict(counts)


def previous_qwen_maxima(output_dir):
    result = []
    for path in Path(output_dir).glob("embeddings/*/*/manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest["model"]["alias"] != "qwen3-0.6B":
            continue
        metadata = pq.ParquetFile(path.parent / "rows.parquet").metadata
        names = pq.ParquetFile(path.parent / "rows.parquet").schema.names
        column = names.index("encoded_tokens")
        maxima = []
        for index in range(metadata.num_row_groups):
            stats = metadata.row_group(index).column(column).statistics
            if stats is not None and stats.has_min_max:
                maxima.append(stats.max)
        result.append(
            {
                "manifest": str(path),
                "kind": manifest["kind"],
                "rows": manifest["rows"],
                "max_encoded_tokens": max(maxima) if maxima else None,
            }
        )
    return result


def prepare(args, config):
    destination = Path(args.output)
    if destination.exists():
        raise ValueError(f"Sample file is immutable and already exists: {destination}")
    chunks = Path(args.chunks) if args.chunks else config.corpus_dir(args.corpus) / "chunks.parquet"
    queries_path = (
        Path(args.queries)
        if args.queries
        else config.relevance_dir(args.corpus) / "queries.parquet"
    )
    corpus_dir, relevance_dir = chunks.parent, queries_path.parent
    started = time.perf_counter()
    documents, counts = select_documents(
        parquet_rows(chunks, ["id", "language", "text", "token_count"])
    )
    queries = [
        {"id": row["id"], "language": row["question_language"], "text": row["question"]}
        for row in parquet_rows(queries_path, ["id", "question_language", "question"])
    ]
    payload = {
        "schema": 1,
        "corpus": args.corpus,
        "corpus_manifest": json.loads((corpus_dir / "manifest.json").read_text()),
        "relevance_manifest": json.loads((relevance_dir / "manifest.json").read_text()),
        "selection": "Per language: longest 64 texts by character count plus 32 deterministic ID-hash samples; all queries. No full-corpus model tokenization.",
        "document_counts_by_language": counts,
        "documents": documents,
        "queries": queries,
        "previous_qwen_maxima": previous_qwen_maxima(
            args.prior_embeddings_root or config.output_dir
        )
        if (args.prior_embeddings_root or config)
        else [],
        "seconds": time.perf_counter() - started,
    }
    if args.prior_qwen_max_tokens:
        payload["previous_qwen_maxima"].append(
            {
                "source": "explicit --prior-qwen-max-tokens from completed audit metadata",
                "max_encoded_tokens": args.prior_qwen_max_tokens,
            }
        )
    payload["fingerprint"] = fingerprint(payload)
    atomic_json(destination, payload)
    print(
        json.dumps(
            {
                "samples": str(destination),
                "documents": len(documents),
                "queries": len(queries),
                "fingerprint": payload["fingerprint"],
            }
        ),
        flush=True,
    )


def nvml_process_memory():
    """Sample this process through nvidia-smi/NVML; never call it a peak."""
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "rows": [
                line
                for line in process.stdout.splitlines()
                if line.split(",", 1)[0].strip() == str(os.getpid())
            ],
            "method": "nvidia-smi NVML process-memory snapshot, not continuous peak",
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"unavailable": str(error)}


def cuda_memory(torch):
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def model_lengths(encoder, records, kind):
    prefix = encoder.spec.document_prefix if kind == "document" else encoder.spec.query_prefix
    lengths = []
    for start in range(0, len(records), 128):
        tokenized = encoder.tokenizer(
            [prefix + row["text"] for row in records[start : start + 128]],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in tokenized["input_ids"])
    return lengths


def stress_length(spec, document_lengths, query_lengths, historical):
    observed = max(document_lengths + query_lengths + [1])
    if spec.alias == "qwen3-0.6B":
        previous = max([row["max_encoded_tokens"] or 0 for row in historical] + [0])
        # The 32K capacity is not a feasible worst-case stress test at batch 32.
        length = min(
            spec.max_length, max(2048, int(math.ceil(max(observed, previous) * 1.10 / 128) * 128))
        )
        return (
            length,
            "At least 2048 tokens and 110% of max(sampled UN document tokens, all UN query tokens, prior completed Qwen audit maxima), rounded up to 128; NOT an exhaustive UN document bound",
        )
    if spec.alias == "bge-m3":
        return (
            min(spec.max_length, max(512, observed)),
            "All document chunks use the identical BGE tokenizer and are bounded at 512 tokens; all actual query lengths included",
        )
    return spec.max_length, "Full configured model token limit at full batch size"


def apply_mixed_padding(inputs, padding_side, pad_token_id):
    """Alternate full and half-length rows, preserving the model's padding side."""
    if padding_side not in {"left", "right"}:
        raise ValueError(f"Unsupported padding side: {padding_side}")
    if pad_token_id is None:
        raise ValueError("Mixed-padding stress requires the production tokenizer's pad token")
    length = inputs["input_ids"].shape[1]
    padding = slice(0, length - length // 2) if padding_side == "left" else slice(length // 2, None)
    inputs["input_ids"][1::2, padding] = pad_token_id
    inputs["attention_mask"][1::2, padding] = 0


def synthetic_forward(encoder, length, batch_size, *, mixed_padding=False):
    torch = encoder.torch
    token = encoder.tokenizer("document", add_special_tokens=False)["input_ids"][0]
    inputs = {
        "input_ids": torch.full(
            (batch_size, length), token, dtype=torch.long, device=encoder.device
        ),
        "attention_mask": torch.ones((batch_size, length), dtype=torch.long, device=encoder.device),
    }
    if "token_type_ids" in encoder.tokenizer.model_input_names:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    if mixed_padding:
        apply_mixed_padding(inputs, encoder.tokenizer.padding_side, encoder.tokenizer.pad_token_id)
    with torch.inference_mode():
        output = (
            encoder.model(inputs)["sentence_embedding"]
            if encoder.spec.backend == "sentence_transformers"
            else encoder.model(**inputs).last_hidden_state[:, 0, :]
        )
        if tuple(output.shape) != (batch_size, encoder.spec.dimension):
            raise ValueError(f"Unexpected stress embedding shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all().item():
            raise ValueError("Nonfinite synthetic stress embeddings")


def probe(args, config):
    import torch

    samples = json.loads(Path(args.samples).read_text())
    sample_hash = samples.pop("fingerprint")
    if fingerprint(samples) != sample_hash or samples["corpus"] != args.corpus:
        raise ValueError("Probe sample identity mismatch")
    if (
        config.runtime.device != "cuda"
        or config.runtime.dtype != "float32"
        or config.runtime.batch_size != 32
    ):
        raise ValueError(
            "This campaign probe requires the actual CUDA/float32/batch32 configuration"
        )
    if not torch.cuda.is_available():
        raise ValueError("GPU probe requires a compute allocation with CUDA")
    registry = load_registry(config.model_registry)
    aliases = args.models or config.models
    if set(aliases) - set(config.models):
        raise ValueError("Probe models must belong to the selected campaign")
    properties = torch.cuda.get_device_properties(0)
    if args.allocator_budget_gib is not None:
        budget = int(args.allocator_budget_gib * 1024**3)
        if not 0 < budget <= properties.total_memory:
            raise ValueError(
                "Allocator budget must be positive and no larger than visible device memory"
            )
        torch.cuda.set_per_process_memory_fraction(budget / properties.total_memory, device=0)
    report = {
        "schema": 1,
        "probe_version": 2,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "samples_fingerprint": sample_hash,
        "corpus": args.corpus,
        "runtime": asdict(config.runtime),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        },
        "allocator_budget_gib": args.allocator_budget_gib,
        "allocator_budget_caveat": "A PyTorch allocator limit on a larger MIG device does not emulate target MIG compute resources or cap CUDA-context/non-PyTorch allocations; actual memory snapshots are retained.",
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_GPUS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "methodology": "Explicit sizing experiment. No benchmark embeddings published. Real sample encode throughput includes production truncation audits. Synthetic stress uses the same published forward/pooling stack but excludes tokenization/audits. Qwen document sampling is not exhaustive proof.",
        "models": [],
    }
    atomic_json(args.output, report)
    for alias in aliases:
        encoder = None
        tests = []
        action = None
        result = {
            "alias": alias,
            "model": asdict(registry[alias]),
            "status": "running",
            "tests": [],
        }
        report["models"].append(result)
        started = time.perf_counter()
        atomic_json(args.output, report)
        try:
            torch.cuda.empty_cache()
            result["before_load_memory"] = cuda_memory(torch)
            torch.cuda.reset_peak_memory_stats()
            encoder = HuggingFaceEncoder(registry[alias], config.runtime)
            torch.cuda.synchronize()
            result["load_seconds"] = time.perf_counter() - started
            result["loaded_memory"] = cuda_memory(torch)
            result["nvml_after_load"] = nvml_process_memory()
            lengths = {
                kind: model_lengths(encoder, samples[key], kind)
                for kind, key in (("document", "documents"), ("query", "queries"))
            }
            result["lengths"] = {
                kind: {
                    "count": len(values),
                    "max": max(values, default=0),
                    "mean": sum(values) / len(values) if values else None,
                }
                for kind, values in lengths.items()
            }
            length, rationale = stress_length(
                encoder.spec, lengths["document"], lengths["query"], samples["previous_qwen_maxima"]
            )
            result["stress_length"] = length
            result["stress_rationale"] = rationale
            tests = [
                (
                    "synthetic_full_batch",
                    "forward_only",
                    lambda encoder=encoder, length=length: synthetic_forward(
                        encoder, length, config.runtime.batch_size
                    ),
                ),
                (
                    "synthetic_mixed_padding",
                    "forward_only",
                    lambda encoder=encoder, length=length: synthetic_forward(
                        encoder, length, config.runtime.batch_size, mixed_padding=True
                    ),
                ),
            ]
            for kind, key in (("document", "documents"), ("query", "queries")):
                records = samples[key]
                if not records:
                    continue
                longest = max(range(len(records)), key=lambda index: lengths[kind][index])
                texts = [records[longest]["text"]] * config.runtime.batch_size
                tests.append(
                    (
                        f"{kind}_longest_repeated",
                        "encode_with_audit",
                        lambda texts=texts, kind=kind, encoder=encoder: encoder.encode(texts, kind),
                    )
                )
                # One deterministic representative sample batch per input kind.
                pool = [
                    row for row in records if "deterministic_sample" in row.get("selection", [])
                ] or records
                texts = [
                    pool[index * len(pool) // config.runtime.batch_size]["text"]
                    for index in range(config.runtime.batch_size)
                ]
                if kind == "document":
                    mixed = [records[longest]["text"], *texts[1:]]
                    tests.append(
                        (
                            "document_longest_mixed",
                            "encode_with_audit",
                            lambda texts=mixed, encoder=encoder: encoder.encode(texts, "document"),
                        )
                    )
                tests.append(
                    (
                        f"{kind}_representative",
                        "encode_with_audit",
                        lambda texts=texts, kind=kind, encoder=encoder: encoder.encode(texts, kind),
                    )
                )
            for name, method, action in tests:
                test = {
                    "name": name,
                    "method": method,
                    "status": "running",
                    "batch_size": config.runtime.batch_size,
                }
                result["tests"].append(test)
                atomic_json(args.output, report)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                test_started = time.perf_counter()
                try:
                    output = action()
                    torch.cuda.synchronize()
                    test["seconds"] = time.perf_counter() - test_started
                    test["inputs_per_second"] = (
                        config.runtime.batch_size / test["seconds"]
                        if method == "encode_with_audit"
                        else None
                    )
                    if method == "encode_with_audit":
                        test["input_tokens"] = sum(audit.input_tokens for audit in output.audits)
                        test["encoded_tokens"] = sum(
                            audit.encoded_tokens for audit in output.audits
                        )
                        test["encoded_lengths"] = [audit.encoded_tokens for audit in output.audits]
                    del output
                    test["status"] = "success"
                except Exception as error:
                    test["status"] = (
                        "oom" if isinstance(error, torch.OutOfMemoryError) else "failed"
                    )
                    test["error"] = str(error)
                    test["traceback"] = traceback.format_exc()
                    test["seconds"] = time.perf_counter() - test_started
                finally:
                    test["memory"] = cuda_memory(torch)
                    test["nvml_after_test"] = nvml_process_memory()
                    atomic_json(args.output, report)
                if test["status"] != "success":
                    gc.collect()
                    torch.cuda.empty_cache()
            result["status"] = (
                "success"
                if all(test["status"] == "success" for test in result["tests"])
                else "failed"
            )
        except Exception as error:
            result["status"] = "oom" if isinstance(error, torch.OutOfMemoryError) else "failed"
            result["error"] = str(error)
            result["traceback"] = traceback.format_exc()
        finally:
            result["seconds"] = time.perf_counter() - started
            result["finished_at"] = datetime.now(UTC).isoformat()
            tests.clear()
            action = None
            encoder = None
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(args.output, report)
    report["status"] = (
        "success" if all(model["status"] == "success" for model in report["models"]) else "failed"
    )
    report["finished_at"] = datetime.now(UTC).isoformat()
    atomic_json(args.output, report)
    print(json.dumps({"output": args.output, "status": report["status"]}), flush=True)
    return 0 if report["status"] == "success" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "probe"))
    parser.add_argument("--config")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples")
    parser.add_argument(
        "--chunks", help="Explicit prepared chunks.parquet for local sample preparation"
    )
    parser.add_argument(
        "--queries", help="Explicit prepared queries.parquet for local sample preparation"
    )
    parser.add_argument(
        "--prior-embeddings-root", help="Artifact root containing embeddings/*/*/rows.parquet"
    )
    parser.add_argument("--prior-qwen-max-tokens", type=int)
    parser.add_argument("--allocator-budget-gib", type=float)
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()
    if args.mode == "probe" and not args.samples:
        parser.error("probe requires --samples")
    if args.mode == "probe" and not args.config:
        parser.error("probe requires --config")
    if args.mode == "prepare" and not args.config and not (args.chunks and args.queries):
        parser.error("prepare requires --config or both --chunks and --queries")
    config = load_config(args.config) if args.config else None
    return prepare(args, config) if args.mode == "prepare" else probe(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
