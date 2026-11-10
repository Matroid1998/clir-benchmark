"""Offline, lossless analysis indexes and MTEB-style custom-task result exports.

MTEB TaskResult reference: https://github.com/embeddings-benchmark/mteb/blob/main/
mteb/results/task_result.py. This package computes its own metrics, so exports
deliberately do not claim an MTEB engine version or an end-to-end evaluation time.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import CorpusStore, ParquetSink, rows
from .config import Config
from .encoders import ModelSpec
from .errors import ValidationError
from .fingerprints import cached_file_digest as file_digest
from .util import digest, new_directory, read_json, write_json

MTEB_SCHEMA_REFERENCE = (
    "https://github.com/embeddings-benchmark/mteb/blob/main/mteb/results/task_result.py"
)
# Defaults for the languages in the three corpora. Original BCP-47 tags are
# retained alongside these display codes; an explicit script always wins.
LANGUAGE_SCRIPTS = dict(
    en="eng-Latn",
    de="deu-Latn",
    fr="fra-Latn",
    es="spa-Latn",
    pt="por-Latn",
    it="ita-Latn",
    nl="nld-Latn",
    da="dan-Latn",
    sv="swe-Latn",
    fi="fin-Latn",
    et="est-Latn",
    lv="lav-Latn",
    lt="lit-Latn",
    pl="pol-Latn",
    cs="ces-Latn",
    sk="slk-Latn",
    sl="slv-Latn",
    hr="hrv-Latn",
    ro="ron-Latn",
    hu="hun-Latn",
    mt="mlt-Latn",
    ga="gle-Latn",
    el="ell-Grek",
    bg="bul-Cyrl",
    ru="rus-Cyrl",
    ar="ara-Arab",
    zh="zho-Hans",
    ja="jpn-Jpan",
    ko="kor-Hang",
    tr="tur-Latn",
    uk="ukr-Cyrl",
    he="heb-Hebr",
    hi="hin-Deva",
    no="nor-Latn",
)


def metric_name(name: str) -> str:
    """Keep familiar MTEB retrieval names without changing metric values."""
    prefixes = {
        "nDCG": "ndcg",
        "MRR": "mrr",
        "Recall": "recall",
        "Hit": "hit",
        "SourceRecall": "source_recall",
        "AllSourceSuccess": "all_source_success",
    }
    metric, cutoff = name.split("@", 1)
    return f"{prefixes[metric]}_at_{cutoff}"


def language_script(tag: str) -> str:
    parts = tag.split("-")
    mapped = LANGUAGE_SCRIPTS.get(parts[0].lower())
    if mapped is None:
        return tag  # Never invent an ISO-639-3 code for an unsupported language.
    script = next((part.title() for part in parts[1:] if len(part) == 4), None)
    return mapped.split("-")[0] + "-" + script if script else mapped


def mteb_result(report: dict) -> dict:
    document_mode = report.get("retrieval_unit", report["run"].get("retrieval_unit")) == "document"
    scores = []
    for setting in report["metrics"]["setting"]:
        tags = sorted({setting["query_language"], *setting["target_languages"].split(",")})
        scores.append(
            {
                "hf_subset": setting["setting_id"],
                "languages": sorted({language_script(tag) for tag in tags}),
                "language_tags": tags,
                "main_score": setting["metrics"]["nDCG@10"],
                "main_score_name": "ndcg_at_10",
                "query_count": setting["query_count"],
                **{metric_name(name): value for name, value in setting["metrics"].items()},
            }
        )
    return {
        "task_name": "CLIRBenchmark_" + report["corpus"] + ("_Document" if document_mode else ""),
        **(
            {"retrieval_unit": "document", "document_identity": "source_unit_id"}
            if document_mode
            else {}
        ),
        # This pins the effective, span-grounded cohort rather than pretending
        # the original HF revision describes a sampled or overlaid dataset.
        "dataset_revision": report["run"]["relevance_hash"],
        "mteb_version": None,
        "scores": {"test": scores},
        "evaluation_time": None,
        "kg_co2_emissions": None,
        "date": None,
        "evaluation_phases": None,
    }


def _json_config(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    return value


def _atomic_json(path: Path, value: dict) -> None:
    # Sidecars are recoverable on the next offline evaluate invocation.
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value.replace("/", "__"))


def _mteb_path(config: Config, report: dict, run_dir: Path) -> Path:
    model = report["run"].get("model")
    repo = model["repo_id"] if model else "clir-benchmark__bm25"
    revision = model["revision"] if model else digest(report["run"].get("bm25"))
    experiment = f"{report['baseline']}__{run_dir.name}"
    return (
        config.output_dir
        / "mteb"
        / _safe_component(repo)
        / _safe_component(revision)
        / "experiments"
        / experiment
        / f"CLIRBenchmark_{report['corpus']}.json"
    )


def _snapshot_local_inputs(config: Config, corpus: str) -> list[Path]:
    """Keep selected upstream rows and subset provenance inside the artifact root."""
    result = []
    dataset = config.datasets[corpus]
    for source in (dataset.qa, dataset.corpus):
        if not source.local_dir:
            continue
        local = Path(source.local_dir)
        identity = source.identity()
        metadata = {}
        # The deterministic subset command publishes these alongside qa/corpus.
        if local.name in {"qa", "corpus"} and (local.parent / "selection.json").is_file():
            for name in ("config.yaml", "manifest.json", "selection.json", "evidence.jsonl"):
                path = local.parent / name
                if path.is_file():
                    metadata[name] = file_digest(path)
        identity = {"source": identity, "subset_metadata": metadata}
        destination = config.output_dir / "input_snapshots" / digest(identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Distinct model jobs share these immutable inputs. Serialize just the
        # initial publication so simultaneous exports do not fail or duplicate
        # a full corpus copy; each report remains independently writable.
        with (destination.parent / f".{destination.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not destination.exists():
                with new_directory(destination) as temporary:
                    files = {
                        **{name: local / name for name in identity["source"]["files"]},
                        **{f"subset/{name}": local.parent / name for name in metadata},
                    }
                    for name, original in files.items():
                        copied = temporary / name
                        copied.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, copied)
                        expected = (
                            metadata[name.removeprefix("subset/")]
                            if name.startswith("subset/")
                            else identity["source"]["files"][name]
                        )
                        if file_digest(copied) != expected:
                            raise ValidationError(f"Local input changed during export: {original}")
                    write_json(temporary / "manifest.json", {"schema": 1, **identity})
            elif read_json(destination / "manifest.json") != {"schema": 1, **identity}:
                raise ValidationError(f"Local input snapshot identity mismatch: {destination}")
        result.append(destination)
    return result


def _dependencies(config: Config, corpus: str, run_dir: Path, report: dict) -> list[Path]:
    """Follow saved dependencies, including all attempts and relevant invocations."""
    from .hyde import hyde_path
    from .pipeline import embedding_path
    from .translation import translation_path

    root = config.output_dir.resolve()
    files: set[Path] = set()
    directories: set[Path] = set()
    pending: list[Path] = []

    def add(path: Path) -> None:
        path = path.resolve()
        if not path.is_relative_to(root) or not path.exists():
            return
        relative = path.relative_to(root)
        if not relative.parts or relative.parts[0] not in {
            "corpus",
            "relevance",
            "runs",
            "embeddings",
            "lexical",
            "reports",
            "translations",
            "hyde",
            "translation_requests",
            "hyde_requests",
            "adore_requests",
            "adore_searches",
            "reranking_pairs",
            "translation_invocations",
            "hyde_invocations",
            "adore_invocations",
            "reranking_invocations",
            "input_snapshots",
        }:
            return
        if path.name.startswith(".") or path.name == "analysis_manifest.json":
            return
        if path.is_dir():
            if path in directories:
                return
            directories.add(path)
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix not in {".lock", ".tmp"}:
                    add(child)
        elif path not in files:
            files.add(path)
            pending.append(path)

    def inspect(value, key=""):
        if isinstance(value, dict):
            for name, item in value.items():
                inspect(item, name)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item, key)
        elif isinstance(value, str):
            if value.startswith(str(root) + os.sep):
                add(Path(value))
            if re.fullmatch(r"[0-9a-f]{64}", value) and any(
                word in key for word in ("request_key", "search_key", "score_key")
            ):
                for namespace in ("translation_requests", "hyde_requests", "adore_requests"):
                    add(root / namespace / f"{value}.json")
                for namespace in ("adore_searches", "reranking_pairs"):
                    add(root / namespace / value)

    def add_run(manifest):
        if manifest.get("parent_run"):
            add(Path(manifest["parent_run"]))
            add_run(manifest["parent_manifest"])
        model = manifest.get("model")
        baseline = manifest.get("baseline", "dense")
        translation_language = manifest.get("translation_language")
        output_language = manifest.get("output_language") or manifest.get("hyde_language")
        if translation_language:
            add(translation_path(config, corpus, translation_language))
        if output_language and baseline.removeprefix("rerank-") == "hyde-dense":
            add(hyde_path(config, corpus, output_language))
        if model:
            spec = ModelSpec(**model)
            add(embedding_path(config, corpus, spec, "document"))
            if not baseline.removeprefix("rerank-").startswith("adore-"):
                add(
                    embedding_path(
                        config,
                        corpus,
                        spec,
                        "query",
                        translation_language=translation_language,
                        hyde_language=output_language,
                    )
                )

    add(config.corpus_dir(corpus))
    add(config.relevance_dir(corpus))
    for path in _snapshot_local_inputs(config, corpus):
        add(path)
    add(run_dir)
    add_run(report["run"])
    inspect(report)

    def follow_pending():
        while pending:
            path = pending.pop()
            if path.suffix == ".json":
                inspect(read_json(path))
            elif path.suffix == ".parquet":
                keys = [
                    name
                    for name in pq.ParquetFile(path).schema_arrow.names
                    if "request_key" in name
                ]
                if keys:
                    for row in rows(path, keys):
                        inspect(row)

    follow_pending()
    # Invocation records are mutable history, separate from the original build
    # accounting. Index only records belonging to dependencies of this result.
    for namespace in (
        "translation_invocations",
        "hyde_invocations",
        "adore_invocations",
        "reranking_invocations",
    ):
        for path in sorted((root / namespace).glob("*.json")):
            record = read_json(path)
            target = record.get("run") or record.get("artifact")
            if target and Path(target).resolve() in directories:
                add(path)
    follow_pending()
    return sorted(files)


RANKING_DETAILS_SCHEMA = pa.schema(
    [
        ("setting_id", pa.string()),
        ("query_id", pa.string()),
        ("query", pa.string()),
        ("query_language", pa.string()),
        ("chunk_id", pa.string()),
        ("rank", pa.int64()),
        ("score", pa.float64()),
        ("relevance", pa.int64()),
        ("required_source_ids", pa.list_(pa.string())),
        ("source_unit_id", pa.string()),
        ("source_id", pa.string()),
        ("parent_id", pa.string()),
        ("document_language", pa.string()),
        ("text", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
    ]
)

DOCUMENT_RANKING_DETAILS_SCHEMA = pa.schema(
    [field for field in RANKING_DETAILS_SCHEMA if field.name != "chunk_id"]
    + [
        ("document_id", pa.string()),
        ("winning_chunk_id", pa.string()),
    ]
)


def _ranking_details(config: Config, corpus: str, run_dir: Path, destination: Path) -> None:
    document_mode = config.retrieval_unit == "document"
    result_id = "document_id" if document_mode else "chunk_id"
    queries = {row["id"]: row for row in rows(config.relevance_dir(corpus) / "queries.parquet")}
    qrels = defaultdict(set)
    for row in rows(config.relevance_dir(corpus) / "qrels.parquet"):
        qrels[(row["setting_id"], row["query_id"], row[result_id])].add(row["required_source_id"])
    with (
        CorpusStore(config.corpus_dir(corpus) / "lookup.sqlite") as store,
        ParquetSink(
            destination,
            DOCUMENT_RANKING_DETAILS_SCHEMA if document_mode else RANKING_DETAILS_SCHEMA,
        ) as sink,
    ):
        for row in rows(run_dir / "run.parquet"):
            chunk_id = row["winning_chunk_id"] if document_mode else row["chunk_id"]
            chunk = store.chunk(chunk_id)
            if chunk is None:
                raise ValidationError(f"Result refers to unknown chunk {chunk_id}")
            source = store.get_source(chunk["source_unit_id"])
            if source is None:
                raise ValidationError(f"Result refers to unknown source {chunk['source_unit_id']}")
            if document_mode and source.id != row["document_id"]:
                raise ValidationError("Winning chunk does not belong to the ranked document")
            query = queries[row["query_id"]]
            required = qrels[(row["setting_id"], row["query_id"], row[result_id])]
            sink.append(
                {
                    **row,
                    "query": query["question"],
                    "query_language": query["question_language"],
                    "relevance": int(bool(required)),
                    "required_source_ids": sorted(required),
                    "source_unit_id": source.id,
                    "source_id": source.source_id,
                    "parent_id": source.parent_id,
                    "document_language": source.language,
                    "text": source.text[chunk["start"] : chunk["end"]],
                    "start": chunk["start"],
                    "end": chunk["end"],
                }
            )


def export_result(config: Config, corpus: str, run_dir: Path, report_dir: Path) -> Path:
    """Export already computed results; never generate, encode, retrieve or rerank."""
    report = read_json(report_dir / "report.json")
    if report["run"].get("retrieval_unit", "chunk") != config.retrieval_unit:
        raise ValidationError("Result retrieval unit does not match the evaluation configuration")
    mteb = mteb_result(report)
    standardized_path = _mteb_path(config, report, run_dir)
    _atomic_json(report_dir / "mteb.json", mteb)
    _atomic_json(standardized_path, mteb)
    resolved = _json_config(asdict(config))
    # No environment variable values, .env contents, HTTP headers or credentials
    # are read. Keep each evaluation's selection without overwriting earlier ones.
    config_path = report_dir / "evaluation_configs" / f"{digest(resolved)}.json"
    if not config_path.exists():
        _atomic_json(config_path, resolved)
    detail_dir = report_dir / "analysis"
    identity = {
        name: file_digest(path)
        for name, path in {
            "run": run_dir / "run.parquet",
            "qrels": config.relevance_dir(corpus) / "qrels.parquet",
            "queries": config.relevance_dir(corpus) / "queries.parquet",
            "corpus": config.corpus_dir(corpus) / "manifest.json",
        }.items()
    }
    if detail_dir.exists():
        saved = read_json(detail_dir / "manifest.json")
        if saved["inputs"] != identity:
            raise ValidationError(
                "Analysis export is stale: rankings, qrels, queries or corpus changed"
            )
        if saved["rankings_sha256"] != file_digest(detail_dir / "rankings.parquet"):
            raise ValidationError("Corrupt analysis ranking export")
    else:
        with new_directory(detail_dir) as temporary:
            _ranking_details(config, corpus, run_dir, temporary / "rankings.parquet")
            write_json(
                temporary / "manifest.json",
                {
                    "schema": 1,
                    "inputs": identity,
                    "rankings_sha256": file_digest(temporary / "rankings.parquet"),
                    "notes": "All saved ranks are retained, with exact source slices and binary qrels. Empty rankings remain represented in per_query.json and eligibility.parquet.",
                },
            )
    paths = _dependencies(config, corpus, run_dir, report)
    paths += [
        p
        for p in report_dir.rglob("*")
        if p.is_file() and p.name != "analysis_manifest.json" and p.suffix != ".tmp"
    ]
    paths.append(standardized_path)
    root = config.output_dir.resolve()
    inventory = [
        {
            "path": str(path.resolve().relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
        for path in sorted(set(paths))
    ]
    _atomic_json(
        report_dir / "analysis_manifest.json",
        {
            "schema": 1,
            "format": "clir-benchmark-analysis-v1",
            "artifact_root": os.path.relpath(root, report_dir),
            "baseline": report["baseline"],
            **(
                {"retrieval_unit": "document", "document_identity": "source_unit_id"}
                if config.retrieval_unit == "document"
                else {}
            ),
            "run_id": run_dir.name,
            "mteb_result": str(standardized_path.resolve().relative_to(root)),
            "mteb_schema_reference": MTEB_SCHEMA_REFERENCE,
            "mteb_notes": "Custom CLIR task; scores.test contains one subset per retrieval setting. Metrics are on the 0-1 scale. main_score is ndcg_at_10. MTEB was not executed, so mteb_version is null; evaluation_time and CO2 are unknown. Language scripts use documented defaults, preserving original language_tags. This is a format-compatible export, not a registered MTEB leaderboard task.",
            "configuration_notes": "evaluation_configs retains every resolved configuration used to score/export this result. Original execution parameters and model identities remain in run and dependency manifests. Environment variable names and paths are configuration; secret values are never loaded by export.",
            "retention_notes": "Retain the entire artifact_root directory. Files are stored once and referenced here by relative path and SHA-256, including source text, queries/evidence, qrels/matches/eligibility, vectors/token audits, rankings, generation payloads/responses and failed/retried attempts, reranker scores, timing/cost records and invocation history. These references do not duplicate shared work or its costs. Original local/HF inputs are identified by dataset manifests.",
            "timing_notes": report.get("timing_notes"),
            "files": inventory,
        },
    )
    return report_dir
