from __future__ import annotations

import csv
import hashlib
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa

from .adapters import (
    DatasetReader,
    apply_evidence_overlay,
    chemistry_pair,
    legal_unit,
    query_from_row,
    select_candidates,
    verify_chemistry_provenance,
)
from .artifacts import (
    CHUNK_SCHEMA,
    ELIGIBILITY_SCHEMA,
    MATCH_SCHEMA,
    QREL_SCHEMA,
    QUERY_SCHEMA,
    RUN_SCHEMA,
    SOURCE_SCHEMA,
    CorpusStore,
    ParquetSink,
    query_record,
    read_queries,
    rows,
    source_record,
)
from .chunking import Chunker, load_tokenizer
from .config import Config
from .embedding import (
    embedding_key,
    encode_records,
    load_embeddings,
    support_exposure,
)
from .encoders import Encoder, HuggingFaceEncoder, ModelSpec, load_registry
from .errors import ValidationError
from .metrics import aggregate, score_query
from .relevance import build_settings, judge
from .retrieval import DenseIndex
from .util import (
    digest,
    hf_token,
    json_dumps,
    new_directory,
    read_json,
    versions,
    write_json,
)


def prepare_corpus(config: Config, corpus: str, *, chunker: Chunker | None = None) -> Path:
    destination = config.corpus_dir(corpus)
    if destination.exists():
        read_json(destination / "manifest.json")
        return destination
    token = hf_token(config.env_file)
    chunker = chunker or Chunker(load_tokenizer(config.chunking, token), config.chunking)
    reader = DatasetReader(config.datasets[corpus].corpus, token)
    started = time.perf_counter()
    with new_directory(destination) as temporary:
        with (
            CorpusStore(temporary / "lookup.sqlite", create=True) as store,
            ParquetSink(temporary / "sources.parquet", SOURCE_SCHEMA) as source_sink,
            ParquetSink(temporary / "chunks.parquet", CHUNK_SCHEMA) as chunk_sink,
            ParquetSink(
                temporary / "upstream_pairs.parquet", pa.schema([("record_json", pa.string())])
            ) as pair_sink,
        ):
            language_counts = Counter()
            pairs = []
            duplicate_count = 0
            configuration = "default" if corpus == "chemistry" else "corpus"
            for row in reader.read(configuration):
                if corpus == "chemistry":
                    units, pair = chemistry_pair(row, reader.spec)
                    pair_sink.append({"record_json": json_dumps(row)})
                    if pair:
                        pairs.append(pair)
                else:
                    units = [legal_unit(row, corpus)]
                for unit in units:
                    if not store.add_source(unit):
                        duplicate_count += 1
                        continue
                    source_sink.append(source_record(unit))
                    language_counts[unit.language] += 1
                    for chunk in chunker.split(unit):
                        store.add_chunk(chunk)
                        chunk_sink.append(asdict(chunk))
                    if source_sink.count % 1000 == 0:
                        store.db.commit()
            if source_sink.count == 0:
                raise ValidationError(f"{corpus}: source corpus is empty")
            store.add_pairs(pairs)
            store.db.commit()
            # Fingerprint content and boundaries, not row order, source revision,
            # QA selection, annotations, or generation provenance.
            content = hashlib.sha256()
            for (chunk_id,) in store.db.execute("SELECT id FROM chunks ORDER BY id"):
                content.update(chunk_id.encode("utf-8") + b"\n")
            fingerprint = digest(
                {
                    "schema": 1,
                    "corpus": corpus,
                    "chunking": chunker.manifest,
                    "chunks_hash": content.hexdigest(),
                }
            )
            manifest = {
                "schema": 1,
                "corpus": corpus,
                "corpus_fingerprint": fingerprint,
                "config_hash": config.corpus_key(corpus),
                "source": reader.spec.identity(),
                "chunking": chunker.manifest,
                "source_count": source_sink.count,
                "chunk_count": chunk_sink.count,
                "languages": dict(sorted(language_counts.items())),
                "duplicate_source_occurrences": duplicate_count,
                "accepted_direct_pairs": len(set(pairs)),
                "seconds": time.perf_counter() - started,
                "versions": versions(),
            }
        write_json(temporary / "manifest.json", manifest)
    return destination


def prepare_relevance(config: Config, corpus: str) -> Path:
    destination = config.relevance_dir(corpus)
    if destination.exists():
        read_json(destination / "manifest.json")
        return destination
    corpus_dir = config.corpus_dir(corpus)
    token = hf_token(config.env_file)
    reader = DatasetReader(config.datasets[corpus].qa, token)
    started = time.perf_counter()
    with CorpusStore(corpus_dir / "lookup.sqlite") as store:
        if corpus == "chemistry":
            verify_chemistry_provenance(reader, store, config.datasets[corpus].corpus)
        queries = [query_from_row(row, corpus) for row in reader.read("qas")]
        queries = apply_evidence_overlay(queries, config.evidence_file)
        selected, selection_audit = select_candidates(queries, config.candidate_mode)
        associations = defaultdict(set)
        known_query_ids = {query.id for query in queries}
        for row in reader.read("source_qrels"):
            if row.get("query_id") not in known_query_ids:
                raise ValidationError(f"source_qrels references unknown QA: {row.get('query_id')}")
            if row.get("score", 0) > 0:
                associations[row["query_id"]].add(row["corpus_id"])
        query_languages = sorted(
            set(config.query_languages) or {q.question_language for q in selected}
        )
        targets = sorted(set(config.target_languages) or set(store.languages()))
        settings = build_settings(corpus, query_languages, targets, config.settings)
        with new_directory(destination) as temporary:
            with ParquetSink(temporary / "queries.parquet", QUERY_SCHEMA) as sink:
                for query in selected:
                    sink.append(query_record(query))
            write_json(temporary / "selection.json", selection_audit)
            status = {}
            with (
                ParquetSink(temporary / "qrels.parquet", QREL_SCHEMA) as qrel_sink,
                ParquetSink(temporary / "matches.parquet", MATCH_SCHEMA) as match_sink,
                ParquetSink(
                    temporary / "eligibility.parquet", ELIGIBILITY_SCHEMA
                ) as eligibility_sink,
            ):
                for setting in settings:
                    judgments = []
                    counts = Counter()
                    reasons = Counter()
                    for query in selected:
                        if query.question_language != setting.query_language:
                            continue
                        judgment = judge(query, setting, store, associations[query.id])
                        judgments.append(judgment)
                        counts[judgment.status] += 1
                        reasons[judgment.reason] += 1
                        eligibility_sink.append(
                            {
                                "setting_id": setting.id,
                                "query_id": query.id,
                                "status": judgment.status,
                                "reason": judgment.reason,
                                "detail_json": json_dumps(judgment.detail),
                                "required_source_ids": judgment.required_source_ids,
                            }
                        )
                    blocked = counts["blocked"] > 0
                    status[setting.id] = {
                        **asdict(setting),
                        "status": "blocked" if blocked else "ready",
                        "counts": dict(counts),
                        "reasons": dict(reasons),
                    }
                    # A partially annotated setting must not publish partial qrels.
                    if blocked:
                        continue
                    export_dir = temporary / "qrels"
                    export_dir.mkdir(exist_ok=True)
                    with (export_dir / f"{setting.id}.tsv").open(
                        "w", encoding="utf-8", newline=""
                    ) as handle:
                        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                        for judgment in judgments:
                            if judgment.status != "eligible":
                                continue
                            for match in judgment.matches:
                                match_sink.append({"setting_id": setting.id, **asdict(match)})
                            for chunk_id, groups in sorted(judgment.relevant.items()):
                                writer.writerow([judgment.query_id, 0, chunk_id, 1])
                                for group in sorted(groups):
                                    qrel_sink.append(
                                        {
                                            "setting_id": setting.id,
                                            "query_id": judgment.query_id,
                                            "chunk_id": chunk_id,
                                            "required_source_id": group,
                                            "relevance": 1,
                                        }
                                    )
            query_fingerprint = digest([(q.id, q.question_language, q.question) for q in selected])
            write_json(temporary / "settings.json", status)
            write_json(
                temporary / "manifest.json",
                {
                    "schema": 1,
                    "corpus": corpus,
                    "config_hash": config.relevance_key(corpus),
                    "corpus_fingerprint": read_json(corpus_dir / "manifest.json")[
                        "corpus_fingerprint"
                    ],
                    "qa_source": reader.spec.identity(),
                    "query_fingerprint": query_fingerprint,
                    "input_query_count": len(queries),
                    "selected_query_count": len(selected),
                    "candidate_mode": config.candidate_mode,
                    "blocked_settings": [
                        name for name, value in status.items() if value["status"] == "blocked"
                    ],
                    "ready_settings": [
                        name for name, value in status.items() if value["status"] == "ready"
                    ],
                    "seconds": time.perf_counter() - started,
                    "versions": versions(),
                },
            )
    return destination


def model_spec(config: Config, alias: str) -> ModelSpec:
    registry = load_registry(config.model_registry)
    if alias not in registry:
        raise ValidationError(f"Unknown model alias {alias!r}. Available: {', '.join(registry)}")
    return registry[alias]


def embedding_path(config: Config, corpus: str, spec: ModelSpec, kind: str) -> Path:
    directory = config.corpus_dir(corpus) if kind == "document" else config.relevance_dir(corpus)
    if not (directory / "manifest.json").is_file():
        raise ValidationError(
            f"Missing {'corpus' if kind == 'document' else 'relevance'} artifacts: {directory}"
        )
    manifest = read_json(directory / "manifest.json")
    fingerprint = manifest["corpus_fingerprint" if kind == "document" else "query_fingerprint"]
    return (
        config.output_dir
        / "embeddings"
        / kind
        / embedding_key(fingerprint, kind, spec, config.runtime)
    )


def embed(
    config: Config,
    corpus: str,
    alias: str,
    *,
    kind: str = "both",
    encoder_factory: Callable = HuggingFaceEncoder,
) -> dict[str, Path]:
    spec = model_spec(config, alias)
    result = {}
    encoder: Encoder | None = None
    kinds = ("document", "query") if kind == "both" else (kind,)
    for input_kind in kinds:
        destination = embedding_path(config, corpus, spec, input_kind)
        if destination.exists():
            load_embeddings(destination, ["id"])
            result[input_kind] = destination
            continue
        encoder = encoder or encoder_factory(spec, config.runtime, hf_token(config.env_file))
        if input_kind == "document":
            source_dir = config.corpus_dir(corpus)
            manifest = read_json(source_dir / "manifest.json")
            count, fingerprint = manifest["chunk_count"], manifest["corpus_fingerprint"]

            def factory(directory=source_dir):
                return rows(directory / "chunks.parquet")
        else:
            source_dir = config.relevance_dir(corpus)
            manifest = read_json(source_dir / "manifest.json")
            count, fingerprint = manifest["selected_query_count"], manifest["query_fingerprint"]

            def factory(directory=source_dir):
                for row in rows(
                    directory / "queries.parquet", ["id", "question", "question_language"]
                ):
                    yield {
                        "id": row["id"],
                        "text": row["question"],
                        "language": row["question_language"],
                    }

        encode_records(
            destination, factory, count, encoder, input_kind, config.runtime, fingerprint
        )
        result[input_kind] = destination
    return result


def run_path(config: Config, corpus: str, spec: ModelSpec) -> Path:
    identity = {
        "schema": 1,
        "relevance": config.relevance_key(corpus),
        "documents": str(embedding_path(config, corpus, spec, "document")),
        "queries": str(embedding_path(config, corpus, spec, "query")),
        "retrieval": "exact_cosine",
        "top_k": 100,
        "query_batch_size": config.runtime.query_batch_size,
        "corpus_block_size": config.runtime.corpus_block_size,
    }
    return config.output_dir / "runs" / corpus / digest(identity)


def retrieve(config: Config, corpus: str, alias: str) -> Path:
    spec = model_spec(config, alias)
    destination = run_path(config, corpus, spec)
    if destination.exists():
        read_json(destination / "manifest.json")
        return destination
    relevance_dir = config.relevance_dir(corpus)
    settings = read_json(relevance_dir / "settings.json")
    documents, doc_rows, doc_manifest = load_embeddings(
        embedding_path(config, corpus, spec, "document"), ["id", "language"]
    )
    queries, query_rows, query_manifest = load_embeddings(
        embedding_path(config, corpus, spec, "query"), ["id"]
    )
    index = DenseIndex(
        documents, [row["id"] for row in doc_rows], [row["language"] for row in doc_rows]
    )
    del doc_rows
    query_index = {row["id"]: index for index, row in enumerate(query_rows)}
    eligible = defaultdict(list)
    for row in rows(relevance_dir / "eligibility.parquet"):
        if row["status"] == "eligible" and settings[row["setting_id"]]["status"] == "ready":
            eligible[row["setting_id"]].append(row["query_id"])
    if not any(item["status"] == "ready" for item in settings.values()):
        raise ValidationError(
            "No ready retrieval settings; supply the missing multilingual annotations first"
        )
    started = time.perf_counter()
    with new_directory(destination) as temporary:
        with ParquetSink(temporary / "run.parquet", RUN_SCHEMA) as sink:
            for setting_id, info in settings.items():
                if info["status"] != "ready":
                    continue
                ids = sorted(eligible[setting_id])
                if set(ids) - query_index.keys():
                    raise ValidationError(f"Missing query embeddings for {setting_id}")
                with (temporary / f"{setting_id}.trec").open("w", encoding="utf-8") as handle:
                    # Select queries in bounded batches; never copy the entire
                    # query matrix just to filter a language setting.
                    for start in range(0, len(ids), config.runtime.query_batch_size):
                        batch_ids = ids[start : start + config.runtime.query_batch_size]
                        vectors = queries[[query_index[query_id] for query_id in batch_ids]]
                        results = index.search(
                            vectors,
                            tuple(info["target_languages"]),
                            query_batch_size=config.runtime.query_batch_size,
                            corpus_block_size=config.runtime.corpus_block_size,
                        )
                        for query_id, ranking in zip(batch_ids, results, strict=True):
                            for rank, (chunk_id, score) in enumerate(ranking, 1):
                                sink.append(
                                    {
                                        "setting_id": setting_id,
                                        "query_id": query_id,
                                        "chunk_id": chunk_id,
                                        "rank": rank,
                                        "score": score,
                                    }
                                )
                                handle.write(
                                    f"{query_id}\tQ0\t{chunk_id}\t{rank}\t{score:.9g}\t{alias}\n"
                                )
        write_json(
            temporary / "manifest.json",
            {
                "schema": 1,
                "corpus": corpus,
                "model": asdict(spec),
                "relevance_hash": config.relevance_key(corpus),
                "corpus_fingerprint": doc_manifest["input_fingerprint"],
                "query_fingerprint": query_manifest["input_fingerprint"],
                "settings": settings,
                "top_k": 100,
                "similarity": "cosine_float32",
                "tie_break": "chunk_id_ascending",
                "runtime": asdict(config.runtime),
                "seconds": time.perf_counter() - started,
                "versions": versions(),
            },
        )
    return destination


def evaluate(config: Config, corpus: str, alias: str) -> Path:
    spec = model_spec(config, alias)
    run_dir = run_path(config, corpus, spec)
    if not (run_dir / "manifest.json").is_file():
        raise ValidationError(
            f"Retrieval run does not exist: {run_dir}. Run retrieve explicitly first."
        )
    destination = config.output_dir / "reports" / corpus / run_dir.name
    if destination.exists():
        read_json(destination / "report.json")
        return destination
    relevance_dir = config.relevance_dir(corpus)
    settings = read_json(relevance_dir / "settings.json")
    queries = {query.id: query for query in read_queries(relevance_dir / "queries.parquet")}
    qrels = defaultdict(lambda: defaultdict(set))
    for row in rows(relevance_dir / "qrels.parquet"):
        qrels[(row["setting_id"], row["query_id"])][row["chunk_id"]].add(row["required_source_id"])
    rankings = defaultdict(list)
    for row in rows(run_dir / "run.parquet"):
        rankings[(row["setting_id"], row["query_id"])].append((row["rank"], row["chunk_id"]))
    per_query = []
    for row in rows(relevance_dir / "eligibility.parquet"):
        setting = settings[row["setting_id"]]
        if row["status"] != "eligible" or setting["status"] != "ready":
            continue
        key = (row["setting_id"], row["query_id"])
        ranking = sorted(rankings[key])
        if [rank for rank, _ in ranking] != list(range(1, len(ranking) + 1)):
            raise ValidationError(f"Invalid run ranks for {key}")
        query = queries[row["query_id"]]
        scores = score_query(
            [chunk_id for _, chunk_id in ranking], qrels[key], set(row["required_source_ids"])
        )
        per_query.append(
            {
                "query_id": query.id,
                "corpus": corpus,
                "regime": setting["kind"],
                "setting_id": setting["id"],
                "query_language": query.question_language,
                "target_languages": ",".join(setting["target_languages"]),
                "generator": query.generator_model_id,
                "persona": query.persona,
                "source_count": len(row["required_source_ids"]),
                "quality_audit_status": query.metadata.get("quality_audit_status"),
                "metrics": scores,
            }
        )
    document_path = embedding_path(config, corpus, spec, "document")
    document_manifest = read_json(document_path / "manifest.json")
    positive_ids = {row["chunk_id"] for row in rows(relevance_dir / "qrels.parquet", ["chunk_id"])}
    document_rows = [
        row for row in rows(document_path / "rows.parquet") if row["id"] in positive_ids
    ]
    query_manifest = read_json(embedding_path(config, corpus, spec, "query") / "manifest.json")
    report = {
        "schema": 1,
        "corpus": corpus,
        "model": alias,
        "run": read_json(run_dir / "manifest.json"),
        "settings": settings,
        "candidate_selection": read_json(relevance_dir / "manifest.json"),
        "metrics": aggregate(per_query),
        "truncation": {
            "documents": document_manifest["truncation"],
            "queries": query_manifest["truncation"],
            "support_exposure": support_exposure(
                rows(relevance_dir / "matches.parquet"), document_rows
            ),
        },
        "versions": versions(),
    }
    with new_directory(destination) as temporary:
        write_json(temporary / "report.json", report)
        write_json(temporary / "per_query.json", per_query)
        flattened = [
            {**{key: value for key, value in row.items() if key != "metrics"}, **row["metrics"]}
            for row in per_query
        ]
        with (temporary / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = (
                list(flattened[0]) if flattened else ["query_id", "corpus", "regime", "setting_id"]
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened)
        setting_rows = [
            {**{key: value for key, value in row.items() if key != "metrics"}, **row["metrics"]}
            for row in report["metrics"]["setting"]
        ]
        with (temporary / "settings.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(setting_rows[0])
                if setting_rows
                else ["corpus", "regime", "setting_id", "query_count"],
            )
            writer.writeheader()
            writer.writerows(setting_rows)
    return destination
