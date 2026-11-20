from __future__ import annotations

import csv
import hashlib
import os
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

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
    DOCUMENT_QREL_SCHEMA,
    DOCUMENT_RUN_SCHEMA,
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
from .chunk_batches import checkpointed_chunks
from .chunking import Chunker, make_chunker
from .config import (
    BASELINES,
    VERIFIED_OVERLAP_POLICY,
    Config,
    SharedSentenceConfig,
    validate_retrieval_baselines,
)
from .embedding import (
    embedding_key,
    encode_records,
    load_embeddings,
    support_exposure,
)
from .encoders import Encoder, HuggingFaceEncoder, ModelSpec, load_registry
from .errors import ValidationError
from .hyde import load_hyde
from .lexical import BM25Index, bm25_path, bm25_scoring
from .metrics import aggregate, score_query
from .relevance import build_settings, judge, judge_documents
from .results import export_result
from .retrieval import DenseIndex, fuse_chunk_ids, reciprocal_rank_fusion
from .translation import load_translations, translation_summary
from .types import SourceUnit
from .util import (
    digest,
    hf_token,
    json_dumps,
    new_directory,
    read_json,
    versions,
    write_json,
)
from .verified_corpora import (
    iter_verified_units,
    verified_overlap_judgment,
    verified_overlap_matches,
    verified_query_from_row,
)


def _corpus_units(
    reader: DatasetReader,
    corpus: str,
    corpus_format: str,
    pair_sink: ParquetSink,
    pairs: list[tuple[str, str]],
) -> Iterator[SourceUnit]:
    if corpus_format == "verified-multilingual-v1":
        for unit in iter_verified_units(reader, corpus):
            for target in unit.metadata.get("paired_document_ids", []):
                pairs.append(tuple(sorted((unit.id, target))))
            yield unit
    elif corpus == "chemistry":
        for row in reader.read("default"):
            units, pair = chemistry_pair(row, reader.spec)
            pair_sink.append({"record_json": json_dumps(row)})
            if pair:
                pairs.append(pair)
            yield from units
    else:
        for row in reader.read("corpus"):
            yield legal_unit(row, corpus)


def prepare_corpus(
    config: Config,
    corpus: str,
    *,
    chunker: Chunker | None = None,
    work_dir: Path | None = None,
    workers: int = 1,
) -> Path:
    if type(workers) is not int or workers < 1:
        raise ValidationError("prepare-corpus requires at least one worker")
    destination = config.corpus_dir(corpus)
    if destination.exists():
        read_json(destination / "manifest.json")
        return destination
    token = hf_token(config.env_file)
    injected_chunker = chunker
    if workers > 1 and chunker is None and isinstance(config.chunking, SharedSentenceConfig):
        from .shared_chunks import shared_identity

        chunk_policy = shared_identity(config.chunking, config.model_registry)
    else:
        chunker = chunker or make_chunker(config, token)
        chunk_policy = chunker.manifest
    chunking_hash = digest(chunk_policy)
    reader = DatasetReader(config.datasets[corpus].corpus, token)
    started = time.perf_counter()
    with new_directory(destination) as staging, ExitStack() as work:
        temporary = staging
        if work_dir is not None:
            temporary = Path(
                work.enter_context(TemporaryDirectory(prefix=f"chunks-{corpus}-", dir=work_dir))
            )
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
            corpus_format = config.datasets[corpus].corpus_format
            units = _corpus_units(reader, corpus, corpus_format, pair_sink, pairs)
            if workers > 1:
                checkpoints = (
                    config.output_dir
                    / "chunk-checkpoints"
                    / corpus
                    / destination.name
                    / chunking_hash
                )
                prepared = checkpointed_chunks(
                    units, config, injected_chunker, chunking_hash, checkpoints, workers
                )
            else:
                prepared = ((unit, None) for unit in units)
            for unit, chunks in prepared:
                if not store.add_source(unit):
                    duplicate_count += 1
                    continue
                source_sink.append(source_record(unit))
                language_counts[unit.language] += 1
                for chunk in chunker.split(unit) if chunks is None else chunks:
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
                    "chunking": chunk_policy,
                    "chunks_hash": content.hexdigest(),
                }
            )
            manifest = {
                "schema": 1,
                "corpus": corpus,
                "corpus_fingerprint": fingerprint,
                "config_hash": destination.name,
                "source": config.corpus_source_identity(corpus),
                "corpus_format": corpus_format,
                "chunking": chunk_policy,
                "source_count": source_sink.count,
                "chunk_count": chunk_sink.count,
                "languages": dict(sorted(language_counts.items())),
                "duplicate_source_occurrences": duplicate_count,
                "accepted_direct_pairs": len(set(pairs)),
                "seconds": time.perf_counter() - started,
                "versions": versions(),
            }
        if work_dir is not None:
            copy_start = time.perf_counter()
            shutil.copytree(temporary, staging, dirs_exist_ok=True)
            manifest["publication_copy_seconds"] = time.perf_counter() - copy_start
        write_json(staging / "manifest.json", manifest)
    return destination


def prepare_relevance(config: Config, corpus: str, *, work_dir: Path | None = None) -> Path:
    destination = config.relevance_dir(corpus)
    if destination.exists():
        read_json(destination / "manifest.json")
        return destination
    corpus_dir = config.corpus_dir(corpus)
    token = hf_token(config.env_file)
    reader = DatasetReader(config.datasets[corpus].qa, token)
    started = time.perf_counter()
    with ExitStack() as stack:
        database = corpus_dir / "lookup.sqlite"
        if work_dir is not None:
            local = Path(
                stack.enter_context(TemporaryDirectory(dir=work_dir, prefix="clir-qrels-"))
            )
            shutil.copyfile(database, local / "lookup.sqlite")
            database = local / "lookup.sqlite"
        store = stack.enter_context(CorpusStore(database))
        verified = (
            config.datasets[corpus].corpus_format == "verified-multilingual-v1"
            and config.retrieval_unit == "chunk"
        )
        if corpus == "chemistry" and not verified:
            verify_chemistry_provenance(reader, store, config.datasets[corpus].corpus)
        raw_queries = reader.read("qas")
        if config.candidate_mode == "best-per-document":
            raw_queries = list(raw_queries)
        parse_query = verified_query_from_row if verified else query_from_row
        queries = [parse_query(row, corpus) for row in raw_queries]
        queries = apply_evidence_overlay(queries, config.evidence_file)
        source_ids = None
        if config.candidate_mode == "best-per-document":
            source_ids = {}
            for unit_id in {query.source_unit_id for query in queries}:
                unit = store.get_source(unit_id)
                if unit is None:
                    raise ValidationError(f"Missing canonical document mapping for {unit_id}")
                source_ids[unit_id] = unit.source_id
        selected, selection_audit = select_candidates(
            queries,
            config.candidate_mode,
            source_ids=source_ids,
            policy=config.candidate_selection,
        )
        document_mode = config.retrieval_unit == "document"
        judge_query = judge_documents if document_mode else judge
        result_id = "document_id" if document_mode else "chunk_id"
        associations = defaultdict(set)
        known_query_ids = {query.id for query in queries}
        verified_matches = verified_overlap_matches(selected, store) if verified else {}
        raw_associations = () if verified else reader.read("source_qrels")
        if config.candidate_mode == "best-per-document":
            raw_associations = list(raw_associations)
        for row in raw_associations:
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
            if config.candidate_mode == "best-per-document":
                with ParquetSink(temporary / "input_queries.parquet", QUERY_SCHEMA) as sink:
                    for query in queries:
                        sink.append(query_record(query))
                # Retain raw grader/provenance fields as well as normalized query
                # records, including candidates rejected before evaluation.
                for name, records in (
                    ("input_qas.jsonl", raw_queries),
                    ("input_source_qrels.jsonl", raw_associations),
                ):
                    with (temporary / name).open("w", encoding="utf-8") as handle:
                        for record in records:
                            handle.write(json_dumps(record) + "\n")
            with ParquetSink(temporary / "queries.parquet", QUERY_SCHEMA) as sink:
                for query in selected:
                    sink.append(query_record(query))
            write_json(temporary / "selection.json", selection_audit)
            status = {}
            with (
                ParquetSink(
                    temporary / "qrels.parquet",
                    DOCUMENT_QREL_SCHEMA if document_mode else QREL_SCHEMA,
                ) as qrel_sink,
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
                        judgment = (
                            verified_overlap_judgment(query, setting, verified_matches[query.id])
                            if verified
                            else judge_query(query, setting, store, associations[query.id])
                        )
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
                                record = asdict(match)
                                record["annotation_json"] = json_dumps(record.pop("annotation"))
                                match_sink.append({"setting_id": setting.id, **record})
                            for chunk_id, groups in sorted(judgment.relevant.items()):
                                writer.writerow([judgment.query_id, 0, chunk_id, 1])
                                for group in sorted(groups):
                                    qrel_sink.append(
                                        {
                                            "setting_id": setting.id,
                                            "query_id": judgment.query_id,
                                            result_id: chunk_id,
                                            "required_source_id": group,
                                            "relevance": 1,
                                        }
                                    )
            query_fingerprint = digest([(q.id, q.question_language, q.question) for q in selected])
            write_json(temporary / "settings.json", status)
            write_json(
                temporary / "manifest.json",
                {
                    "schema": 2,
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
                    **(
                        {"candidate_selection": asdict(config.candidate_selection)}
                        if config.candidate_mode == "best-per-document"
                        else {}
                    ),
                    **(
                        {
                            "retrieval_unit": "document",
                            "document_identity": "source_unit_id",
                            "relevance_basis": "known_source_associations",
                        }
                        if document_mode
                        else {}
                    ),
                    "selected_query_languages": dict(
                        sorted(Counter(q.question_language for q in selected).items())
                    ),
                    "alignment_contracts": dict(
                        Counter(q.evidence_alignment_contract or "unspecified" for q in selected)
                    ),
                    "alignment_eligibility": dict(
                        Counter(q.evidence_alignment_eligibility or "unspecified" for q in selected)
                    ),
                    **({"relevance_basis": VERIFIED_OVERLAP_POLICY} if verified else {}),
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


def _require_query_texts(
    artifact: Path, records: list[dict], *, score_refusals: bool = False
) -> None:
    """Reject unusable text unless it is an explicitly recorded, scoreable refusal."""
    missing = [
        row["id"]
        for row in records
        if (
            row.get("status") == "provider_refused"
            or not isinstance(row.get("text"), str)
            or not row["text"].strip()
        )
        and not (score_refusals and row.get("status") == "provider_refused" and row["text"] is None)
    ]
    if missing:
        raise ValidationError(
            f"{artifact}: {len(missing)} of {len(records)} queries lack usable generated text. "
            f"Retrieval requires the complete cohort. Affected IDs (first 10): {', '.join(missing[:10])}"
        )


def embedding_path(
    config: Config,
    corpus: str,
    spec: ModelSpec,
    kind: str,
    *,
    translation_language: str | None = None,
    hyde_language: str | None = None,
) -> Path:
    if translation_language is not None and hyde_language is not None:
        raise ValidationError("Choose either translation_language or hyde_language for embedding")
    if kind == "query" and hyde_language is not None:
        path, records, manifest = load_hyde(config, corpus, hyde_language)
        _require_query_texts(path, records, score_refusals=config.score_generation_refusals)
        return (
            config.output_dir
            / "embeddings"
            / "hyde"
            / embedding_key(manifest["passage_fingerprint"], "document", spec, config.runtime)
        )
    if kind == "query" and translation_language is not None:
        path, records, manifest = load_translations(config, corpus, translation_language)
        _require_query_texts(path, records, score_refusals=config.score_generation_refusals)
        return (
            config.output_dir
            / "embeddings"
            / kind
            / embedding_key(manifest["query_fingerprint"], kind, spec, config.runtime)
        )
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
    translation_language: str | None = None,
    hyde_language: str | None = None,
) -> dict[str, Path]:
    if translation_language is not None and hyde_language is not None:
        raise ValidationError("Choose either translation_language or hyde_language for embedding")
    if kind == "document" and hyde_language is not None:
        raise ValidationError(
            "hyde_language applies to query-side passages; use kind=query or both"
        )
    spec = model_spec(config, alias)
    result = {}
    encoder: Encoder | None = None
    kinds = ("document", "query") if kind == "both" else (kind,)
    destinations = {
        input_kind: embedding_path(
            config,
            corpus,
            spec,
            input_kind,
            translation_language=translation_language,
            hyde_language=hyde_language,
        )
        for input_kind in kinds
    }
    for input_kind, destination in destinations.items():
        require_complete = (
            input_kind == "document"
            and isinstance(config.chunking, SharedSentenceConfig)
            and alias in config.chunking.models
        )
        if destination.exists():
            _, _, cached = load_embeddings(destination, ["id"])
            if require_complete and cached["truncation"]["all"]["truncated"]:
                raise ValidationError(
                    f"Shared document embeddings contain truncated inputs: {destination}"
                )
            result[input_kind] = destination
            continue
        encoder = encoder or encoder_factory(spec, config.runtime, hf_token(config.env_file))
        if input_kind == "document":
            source_dir = config.corpus_dir(corpus)
            manifest = read_json(source_dir / "manifest.json")
            count, fingerprint = manifest["chunk_count"], manifest["corpus_fingerprint"]

            def factory(directory=source_dir):
                return rows(directory / "chunks.parquet")
        elif hyde_language is not None:
            _, passages, manifest = load_hyde(config, corpus, hyde_language)
            passages = [row for row in passages if row["text"] is not None]
            count, fingerprint = len(passages), manifest["passage_fingerprint"]

            def factory(records=passages):
                return iter(records)
        elif translation_language is not None:
            _, translated, manifest = load_translations(config, corpus, translation_language)
            translated = [row for row in translated if row["text"] is not None]
            count, fingerprint = len(translated), manifest["query_fingerprint"]

            def factory(records=translated):
                return iter(records)
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

        encoder_kind = "document" if hyde_language is not None else input_kind
        encode_records(
            destination,
            factory,
            count,
            encoder,
            encoder_kind,
            config.runtime,
            fingerprint,
            require_complete_inputs=require_complete,
        )
        result[input_kind] = destination
    return result


def run_path(
    config: Config,
    corpus: str,
    spec: ModelSpec | None = None,
    *,
    baseline: str = "dense",
    translation_language: str | None = None,
    hyde_language: str | None = None,
) -> Path:
    validate_retrieval_baselines(config, (baseline,))
    if baseline not in BASELINES:
        raise ValidationError(f"Unknown retrieval baseline: {baseline}")
    if baseline.startswith("rerank-"):
        from .rerank_pipeline import rerank_run_path

        return rerank_run_path(
            config,
            corpus,
            spec,
            baseline=baseline,
            translation_language=translation_language,
            hyde_language=hyde_language,
        )
    if baseline.startswith("adore-"):
        from .adore_pipeline import adore_run_path

        if translation_language is not None or hyde_language is not None:
            raise ValidationError(
                "ADORE preserves the original query language; no language overrides"
            )
        return adore_run_path(config, corpus, spec, baseline=baseline)
    if baseline.startswith("hyde-"):
        if hyde_language is None or translation_language is not None:
            raise ValidationError(f"{baseline} requires hyde_language and no translation_language")
        if spec is None and baseline != "hyde-bm25":
            raise ValidationError(f"{baseline} requires a dense model")
        path, records, manifest = load_hyde(config, corpus, hyde_language)
        _require_query_texts(path, records, score_refusals=config.score_generation_refusals)
        identity = {
            "schema": 1,
            "baseline": baseline,
            "output_language": hyde_language,
            "relevance": config.relevance_key(corpus),
            "passages": str(path),
            "passage_fingerprint": manifest["passage_fingerprint"],
            "top_k": 100,
            "query_batch_size": config.runtime.query_batch_size,
        }
        if spec is not None:
            identity.update(
                documents=str(embedding_path(config, corpus, spec, "document")),
                queries=str(
                    embedding_path(config, corpus, spec, "query", hyde_language=hyde_language)
                ),
                encoder_kind="document",
                retrieval="exact_cosine",
                corpus_block_size=config.runtime.corpus_block_size,
            )
        if baseline in {"hyde-bm25", "hyde-hybrid"}:
            identity.update(index=str(bm25_path(config, corpus)), bm25=bm25_scoring(config.bm25))
        if baseline == "hyde-hybrid":
            identity["fusion"] = {"method": "rrf", **asdict(config.hybrid)}
        if config.cache_hybrid_candidates:
            identity["candidate_depth"] = config.hybrid.candidate_k
        if config.score_generation_refusals:
            identity["refused_query_policy"] = "empty_ranking"
        return config.output_dir / "runs" / corpus / digest(identity)
    if hyde_language is not None:
        raise ValidationError("hyde_language requires a hyde-* baseline")
    if baseline.startswith("translate-"):
        if translation_language is None:
            raise ValidationError(f"{baseline} requires a translation_language")
        path, records, manifest = load_translations(config, corpus, translation_language)
        _require_query_texts(path, records, score_refusals=config.score_generation_refusals)
        identity = {
            "schema": 1,
            "baseline": baseline,
            "translation_language": translation_language,
            "translations": str(path),
            "query_fingerprint": manifest["query_fingerprint"],
            "original_run": str(
                run_path(config, corpus, spec, baseline=baseline.removeprefix("translate-"))
            ),
            "query_embeddings": str(
                embedding_path(
                    config, corpus, spec, "query", translation_language=translation_language
                )
            )
            if spec
            else None,
        }
        if config.score_generation_refusals:
            identity["refused_query_policy"] = "empty_ranking"
        return config.output_dir / "runs" / corpus / digest(identity)
    if translation_language is not None:
        raise ValidationError("translation_language requires a translate-* baseline")
    if baseline != "bm25" and spec is None:
        raise ValidationError(f"{baseline} retrieval requires a dense model")
    if baseline != "dense":
        identity = {
            "schema": 1,
            "baseline": baseline,
            "relevance": config.relevance_key(corpus),
            "index": str(bm25_path(config, corpus)),
            "bm25": bm25_scoring(config.bm25),
            "top_k": 100,
        }
        if config.cache_hybrid_candidates:
            identity["candidate_cache"] = {
                "depth": config.hybrid.candidate_k,
                "batch_size": config.runtime.query_batch_size,
            }
        if baseline == "hybrid":
            identity["dense"] = str(run_path(config, corpus, spec))
            identity["fusion"] = {"method": "rrf", **asdict(config.hybrid)}
        return config.output_dir / "runs" / corpus / digest(identity)
    # Preserve existing dense-run identities when adding lexical baselines.
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
    if config.cache_hybrid_candidates:
        identity["candidate_cache"] = {"depth": config.hybrid.candidate_k}
    return config.output_dir / "runs" / corpus / digest(identity)


def retrieval_spec(config: Config, alias: str | None, baseline: str) -> ModelSpec | None:
    validate_retrieval_baselines(config, (baseline,))
    if baseline not in BASELINES:
        raise ValidationError(f"Unknown retrieval baseline: {baseline}")
    if (
        baseline.removeprefix("rerank-")
        .removeprefix("translate-")
        .removeprefix("hyde-")
        .removeprefix("adore-")
        == "bm25"
    ):
        if alias is not None:
            raise ValidationError("BM25 does not use a dense model; omit the model alias")
        return None
    if alias is None:
        raise ValidationError(f"{baseline} requires a dense model alias")
    return model_spec(config, alias)


def retrieve(
    config: Config,
    corpus: str,
    alias: str | None = None,
    *,
    baseline: str = "dense",
    translation_language: str | None = None,
    hyde_language: str | None = None,
) -> Path:
    spec = retrieval_spec(config, alias, baseline)
    destination = run_path(
        config,
        corpus,
        spec,
        baseline=baseline,
        translation_language=translation_language,
        hyde_language=hyde_language,
    )
    if baseline.startswith("rerank-"):
        from .rerank_pipeline import load_rerank_run

        load_rerank_run(destination)
        return destination
    if baseline.startswith("adore-"):
        from .adore_pipeline import load_adore_run

        load_adore_run(destination)
        return destination
    if destination.exists():
        read_json(destination / "manifest.json")
        return destination
    relevance_dir = config.relevance_dir(corpus)
    settings = read_json(relevance_dir / "settings.json")
    eligible = defaultdict(list)
    for row in rows(relevance_dir / "eligibility.parquet"):
        if row["status"] == "eligible" and settings[row["setting_id"]]["status"] == "ready":
            eligible[row["setting_id"]].append(row["query_id"])
    if not any(item["status"] == "ready" for item in settings.values()):
        raise ValidationError(
            "No ready retrieval settings; supply the missing multilingual annotations first"
        )
    started = time.perf_counter()
    document_mode = config.retrieval_unit == "document"
    backend = baseline.removeprefix("translate-").removeprefix("hyde-")
    translated = None
    translation_manifest = None
    hyde_manifest = None
    passages = None
    if hyde_language is not None:
        _, passages, hyde_manifest = load_hyde(config, corpus, hyde_language)
    if translation_language is not None:
        _, translated, translation_manifest = load_translations(
            config, corpus, translation_language
        )
    generated = translated if translated is not None else passages
    refused = {row["id"] for row in generated or () if row.get("status") == "provider_refused"}
    cache_candidates = config.cache_hybrid_candidates
    fuse_cached = cache_candidates and backend == "hybrid"
    prefix = baseline.removesuffix(backend)
    component_paths = {}
    components = {}
    with ExitStack() as resources:
        if fuse_cached:
            for name, component_spec in (("dense", spec), ("bm25", None)):
                path = run_path(
                    config,
                    corpus,
                    component_spec,
                    baseline=prefix + name,
                    translation_language=translation_language,
                    hyde_language=hyde_language,
                )
                if not (path / "candidates.parquet").is_file():
                    raise ValidationError(
                        f"Prepare the {name} candidate run before hybrid retrieval: {path}"
                    )
                component_paths[name] = str(path)
                components[name] = rows(path / "candidates.parquet")

        index = None
        lexical = None
        query_index = {}
        query_texts = {}
        if spec is not None and not fuse_cached:
            documents, doc_rows, _ = load_embeddings(
                embedding_path(config, corpus, spec, "document"),
                ["id", "language", "source_unit_id"] if document_mode else ["id", "language"],
            )
            query_vectors, query_rows, _ = load_embeddings(
                embedding_path(
                    config,
                    corpus,
                    spec,
                    "query",
                    translation_language=translation_language,
                    hyde_language=hyde_language,
                ),
                ["id"],
            )
            index = DenseIndex(
                documents,
                [row["id"] for row in doc_rows],
                [row["language"] for row in doc_rows],
                **(
                    {"document_ids": [row["source_unit_id"] for row in doc_rows]}
                    if document_mode
                    else {}
                ),
            )
            del doc_rows
            query_index = {row["id"]: index for index, row in enumerate(query_rows)}
        lexical_manifest = None
        if backend in {"bm25", "hybrid"}:
            path = bm25_path(config, corpus)
            if not (path / "manifest.json").is_file() or not (path / "index.sqlite").is_file():
                raise ValidationError(f"BM25 index is not prepared: {path}. Run index-bm25 first.")
            lexical_manifest = read_json(path / "manifest.json")
        if backend in {"bm25", "hybrid"} and not fuse_cached:
            local = os.environ.get("SLURM_TMPDIR")
            if cache_candidates and local:
                copy = Path(
                    resources.enter_context(TemporaryDirectory(dir=local, prefix="bm25-search-"))
                )
                shutil.copyfile(path / "index.sqlite", copy / "index.sqlite")
                shutil.copyfile(path / "manifest.json", copy / "manifest.json")
                path = copy
            lexical = resources.enter_context(BM25Index(path, config.bm25))
            query_texts = {
                row["id"]: row["question"]
                for row in rows(relevance_dir / "queries.parquet", ["id", "question"])
            }
            if generated is not None:
                query_texts = {
                    row["id"]: row["text"] for row in generated if row["id"] not in refused
                }
        tag = "bm25" if backend == "bm25" else f"hybrid-{alias}" if backend == "hybrid" else alias
        if translation_language is not None:
            tag = f"translate-{translation_language}-{tag}"
        if hyde_language is not None:
            tag = f"hyde-{hyde_language}-{tag}"
        if document_mode:
            tag = f"document-{tag}"
        candidate_k = config.hybrid.candidate_k if backend == "hybrid" or cache_candidates else 100
        temporary = resources.enter_context(new_directory(destination))
        timing_schema = pa.schema(
            [
                ("setting_id", pa.string()),
                ("query_id", pa.string()),
                ("retrieval_seconds_amortized", pa.float64()),
            ]
        )
        if fuse_cached:
            timing_schema = pa.schema(
                [
                    *timing_schema,
                    ("dense_seconds_amortized", pa.float64()),
                    ("bm25_seconds_amortized", pa.float64()),
                    ("fusion_seconds_amortized", pa.float64()),
                ]
            )
        with (
            ExitStack() as writers,
            ParquetSink(
                temporary / "run.parquet", DOCUMENT_RUN_SCHEMA if document_mode else RUN_SCHEMA
            ) as sink,
            ParquetSink(temporary / "timings.parquet", timing_schema) as timing_sink,
        ):
            candidates = None
            if cache_candidates and not fuse_cached:
                schema = pa.schema(
                    [
                        ("setting_id", pa.string()),
                        ("query_id", pa.string()),
                        ("chunk_ids", pa.list_(pa.string())),
                        ("seconds", pa.float64()),
                    ]
                )
                candidates = writers.enter_context(
                    ParquetSink(temporary / "candidates.parquet", schema, batch_size=16)
                )
            for setting_id, info in settings.items():
                if info["status"] != "ready":
                    continue
                ids = sorted(eligible[setting_id])
                if index is not None and set(ids) - refused - query_index.keys():
                    raise ValidationError(f"Missing query embeddings for {setting_id}")
                if lexical is not None and set(ids) - refused - query_texts.keys():
                    raise ValidationError(f"Missing query text for {setting_id}")
                with (temporary / f"{setting_id}.trec").open("w", encoding="utf-8") as handle:
                    # Select queries in bounded batches; never copy the entire
                    # query matrix just to filter a language setting.
                    for start in range(0, len(ids), config.runtime.query_batch_size):
                        batch_ids = ids[start : start + config.runtime.query_batch_size]
                        search_ids = [query_id for query_id in batch_ids if query_id not in refused]
                        languages = tuple(info["target_languages"])
                        cached = {}
                        for name, stream in components.items():
                            cached[name] = []
                            for query_id in batch_ids:
                                record = next(stream, None)
                                if record is None or (record["setting_id"], record["query_id"]) != (
                                    setting_id,
                                    query_id,
                                ):
                                    raise ValidationError(
                                        f"{name} candidate cohort differs at {setting_id}/{query_id}"
                                    )
                                cached[name].append(record)
                        batch_started = time.perf_counter()
                        dense_results = (
                            (index.search_documents if document_mode else index.search)(
                                query_vectors[[query_index[query_id] for query_id in search_ids]],
                                languages,
                                k=candidate_k,
                                query_batch_size=config.runtime.query_batch_size,
                                corpus_block_size=config.runtime.corpus_block_size,
                            )
                            if index is not None
                            else ([] for _ in search_ids)
                        )
                        lexical_results = (
                            lexical.search(
                                (query_texts[query_id] for query_id in search_ids),
                                languages,
                                k=candidate_k,
                            )
                            if lexical is not None
                            else ([] for _ in search_ids)
                        )
                        batch_rankings = {}
                        if fuse_cached:
                            for query_id, dense_row, bm25_row in zip(
                                batch_ids, cached["dense"], cached["bm25"], strict=True
                            ):
                                ranking = fuse_chunk_ids(
                                    dense_row["chunk_ids"], bm25_row["chunk_ids"], config.hybrid
                                )
                                if query_id in refused and ranking:
                                    raise ValidationError(
                                        f"Refused query has cached candidates: {query_id}"
                                    )
                                batch_rankings[query_id] = ranking
                        else:
                            for query_id, dense_ranking, lexical_ranking in zip(
                                search_ids, dense_results, lexical_results, strict=True
                            ):
                                ranking = (
                                    reciprocal_rank_fusion(
                                        dense_ranking, lexical_ranking, config.hybrid
                                    )
                                    if backend == "hybrid"
                                    else lexical_ranking
                                    if backend == "bm25"
                                    else dense_ranking
                                )
                                batch_rankings[query_id] = ranking
                        batch_seconds = time.perf_counter() - batch_started
                        for position, query_id in enumerate(batch_ids):
                            ranking = batch_rankings.get(query_id, [])
                            seconds = (
                                None if query_id in refused else batch_seconds / len(search_ids)
                            )
                            component_times = {}
                            if fuse_cached:
                                component_times = {
                                    f"{name}_seconds_amortized": cached[name][position]["seconds"]
                                    for name in components
                                }
                                component_times["fusion_seconds_amortized"] = seconds
                                if query_id not in refused:
                                    seconds = sum(component_times.values())
                            if candidates is not None:
                                candidates.append(
                                    {
                                        "setting_id": setting_id,
                                        "query_id": query_id,
                                        "chunk_ids": [item[0] for item in ranking],
                                        "seconds": seconds,
                                    }
                                )
                                ranking = ranking[:100]
                            timing_sink.append(
                                {
                                    "setting_id": setting_id,
                                    "query_id": query_id,
                                    "retrieval_seconds_amortized": seconds,
                                    **component_times,
                                }
                            )
                            for rank, item in enumerate(ranking, 1):
                                if document_mode:
                                    result_id, winning_chunk_id, score = item
                                    identity = {
                                        "document_id": result_id,
                                        "winning_chunk_id": winning_chunk_id,
                                    }
                                else:
                                    result_id, score = item
                                    identity = {"chunk_id": result_id}
                                sink.append(
                                    {
                                        "setting_id": setting_id,
                                        "query_id": query_id,
                                        **identity,
                                        "rank": rank,
                                        "score": score,
                                    }
                                )
                                handle.write(
                                    f"{query_id}\tQ0\t{result_id}\t{rank}\t{score:.17g}\t{tag}\n"
                                )
            if any(next(stream, None) is not None for stream in components.values()):
                raise ValidationError("Unexpected trailing hybrid candidates")
        write_json(
            temporary / "manifest.json",
            {
                "schema": 2,
                "corpus": corpus,
                "baseline": baseline,
                **(
                    {
                        "retrieval_unit": "document",
                        "document_identity": "source_unit_id",
                        "document_scoring": "maximum_chunk_similarity_before_top_k",
                        "tie_breaking": "document_id_then_winning_chunk_id",
                    }
                    if document_mode
                    else {}
                ),
                "translation_language": translation_language,
                "translation": translation_manifest,
                **(
                    {"refused_query_policy": "empty_ranking", "refused_query_count": len(refused)}
                    if generated is not None and config.score_generation_refusals
                    else {}
                ),
                **(
                    {
                        "output_language": hyde_language,
                        "hyde": hyde_manifest,
                        **(
                            {
                                "search_vector_source": "hypothetical_document",
                                "search_encoder_kind": "document",
                            }
                            if spec is not None
                            else {}
                        ),
                        **({"bm25_input": "hypothetical_document"} if backend != "dense" else {}),
                    }
                    if hyde_manifest is not None
                    else {}
                ),
                "timing_contract": (
                    "sum of cached component search wall times and current fusion wall time, each amortized per batch; excludes artifact I/O; reconstructed serial total"
                    if fuse_cached
                    else "search and fusion wall time per batch divided by batch size; excludes model/index loading and result serialization"
                ),
                **(
                    {"candidate_depth": candidate_k, "component_runs": component_paths}
                    if cache_candidates
                    else {}
                ),
                "run_name": tag,
                "model": asdict(spec) if spec is not None else None,
                "relevance_hash": config.relevance_key(corpus),
                "corpus_fingerprint": read_json(config.corpus_dir(corpus) / "manifest.json")[
                    "corpus_fingerprint"
                ],
                "query_fingerprint": read_json(relevance_dir / "manifest.json")[
                    "query_fingerprint"
                ],
                "settings": settings,
                "top_k": 100,
                "similarity": "rrf"
                if backend == "hybrid"
                else "bm25_float64"
                if backend == "bm25"
                else "cosine_float32",
                "bm25": bm25_scoring(config.bm25) if lexical_manifest is not None else None,
                "lexical_index": {
                    "path": str(bm25_path(config, corpus)),
                    "manifest": lexical_manifest,
                }
                if lexical_manifest is not None
                else None,
                "fusion": {"method": "rrf", **asdict(config.hybrid)}
                if backend == "hybrid"
                else None,
                "tie_break": "document_id_then_winning_chunk_id_ascending"
                if document_mode
                else "chunk_id_ascending",
                "runtime": asdict(config.runtime),
                "seconds": time.perf_counter() - started,
                "versions": versions(),
            },
        )
    return destination


def evaluate(
    config: Config,
    corpus: str,
    alias: str | None = None,
    *,
    baseline: str = "dense",
    translation_language: str | None = None,
    hyde_language: str | None = None,
) -> Path:
    validate_retrieval_baselines(config, (baseline,))
    if baseline.startswith("rerank-"):
        from .rerank_pipeline import evaluate_rerank

        return evaluate_rerank(
            config,
            corpus,
            alias,
            baseline=baseline,
            translation_language=translation_language,
            hyde_language=hyde_language,
        )
    if baseline.startswith("adore-"):
        from .adore_pipeline import evaluate_adore

        if translation_language is not None or hyde_language is not None:
            raise ValidationError(
                "ADORE preserves the original query language; no language overrides"
            )
        return evaluate_adore(config, corpus, alias, baseline=baseline)
    spec = retrieval_spec(config, alias, baseline)
    run_dir = run_path(
        config,
        corpus,
        spec,
        baseline=baseline,
        translation_language=translation_language,
        hyde_language=hyde_language,
    )
    if not (run_dir / "manifest.json").is_file():
        raise ValidationError(
            f"Retrieval run does not exist: {run_dir}. Run retrieve explicitly first."
        )
    destination = config.output_dir / "reports" / corpus / run_dir.name
    if destination.exists():
        read_json(destination / "report.json")
        return export_result(config, corpus, run_dir, destination)
    relevance_dir = config.relevance_dir(corpus)
    document_mode = config.retrieval_unit == "document"
    result_id = "document_id" if document_mode else "chunk_id"
    settings = read_json(relevance_dir / "settings.json")
    queries = {query.id: query for query in read_queries(relevance_dir / "queries.parquet")}
    translated = {}
    hypothetical = {}
    hyde_manifest = None
    if hyde_language is not None:
        _, passage_rows, hyde_manifest = load_hyde(config, corpus, hyde_language)
        hypothetical = {row["id"]: row for row in passage_rows}
    if translation_language is not None:
        _, translated_rows, _ = load_translations(config, corpus, translation_language)
        translated = {row["id"]: row for row in translated_rows}
    timings = (
        {
            (row["setting_id"], row["query_id"]): row["retrieval_seconds_amortized"]
            for row in rows(run_dir / "timings.parquet")
        }
        if (run_dir / "timings.parquet").is_file()
        else {}
    )
    query_encoding_seconds = 0.0
    if spec is not None:
        query_manifest = read_json(
            embedding_path(
                config,
                corpus,
                spec,
                "query",
                translation_language=translation_language,
                hyde_language=hyde_language,
            )
            / "manifest.json"
        )
        query_encoding_seconds = (
            query_manifest["seconds"] / query_manifest["rows"] if query_manifest["rows"] else 0.0
        )
    qrels = defaultdict(lambda: defaultdict(set))
    for row in rows(relevance_dir / "qrels.parquet"):
        qrels[(row["setting_id"], row["query_id"])][row[result_id]].add(row["required_source_id"])
    rankings = defaultdict(list)
    for row in rows(run_dir / "run.parquet"):
        rankings[(row["setting_id"], row["query_id"])].append((row["rank"], row[result_id]))
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
                "baseline": baseline,
                **({"retrieval_unit": "document"} if document_mode else {}),
                "translation_language": translation_language,
                "translation_status": translated[query.id]["status"] if translated else "original",
                "translation_seconds": translated[query.id]["translation_seconds"]
                if translated
                else 0.0,
                "translation_cost_usd": translated[query.id]["estimated_cost_usd"]
                if translated
                else 0.0,
                **(
                    {
                        "translation_request_key": translated[query.id]["request_key"],
                        "translation_attempts": translated[query.id]["attempts"],
                        "translation_requested_model": translated[query.id].get("requested_model"),
                        "translation_returned_model": translated[query.id].get("returned_model"),
                        "translation_api_cost_usd": translated[query.id].get("api_cost_usd"),
                        **{
                            f"translation_{name}": translated[query.id][name]
                            for name in ("input_tokens", "cached_input_tokens", "output_tokens")
                        },
                    }
                    if translated
                    else {}
                ),
                "retrieval_seconds_amortized": timings.get(key),
                "query_encoding_seconds_amortized": None
                if (translated or hypothetical).get(query.id, {}).get("status")
                == "provider_refused"
                else query_encoding_seconds,
                **(
                    {
                        "output_language": hyde_language,
                        "generation_model": hypothetical[query.id].get(
                            "requested_model", hyde_manifest["configuration"]["model"]
                        ),
                        "generation_returned_model": hypothetical[query.id].get("returned_model"),
                        "generation_api_cost_usd": hypothetical[query.id].get("api_cost_usd"),
                        "generation_status": hypothetical[query.id].get("status", "generated"),
                        "generation_seconds": hypothetical[query.id]["generation_seconds"],
                        "generation_cost_usd": hypothetical[query.id]["estimated_cost_usd"],
                        "generation_attempts": hypothetical[query.id]["attempts"],
                        "generation_request_key": hypothetical[query.id]["request_key"],
                        **{
                            f"generation_{name}": hypothetical[query.id][name]
                            for name in ("input_tokens", "cached_input_tokens", "output_tokens")
                        },
                    }
                    if hypothetical
                    else {}
                ),
                "model": alias,
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
    truncation = {
        "applicable": False,
        "reason": "BM25 uses full chunk and query text; no input truncation",
    }
    if spec is not None:
        document_path = embedding_path(config, corpus, spec, "document")
        document_manifest = read_json(document_path / "manifest.json")
        if document_mode:
            exposure = {
                "applicable": False,
                "reason": "Document relevance uses known sources, not exact evidence spans",
            }
        else:
            positive_ids = {
                row["chunk_id"] for row in rows(relevance_dir / "qrels.parquet", ["chunk_id"])
            }
            document_rows = [
                row for row in rows(document_path / "rows.parquet") if row["id"] in positive_ids
            ]
            exposure = support_exposure(rows(relevance_dir / "matches.parquet"), document_rows)
        truncation = {
            "applicable": True,
            "scope": "dense_component"
            if baseline.removeprefix("translate-").removeprefix("hyde-") == "hybrid"
            else "dense",
            "documents": document_manifest["truncation"],
            "queries": query_manifest["truncation"],
            "support_exposure": exposure,
        }
        if hypothetical:
            truncation["hypothetical_documents"] = truncation.pop("queries")
            truncation["search_encoder_kind"] = "document"
    report = {
        "schema": 2,
        "corpus": corpus,
        "baseline": baseline,
        **(
            {
                "retrieval_unit": "document",
                "document_identity": "source_unit_id",
                "relevance_basis": "known_source_associations",
            }
            if document_mode
            else {}
        ),
        "translation_language": translation_language,
        "translation_summary": translation_summary(list(translated.values()))
        if translated
        else None,
        **(
            {
                "output_language": hyde_language,
                "generation_model": hyde_manifest["configuration"]["model"],
                "hyde_summary": hyde_manifest["summary"],
                "generation_timing_notes": "HyDE generation time includes all recorded attempts even on cache reuse. Hypothetical-passage encoding uses the document adapter; its offline build time is amortized over inputs. Costs are API estimates; local encoding and retrieval compute are unpriced.",
            }
            if hypothetical
            else {}
        ),
        "timing_notes": "Translation time includes all recorded API attempts, even on cache reuse. Retrieval time is amortized over each batch. Query encoding time is the cached embedding build wall time divided by its row count. These are offline component measurements, not measured online end-to-end latency.",
        "model": alias,
        "run": read_json(run_dir / "manifest.json"),
        "settings": settings,
        "candidate_selection": read_json(relevance_dir / "manifest.json"),
        "metrics": aggregate(per_query),
        "truncation": truncation,
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
            {
                "baseline": baseline,
                "model": alias,
                **(
                    {
                        "output_language": hyde_language,
                        "generation_model": hyde_manifest["configuration"]["model"],
                    }
                    if hypothetical
                    else {}
                ),
                **{key: value for key, value in row.items() if key != "metrics"},
                **row["metrics"],
            }
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
    return export_result(config, corpus, run_dir, destination)
