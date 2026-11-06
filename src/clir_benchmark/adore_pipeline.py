from __future__ import annotations

import csv
import time
import uuid
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa

from .adore import AdoreSession, prompts_identity
from .artifacts import RUN_SCHEMA, CorpusStore, ParquetSink, read_queries, rows
from .config import Config
from .embedding import embedding_key, encode_records, load_embeddings, support_exposure
from .encoders import HuggingFaceEncoder, ModelSpec
from .errors import ValidationError
from .generation import nullable_sum
from .lexical import BM25Index, bm25_path, bm25_scoring
from .metrics import aggregate, score_query
from .retrieval import DenseIndex, reciprocal_rank_fusion
from .util import digest, file_digest, hf_token, new_directory, read_json, versions, write_json


def adore_identity(config: Config, corpus: str, spec: ModelSpec | None, baseline: str) -> dict:
    from .pipeline import embedding_path

    backend = baseline.removeprefix("adore-")
    if baseline not in {"adore-bm25", "adore-dense", "adore-hybrid"}:
        raise ValidationError(f"Unknown ADORE baseline: {baseline}")
    if (backend == "bm25") != (spec is None):
        raise ValidationError("ADORE BM25 takes no model; dense/hybrid require a model")
    corpus_manifest = read_json(config.corpus_dir(corpus) / "manifest.json")
    return {
        "schema": 1,
        "baseline": baseline,
        "corpus_fingerprint": corpus_manifest["corpus_fingerprint"],
        "configuration": asdict(config.adore),
        "prompts": prompts_identity(),
        "language_policy": "original_query_language_only",
        "ranking_policy": "last_expanded_query_no_inter_round_fusion",
        "model": asdict(spec) if spec else None,
        "documents": str(embedding_path(config, corpus, spec, "document")) if spec else None,
        "bm25": bm25_scoring(config.bm25) if backend in {"bm25", "hybrid"} else None,
        "index": str(bm25_path(config, corpus)) if backend in {"bm25", "hybrid"} else None,
        "fusion": asdict(config.hybrid) if backend == "hybrid" else None,
        "runtime": asdict(config.runtime),
        "top_k": 100,
        "versions": versions(),
    }


def adore_run_path(
    config: Config,
    corpus: str,
    spec: ModelSpec | None = None,
    *,
    baseline: str = "adore-bm25",
) -> Path:
    identity = {
        **adore_identity(config, corpus, spec, baseline),
        "relevance": config.relevance_key(corpus),
    }
    return config.output_dir / "runs" / corpus / digest(identity)


def load_adore_run(path: Path) -> dict:
    if not (path / "manifest.json").is_file():
        raise ValidationError(f"ADORE run is not prepared: {path}. Run run-adore explicitly first.")
    manifest = read_json(path / "manifest.json")
    for name, expected in manifest["files"].items():
        if not (path / name).is_file() or file_digest(path / name) != expected:
            raise ValidationError(f"Corrupt ADORE run artifact: {path / name}")
    return manifest


class _SearchStages:
    """Explicit per-expansion encoding/search stages with reusable immutable artifacts."""

    def __init__(self, config, corpus, spec, baseline, resources, encoder_factory):
        from .pipeline import embedding_path

        self.config, self.corpus, self.spec = config, corpus, spec
        self.backend = baseline.removeprefix("adore-")
        self.encoder_factory, self.encoder = encoder_factory, None
        self.identity = adore_identity(config, corpus, spec, baseline)
        # Search is independent of LLM model, pricing, round budget, qrels and query IDs.
        self.search_identity = {
            key: self.identity[key]
            for key in (
                "schema",
                "baseline",
                "corpus_fingerprint",
                "model",
                "documents",
                "bm25",
                "index",
                "fusion",
                "runtime",
                "top_k",
                "versions",
            )
        }
        self.index = self.lexical = None
        self.searches, self.embeddings = {}, {}
        self.new_searches, self.new_embeddings = set(), set()
        self.search_cache_hits = self.embedding_cache_hits = 0
        if spec:
            vectors, metadata, _ = load_embeddings(
                embedding_path(config, corpus, spec, "document"), ["id", "language"]
            )
            self.index = DenseIndex(
                vectors, [row["id"] for row in metadata], [row["language"] for row in metadata]
            )
        if self.backend in {"bm25", "hybrid"}:
            self.lexical = resources.enter_context(
                BM25Index(bm25_path(config, corpus), config.bm25)
            )

    def search(self, text, language, target_languages):
        record = {"id": digest([language, text]), "language": language, "text": text}
        key = digest({**self.search_identity, "query": record, "languages": target_languages})
        destination = self.config.output_dir / "adore_searches" / key
        if destination.exists():
            result = read_json(destination / "result.json")
            if result["identity"] != key or digest(result["ranking"]) != result["ranking_digest"]:
                raise ValidationError(f"Corrupt ADORE search artifact: {destination}")
            self.search_cache_hits += 1
            self.searches[key] = result
            if result["query_embedding_path"]:
                embedding_path = Path(result["query_embedding_path"])
                _, _, manifest = load_embeddings(embedding_path)
                self.embeddings[str(embedding_path)] = manifest
                self.embedding_cache_hits += 1
            return {**result, "cached": True, "embedding_cached": bool(self.spec)}
        vectors = None
        query_embedding_path = None
        embedding_cached = False
        if self.spec:
            fingerprint = digest([(record["id"], record["language"], record["text"])])
            query_embedding_path = (
                self.config.output_dir
                / "embeddings"
                / "adore"
                / embedding_key(fingerprint, "query", self.spec, self.config.runtime)
            )
            embedding_cached = query_embedding_path.exists()
            if not embedding_cached:
                if self.encoder is None:
                    self.encoder = self.encoder_factory(
                        self.spec, self.config.runtime, hf_token(self.config.env_file)
                    )
                encode_records(
                    query_embedding_path,
                    lambda: iter([record]),
                    1,
                    self.encoder,
                    "query",
                    self.config.runtime,
                    fingerprint,
                )
                self.new_embeddings.add(str(query_embedding_path))
            else:
                self.embedding_cache_hits += 1
            vectors, _, manifest = load_embeddings(query_embedding_path)
            self.embeddings[str(query_embedding_path)] = manifest
        candidate_k = self.config.hybrid.candidate_k if self.backend == "hybrid" else 100
        started = time.perf_counter()
        dense = (
            next(
                self.index.search(
                    vectors,
                    target_languages,
                    k=candidate_k,
                    query_batch_size=self.config.runtime.query_batch_size,
                    corpus_block_size=self.config.runtime.corpus_block_size,
                )
            )
            if self.index is not None
            else []
        )
        lexical = (
            next(self.lexical.search([text], target_languages, k=candidate_k))
            if self.lexical
            else []
        )
        ranking = (
            reciprocal_rank_fusion(dense, lexical, self.config.hybrid)
            if self.backend == "hybrid"
            else lexical
            if self.backend == "bm25"
            else dense
        )
        seconds = time.perf_counter() - started
        result = {
            "identity": key,
            "ranking": ranking,
            "ranking_digest": digest(ranking),
            "retrieval_seconds": seconds,
            "query_embedding_path": str(query_embedding_path) if query_embedding_path else None,
        }
        with new_directory(destination) as temporary:
            write_json(temporary / "result.json", result)
        self.searches[key] = result
        self.new_searches.add(key)
        return {**result, "cached": False, "embedding_cached": embedding_cached}

    def totals(self, search_keys=None, *, current=False):
        keys = set(self.searches if search_keys is None else search_keys)
        embedding_paths = {
            self.searches[key]["query_embedding_path"]
            for key in keys
            if self.searches[key]["query_embedding_path"]
        }
        if current:
            keys &= self.new_searches
            # A newly encoded input still incurred work if its subsequent search failed.
            embedding_paths = (
                self.new_embeddings
                if search_keys is None
                else embedding_paths & self.new_embeddings
            )
        return {
            "retrieval_seconds": nullable_sum(
                self.searches[key]["retrieval_seconds"] for key in keys
            ),
            "query_encoding_seconds": nullable_sum(
                self.embeddings[path]["seconds"] for path in embedding_paths
            ),
            "search_count": len(keys),
            "query_embedding_count": len(embedding_paths),
        }


def run_adore(
    config: Config,
    corpus: str,
    alias: str | None = None,
    *,
    baseline: str = "adore-bm25",
    client_factory=None,
    encoder_factory=HuggingFaceEncoder,
    retry_failed: bool = False,
) -> Path:
    from .pipeline import retrieval_spec

    spec = retrieval_spec(config, alias, baseline)
    destination = adore_run_path(config, corpus, spec, baseline=baseline)
    invocation_path = config.output_dir / "adore_invocations" / f"{uuid.uuid4().hex}.json"
    invocation = {
        "schema": 1,
        "corpus": corpus,
        "baseline": baseline,
        "model": alias,
        "run": str(destination),
        "artifact_reused": False,
        "status": "failed",
    }
    session, stages = None, None
    started = time.perf_counter()
    try:
        if destination.exists():
            load_adore_run(destination)
            invocation.update(
                {
                    "artifact_reused": True,
                    "status": "success",
                    "new_work": {
                        "attempts": 0,
                        "api_attempts": 0,
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "output_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "generation_seconds": 0.0,
                    },
                    "new_retrieval_seconds": 0.0,
                    "new_query_encoding_seconds": 0.0,
                }
            )
            return destination
        relevance_dir = config.relevance_dir(corpus)
        settings = read_json(relevance_dir / "settings.json")
        eligible = defaultdict(list)
        for row in rows(relevance_dir / "eligibility.parquet"):
            if row["status"] == "eligible" and settings[row["setting_id"]]["status"] == "ready":
                eligible[row["setting_id"]].append(row["query_id"])
        if not any(eligible.values()):
            raise ValidationError("No eligible queries in ready settings for ADORE")
        queries = {
            row["id"]: row
            for row in rows(
                relevance_dir / "queries.parquet", ["id", "question", "question_language"]
            )
        }
        with ExitStack() as resources:
            # Check expensive shared prerequisites before making a paid generation call.
            store = resources.enter_context(
                CorpusStore(config.corpus_dir(corpus) / "lookup.sqlite")
            )
            stages = _SearchStages(config, corpus, spec, baseline, resources, encoder_factory)
            options = {"client_factory": client_factory} if client_factory is not None else {}
            session = AdoreSession(config, corpus, retry_failed=retry_failed, **options)
            traces = []
            temporary = resources.enter_context(new_directory(destination))
            with ParquetSink(temporary / "run.parquet", RUN_SCHEMA) as sink:
                for setting_id in sorted(eligible):
                    setting = settings[setting_id]
                    with (temporary / f"{setting_id}.trec").open("w", encoding="utf-8") as trec:
                        for query_id in sorted(eligible[setting_id]):
                            query = queries[query_id]
                            loop_started = time.perf_counter()
                            feedback, judgments_by_id = [], {}
                            rounds, request_keys, search_keys = [], set(), set()
                            previous_top, unchanged = None, 0
                            any_cached = False
                            stop_reason = "max_rounds"
                            for round_number in range(1, config.adore.max_rounds + 1):
                                expansion = session.expand(query, round_number, feedback)
                                request_keys.add(expansion["request_key"])
                                any_cached |= expansion["cached"] or expansion["attempts"] > 1
                                search = stages.search(
                                    expansion["text"],
                                    query["question_language"],
                                    tuple(setting["target_languages"]),
                                )
                                search_keys.add(search["identity"])
                                any_cached |= search["cached"] or search["embedding_cached"]
                                ranking = search["ranking"]
                                top = [item[0] for item in ranking[: config.adore.feedback_k]]
                                current_judgments = []
                                for chunk_id in top:
                                    if chunk_id not in judgments_by_id:
                                        chunk = store.chunk(chunk_id)
                                        if chunk is None:
                                            raise ValidationError(
                                                f"Retrieved unknown chunk: {chunk_id}"
                                            )
                                        source = store.get_source(chunk["source_unit_id"])
                                        if source is None:
                                            raise ValidationError(
                                                f"Missing source for chunk: {chunk_id}"
                                            )
                                        document = {
                                            "id": chunk["id"],
                                            "language": chunk["language"],
                                            "text": source.text[chunk["start"] : chunk["end"]],
                                        }
                                        judgment = session.judge(query, document)
                                        request_keys.add(judgment["request_key"])
                                        any_cached |= judgment["cached"] or judgment["attempts"] > 1
                                        judgments_by_id[chunk_id] = {**document, **judgment}
                                    current_judgments.append(judgments_by_id[chunk_id])
                                feedback = [judgments_by_id[key] for key in sorted(judgments_by_id)]
                                unchanged = unchanged + 1 if top == previous_top else 0
                                previous_top = top
                                rounds.append(
                                    {
                                        "round": round_number,
                                        "expansion": expansion,
                                        "ranking": ranking,
                                        "judgments": current_judgments,
                                        "search_key": search["identity"],
                                        "query_embedding_path": search["query_embedding_path"],
                                        "retrieval_seconds": search["retrieval_seconds"],
                                        "search_cached": search["cached"],
                                    }
                                )
                                if len(top) == config.adore.feedback_k and all(
                                    item["score"] == 3 for item in current_judgments
                                ):
                                    stop_reason = "all_feedback_max_relevance"
                                    break
                                if unchanged >= 2:
                                    stop_reason = "unchanged_top_documents"
                                    break
                            accounting = session.summary(request_keys)
                            components = stages.totals(search_keys)
                            loop_seconds = time.perf_counter() - loop_started
                            trace = {
                                "setting_id": setting_id,
                                "query_id": query_id,
                                "output_language": query["question_language"],
                                "final_query": expansion["text"],
                                "rounds": rounds,
                                "rounds_completed": len(rounds),
                                "stop_reason": stop_reason,
                                "request_keys": sorted(request_keys),
                                "search_keys": sorted(search_keys),
                                "generation_accounting": accounting,
                                **components,
                                "component_seconds": nullable_sum(
                                    [
                                        accounting["generation_seconds"],
                                        components["retrieval_seconds"],
                                        components["query_encoding_seconds"],
                                    ]
                                ),
                                "query_loop_seconds_current_invocation": loop_seconds,
                                "end_to_end_seconds": None if any_cached else loop_seconds,
                                "cached_work_used": any_cached,
                            }
                            traces.append(trace)
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
                                trec.write(
                                    f"{query_id}\tQ0\t{chunk_id}\t{rank}\t{score:.17g}\t{baseline}\n"
                                )
            write_json(temporary / "traces.json", traces)
            timing_schema = pa.schema(
                [("setting_id", pa.string()), ("query_id", pa.string())]
                + [
                    (name, pa.float64())
                    for name in (
                        "retrieval_seconds",
                        "query_encoding_seconds",
                        "component_seconds",
                        "end_to_end_seconds",
                        "query_loop_seconds_current_invocation",
                    )
                ]
            )
            with ParquetSink(temporary / "timings.parquet", timing_schema) as sink:
                for trace in traces:
                    sink.append(trace)
            corpus_manifest = read_json(config.corpus_dir(corpus) / "manifest.json")
            relevance_manifest = read_json(relevance_dir / "manifest.json")
            write_json(
                temporary / "manifest.json",
                {
                    **adore_identity(config, corpus, spec, baseline),
                    "corpus": corpus,
                    "selected_baselines": list(config.baselines),
                    "executed_baseline": baseline,
                    "relevance_hash": config.relevance_key(corpus),
                    "corpus_fingerprint": corpus_manifest["corpus_fingerprint"],
                    "query_fingerprint": relevance_manifest["query_fingerprint"],
                    "settings": settings,
                    "adore_summary": session.summary(),
                    "component_summary": stages.totals(),
                    "request_keys": sorted(
                        {key for trace in traces for key in trace["request_keys"]}
                    ),
                    "query_embedding_paths": sorted(stages.embeddings),
                    "query_setting_count": len(traces),
                    "query_count": len({trace["query_id"] for trace in traces}),
                    "timing_contract": (
                        "Generation includes original recorded attempts, including failed/retried attempts. "
                        "Encoding is original per-input artifact build time; search is measured per expansion. "
                        "Component totals deduplicate shared requests, searches and embeddings. Query end_to_end_seconds "
                        "measures the query loop excluding shared index setup, including lazy encoder initialization; "
                        "it is null when any required work was cached or an API request has historical failed attempts. "
                        "Local compute costs are unpriced. "
                        "Current invocation work is recorded separately in adore_invocations."
                    ),
                    "seconds": time.perf_counter() - started,
                    "files": {
                        path.name: file_digest(path)
                        for path in temporary.iterdir()
                        if path.is_file()
                    },
                },
            )
        invocation["status"] = "success"
        return destination
    except BaseException as exc:
        invocation["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        invocation["error_type"] = type(exc).__name__
        raise
    finally:
        invocation["seconds"] = time.perf_counter() - started
        if session is not None:
            invocation.update(session.invocation_summary())
        if stages is not None:
            current = stages.totals(current=True)
            invocation.update(
                {
                    "new_retrieval_seconds": current["retrieval_seconds"],
                    "new_query_encoding_seconds": current["query_encoding_seconds"],
                    "new_search_count": current["search_count"],
                    "new_query_embedding_count": current["query_embedding_count"],
                    "search_cache_hits": stages.search_cache_hits,
                    "embedding_cache_hits": stages.embedding_cache_hits,
                }
            )
        write_json(invocation_path, invocation)


def evaluate_adore(config: Config, corpus: str, alias=None, *, baseline="adore-bm25") -> Path:
    from .pipeline import embedding_path, retrieval_spec

    spec = retrieval_spec(config, alias, baseline)
    run_dir = adore_run_path(config, corpus, spec, baseline=baseline)
    manifest = load_adore_run(run_dir)
    destination = config.output_dir / "reports" / corpus / run_dir.name
    if destination.exists():
        read_json(destination / "report.json")
        return destination
    relevance_dir = config.relevance_dir(corpus)
    settings = read_json(relevance_dir / "settings.json")
    queries = {query.id: query for query in read_queries(relevance_dir / "queries.parquet")}
    traces = {
        (row["setting_id"], row["query_id"]): row for row in read_json(run_dir / "traces.json")
    }
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
        if key not in traces or [rank for rank, _ in ranking] != list(range(1, len(ranking) + 1)):
            raise ValidationError(f"Missing ADORE trace or invalid run ranks for {key}")
        query, trace = queries[row["query_id"]], traces[key]
        accounting = trace["generation_accounting"]
        per_query.append(
            {
                "query_id": query.id,
                "baseline": baseline,
                "model": alias,
                "corpus": corpus,
                "regime": setting["kind"],
                "setting_id": setting["id"],
                "query_language": query.question_language,
                "output_language": trace["output_language"],
                "target_languages": ",".join(setting["target_languages"]),
                "generator": query.generator_model_id,
                "persona": query.persona,
                "source_count": len(row["required_source_ids"]),
                "quality_audit_status": query.metadata.get("quality_audit_status"),
                "generation_model": config.adore.model,
                "generation_seconds": accounting["generation_seconds"],
                "generation_cost_usd": accounting["estimated_cost_usd"],
                "generation_attempts": accounting["attempts"],
                **{
                    f"generation_{name}": accounting[name]
                    for name in ("input_tokens", "cached_input_tokens", "output_tokens")
                },
                **{
                    name: trace[name]
                    for name in (
                        "retrieval_seconds",
                        "query_encoding_seconds",
                        "component_seconds",
                        "end_to_end_seconds",
                        "query_loop_seconds_current_invocation",
                        "request_keys",
                        "stop_reason",
                    )
                },
                "rounds": trace["rounds_completed"],
                "metrics": score_query(
                    [chunk_id for _, chunk_id in ranking],
                    qrels[key],
                    set(row["required_source_ids"]),
                ),
            }
        )
    truncation = {"applicable": False, "reason": "BM25 consumes full queries and chunks"}
    if spec:
        document_path = embedding_path(config, corpus, spec, "document")
        positive_ids = {
            row["chunk_id"] for row in rows(relevance_dir / "qrels.parquet", ["chunk_id"])
        }
        document_rows = [
            row for row in rows(document_path / "rows.parquet") if row["id"] in positive_ids
        ]
        truncation = {
            "applicable": True,
            "scope": "dense_component" if baseline == "adore-hybrid" else "dense",
            "documents": read_json(document_path / "manifest.json")["truncation"],
            "expanded_queries": {
                path: read_json(Path(path) / "manifest.json")["truncation"]
                for path in manifest["query_embedding_paths"]
            },
            "support_exposure": support_exposure(
                rows(relevance_dir / "matches.parquet"), document_rows
            ),
        }
    report = {
        "schema": 2,
        "corpus": corpus,
        "baseline": baseline,
        "model": alias,
        "generation_model": config.adore.model,
        "adore_summary": manifest["adore_summary"],
        "component_summary": manifest["component_summary"],
        "run": manifest,
        "timing_notes": manifest["timing_contract"],
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
        _write_csv(temporary / "per_query.csv", flattened, ["query_id", "setting_id"])
        setting_rows = [
            {
                "baseline": baseline,
                "model": alias,
                **{key: value for key, value in row.items() if key != "metrics"},
                **row["metrics"],
            }
            for row in report["metrics"]["setting"]
        ]
        _write_csv(
            temporary / "settings.csv",
            setting_rows,
            ["corpus", "regime", "setting_id", "query_count"],
        )
    return destination


def _write_csv(path, records, empty_fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else empty_fields)
        writer.writeheader()
        writer.writerows(records)
