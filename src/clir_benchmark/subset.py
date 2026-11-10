"""Explicit, reproducible small legal-corpus subsets for execution smoke tests.

The upstream QA/corpus shards are read once, but only referenced source units and
a bounded distractor reservoir are retained. No embedding or generation runs here.
"""

from __future__ import annotations

import heapq
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .adapters import (
    DatasetReader,
    apply_evidence_overlay,
    canonical_source_ids,
    legal_unit,
    query_from_row,
    select_candidates,
)
from .artifacts import CorpusStore
from .chunking import Chunker, load_tokenizer
from .config import Config
from .errors import ValidationError
from .evidence import parse_annotations
from .relevance import build_settings, judge, judge_documents
from .util import digest, file_digest, hf_token, new_directory, read_json, versions, write_json


def _plain(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _parquet(path: Path, records: list[dict], *, schema=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path, compression="zstd")


def prepare_subset(
    config: Config,
    corpus: str,
    *,
    max_questions: int = 2,
    max_sources: int = 12,
    seed: int = 0,
    destination: Path | None = None,
    chunker: Chunker | None = None,
) -> Path:
    """Save exact upstream rows, an auditable selection, and a runnable config.

    Chunk questions must be eligible in every requested setting; document questions
    need an available indexed source in at least one setting. Required and annotated
    sources are retained even outside the target language pool. In ``best`` mode,
    complete candidate groups accompany each winner, preserving the selection
    algorithm. Invalid *unselected* upstream groups are explicitly listed in the
    selection audit; no replacement candidate is substituted for a dropped winner.
    """
    started = time.perf_counter()
    if corpus not in {"eurlex", "un"}:
        raise ValidationError("Small subset preparation currently supports eurlex and un")
    if corpus not in config.datasets:
        raise ValidationError(f"Corpus is not configured: {corpus}")
    for name, value in (("max_questions", max_questions), ("max_sources", max_sources)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError("seed must be an integer")
    resolved = _plain(asdict(config))
    identity_config = dict(resolved)
    if config.candidate_mode != "best-per-document":
        identity_config.pop("candidate_selection")
    if config.retrieval_unit == "chunk":
        identity_config.pop("retrieval_unit")
    identity = {
        "schema": 1,
        "method": "eligible-question-hash-order-with-referenced-sources-v1",
        "corpus": corpus,
        "max_questions": max_questions,
        "max_sources": max_sources,
        "seed": seed,
        "config": identity_config,
        "source_identities": {
            kind: getattr(config.datasets[corpus], kind).identity() for kind in ("qa", "corpus")
        },
        "evidence_hash": file_digest(config.evidence_file) if config.evidence_file else None,
    }
    destination = (
        Path(destination).resolve()
        if destination is not None
        else config.output_dir / "subsets" / corpus / digest(identity)
    )
    if destination.exists():
        manifest = read_json(destination / "manifest.json")
        if manifest["identity"] != identity:
            raise ValidationError(f"Subset destination has a different selection: {destination}")
        for name, expected in manifest["files"].items():
            if file_digest(destination / name) != expected:
                raise ValidationError(f"Subset artifact changed: {name}")
        return destination

    token = hf_token(config.env_file)
    reader = DatasetReader(config.datasets[corpus].qa, token)
    raw_queries = list(reader.read("qas"))
    queries = apply_evidence_overlay(
        [query_from_row(row, corpus) for row in raw_queries], config.evidence_file
    )
    raw_by_id = {row["_id"]: row for row in raw_queries}
    if len(raw_by_id) != len(raw_queries):
        raise ValidationError("Duplicate QA IDs in subset input")
    source_reader = DatasetReader(config.datasets[corpus].corpus, token)
    document_selection = config.candidate_mode == "best-per-document"
    document_retrieval = config.retrieval_unit == "document"
    source_ids = (
        canonical_source_ids(
            source_reader, corpus, {unit for query in queries for unit in query.required_units}
        )
        if document_selection or document_retrieval
        else {}
    )
    groups = defaultdict(list)
    for query in queries:
        groups[query.task_id].append(query)
    candidates, selection_audit, rejected_groups = [], [], []
    if document_selection:
        winners, selection_audit = select_candidates(
            queries, config.candidate_mode, source_ids=source_ids, policy=config.candidate_selection
        )
        candidates = [
            query
            for query in winners
            if not config.query_languages or query.question_language in config.query_languages
        ]
        rejected_groups = [
            {"canonical_source_id": row["canonical_source_id"], "reason": row["status"]}
            for row in selection_audit
            if not row["selected_ids"]
        ]
    else:
        for task_id, group in sorted(groups.items()):
            try:
                winners, audit = select_candidates(group, config.candidate_mode)
            except ValidationError as exc:
                rejected_groups.append({"task_id": task_id, "reason": str(exc)})
                continue
            candidates.extend(
                query
                for query in winners
                if not config.query_languages or query.question_language in config.query_languages
            )
            selection_audit.extend(audit)
    candidates.sort(key=lambda query: (digest([seed, query.id]), query.id))
    if not candidates:
        raise ValidationError("No valid candidate groups match the subset query languages")
    needed_ids = {
        unit_id
        for query in candidates
        for unit_id in (
            *query.required_units,
            *([] if document_retrieval else parse_annotations(query)),
        )
    }
    candidate_source_units = defaultdict(set)
    if document_selection:
        for query in queries:
            candidate_source_units[source_ids[query.source_unit_id]].add(query.source_unit_id)
        for query in candidates:
            needed_ids.update(candidate_source_units[source_ids[query.source_unit_id]])
    needed_canonical = {source_ids[unit_id] for unit_id in needed_ids if unit_id in source_ids}
    source_qrels = list(reader.read("source_qrels"))
    associations = defaultdict(set)
    for row in source_qrels:
        if row.get("score", 0) > 0:
            associations[row["query_id"]].add(row["corpus_id"])
    raw_sources, reservoir = {}, []
    reservoir_rows = {}
    upstream_languages = Counter()
    # The bounded reservoir avoids retaining the full original corpus in RAM or
    # preparing its chunks. A few extra candidates allow exclusion of positives.
    reservoir_size = max_sources * 4
    upstream_source_count = 0
    for row in source_reader.read("corpus"):
        upstream_source_count += 1
        source_id = row.get("_id")
        language = row.get("language")
        upstream_languages[language] += 1
        if source_id in needed_ids or (
            (document_selection or document_retrieval)
            and row.get("source_id") in needed_canonical
            and (not config.target_languages or language in config.target_languages)
        ):
            previous = raw_sources.get(source_id)
            if previous is not None and any(
                previous.get(name) != row.get(name) for name in ("source_id", "language", "text")
            ):
                raise ValidationError(f"Conflicting duplicate source unit: {source_id}")
            raw_sources.setdefault(source_id, row)
        elif not config.target_languages or language in config.target_languages:
            previous = reservoir_rows.get(source_id)
            if previous is not None:
                if any(
                    previous.get(name) != row.get(name)
                    for name in ("source_id", "language", "text")
                ):
                    raise ValidationError(f"Conflicting duplicate source unit: {source_id}")
                continue
            priority = -int(digest([seed, source_id]), 16)
            entry = (priority, source_id, row)
            if len(reservoir) < reservoir_size:
                heapq.heappush(reservoir, entry)
                reservoir_rows[source_id] = row
            elif entry[:2] > reservoir[0][:2]:
                _, removed_id, _ = heapq.heapreplace(reservoir, entry)
                del reservoir_rows[removed_id]
                reservoir_rows[source_id] = row
    targets = list(config.target_languages) or sorted(upstream_languages)
    for _, source_id, row in reservoir:
        raw_sources.setdefault(source_id, row)
    chunker = chunker or Chunker(load_tokenizer(config.chunking, token), config.chunking)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    selected, selected_sources, skipped = [], set(), []
    selected_settings = {}
    judge_query = judge_documents if document_retrieval else judge
    with TemporaryDirectory(prefix=".subset-selection-", dir=config.output_dir) as temporary:
        with CorpusStore(Path(temporary) / "sources.sqlite", create=True) as store:
            for row in raw_sources.values():
                store.add_source(legal_unit(row, corpus))
            chunked = set()
            for query in candidates:
                required = set(query.required_units)
                if not document_retrieval:
                    required.update(parse_annotations(query))
                if document_selection:
                    required.update(candidate_source_units[source_ids[query.source_unit_id]])
                if required - raw_sources.keys():
                    skipped.append({"query_id": query.id, "reason": "missing_required_source"})
                    continue
                # Include all available target variants of each required source;
                # annotations also retain references outside the retrieval pool.
                for unit_id in query.required_units:
                    required.update(
                        variant
                        for variant in store.variants(store.get_source(unit_id))
                        if store.get_source(variant).language in targets
                    )
                if len(selected_sources | required) > max_sources:
                    skipped.append({"query_id": query.id, "reason": "source_budget"})
                    continue
                for unit_id in sorted(required - chunked):
                    for chunk in chunker.split(store.get_source(unit_id)):
                        store.add_chunk(chunk)
                    chunked.add(unit_id)
                settings = build_settings(
                    corpus, [query.question_language], targets, config.settings
                )
                judgments = [
                    (
                        setting,
                        judge_query(query, setting, store, associations[query.id] & required),
                    )
                    for setting in settings
                ]
                failures = [
                    {"setting_id": setting.id, "status": result.status, "reason": result.reason}
                    for setting, result in judgments
                    if result.status != "eligible"
                ]
                if failures and (not document_retrieval or len(failures) == len(judgments)):
                    skipped.append(
                        {"query_id": query.id, "reason": "ineligible", "settings": failures}
                    )
                    continue
                selected.append(query)
                selected_sources.update(required)
                selected_settings[query.id] = [
                    setting.id for setting, result in judgments if result.status == "eligible"
                ]
                if len(selected) == max_questions:
                    break
            if len(selected) != max_questions:
                raise ValidationError(
                    f"Found only {len(selected)} eligible questions within {max_sources} sources; "
                    "increase the source limit or change the language/settings selection"
                )
            positive_canonical = {
                store.get_source(unit_id).source_id for unit_id in selected_sources
            }
            distractors = sorted(
                (
                    unit_id
                    for unit_id, row in raw_sources.items()
                    if unit_id not in selected_sources
                    and row["language"] in targets
                    and row["source_id"] not in positive_canonical
                ),
                key=lambda unit_id: (digest([seed, unit_id]), unit_id),
            )
            distractor_ids = distractors[: max_sources - len(selected_sources)]
            selected_sources.update(distractor_ids)
            # Other selected questions and distractors can introduce retained
            # associations absent from this query's own required-source set.
            # Match the exact associations prepare_relevance will validate.
            for query in selected:
                for setting in build_settings(
                    corpus, [query.question_language], targets, config.settings
                ):
                    judge_query(query, setting, store, associations[query.id] & selected_sources)

    chosen_ids = {query.id for query in selected}
    chosen_tasks = {query.task_id for query in selected}
    chosen_documents = (
        {source_ids[query.source_unit_id] for query in selected} if document_selection else set()
    )
    retained_queries = [
        query
        for query in queries
        if (
            source_ids[query.source_unit_id] in chosen_documents
            if document_selection
            else query.task_id in chosen_tasks
            if config.candidate_mode == "best"
            else query.id in chosen_ids
        )
    ]
    retained_ids = {query.id for query in retained_queries}
    original_qrels = [row for row in source_qrels if row["query_id"] in retained_ids]
    subset_qrels = [row for row in original_qrels if row["corpus_id"] in selected_sources]
    output_config = {"version": 1, **resolved}
    output_config["output_dir"] = str(destination / "artifacts")
    output_config["query_languages"] = sorted({query.question_language for query in selected})
    output_config["target_languages"] = targets
    output_config["datasets"] = {
        corpus: {
            kind: {
                **asdict(getattr(config.datasets[corpus], kind)),
                "local_dir": str(destination / kind),
            }
            for kind in ("qa", "corpus")
        }
    }
    with new_directory(destination) as temporary:
        _parquet(
            temporary / "qa/data/qas/train-00000.parquet",
            [raw_by_id[query.id] for query in sorted(retained_queries, key=lambda query: query.id)],
        )
        _parquet(
            temporary / "qa/data/source_qrels/train-00000.parquet",
            subset_qrels,
            schema=pa.schema(
                [("query_id", pa.string()), ("corpus_id", pa.string()), ("score", pa.int64())]
            )
            if not subset_qrels
            else None,
        )
        _parquet(
            temporary / "corpus/data/corpus/train-00000.parquet",
            [raw_sources[unit_id] for unit_id in sorted(selected_sources)],
        )
        if config.evidence_file:
            # Freeze only retained QA overrides, avoiding dependence on mutable
            # external overlays or unrelated QA IDs during later validation.
            overlay = temporary / "evidence.jsonl"
            import json

            with (
                config.evidence_file.open(encoding="utf-8") as source,
                overlay.open("w", encoding="utf-8") as target,
            ):
                for line in source:
                    if (
                        line.strip()
                        and json.loads(line).get("query_id", json.loads(line).get("_id"))
                        in retained_ids
                    ):
                        target.write(line if line.endswith("\n") else line + "\n")
            output_config["evidence_file"] = str(destination / "evidence.jsonl")
        write_json(
            temporary / "selection.json",
            {
                "selected_query_ids": sorted(chosen_ids),
                "retained_candidate_ids": sorted(retained_ids),
                "selected_source_ids": sorted(selected_sources),
                "distractor_source_ids": distractor_ids,
                "eligible_settings": selected_settings,
                "candidate_groups": [
                    row
                    for row in selection_audit
                    if (
                        row["canonical_source_id"] in chosen_documents
                        if document_selection
                        else row["task_id"] in chosen_tasks
                    )
                ],
                "rejected_upstream_groups": rejected_groups,
                "skipped_candidates": skipped,
                "original_source_qrels": original_qrels,
                "omitted_source_qrels": [
                    row for row in original_qrels if row["corpus_id"] not in selected_sources
                ],
            },
        )
        (temporary / "config.yaml").write_text(
            yaml.safe_dump(output_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        write_json(
            temporary / "manifest.json",
            {
                "identity": identity,
                "purpose": "execution smoke test; not representative retrieval effectiveness",
                "selection_policy": "seeded hash order; every selected winner eligible in at least one available document setting; retain exact source and QA rows"
                if document_retrieval
                else "seeded hash order; every selected winner eligible in every requested setting; retain exact source and QA rows",
                "association_policy": "source_qrels restricted to retained corpus; original rows retained in selection.json",
                "question_count": len(selected),
                "retained_candidate_count": len(retained_queries),
                "source_count": len(selected_sources),
                "distractor_count": len(distractor_ids),
                "upstream_source_count": upstream_source_count,
                "upstream_source_languages": dict(sorted(upstream_languages.items())),
                "chunking": chunker.manifest,
                "seconds": time.perf_counter() - started,
                "timing_method": "measured preparation wall time including input reads and tokenizer loading; no model inference",
                "versions": versions(),
                "files": {
                    str(path.relative_to(temporary)): file_digest(path)
                    for path in sorted(temporary.rglob("*"))
                    if path.is_file()
                },
            },
        )
    return destination
