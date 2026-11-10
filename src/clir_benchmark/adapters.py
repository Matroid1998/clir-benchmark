from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

from .artifacts import CorpusStore, rows
from .config import CandidateSelectionConfig, SourceSpec, language_tag
from .errors import ValidationError
from .evidence import parse_annotations
from .types import Query, SourceUnit
from .util import json_dumps, object_field


class DatasetReader:
    def __init__(self, spec: SourceSpec, token: str | None = None):
        self.spec = spec
        self.token = token
        self._files: list[str] | None = None

    def read(self, configuration: str) -> Iterator[dict]:
        pattern = (
            "data/train-*.parquet"
            if configuration == "default"
            else f"data/{configuration}/*.parquet"
        )
        if self.spec.local_dir:
            paths = sorted(Path(self.spec.local_dir).glob(pattern))
        else:
            from huggingface_hub import HfApi, hf_hub_download

            if self._files is None:
                self._files = HfApi(token=self.token).list_repo_files(
                    self.spec.repo_id, repo_type="dataset", revision=self.spec.revision
                )
            paths = [
                Path(
                    hf_hub_download(
                        self.spec.repo_id,
                        name,
                        repo_type="dataset",
                        revision=self.spec.revision,
                        token=self.token,
                    )
                )
                for name in sorted(self._files)
                if fnmatch.fnmatch(name, pattern)
            ]
        if not paths:
            raise ValidationError(
                f"{self.spec.repo_id}@{self.spec.revision}: no {configuration!r} Parquet files ({pattern})"
            )
        for path in paths:
            yield from rows(path)


def required_string(row: dict, key: str, label: str, *, allow_empty: bool = False) -> str:
    value = row.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValidationError(
            f"{label}: {key} must be {'a string' if allow_empty else 'a nonempty string'}"
        )
    return value


def legal_unit(row: dict, corpus: str) -> SourceUnit:
    label = f"{corpus} source {row.get('_id', '<missing ID>')}"
    return SourceUnit(
        id=required_string(row, "_id", label),
        corpus=corpus,
        source_id=required_string(row, "source_id", label),
        parent_id=row.get("parent_id") or "",
        language=language_tag(required_string(row, "language", label), label),
        text=required_string(row, "text", label, allow_empty=True),
        unit_type=required_string(row, "unit_type", label),
        title=row.get("title") or "",
        metadata=object_field(row.get("metadata_json"), label + " metadata"),
    )


def chemistry_pair(row: dict, spec: SourceSpec) -> tuple[list[SourceUnit], tuple[str, str] | None]:
    publication = required_string(row, "doc_id", "chemistry pair")
    units = []
    for side in ("source", "target"):
        language = language_tag(required_string(row, side + "_language", publication), publication)
        text = required_string(row, side + "_text", publication, allow_empty=True)
        units.append(
            SourceUnit(
                id=f"{publication}_{language}",
                corpus="chemistry",
                source_id=publication,
                parent_id=publication,
                language=language,
                text=text,
                unit_type="abstract",
                metadata={
                    "upstream_dataset": spec.repo_id,
                    "upstream_revision": spec.revision,
                    "upstream_group_key": row.get("group_key"),
                    "upstream_example_id": row.get("example_id"),
                    "field": row.get("field", "abstract"),
                    "publication_date": row.get("pub_date"),
                },
            )
        )
    if units[0].language == units[1].language:
        raise ValidationError(f"{publication}: translation pair has identical language labels")
    accepted = row.get("pair_ok") is True and row.get("judge_verdict") in {"exact", "high"}
    return units, (units[0].id, units[1].id) if accepted else None


def query_from_row(row: dict, corpus: str) -> Query:
    label = f"{corpus} QA {row.get('_id', '<missing ID>')}"
    query_id = required_string(row, "_id", label)
    if any(character.isspace() for character in query_id):
        raise ValidationError(f"{label}: QA IDs cannot contain whitespace in TREC exports")
    for field in ("answer", "evidence"):
        if row.get(field) is not None and not isinstance(row[field], str):
            raise ValidationError(f"{label}: {field} must be a verbatim string or null")
    metadata = object_field(row.get("metadata_json"), label + " metadata")
    provenance = object_field(row.get("provenance_json"), label + " provenance")
    generator = required_string(row, "generator_model_id", label)
    task = metadata.get("decider_routing_task") if corpus == "chemistry" else provenance.get("task")
    if not task or not provenance.get("file"):
        raise ValidationError(
            f"{label}: missing generation task identity or originating provenance file"
        )
    if isinstance(task, str) and task.startswith("["):
        import json

        try:
            task = json.loads(task)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{label}: malformed chemistry routing task") from exc
    task_id = json_dumps([corpus, provenance["file"], task, generator])
    marked = row.get("is_best")
    if marked is not None and not isinstance(marked, bool):
        raise ValidationError(f"{label}: is_best must be boolean or null")
    rank = row.get("candidate_rank")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
        raise ValidationError(f"{label}: candidate_rank must be a positive integer or null")
    score = row.get("total_score")
    if score is not None:
        try:
            score = float(score)
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"{label}: invalid grader total_score") from exc
        if not math.isfinite(score):
            raise ValidationError(f"{label}: grader total_score must be finite")
    involved = row.get("involved_source_unit_ids") or []
    if not isinstance(involved, list) or any(
        not isinstance(item, str) or not item for item in involved
    ):
        raise ValidationError(f"{label}: involved_source_unit_ids must be a list of IDs")
    source_id = required_string(row, "source_unit_id", label)
    unresolved = bool(row.get("unresolved_article_references") or row.get("multi_article_conflict"))
    if row.get("multi_article") and len(set([source_id, *involved])) < 2:
        unresolved = True
    retained = {
        key: row.get(key)
        for key in (
            "quality_audit_status",
            "grading_status",
            "faith_overall",
            "qual_overall",
            "generation_target_id",
            "evidence_language",
        )
    }
    retained["provenance"] = provenance
    query = Query(
        id=query_id,
        corpus=corpus,
        question=required_string(row, "question", label),
        question_language=language_tag(required_string(row, "question_language", label), label),
        context_language=language_tag(
            row.get("context_language") or row["question_language"], label
        ),
        answer=row.get("answer") or "",
        evidence=row.get("evidence") or "",
        source_unit_id=source_id,
        involved_source_unit_ids=tuple(involved),
        generator_model_id=generator,
        persona=row.get("persona") or row.get("mode") or "unknown",
        task_id=task_id,
        is_best=marked,
        candidate_rank=rank,
        total_score=score,
        evidence_all_languages=object_field(
            row.get("evidence_all_languages"), label + " evidence_all_languages"
        ),
        evidence_alignment_status=object_field(
            row.get("evidence_alignment_status"), label + " evidence_alignment_status"
        ),
        evidence_alignment_contract=row.get("evidence_alignment_contract"),
        evidence_alignment_eligibility=row.get("evidence_alignment_eligibility"),
        unresolved_sources=unresolved,
        metadata=retained,
    )
    parse_annotations(query)
    return query


def canonical_source_ids(
    reader: DatasetReader, corpus: str, source_unit_ids: set[str]
) -> dict[str, str]:
    """Read document identities without preparing chunks or loading a model."""
    result = {}
    configuration = "default" if corpus == "chemistry" else "corpus"
    for row in reader.read(configuration):
        units = (
            chemistry_pair(row, reader.spec)[0]
            if corpus == "chemistry"
            else [legal_unit(row, corpus)]
        )
        for unit in units:
            if unit.id not in source_unit_ids:
                continue
            previous = result.setdefault(unit.id, unit.source_id)
            if previous != unit.source_id:
                raise ValidationError(f"Conflicting canonical source identity for {unit.id}")
    return result


def _select_per_document(
    queries: list[Query], source_ids: Mapping[str, str] | None, policy: CandidateSelectionConfig
) -> tuple[list[Query], list[dict]]:
    if source_ids is None:
        raise ValidationError("best-per-document requires canonical source IDs from the corpus")
    groups: dict[tuple[str, str], list[Query]] = {}
    ids = set()
    for query in queries:
        if query.id in ids:
            raise ValidationError(f"Duplicate QA ID: {query.id}")
        ids.add(query.id)
        source_id = source_ids.get(query.source_unit_id)
        if not isinstance(source_id, str) or not source_id:
            raise ValidationError(
                f"QA {query.id}: missing canonical source mapping for {query.source_unit_id}"
            )
        if query.total_score is not None and (
            isinstance(query.total_score, bool)
            or not isinstance(query.total_score, (int, float))
            or not math.isfinite(query.total_score)
        ):
            raise ValidationError(f"QA {query.id}: grader total_score must be finite and numeric")
        groups.setdefault((query.corpus, source_id), []).append(query)

    selected, audit = [], []
    preferred = set(policy.preferred_generators)
    for (corpus, source_id), candidates in sorted(groups.items()):
        candidates.sort(key=lambda query: query.id)
        scored = [query for query in candidates if query.total_score is not None]
        maximum = max(query.total_score for query in scored) if scored else None
        tied = [query for query in scored if query.total_score == maximum]
        preferred_tied = [query for query in tied if query.generator_model_id in preferred]
        pool = preferred_tied or tied
        group_seed = int(
            hashlib.sha256(
                json.dumps(
                    [policy.seed, corpus, source_id], ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            16,
        )
        winner = random.Random(group_seed).choice(pool) if pool else None
        if winner is not None:
            selected.append(winner)
        details = []
        for query in candidates:
            disposition = (
                "missing_score"
                if query.total_score is None
                else "below_maximum"
                if query.total_score != maximum
                else "selected"
                if query == winner
                else "not_preferred_tie"
                if preferred_tied and query.generator_model_id not in preferred
                else "tie_not_selected"
            )
            details.append(
                {
                    "query_id": query.id,
                    "source_unit_id": query.source_unit_id,
                    "canonical_source_id": source_id,
                    "generator_model_id": query.generator_model_id,
                    "question_language": query.question_language,
                    "total_score": query.total_score,
                    "disposition": disposition,
                }
            )
        audit.append(
            {
                "schema": 1,
                "corpus": corpus,
                "canonical_source_id": source_id,
                "candidate_count": len(candidates),
                "scored_candidate_count": len(scored),
                "excluded_missing_score_count": len(candidates) - len(scored),
                "selected_ids": [winner.id] if winner else [],
                "selected_language": winner.question_language if winner else None,
                "selected_generator": winner.generator_model_id if winner else None,
                "method": "maximum_score_preferred_tier_seeded_random",
                "status": "selected" if winner else "excluded_no_scored_candidates",
                "maximum_score": maximum,
                "maximum_score_ids": [query.id for query in tied],
                "preferred_tied_ids": [query.id for query in preferred_tied],
                "random_pool_ids": [query.id for query in pool],
                "selection_seed": policy.seed,
                "group_seed": group_seed,
                "random_algorithm": "python.random.Random.choice(sorted_query_ids)-v1",
                "preferred_generators": list(policy.preferred_generators),
                "language_counts": dict(
                    sorted(Counter(query.question_language for query in candidates).items())
                ),
                "generator_counts": dict(
                    sorted(Counter(query.generator_model_id for query in candidates).items())
                ),
                "candidates": details,
            }
        )
    return sorted(selected, key=lambda query: query.id), audit


def select_candidates(
    queries: list[Query],
    mode: str = "best",
    *,
    source_ids: Mapping[str, str] | None = None,
    policy: CandidateSelectionConfig | None = None,
) -> tuple[list[Query], list[dict]]:
    if mode == "best-per-document":
        return _select_per_document(queries, source_ids, policy or CandidateSelectionConfig())
    if mode not in {"all", "best"}:
        raise ValidationError(f"Unknown candidate mode: {mode}")
    groups: dict[tuple, list[Query]] = {}
    ids = set()
    for query in queries:
        if query.id in ids:
            raise ValidationError(f"Duplicate QA ID: {query.id}")
        ids.add(query.id)
        if not query.task_id:
            raise ValidationError(f"QA {query.id}: missing generation task identity")
        groups.setdefault((query.corpus, query.task_id, query.generator_model_id), []).append(query)
    selected, audit = [], []
    for key, candidates in sorted(groups.items()):
        marked = [q for q in candidates if q.is_best is True]
        if len(marked) > 1:
            raise ValidationError(
                f"Multiple is_best winners for task {key}: {[q.id for q in marked]}"
            )
        if mode == "all":
            winners, method = candidates, "all"
        elif marked:
            winners, method = marked, "exported_is_best"
        else:
            if any(q.total_score is None for q in candidates):
                raise ValidationError(
                    f"Task {key}: missing grader total_score required to select best candidate"
                )
            winners = [
                min(
                    candidates,
                    key=lambda q: (
                        -q.total_score,
                        q.candidate_rank if q.candidate_rank is not None else math.inf,
                        q.id,
                    ),
                )
            ]
            method = "grader_total_score"
        selected.extend(winners)
        audit.append(
            {
                "task_id": key[1],
                "generator_model_id": key[2],
                "candidate_count": len(candidates),
                "selected_ids": [q.id for q in winners],
                "method": method,
            }
        )
    return sorted(selected, key=lambda q: q.id), audit


def apply_evidence_overlay(queries: list[Query], path: Path | None) -> list[Query]:
    if path is None:
        return queries
    import json

    overlay = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = record.get("query_id", record.get("_id"))
            if not isinstance(query_id, str) or query_id in overlay:
                raise ValidationError(
                    f"Evidence overlay line {line_no}: missing or duplicate QA ID"
                )
            if "evidence_all_languages" not in record:
                raise ValidationError(
                    f"Evidence overlay line {line_no}: missing evidence_all_languages"
                )
            overlay[query_id] = {
                name: object_field(record.get(name), f"evidence overlay {name}")
                for name in ("evidence_all_languages", "evidence_alignment_status")
            }
            for name in ("evidence_alignment_contract", "evidence_alignment_eligibility"):
                overlay[query_id][name] = record.get(name)
    # An overlay may cover several corpora. Unknown IDs are checked across all
    # configured corpora by the validation stage, not discarded within one adapter.
    result = []
    for query in queries:
        # A replacement must not inherit statuses/exclusions for old annotations.
        updated = replace(query, **overlay[query.id]) if query.id in overlay else query
        parse_annotations(updated)
        result.append(updated)
    return result


def verify_chemistry_provenance(
    reader: DatasetReader, store: CorpusStore, corpus_spec: SourceSpec
) -> None:
    for row in reader.read("corpus"):
        unit_id = required_string(row, "_id", "chemistry QA corpus")
        unit = store.get_source(unit_id)
        metadata = object_field(row.get("metadata_json"), unit_id + " metadata")
        if not unit or unit.text != row.get("text") or unit.language != row.get("language"):
            raise ValidationError(
                f"Chemistry QA source {unit_id} is absent or differs verbatim from the full corpus"
            )
        if (
            metadata.get("upstream_revision") != corpus_spec.revision
            or metadata.get("upstream_dataset") != corpus_spec.repo_id
        ):
            raise ValidationError(
                f"Chemistry QA source {unit_id}: upstream dataset/revision provenance does not match configured corpus"
            )
