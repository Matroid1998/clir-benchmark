from __future__ import annotations

from collections.abc import Iterator

from .artifacts import CorpusStore
from .errors import ValidationError
from .evidence import Annotation, alignment_excluded, parse_annotations
from .types import DocumentJudgment, Judgment, Match, Query, Setting


def occurrences(text: str, span: str) -> Iterator[tuple[int, int]]:
    if not isinstance(span, str):
        raise ValidationError("Answer and evidence spans must be strings")
    if span == "":
        return
    start = text.find(span)
    while start >= 0:
        yield start, start + len(span)
        start = text.find(span, start + 1)


def build_settings(
    corpus: str, query_languages: list[str], target_languages: list[str], kinds: tuple[str, ...]
) -> list[Setting]:
    settings = []
    for query_language in sorted(set(query_languages)):
        if "pairwise" in kinds:
            for target in sorted(set(target_languages)):
                kind = "monolingual" if target == query_language else "cross_language"
                settings.append(
                    Setting(
                        f"{corpus}__{query_language}__{target}",
                        corpus,
                        kind,
                        query_language,
                        (target,),
                    )
                )
        if "mixed" in kinds:
            settings.append(
                Setting(
                    f"{corpus}__{query_language}__mixed",
                    corpus,
                    "mixed",
                    query_language,
                    tuple(sorted(set(target_languages))),
                )
            )
    return settings


def annotation_records(
    query: Query, allowed: set[str], store: CorpusStore
) -> dict[str, Annotation]:
    records = parse_annotations(query)
    for unit_id, annotation in records.items():
        if unit_id not in allowed:
            raise ValidationError(
                f"QA {query.id}: annotation source {unit_id!r} is not an associated source"
            )
        unit = store.get_source(unit_id)
        if unit is None or unit.language != annotation.language:
            raise ValidationError(
                f"QA {query.id}: annotation language/source mismatch for {unit_id}"
            )
    return records


def judge(
    query: Query, setting: Setting, store: CorpusStore, source_associations: set[str] | None = None
) -> Judgment:
    required = list(query.required_units)
    if query.unresolved_sources:
        return Judgment(query.id, "dropped", "unresolved_source_involvement", {}, required)
    groups = {}
    canonical_groups = {}
    for unit_id in required:
        unit = store.get_source(unit_id)
        if unit is None:
            return Judgment(
                query.id, "dropped", "missing_source_unit", {"source_unit_id": unit_id}, required
            )
        canonical = (unit.corpus, unit.source_id)
        if canonical in canonical_groups:
            first_group = canonical_groups[canonical]
            if unit_id not in groups[first_group]:
                raise ValidationError(
                    f"QA {query.id}: involved language versions are not directly associated: {first_group}, {unit_id}"
                )
            continue
        canonical_groups[canonical] = unit_id
        groups[unit_id] = store.variants(unit)
    required = list(groups)
    allowed = set().union(*map(set, groups.values()))
    # Provenance qrels are associations only. An asserted association must resolve
    # through the canonical legal source or a direct accepted chemistry pair.
    unexpected = (source_associations or set()) - allowed
    if unexpected:
        raise ValidationError(
            f"QA {query.id}: source_qrels contains unsupported associations: {sorted(unexpected)}"
        )
    annotations = annotation_records(query, allowed, store)
    targets = {}
    for group, variants in groups.items():
        targets[group] = [
            store.get_source(unit_id)
            for unit_id in variants
            if store.get_source(unit_id).language in setting.target_languages
        ]
        if not targets[group]:
            return Judgment(
                query.id,
                "dropped",
                "unavailable_language_version",
                {"required_source_id": group, "target_languages": setting.target_languages},
                required,
            )
    missing = sorted(
        {
            unit.id
            for units in targets.values()
            for unit in units
            if unit.language != query.context_language
            and not (unit.id in annotations and annotations[unit.id].supplied)
            and not alignment_excluded(query, unit.language)
        }
    )
    if missing:
        return Judgment(
            query.id,
            "blocked",
            "missing_multilingual_annotations",
            {"source_unit_ids": missing},
            required,
        )
    matches = []
    unsupported = []
    for group, units in targets.items():
        group_matches = []
        found_in_source = False
        for unit in units:
            spans = [("answer", query.answer, {}), ("evidence", query.evidence, {})]
            if unit.id in annotations:
                spans.extend(annotations[unit.id].spans)
            chunks = store.chunks(unit)
            for field, span, annotation in spans:
                for start, end in occurrences(unit.text, span):
                    found_in_source = True
                    for chunk in chunks:
                        if chunk.start <= start and end <= chunk.end:
                            group_matches.append(
                                Match(
                                    query.id,
                                    chunk.id,
                                    group,
                                    unit.id,
                                    field,
                                    span,
                                    start,
                                    end,
                                    annotation,
                                )
                            )
        if not group_matches:
            unsupported.append(
                {
                    "required_source_id": group,
                    "reason": "span_crosses_chunk_boundaries"
                    if found_in_source
                    else "no_exact_span",
                    "alignment_status": {
                        unit.id: annotations[unit.id].statuses
                        for unit in units
                        if unit.id in annotations and annotations[unit.id].statuses
                    },
                    "alignment_eligibility": query.evidence_alignment_eligibility,
                }
            )
        matches.extend(group_matches)
    if unsupported:
        return Judgment(
            query.id,
            "dropped",
            "incomplete_source_support",
            {"unsupported": unsupported},
            required,
            matches,
        )
    return Judgment(query.id, "eligible", "complete_source_support", {}, required, matches)


def judge_documents(
    query: Query, setting: Setting, store: CorpusStore, source_associations: set[str] | None = None
) -> DocumentJudgment:
    """Label source language versions from provenance, without inspecting evidence.

    Legal variants share a canonical source; chemistry variants are restricted
    to the original source and its directly accepted translation pairs. Each
    variant is a separate relevant document, while required-source groups keep
    source-recall metrics independent of the number of available translations.
    """
    required = list(query.required_units)
    if query.unresolved_sources:
        return DocumentJudgment(query.id, "dropped", "unresolved_source_involvement", {}, required)
    groups = {}
    canonical_groups = {}
    for unit_id in required:
        unit = store.get_source(unit_id)
        if unit is None:
            return DocumentJudgment(
                query.id, "dropped", "missing_source_unit", {"source_unit_id": unit_id}, required
            )
        canonical = (unit.corpus, unit.source_id)
        if canonical in canonical_groups:
            first_group = canonical_groups[canonical]
            if unit_id not in groups[first_group]:
                raise ValidationError(
                    f"QA {query.id}: involved language versions are not directly associated: {first_group}, {unit_id}"
                )
            continue
        canonical_groups[canonical] = unit_id
        groups[unit_id] = store.variants(unit)
    required = list(groups)
    allowed = set().union(*map(set, groups.values()))
    unexpected = (source_associations or set()) - allowed
    if unexpected:
        raise ValidationError(
            f"QA {query.id}: source_qrels contains unsupported associations: {sorted(unexpected)}"
        )
    relevant: dict[str, set[str]] = {}
    for group, variants in groups.items():
        targets = [
            unit_id
            for unit_id in variants
            if store.get_source(unit_id).language in setting.target_languages
        ]
        if not targets:
            return DocumentJudgment(
                query.id,
                "dropped",
                "unavailable_language_version",
                {"required_source_id": group, "target_languages": setting.target_languages},
                required,
            )
        indexed = [unit_id for unit_id in targets if store.has_chunks(unit_id)]
        if not indexed:
            return DocumentJudgment(
                query.id,
                "dropped",
                "missing_indexed_document",
                {"required_source_id": group, "source_unit_ids": sorted(targets)},
                required,
            )
        for document_id in sorted(indexed):
            relevant.setdefault(document_id, set()).add(group)
    return DocumentJudgment(query.id, "eligible", "known_source_document", {}, required, relevant)
