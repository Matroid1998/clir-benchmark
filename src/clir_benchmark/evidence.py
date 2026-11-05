"""Read published span annotations and the original source-record overlay format.

Alignment outcomes describe annotation availability, never retrieval relevance.
Only literal spans are handed to the matcher; status reasons are provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ValidationError
from .types import Query

ALIGNMENT_CONTRACT = "cross_language_evidence_terra_v1"
EMPTY_ALIGNMENT_OUTCOMES = frozenset({"not_found", "uncertain", "invalid"})
EXCLUDED_ALIGNMENT_OUTCOMES = frozenset({"no_exact_source_passage", "quarantined_source_language"})


@dataclass
class Annotation:
    language: str
    source_unit_id: str
    spans: list[tuple[str, str, dict]] = field(default_factory=list)
    supplied: bool = False
    statuses: list[dict] = field(default_factory=list)


def parse_annotations(query: Query) -> dict[str, Annotation]:
    """Validate shapes without requiring a downloaded corpus or trusting flags."""
    label = f"QA {query.id}"
    for name in ("evidence_alignment_contract", "evidence_alignment_eligibility"):
        value = getattr(query, name)
        if value is not None and not isinstance(value, str):
            raise ValidationError(f"{label}: {name} must be a string or null")
    result: dict[str, Annotation] = {}
    aggregates = set()
    for name in ("evidence_all_languages", "evidence_alignment_status"):
        mapping = getattr(query, name)
        if not isinstance(mapping, dict):
            raise ValidationError(f"{label}: {name} must map languages to lists of source records")
        for language, entries in mapping.items():
            if not isinstance(language, str) or not language or not isinstance(entries, list):
                raise ValidationError(
                    f"{label}: {name} must map languages to lists of source records"
                )
            for index, entry in enumerate(entries):
                pointer = f"{name}.{language}[{index}]"
                if not isinstance(entry, dict):
                    raise ValidationError(f"{label}: {pointer} must be an object")
                unit_id = entry.get("source_unit_id")
                if not isinstance(unit_id, str) or not unit_id:
                    raise ValidationError(f"{label}: {pointer} needs a nonempty source_unit_id")
                # Generated spans often have language=null. The enclosing key
                # supplies the language; relevance later checks the corpus unit.
                if entry.get("language") not in (None, language):
                    raise ValidationError(
                        f"{label}: annotation language/source mismatch at {pointer}"
                    )
                annotation = result.setdefault(unit_id, Annotation(language, unit_id))
                if annotation.language != language:
                    raise ValidationError(
                        f"{label}: annotation language/source mismatch for {unit_id}"
                    )
                if name == "evidence_alignment_status":
                    status = entry.get("status")
                    if not isinstance(status, str) or not status:
                        raise ValidationError(
                            f"{label}: {pointer}.status must be a nonempty string"
                        )
                    annotation.statuses.append(entry)
                    # Positive status without a span is incomplete. An explicit
                    # failed attempt is supplied annotation with no positive span.
                    annotation.supplied |= status in EMPTY_ALIGNMENT_OUTCOMES
                    continue
                if "text" in entry:
                    if not isinstance(entry["text"], str):
                        raise ValidationError(f"{label}: {pointer}.text must be a verbatim string")
                    annotation.spans.append((f"{pointer}.text", entry["text"], entry))
                else:
                    if unit_id in aggregates:
                        raise ValidationError(f"{label}: duplicate annotation record for {unit_id}")
                    aggregates.add(unit_id)
                    spans = entry.get("evidence")
                    if not isinstance(spans, list) or any(
                        not isinstance(span, str) for span in spans
                    ):
                        raise ValidationError(
                            f"{label}: evidence for {unit_id} must be a list of verbatim strings (possibly empty)"
                        )
                    answer = entry.get("answer")
                    if answer is not None and not isinstance(answer, str):
                        raise ValidationError(
                            f"{label}: answer for {unit_id} must be a string or null"
                        )
                    annotation.spans.extend(
                        (f"{pointer}.evidence[{i}]", span, entry) for i, span in enumerate(spans)
                    )
                    annotation.spans.append((f"{pointer}.answer", answer or "", entry))
                annotation.supplied = True
    return result


def alignment_excluded(query: Query, language: str) -> bool:
    """A completed native pass may explicitly exclude a QA without inventing spans.

    This is an annotation outcome, not a QA quality filter: raw fields still
    qualify. It applies only to languages explicitly covered by that pass.
    Unknown outcomes, absent languages and bare empty lists remain incomplete.
    """
    return (
        query.evidence_alignment_contract == ALIGNMENT_CONTRACT
        and query.evidence_alignment_eligibility in EXCLUDED_ALIGNMENT_OUTCOMES
        and language in query.evidence_all_languages
        and language in query.evidence_alignment_status
    )
