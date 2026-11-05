from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceUnit:
    id: str
    corpus: str
    source_id: str
    parent_id: str
    language: str
    text: str
    unit_type: str
    title: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    source_unit_id: str
    source_id: str
    parent_id: str
    corpus: str
    language: str
    text: str
    start: int
    end: int
    token_count: int
    index: int
    overlap_chars: int
    overlap_tokens: int
    chunking_hash: str


@dataclass(frozen=True)
class Query:
    id: str
    corpus: str
    question: str
    question_language: str
    context_language: str
    answer: str
    evidence: str
    source_unit_id: str
    involved_source_unit_ids: tuple[str, ...]
    generator_model_id: str
    persona: str
    task_id: str
    is_best: bool | None
    candidate_rank: int | None
    total_score: float | None
    evidence_all_languages: dict = field(default_factory=dict)
    unresolved_sources: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def required_units(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.source_unit_id, *self.involved_source_unit_ids)))


@dataclass(frozen=True)
class Setting:
    id: str
    corpus: str
    kind: str
    query_language: str
    target_languages: tuple[str, ...]


@dataclass(frozen=True)
class Match:
    query_id: str
    chunk_id: str
    required_source_id: str
    source_unit_id: str
    field: str
    span: str
    start: int
    end: int


@dataclass
class Judgment:
    query_id: str
    status: str
    reason: str
    detail: dict
    required_source_ids: list[str]
    matches: list[Match] = field(default_factory=list)

    @property
    def relevant(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for match in self.matches:
            result.setdefault(match.chunk_id, set()).add(match.required_source_id)
        return result
