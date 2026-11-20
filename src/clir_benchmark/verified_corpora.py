"""Expand the verified releases' multilingual corpus records into source editions."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from .adapters import DatasetReader, required_string
from .artifacts import CorpusStore
from .config import Config, language_tag
from .errors import ValidationError
from .types import Chunk, Judgment, Match, Query, Setting, SourceUnit
from .util import digest, hf_token, json_dumps, new_directory, read_json, write_json


def map_support(row: dict, unit: SourceUnit, chunks: list[Chunk]) -> dict:
    """Remap the release's exact source coordinates; keep partial hits distinct from positives."""
    start, end = row["start"], row["end"]
    if (
        row["match_status"] != "exact"
        or row["source_unit_id"] != unit.id
        or row["corpus_id"] != unit.source_id
        or row["language"] != unit.language
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(unit.text)
        or unit.text[start:end] != row["text"]
    ):
        raise ValidationError(f"Evidence offsets do not match {row['query_id']} in {unit.id}")
    ids, pieces = [], []
    covered_to = start
    for chunk in chunks:
        if chunk.source_unit_id != unit.id or unit.text[chunk.start : chunk.end] != chunk.text:
            raise ValidationError(f"Chunk source mismatch for {chunk.id}")
        left, right = max(start, chunk.start), min(end, chunk.end)
        if left < right:
            if left > covered_to:
                raise ValidationError(f"Gap in chunk coverage for {unit.id}")
            covered_to = max(right, covered_to)
            pieces.append(
                {
                    "chunk_id": chunk.id,
                    "source_start": left,
                    "source_end": right,
                    "chunk_start": left - chunk.start,
                    "chunk_end": right - chunk.start,
                }
            )
            if left == start and right == end:
                ids.append(chunk.id)
    if covered_to != end:
        raise ValidationError(f"Incomplete source coverage for {row['query_id']}")
    return {
        key: row[key]
        for key in (
            "query_id",
            "span_index",
            "source_unit_id",
            "language",
            "corpus_id",
            "start",
            "end",
        )
    } | {
        "complete_chunk_ids": ids,
        "intersections": pieces,
    }


def prepare_verified_span_map(config: Config, corpus: str) -> Path:
    """Publish known original-edition positives and all boundary intersections."""
    data = config.datasets[corpus]
    if data.corpus_format != "verified-multilingual-v1":
        raise ValidationError("map-verified-spans requires a verified multilingual release")
    directory = config.corpus_dir(corpus)
    if not (directory / "manifest.json").is_file():
        raise ValidationError(f"Prepare the shared corpus before mapping spans for {corpus}")
    fingerprint = read_json(directory / "manifest.json")["corpus_fingerprint"]
    identity = {
        "schema": 1,
        "corpus_fingerprint": fingerprint,
        "qa": data.qa.identity(("evidence",)),
    }
    dest = config.output_dir / "verified-spans" / corpus / digest(identity)
    if dest.is_dir():
        read_json(dest / "manifest.json")
        return dest
    reader = DatasetReader(data.qa, hf_token(config.env_file))
    t0 = time.perf_counter()
    with new_directory(dest) as temp, CorpusStore(directory / "lookup.sqlite") as store:
        seen, positives = set(), set()
        outcomes = Counter()
        with (temp / "span-map.jsonl").open("w", encoding="utf-8") as out:
            for row in reader.read("evidence"):
                unit = store.get_source(row["source_unit_id"])
                if unit is None:
                    raise ValidationError(
                        f"Evidence references missing edition {row['source_unit_id']}"
                    )
                key = (row["query_id"], unit.id, row["span_index"])
                if key in seen:
                    raise ValidationError(f"Duplicate verified span {key}")
                seen.add(key)
                item = map_support(row, unit, store.chunks(unit))
                complete = item["complete_chunk_ids"]
                outcomes["contained" if complete else "split_across_chunks"] += 1
                positives.update((row["query_id"], chunk) for chunk in complete)
                out.write(json_dumps(item) + "\n")
        with (temp / "known-chunk-qrels.tsv").open("w", encoding="utf-8") as out:
            out.write("query_id\tchunk_id\tscore\n")
            for query, chunk in sorted(positives):
                out.write(f"{query}\t{chunk}\t1\n")
        write_json(
            temp / "manifest.json",
            {
                **identity,
                "corpus": corpus,
                "spans": len(seen),
                "query_chunk_positives": len(positives),
                "outcomes": dict(outcomes),
                "seconds": time.perf_counter() - t0,
            },
        )
    return dest


def iter_verified_units(reader: DatasetReader, corpus: str) -> Iterator[SourceUnit]:
    """Read each published edition, including EUR-Lex annexes and patent title-only editions."""
    configurations = ("corpus", "annexes") if corpus == "eurlex" else ("corpus",)
    for configuration in configurations:
        for row in reader.read(configuration):
            source_id = required_string(row, "_id", f"{corpus} {configuration}")
            label = f"{corpus} {configuration} {source_id}"
            texts = row.get("text")
            unit_ids = row.get("source_unit_ids")
            languages = row.get("available_languages")
            if not isinstance(texts, dict) or not isinstance(unit_ids, dict):
                raise ValidationError(f"{label}: expected language-keyed text and source_unit_ids")
            if not isinstance(languages, list) or not languages:
                raise ValidationError(f"{label}: expected a nonempty available_languages list")
            for language in languages:
                language_tag(language, label)
            if len(languages) != len(set(languages)):
                raise ValidationError(f"{label}: duplicate language in available_languages")
            available = set(languages)
            for field, values in (("text", texts), ("source_unit_ids", unit_ids)):
                if {key for key, value in values.items() if value is not None} != available:
                    raise ValidationError(f"{label}: {field} does not match available_languages")

            by_language = {}
            for key in (
                "title",
                "title_source",
                "abstract_source",
                "edition_sources",
                "paired_document_ids",
                "accepted_pair_ids",
            ):
                values = row.get(key)
                if values is not None:
                    if not isinstance(values, dict):
                        raise ValidationError(f"{label}: expected language-keyed {key}")
                    by_language[key] = values
            if corpus == "chemistry":
                unit_type = "patent_publication"
                parent_id = source_id
            else:
                unit_type = required_string(row, "unit_type", label)
                parent_id = required_string(row, "parent_document_id", label)
            metadata = {"configuration": configuration}
            for key in (
                "publication_date",
                "publication_authority",
                "source_family_ids",
                "sources",
                "ipc_codes",
                "cpc_codes",
                "year",
                "source_date",
                "source_cohort",
                "article_number",
                "symbol",
            ):
                if row.get(key) is not None:
                    metadata[key] = row[key]
            for language in languages:
                edition_label = f"{label} {language}"
                edition_metadata = {
                    key: values[language]
                    for key, values in by_language.items()
                    if values.get(language) is not None
                }
                title = edition_metadata.pop("title", None)
                if title is not None and not isinstance(title, str):
                    raise ValidationError(f"{edition_label}: expected a string title or null")
                partners = edition_metadata.get("paired_document_ids", [])
                if not isinstance(partners, list) or any(
                    not isinstance(value, str) or not value for value in partners
                ):
                    raise ValidationError(
                        f"{edition_label}: expected a list of partner edition IDs"
                    )
                yield SourceUnit(
                    id=required_string(unit_ids, language, edition_label),
                    corpus=corpus,
                    source_id=source_id,
                    parent_id=parent_id,
                    language=language,
                    text=required_string(texts, language, edition_label, allow_empty=True),
                    unit_type=unit_type,
                    title=title or "",
                    metadata={**metadata, **edition_metadata},
                )


def verified_query_from_row(row: dict, corpus: str) -> Query:
    query_id = required_string(row, "_id", corpus)
    label = f"{corpus} QA {query_id}"
    language = language_tag(required_string(row, "question_language", label), label)
    evidence = row.get("evidence_all_languages")
    if not isinstance(evidence, dict) or not any(evidence.values()):
        raise ValidationError(f"{label}: multilingual supporting spans are missing")
    return Query(
        id=query_id,
        corpus=corpus,
        question=required_string(row, "question", label),
        question_language=language,
        context_language=language_tag(row.get("context_language") or language, label),
        source_unit_id=required_string(row, "source_unit_id", label),
        answer=row.get("answer") or "",
        evidence=row.get("evidence") or "",
        evidence_all_languages=evidence,
        evidence_alignment_status=row.get("evidence_alignment_status") or {},
        evidence_alignment_eligibility=row.get("evidence_alignment_eligibility"),
        evidence_alignment_contract=row.get("evidence_alignment_contract"),
        involved_source_unit_ids=(),
        generator_model_id=row.get("generator_model_id") or "",
        persona=row.get("persona") or row.get("mode") or "",
        # Published QAs are the frozen candidates; legacy generation-task files are not required.
        task_id=query_id,
        is_best=row.get("is_best"),
        candidate_rank=row.get("candidate_rank"),
        total_score=row.get("total_score"),
    )


def verified_overlap_matches(queries: list[Query], store: CorpusStore) -> dict:
    """Validate source quotations once and label all positive-length chunk intersections."""

    @lru_cache(maxsize=24)
    def edition(unit_id):
        unit = store.get_source(unit_id)
        if unit is None:
            raise ValidationError(f"Missing supporting edition {unit_id}")
        bounds = list(
            store.db.execute(
                "SELECT id,start,end FROM chunks WHERE source_unit_id=? ORDER BY start", (unit_id,)
            )
        )
        return unit, bounds

    result = {}
    for query in queries:
        snippets = [
            (language, s)
            for language, entries in query.evidence_all_languages.items()
            for s in entries or []
        ]
        anchors = {s["anchor_id"]: s for _, s in snippets if s.get("is_original") is True}
        by_language = defaultdict(list)
        for language, snippet in snippets:
            unit_id = required_string(snippet, "source_unit_id", query.id)
            text = required_string(snippet, "text", query.id)
            unit, chunks = edition(unit_id)
            anchor = anchors.get(snippet.get("anchor_id"))
            if (
                not anchor
                or snippet.get("match_status") != "exact"
                or unit.corpus != query.corpus
                or unit.language != language
                or snippet.get("language") != language
                or snippet.get("corpus_id") != unit.source_id
                or snippet.get("config") != unit.metadata.get("configuration")
                or (anchor["config"], anchor["corpus_id"])
                != (snippet["config"], snippet["corpus_id"])
            ):
                raise ValidationError(
                    f"{query.id}: evidence edition or anchor mismatch in {unit_id}"
                )
            start, end = snippet.get("start"), snippet.get("end")
            if start is None and end is None:
                starts = snippet.get("occurrence_starts")
                if not isinstance(starts, list) or not starts:
                    raise ValidationError(f"{query.id}: missing occurrence positions for {unit_id}")
            elif type(start) is int and type(end) is int and end == start + len(text):
                starts = [start]
            else:
                raise ValidationError(f"{query.id}: invalid evidence range in {unit_id}")
            for pos in starts:
                if type(pos) is not int or pos < 0 or unit.text[pos : pos + len(text)] != text:
                    raise ValidationError(f"{query.id}: quotation mismatch in {unit_id} at {pos}")
                stop = pos + len(text)
                covered = pos
                for chunk_id, chunk_start, chunk_end in chunks:
                    if chunk_start >= stop:
                        break
                    left, right = max(pos, chunk_start), min(stop, chunk_end)
                    if left >= right:
                        continue
                    if left > covered:
                        raise ValidationError(f"{query.id}: gap in chunk coverage in {unit_id}")
                    covered = max(covered, right)
                    by_language[language].append(
                        Match(
                            query.id,
                            chunk_id,
                            anchor["source_unit_id"],
                            unit_id,
                            "evidence_all_languages",
                            unit.text[left:right],
                            left,
                            right,
                            {
                                "anchor_id": snippet["anchor_id"],
                                "snippet_start": pos,
                                "snippet_end": stop,
                                "is_original": snippet["is_original"],
                                "validation_status": snippet.get("validation_status"),
                            },
                        )
                    )
                if covered != stop:
                    raise ValidationError(f"{query.id}: incomplete chunk coverage in {unit_id}")
        result[query.id] = by_language
    return result


def verified_overlap_judgment(query: Query, setting: Setting, by_language: dict) -> Judgment:
    matches = [match for lang in setting.target_languages for match in by_language.get(lang, ())]
    if not matches:
        return Judgment(
            query.id,
            "dropped",
            "no_evidence_in_target_languages",
            {"target_languages": setting.target_languages},
            [],
        )
    return Judgment(
        query.id,
        "eligible",
        "verified_chunk_overlap",
        {},
        sorted({match.required_source_id for match in matches}),
        matches,
    )
