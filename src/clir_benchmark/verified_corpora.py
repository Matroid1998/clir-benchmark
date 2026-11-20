"""Expand the verified releases' multilingual corpus records into source editions."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from .adapters import DatasetReader, required_string
from .artifacts import CorpusStore
from .config import Config, language_tag
from .errors import ValidationError
from .types import Chunk, SourceUnit
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
