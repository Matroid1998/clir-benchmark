"""Expand the verified releases' multilingual corpus records into source editions."""

from __future__ import annotations

from collections.abc import Iterator

from .adapters import DatasetReader, required_string
from .config import language_tag
from .errors import ValidationError
from .types import SourceUnit


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
