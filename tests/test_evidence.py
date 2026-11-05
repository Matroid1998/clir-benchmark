import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import query, source
from test_relevance import make_store, setting

from clir_benchmark.adapters import apply_evidence_overlay, query_from_row
from clir_benchmark.artifacts import QUERY_SCHEMA, ParquetSink, query_record, read_queries
from clir_benchmark.errors import ValidationError
from clir_benchmark.evidence import ALIGNMENT_CONTRACT, parse_annotations
from clir_benchmark.relevance import judge


def native_span(text="preuve", source_unit_id="a:fr", **changes):
    return {
        "source_unit_id": source_unit_id,
        "language": None,
        "text": text,
        "anchor_id": "anchor-1",
        "origin": "cross_language_alignment",
        "alignment_id": "alignment-1",
        "validation_status": "pending_review",
        "match_status": "exact",
        "is_original": False,
        "model": "alignment-model",
        **changes,
    }


def native_query(spans=None, statuses=None, **changes):
    return query(
        evidence_all_languages={"fr": spans if spans is not None else [native_span()]},
        evidence_alignment_status={"fr": statuses or []},
        evidence_alignment_contract=ALIGNMENT_CONTRACT,
        evidence_alignment_eligibility="eligible",
        **changes,
    )


def test_published_spans_are_independent_and_preserve_provenance(tmp_path, chunker):
    units = [source(), source("a:fr", "preuve\nélément preuve", "fr")]
    spans = [native_span(), native_span("élément", anchor_id="anchor-2")]
    with make_store(tmp_path, chunker, units) as store:
        result = judge(native_query(spans), setting("fr"), store)
        assert result.status == "eligible"
        assert {(m.span, m.start, m.end) for m in result.matches} == {
            ("preuve", 0, 6),
            ("élément", 7, 14),
            ("preuve", 15, 21),
        }
        assert {m.annotation["anchor_id"] for m in result.matches} == {"anchor-1", "anchor-2"}
        assert all(m.annotation["validation_status"] == "pending_review" for m in result.matches)
        assert {m.field for m in result.matches} == {
            "evidence_all_languages.fr[0].text",
            "evidence_all_languages.fr[1].text",
        }
        assert all(units[1].text[m.start : m.end] == m.span for m in result.matches)


@pytest.mark.parametrize("text", ["Preuve", "preuve ", "préuve"])
def test_native_exact_flags_cannot_create_relevance(tmp_path, chunker, text):
    with make_store(tmp_path, chunker, [source(), source("a:fr", "preuve", "fr")]) as store:
        q = native_query(
            [native_span(text)],
            [{"source_unit_id": "a:fr", "status": "matched", "exact_match": True}],
        )
        result = judge(q, setting("fr"), store)
        assert result.status == "dropped"
        assert not result.matches


def test_native_span_must_fit_a_chunk(tmp_path, chunker):
    with make_store(
        tmp_path, chunker, [source(), source("a:fr", "0123456789abcdef", "fr")]
    ) as store:
        result = judge(native_query([native_span("56789abc")]), setting("fr"), store)
        assert result.status == "dropped"
        assert result.detail["unsupported"][0]["reason"] == "span_crosses_chunk_boundaries"


@pytest.mark.parametrize("status", ["not_found", "uncertain", "invalid"])
def test_failed_alignments_are_supplied_but_reasons_are_not_spans(tmp_path, chunker, status):
    outcome = {"source_unit_id": "a:fr", "status": status, "reason": "preuve"}
    q = native_query([], [outcome])
    with make_store(tmp_path, chunker, [source(), source("a:fr", "preuve", "fr")]) as store:
        result = judge(q, setting("fr"), store)
        assert result.status == "dropped"
        assert not result.matches
        assert result.detail["unsupported"][0]["alignment_status"] == {"a:fr": [outcome]}
        assert judge(q, setting("en", "fr"), store).status == "eligible"
        # A failed alignment never suppresses an independently matching raw field.
        assert judge(replace(q, answer="preuve"), setting("fr"), store).status == "eligible"


@pytest.mark.parametrize(
    "statuses",
    [
        [],
        [{"source_unit_id": "a:fr", "status": "matched"}],
        [{"source_unit_id": "a:fr", "status": "pending"}],
    ],
)
def test_empty_language_lists_and_incomplete_outcomes_block(tmp_path, chunker, statuses):
    with make_store(tmp_path, chunker, [source(), source("a:fr", "preuve", "fr")]) as store:
        q = native_query([], statuses)
        assert judge(q, setting("fr"), store).reason == "missing_multilingual_annotations"
        assert judge(q, setting("de"), store).reason == "unavailable_language_version"


@pytest.mark.parametrize("eligibility", ["no_exact_source_passage", "quarantined_source_language"])
def test_completed_alignment_exclusions_do_not_filter_raw_fields(tmp_path, chunker, eligibility):
    units = [source(), source("a:fr", "preuve", "fr"), source("a:de", "Beleg", "de")]
    q = replace(native_query([]), evidence_alignment_eligibility=eligibility)
    with make_store(tmp_path, chunker, units) as store:
        assert judge(q, setting("fr"), store).status == "dropped"
        assert judge(replace(q, answer="preuve"), setting("fr"), store).status == "eligible"
        # Only explicitly covered languages of the known contract are complete.
        assert judge(q, setting("de"), store).status == "blocked"
        assert (
            judge(replace(q, evidence_alignment_contract="unknown"), setting("fr"), store).status
            == "blocked"
        )


@pytest.mark.parametrize("status_only", [True, False])
@pytest.mark.parametrize("unit_id,language", [("unrelated:fr", "fr"), ("a:fr", "de")])
def test_native_spans_and_statuses_require_associated_source_language(
    tmp_path, chunker, status_only, unit_id, language
):
    q = query(
        **{
            "evidence_alignment_status" if status_only else "evidence_all_languages": {
                language: [
                    {"source_unit_id": unit_id, "status": "not_found"}
                    if status_only
                    else native_span(source_unit_id=unit_id)
                ]
            }
        }
    )
    with make_store(tmp_path, chunker, [source(), source("a:fr", "preuve", "fr")]) as store:
        with pytest.raises(ValidationError, match="associated source|language/source mismatch"):
            judge(q, setting("fr"), store)


@pytest.mark.parametrize(
    "span", [native_span(text=None), native_span(language="de"), native_span(source_unit_id=None)]
)
def test_native_shape_validation_is_actionable(span):
    with pytest.raises(ValidationError, match="QA q1"):
        parse_annotations(native_query([span]))


def test_multi_source_native_support_is_required_separately(tmp_path, chunker):
    units = [
        source(),
        source("a:fr", "preuve", "fr"),
        source("b:en", "other", source_id="b"),
        source("b:fr", "soutien", "fr", source_id="b"),
    ]
    q = native_query(involved_source_unit_ids=("a:en", "b:en"))
    with make_store(tmp_path, chunker, units) as store:
        assert judge(q, setting("fr"), store).status == "blocked"
        q = replace(
            q, evidence_alignment_status={"fr": [{"source_unit_id": "b:fr", "status": "not_found"}]}
        )
        assert judge(q, setting("fr"), store).status == "dropped"
        q = replace(
            q, evidence_all_languages={"fr": [native_span(), native_span("soutien", "b:fr")]}
        )
        result = judge(q, setting("fr"), store)
        assert result.status == "eligible"
        assert set().union(*result.relevant.values()) == {"a:en", "b:en"}


def test_adapter_and_parquet_preserve_all_alignment_fields(tmp_path):
    native = native_query(
        statuses=[{"source_unit_id": "a:fr", "status": "matched", "request_hash": "request"}]
    )
    row = {
        "_id": "q1",
        "question": "Find alpha",
        "question_language": "en",
        "source_unit_id": "a:en",
        "generator_model_id": "generator",
        "provenance_json": '{"file":"run.json","task":"task-1"}',
        **{
            name: getattr(native, name)
            for name in (
                "evidence_all_languages",
                "evidence_alignment_status",
                "evidence_alignment_contract",
                "evidence_alignment_eligibility",
            )
        },
        # Language-only convenience fields cannot invent source ownership.
        "evidence_fr": ["unattributed"],
    }
    adapted = query_from_row(row, "eurlex")
    assert adapted.evidence_all_languages == native.evidence_all_languages
    assert adapted.evidence_alignment_status == native.evidence_alignment_status
    path = tmp_path / "queries.parquet"
    with ParquetSink(path, QUERY_SCHEMA) as sink:
        sink.append(query_record(adapted))
    assert read_queries(path) == [adapted]
    assert "unattributed" not in str(parse_annotations(adapted))


def test_old_query_artifacts_remain_readable(tmp_path):
    q = query()
    record = {k: v for k, v in query_record(q).items() if not k.startswith("evidence_alignment_")}
    path = tmp_path / "old.parquet"
    pq.write_table(pa.Table.from_pylist([record]), path)
    assert read_queries(path) == [q]


def test_overlays_replace_spans_and_alignment_state_together(tmp_path):
    original = replace(native_query([]), evidence_alignment_eligibility="no_exact_source_passage")
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "evidence_all_languages": {"fr": [{"source_unit_id": "a:fr", "evidence": []}]},
            }
        )
    )
    overlaid = apply_evidence_overlay([original], path)[0]
    assert overlaid.answer == original.answer
    assert overlaid.evidence_alignment_status == {}
    assert overlaid.evidence_alignment_eligibility is None
    assert overlaid.evidence_alignment_contract is None
    fields = {
        name: getattr(original, name)
        for name in (
            "evidence_all_languages",
            "evidence_alignment_status",
            "evidence_alignment_contract",
            "evidence_alignment_eligibility",
        )
    }
    path.write_text(json.dumps({"query_id": "q1", **fields}))
    assert apply_evidence_overlay([original], path) == [original]


def test_legacy_aggregate_duplicates_still_fail():
    entry = {"source_unit_id": "a:fr", "evidence": []}
    with pytest.raises(ValidationError, match="duplicate annotation"):
        parse_annotations(query(evidence_all_languages={"fr": [entry, entry]}))
