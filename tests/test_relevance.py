from dataclasses import replace

import pytest
from conftest import query, source

from clir_benchmark.artifacts import CorpusStore
from clir_benchmark.errors import ValidationError
from clir_benchmark.relevance import build_settings, judge, judge_documents, occurrences
from clir_benchmark.types import Setting


def make_store(tmp_path, chunker, units):
    store = CorpusStore(tmp_path / "lookup.sqlite", create=True)
    for unit in units:
        store.add_source(unit)
        for chunk in chunker.split(unit):
            store.add_chunk(chunk)
    store.db.commit()
    return store


def setting(*languages):
    return Setting(
        "test", "eurlex", "mixed" if len(languages) > 1 else "pairwise", "en", tuple(languages)
    )


@pytest.mark.parametrize(
    "answer,evidence", [("alpha", "no match"), ("no match", "alpha"), ("", "alpha"), ("alpha", "")]
)
def test_either_nonempty_field_qualifies(tmp_path, chunker, answer, evidence):
    with make_store(tmp_path, chunker, [source()]) as store:
        result = judge(query(answer=answer, evidence=evidence), setting("en"), store)
        assert result.status == "eligible"
        assert all(match.span == "alpha" for match in result.matches)


@pytest.mark.parametrize("span", ["Alpha", "alpha  beta", "alpha\nbeta", "<p>alpha</p>", "álpha"])
def test_no_normalization(tmp_path, chunker, span):
    with make_store(tmp_path, chunker, [source()]) as store:
        assert judge(query(answer=span), setting("en"), store).status == "dropped"


def test_empty_fields_never_match_but_nonempty_whitespace_is_literal(tmp_path, chunker):
    with make_store(tmp_path, chunker, [source()]) as store:
        assert judge(query(answer="", evidence=""), setting("en"), store).status == "dropped"
        assert judge(query(answer=" "), setting("en"), store).status == "eligible"


def test_repeated_and_overlapping_occurrences(tmp_path, chunker):
    unit = source(text="a" * 24)
    assert list(occurrences("aaa", "aa")) == [(0, 2), (1, 3)]
    with make_store(tmp_path, chunker, [unit]) as store:
        result = judge(query(answer="aa"), setting("en"), store)
        assert set(result.relevant) == {chunk.id for chunk in store.chunks(unit)}
        assert len(result.matches) > len(result.relevant)


def test_boundary_crossing_is_not_a_hit(tmp_path, chunker):
    with make_store(tmp_path, chunker, [source(text="abcdefghijklmnop")]) as store:
        result = judge(query(answer="abcdefghijkl"), setting("en"), store)
        assert result.status == "dropped"
        assert result.detail["unsupported"][0]["reason"] == "span_crosses_chunk_boundaries"


def test_no_matches_outside_associated_sources(tmp_path, chunker):
    units = [source(text="nothing"), source("b:en", "alpha", source_id="b")]
    with make_store(tmp_path, chunker, units) as store:
        result = judge(query(), setting("en"), store)
        assert result.status == "dropped"
        assert store.get_source("b:en") is not None


def test_every_indispensable_source_required(tmp_path, chunker):
    units = [source(text="alpha"), source("b:en", "beta", source_id="b")]
    with make_store(tmp_path, chunker, units) as store:
        multi = query(involved_source_unit_ids=("a:en", "b:en"), evidence="beta")
        result = judge(multi, setting("en"), store)
        assert result.status == "eligible"
        assert set().union(*result.relevant.values()) == {"a:en", "b:en"}
        assert judge(replace(multi, evidence="missing"), setting("en"), store).status == "dropped"


def test_missing_annotations_block_and_explicit_empty_can_drop(tmp_path, chunker):
    units = [source(), source("a:fr", "preuve", "fr")]
    with make_store(tmp_path, chunker, units) as store:
        assert judge(query(), setting("en"), store).status == "eligible"
        assert judge(query(), setting("fr"), store).status == "blocked"
        assert judge(query(), setting("en", "fr"), store).status == "blocked"
        annotated = query(
            evidence_all_languages={"fr": [{"source_unit_id": "a:fr", "evidence": []}]}
        )
        assert judge(annotated, setting("fr"), store).status == "dropped"
        assert judge(annotated, setting("en", "fr"), store).status == "eligible"
        annotated = replace(
            annotated,
            evidence_all_languages={"fr": [{"source_unit_id": "a:fr", "evidence": ["preuve"]}]},
        )
        assert judge(annotated, setting("fr"), store).status == "eligible"
        assert judge(annotated, setting("de"), store).reason == "unavailable_language_version"


def test_translated_source_specific_optional_answer(tmp_path, chunker):
    with make_store(tmp_path, chunker, [source(), source("a:fr", "preuve", "fr")]) as store:
        q = query(
            evidence_all_languages={
                "fr": [{"source_unit_id": "a:fr", "evidence": [], "answer": "preuve"}]
            }
        )
        result = judge(q, setting("fr"), store)
        assert result.status == "eligible"
        assert result.matches[0].field.endswith(".answer")


@pytest.mark.parametrize(
    "record",
    [
        {"fr": [{"source_unit_id": "b:fr", "evidence": ["preuve"]}]},
        {"de": [{"source_unit_id": "a:fr", "evidence": ["preuve"]}]},
        {"fr": [{"source_unit_id": "a:fr", "evidence": "preuve"}]},
    ],
)
def test_annotation_validation(tmp_path, chunker, record):
    with make_store(tmp_path, chunker, [source(), source("a:fr", "preuve", "fr")]) as store:
        with pytest.raises(ValidationError):
            judge(query(evidence_all_languages=record), setting("fr"), store)


def test_chemistry_direct_pairs_do_not_expand_transitively(tmp_path, chunker):
    units = [
        source("p_en", "alpha", "en", "p", "chemistry"),
        source("p_fr", "preuve", "fr", "p", "chemistry"),
        source("p_zh", "中文", "zh", "p", "chemistry"),
    ]
    with make_store(tmp_path, chunker, units) as store:
        store.add_pairs([("p_en", "p_fr"), ("p_fr", "p_zh")])
        assert set(store.variants(units[0])) == {"p_en", "p_fr"}
        q = query(corpus="chemistry", source_unit_id="p_en", involved_source_unit_ids=("p_en",))
        result = judge(q, setting("zh"), store)
        assert result.reason == "unavailable_language_version"


def test_settings_separate_monolingual_crosslingual_and_mixed():
    settings = build_settings("un", ["en"], ["en", "fr"], ("pairwise", "mixed"))
    assert [item.kind for item in settings] == ["monolingual", "cross_language", "mixed"]


def test_language_versions_are_one_required_source(tmp_path, chunker):
    units = [source(), source("a:fr", "preuve", "fr")]
    with make_store(tmp_path, chunker, units) as store:
        q = query(involved_source_unit_ids=("a:en", "a:fr"))
        result = judge(q, setting("en"), store)
        assert result.status == "eligible"
        assert result.required_source_ids == ["a:en"]


def test_mixed_support_can_come_from_different_languages(tmp_path, chunker):
    units = [
        source(text="alpha"),
        source("a:fr", "no support", "fr"),
        source("b:en", "no support", source_id="b"),
        source("b:fr", "preuve", "fr", source_id="b"),
    ]
    with make_store(tmp_path, chunker, units) as store:
        q = query(
            involved_source_unit_ids=("a:en", "b:en"),
            evidence_all_languages={
                "fr": [
                    {"source_unit_id": "a:fr", "evidence": []},
                    {"source_unit_id": "b:fr", "evidence": ["preuve"]},
                ]
            },
        )
        assert judge(q, setting("en"), store).status == "dropped"
        assert judge(q, setting("fr"), store).status == "dropped"
        result = judge(q, setting("en", "fr"), store)
        assert result.status == "eligible"
        assert {match.source_unit_id for match in result.matches} == {"a:en", "b:fr"}


def test_raw_answer_still_counts_with_an_explicit_empty_translation_record(tmp_path, chunker):
    with make_store(
        tmp_path, chunker, [source(text="2020"), source("a:fr", "2020", "fr")]
    ) as store:
        q = query(
            answer="2020",
            evidence_all_languages={"fr": [{"source_unit_id": "a:fr", "evidence": []}]},
        )
        assert judge(q, setting("fr"), store).status == "eligible"


@pytest.mark.parametrize(
    "answer,evidence", [("", ""), ("not present", "not present"), ("abcdefghijkl", "")]
)
def test_document_judgment_does_not_require_exact_chunk_spans(tmp_path, chunker, answer, evidence):
    with make_store(tmp_path, chunker, [source(text="abcdefghijklmnop")]) as store:
        q = query(answer=answer, evidence=evidence)
        assert judge(q, setting("en"), store).status == "dropped"
        result = judge_documents(q, setting("en"), store)
        assert result.status == "eligible"
        assert result.relevant == {"a:en": {"a:en"}}
        assert result.matches == []


def test_document_variants_need_no_evidence_alignment(tmp_path, chunker):
    units = [source(), source("a:fr", "preuve", "fr")]
    with make_store(tmp_path, chunker, units) as store:
        q = query(answer="", involved_source_unit_ids=("a:en", "a:fr"))
        result = judge_documents(q, setting("en", "fr"), store)
        assert result.required_source_ids == ["a:en"]
        assert result.relevant == {"a:en": {"a:en"}, "a:fr": {"a:en"}}
        assert judge_documents(q, setting("fr"), store).relevant == {"a:fr": {"a:en"}}
        assert judge_documents(q, setting("de"), store).reason == "unavailable_language_version"


def test_document_chemistry_relevance_only_uses_direct_accepted_pairs(tmp_path, chunker):
    units = [
        source("p_en", "alpha", "en", "p", "chemistry"),
        source("p_fr", "preuve", "fr", "p", "chemistry"),
        source("p_zh", "中文", "zh", "p", "chemistry"),
    ]
    with make_store(tmp_path, chunker, units) as store:
        store.add_pairs([("p_en", "p_fr"), ("p_fr", "p_zh")])
        q = query(corpus="chemistry", source_unit_id="p_en", involved_source_unit_ids=())
        assert judge_documents(q, setting("en", "fr", "zh"), store).relevant == {
            "p_en": {"p_en"},
            "p_fr": {"p_en"},
        }
        assert judge_documents(q, setting("zh"), store).reason == "unavailable_language_version"
        with pytest.raises(ValidationError, match="unsupported associations"):
            judge_documents(q, setting("en"), store, {"p_zh"})
        with pytest.raises(ValidationError, match="not directly associated"):
            judge_documents(replace(q, involved_source_unit_ids=("p_zh",)), setting("en"), store)


def test_document_judgment_requires_all_known_indispensable_sources(tmp_path, chunker):
    units = [source(), source("b:fr", "preuve", "fr", "b")]
    with make_store(tmp_path, chunker, units) as store:
        q = query(involved_source_unit_ids=("b:fr",), answer="")
        result = judge_documents(q, setting("en", "fr"), store)
        assert result.status == "eligible"
        assert result.relevant == {"a:en": {"a:en"}, "b:fr": {"b:fr"}}
        assert judge_documents(q, setting("en"), store).reason == "unavailable_language_version"
        assert (
            judge_documents(replace(q, unresolved_sources=True), setting("en"), store).reason
            == "unresolved_source_involvement"
        )
        assert (
            judge_documents(replace(q, source_unit_id="missing"), setting("en"), store).reason
            == "missing_source_unit"
        )
        with pytest.raises(ValidationError, match="unsupported associations"):
            judge_documents(q, setting("en", "fr"), store, {"unrelated"})


def test_document_without_indexed_chunks_is_not_a_retrieval_target(tmp_path, chunker):
    with make_store(tmp_path, chunker, [source()]) as store:
        store.db.execute("DELETE FROM chunks")
        result = judge_documents(query(), setting("en"), store)
        assert result.status == "dropped"
        assert result.reason == "missing_indexed_document"
