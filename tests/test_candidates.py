from dataclasses import replace

import pytest
from conftest import query

from clir_benchmark.adapters import query_from_row, select_candidates
from clir_benchmark.config import CandidateSelectionConfig
from clir_benchmark.errors import ValidationError


def test_exported_winner_is_honored():
    first = query(id="first", total_score=20)
    winner = query(id="winner", is_best=True, candidate_rank=2, total_score=19)
    selected, audit = select_candidates([first, winner])
    assert selected == [winner]
    assert audit[0]["method"] == "exported_is_best"


def test_grader_score_then_rank_then_id():
    queries = [
        query(id="a", total_score=10),
        query(id="z", total_score=20, candidate_rank=2),
        query(id="c", total_score=20),
        query(id="b", total_score=20),
    ]
    selected, audit = select_candidates(queries)
    assert selected[0].id == "b"
    assert audit[0]["method"] == "grader_total_score"
    assert len(select_candidates(queries, "all")[0]) == 4


def test_independent_tasks_and_generators():
    queries = [
        query(id="a"),
        query(id="b", generator_model_id="generator-b"),
        query(id="c", task_id="task-c"),
    ]
    assert len(select_candidates(queries)[0]) == 3


@pytest.mark.parametrize(
    "queries,match",
    [
        ([query(is_best=True), query(id="q2", is_best=True)], "Multiple"),
        ([query(task_id="")], "identity"),
        ([query(total_score=None)], "total_score"),
        ([query(), query()], "Duplicate"),
    ],
)
def test_invalid_selection_is_actionable(queries, match):
    with pytest.raises(ValidationError, match=match):
        select_candidates(queries)


def test_all_mode_does_not_require_scores():
    assert select_candidates([query(total_score=None)], "all")[0][0].total_score is None


def test_rank_one_is_not_assumed_best():
    low = query(candidate_rank=1, total_score=10, is_best=False)
    high = replace(low, id="high", candidate_rank=3, total_score=30)
    assert select_candidates([low, high])[0] == [high]


def test_adapter_generation_identity():
    import json

    row = {
        "_id": "q",
        "question": "question",
        "question_language": "en",
        "source_unit_id": "a:en",
        "generator_model_id": "g",
        "provenance_json": json.dumps({"file": "run.sqlite", "task": "task"}),
        "metadata_json": "{}",
    }
    legal = query_from_row(row, "un")
    assert "run.sqlite" in legal.task_id and "task" in legal.task_id
    row["metadata_json"] = json.dumps({"decider_routing_task": '["gp","patent","en"]'})
    chemistry = query_from_row(row, "chemistry")
    assert "patent" in chemistry.task_id
    row["provenance_json"] = "{}"
    with pytest.raises(ValidationError, match="identity"):
        query_from_row(row, "chemistry")


def test_document_selection_compares_all_generators_and_languages():
    low = query(id="low", is_best=True, candidate_rank=1, total_score=31)
    high = query(
        id="high",
        source_unit_id="a:fr",
        question_language="fr",
        total_score=32,
        generator_model_id="another-generator",
        candidate_rank=3,
    )
    other = query(id="other", source_unit_id="b:en", total_score=5)
    selected, audit = select_candidates(
        [low, other, high],
        "best-per-document",
        source_ids={"a:en": "a", "a:fr": "a", "b:en": "b"},
    )
    assert selected == [high, other]
    assert audit[0]["maximum_score_ids"] == ["high"]
    assert audit[0]["language_counts"] == {"en": 1, "fr": 1}
    assert audit[0]["candidates"][1]["disposition"] == "below_maximum"
    assert selected[0] is high  # Preserve the original question and all provenance.


def test_preferred_generators_are_an_equal_tier_only_among_maxima():
    policy = CandidateSelectionConfig()
    candidates = [
        query(id=f"preferred-{i}", generator_model_id=name, total_score=32)
        for i, name in enumerate(policy.preferred_generators)
    ] + [query(id="unpreferred", total_score=32)]
    selected, audit = select_candidates(
        candidates, "best-per-document", source_ids={"a:en": "a"}, policy=policy
    )
    assert len(selected) == 1
    assert audit[0]["random_pool_ids"] == ["preferred-0", "preferred-1", "preferred-2"]
    assert audit[0]["candidates"][-1]["disposition"] == "not_preferred_tie"
    higher = replace(candidates[-1], total_score=33)
    assert select_candidates(
        [*candidates[:-1], higher], "best-per-document", source_ids={"a:en": "a"}
    )[0] == [higher]


def test_document_ties_use_specified_seed_and_do_not_depend_on_input_order():
    import hashlib
    import json
    import random

    candidates = [query(id=f"q{i}", total_score=32) for i in range(7)]
    seed = int(
        hashlib.sha256(
            json.dumps([42, "eurlex", "a"], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        16,
    )
    expected = random.Random(seed).choice(sorted(candidates, key=lambda q: q.id))
    selected, audit = select_candidates(candidates, "best-per-document", source_ids={"a:en": "a"})
    assert selected == [expected]
    assert audit[0]["group_seed"] == seed
    assert select_candidates(
        list(reversed(candidates)), "best-per-document", source_ids={"a:en": "a"}
    ) == (selected, audit)
    extra = query(id="unrelated", source_unit_id="z:fr", question_language="fr")
    expanded, expanded_audit = select_candidates(
        [extra, *candidates], "best-per-document", source_ids={"a:en": "a", "z:fr": "z"}
    )
    assert expected in expanded
    assert expanded_audit[0] == audit[0]


def test_document_selection_audits_null_scores_and_documents_without_scores():
    missing = query(id="missing", total_score=None)
    scored = query(id="scored", total_score=1)
    unscored_document = query(id="unscored-document", source_unit_id="b:en", total_score=None)
    selected, audit = select_candidates(
        [missing, scored, unscored_document],
        "best-per-document",
        source_ids={"a:en": "a", "b:en": "b"},
    )
    assert selected == [scored]
    assert audit[0]["excluded_missing_score_count"] == 1
    assert audit[0]["candidates"][0]["disposition"] == "missing_score"
    assert audit[1]["status"] == "excluded_no_scored_candidates"
    assert audit[1]["maximum_score"] is None
    assert audit[1]["selected_ids"] == audit[1]["random_pool_ids"] == []


@pytest.mark.parametrize("mapping", [None, {}, {"a:en": ""}])
def test_document_selection_requires_canonical_mapping(mapping):
    with pytest.raises(ValidationError, match="canonical source"):
        select_candidates([query()], "best-per-document", source_ids=mapping)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "32"])
def test_document_selection_rejects_nonfinite_or_nonnumeric_score(score):
    with pytest.raises(ValidationError, match="finite and numeric"):
        select_candidates([query(total_score=score)], "best-per-document", source_ids={"a:en": "a"})


def test_document_selection_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="Duplicate QA"):
        select_candidates([query(), query()], "best-per-document", source_ids={"a:en": "a"})


def test_document_groups_do_not_merge_different_corpora():
    first = query()
    second = query(id="second", corpus="un", total_score=1)
    selected, _ = select_candidates([first, second], "best-per-document", source_ids={"a:en": "a"})
    assert selected == [first, second]
